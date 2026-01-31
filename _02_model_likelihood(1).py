import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple

CSV_PATH = "/Users/garytchois/Desktop/vs/2026_MCM_Problem_C_Data_with_week_stats1.csv"

# ---------- 0) 规则：按赛季切换 percent / rank ----------
def season_rule(season: int) -> str:
    if season in (1, 2) or season >= 28:
        return "rank"
    return "percent"

# ---------- 1) 读入 (保持不变) ----------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["season", "celebrity_name"]).reset_index(drop=True)
    df["contestant_id"] = df.groupby("season").cumcount()
    return df

# ---------- 2) 取出每周 judge_total (保持不变) ----------
def get_week_list(df: pd.DataFrame) -> list[int]:
    weeks = []
    for c in df.columns:
        m = re.fullmatch(r"week(\d+)_judge_score_sum", c)
        if m:
            weeks.append(int(m.group(1)))
    return sorted(set(weeks))

def build_judge_total_long(df: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
    rows = []
    for w in weeks:
        col = f"week{w}_judge_score_sum"
        if col not in df.columns: continue
        tmp = df[["season", "contestant_id", "celebrity_name", col]].copy()
        tmp = tmp.rename(columns={col: "judge_total"})
        tmp["week"] = w
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True)
    long["judge_total"] = pd.to_numeric(long["judge_total"], errors="coerce")
    return long

# ---------- 3) 解析 results (保持不变) ----------
def parse_elim_week(results: str) -> int | None:
    m = re.search(r"Eliminated Week (\d+)", str(results))
    return int(m.group(1)) if m else None

def is_withdrew(results: str) -> bool:
    return str(results).strip().lower() == "withdrew"

# ---------- 4) 构造 withdrew (保持不变) ----------
def infer_withdrew_week(df: pd.DataFrame, judge_long: pd.DataFrame, weeks: list[int]) -> dict[tuple[int,int], int]:
    withdrew_people = df[df["results"].apply(is_withdrew)][["season", "contestant_id"]]
    withdrew_set = set(map(tuple, withdrew_people.values.tolist()))
    if not withdrew_set: return {}
    pivot = judge_long.pivot_table(index=["season", "contestant_id"], columns="week", values="judge_total", aggfunc="first")
    out = {}
    for key in withdrew_set:
        if key not in pivot.index: continue
        series = pivot.loc[key]
        active_weeks = [w for w in weeks if (w in series.index and pd.notna(series[w]) and series[w] > 0)]
        if active_weeks: out[key] = max(active_weeks)
    return out

# ---------- 5) 事件对象 (保持不变) ----------
@dataclass
class WeekEvent:
    season: int
    week: int
    rule: str
    active_ids: list[int]
    J: np.ndarray
    zJ: np.ndarray
    j_percent: np.ndarray
    eliminated_ids: list[int]
    skip_likelihood: bool
    note: str

def build_events(df: pd.DataFrame, judge_long: pd.DataFrame, weeks: list[int]) -> list[WeekEvent]:
    df2 = df.copy()
    df2["elim_week"] = df2["results"].apply(parse_elim_week)
    elim_map = (df2.dropna(subset=["elim_week"]).groupby(["season", "elim_week"])["contestant_id"].apply(list).to_dict())
    withdrew_week = infer_withdrew_week(df2, judge_long, weeks)
    active_long = judge_long[(judge_long["judge_total"].notna()) & (judge_long["judge_total"] > 0)]
    roster_map = (active_long.groupby(["season", "week"])["contestant_id"].apply(list).to_dict())
    jt_map = {(int(r.season), int(r.week), int(r.contestant_id)): float(r.judge_total) for r in judge_long.itertuples(index=False) if pd.notna(r.judge_total)}

    events: list[WeekEvent] = []
    seasons = sorted(df2["season"].unique().tolist())
    for s in seasons:
        for w in weeks:
            active = roster_map.get((s, w), [])
            if len(active) == 0: continue
            J = np.array([jt_map.get((s, w, i), np.nan) for i in active], dtype=float)
            m = np.nanmean(J)
            sd = np.nanstd(J)
            zJ = (J - m) / (sd + 1e-8)
            sumJ = np.nansum(J)
            j_percent = J / (sumJ + 1e-12)
            eliminated = elim_map.get((s, w), [])
            rule = season_rule(int(s))
            skip = False
            note = ""
            for (ss, cid), ww in withdrew_week.items():
                if ss == s and ww == w:
                    skip = True
                    note = "withdrew"
                    break
            events.append(WeekEvent(season=int(s), week=int(w), rule=rule, active_ids=active, J=J, zJ=zJ, j_percent=j_percent, eliminated_ids=eliminated, skip_likelihood=skip, note=note))
    return events

# ========= 工具函数 (保持不变) =========
def logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m)) + 1e-300))

def log_softmax(a: np.ndarray) -> np.ndarray:
    return a - logsumexp(a)

def softmax(a: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(a))

def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    uniq = {}
    for i, v in enumerate(values): uniq.setdefault(v, []).append(i)
    for v, idxs in uniq.items():
        if len(idxs) > 1: ranks[idxs] = float(np.mean(ranks[idxs]))
    return ranks

def vote_share(mu_vec: np.ndarray, zJ: np.ndarray, gamma: float) -> np.ndarray:
    return softmax(mu_vec + gamma * zJ)

def hazard_percent(j_percent: np.ndarray, p: np.ndarray, w: float = 0.5) -> np.ndarray:
    combined = w * j_percent + (1.0 - w) * p
    return -combined

def hazard_rank(J: np.ndarray, p: np.ndarray) -> np.ndarray:
    rj = rank_desc(J)
    rp = rank_desc(p)
    return 0.5 * (rj + rp)

# ========= 核心修改区域 =========

def elimination_loglik_for_event(event, mu_s: np.ndarray, gamma: float, kappa: float, w_percent: float = 0.5) -> float:
    """
    通用似然计算，注意这里只接受单一 kappa，具体传哪个由 posterior 决定
    """
    if getattr(event, "skip_likelihood", False): return 0.0
    eliminated_ids = list(event.eliminated_ids)
    if len(eliminated_ids) == 0: return 0.0
    remaining = list(event.active_ids)
    ll = 0.0
    for e_id in eliminated_ids:
        if e_id not in remaining: return -np.inf
        # 映射
        active_map = {cid: i for i, cid in enumerate(event.active_ids)}
        sub = np.array([active_map[cid] for cid in remaining], dtype=int)
        
        mu_vec = mu_s[np.array(remaining, dtype=int)]
        J_sub = event.J[sub]
        zJ_sub = event.zJ[sub]
        jperc_sub = event.j_percent[sub]
        
        p_sub = vote_share(mu_vec, zJ_sub, gamma)
        
        if event.rule == "percent":
            hz = hazard_percent(jperc_sub, p_sub, w=w_percent)
        elif event.rule == "rank":
            hz = hazard_rank(J_sub, p_sub)
        else:
            raise ValueError(f"Unknown rule: {event.rule}")
            
        log_q = log_softmax(kappa * hz)
        e_pos = remaining.index(e_id)
        ll += float(log_q[e_pos])
        remaining.remove(e_id)
    return ll

def center_mu(mu: np.ndarray) -> np.ndarray:
    return mu - np.mean(mu)

def log_prior(mu_by_season: Dict[int, np.ndarray], gamma: float, 
              log_kappa_strict: float, log_kappa_fuzzy: float,
              sigma_mu: float = 3.0) -> float: # <--- 这里的默认值改为了 3.0，允许更大的波动
    """
    Modified Prior:
    1. sum(log_kappa_strict + log_kappa_fuzzy)
    2. sigma_mu relaxed to allow viral stars
    """
    mu_all = np.concatenate([mu_by_season[s] for s in sorted(mu_by_season.keys())])
    lp = 0.0
    lp += -0.5 * np.sum((mu_all / sigma_mu) ** 2) # Relaxed prior
    lp += -0.5 * (gamma ** 2)
    lp += -0.5 * (log_kappa_strict ** 2) # Prior for strict kappa
    lp += -0.5 * (log_kappa_fuzzy ** 2)  # Prior for fuzzy kappa
    return float(lp)

def log_posterior(mu_by_season: Dict[int, np.ndarray], gamma: float, 
                  log_kappa_strict: float, log_kappa_fuzzy: float, 
                  events: list,
                  sigma_mu: float = 3.0, w_percent: float = 0.5) -> float:
    
    kappa_strict = float(np.exp(log_kappa_strict))
    kappa_fuzzy = float(np.exp(log_kappa_fuzzy))
    
    lp = log_prior(mu_by_season, gamma, log_kappa_strict, log_kappa_fuzzy, sigma_mu=sigma_mu)

    ll = 0.0
    for ev in events:
        mu_s = mu_by_season[ev.season]
        
        # --- 核心逻辑: 根据赛季选择 Kappa ---
        # S28+ (包含28) 使用 fuzzy kappa (因为有 judge save)
        # S1-27 使用 strict kappa
        if ev.season >= 28:
            k = kappa_fuzzy
        else:
            k = kappa_strict
            
        ll += elimination_loglik_for_event(ev, mu_s=mu_s, gamma=gamma, kappa=k, w_percent=w_percent)

    return lp + ll