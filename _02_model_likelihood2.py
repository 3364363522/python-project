# _02_model_likelihood.py
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict


# ---------- 0) 规则：按赛季切换 percent / rank ----------
def season_rule(season: int) -> str:
    """
    题面合理假设：
    - S1-2: rank
    - S3-27: percent
    - S28-34: 回到 rank（并且有 bottom2 + judge save）
    """
    if season in (1, 2) or season >= 28:
        return "rank"
    return "percent"


# ---------- 1) 读入 ----------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["season", "celebrity_name"]).reset_index(drop=True)
    df["contestant_id"] = df.groupby("season").cumcount()
    return df


# ---------- 2) 取 week{k} 列表 ----------
def get_week_list(df: pd.DataFrame) -> list[int]:
    weeks = []
    for c in df.columns:
        m = re.fullmatch(r"week(\d+)_judge_score_sum", c)
        if m:
            weeks.append(int(m.group(1)))
    return sorted(set(weeks))


# ---------- 3) judge_total long ----------
def build_judge_total_long(df: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
    rows = []
    for w in weeks:
        col = f"week{w}_judge_score_sum"
        if col not in df.columns:
            continue
        tmp = df[["season", "contestant_id", "celebrity_name", col]].copy()
        tmp = tmp.rename(columns={col: "judge_total"})
        tmp["week"] = w
        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)
    long["judge_total"] = pd.to_numeric(long["judge_total"], errors="coerce")
    return long


# ---------- 4) 解析 results：淘汰周 / withdrew ----------
def parse_elim_week(results: str) -> int | None:
    m = re.search(r"Eliminated Week (\d+)", str(results))
    return int(m.group(1)) if m else None


def is_withdrew(results: str) -> bool:
    return str(results).strip().lower() == "withdrew"


# ---------- 5) 构造 withdrew 的“发生周” ----------
def infer_withdrew_week(df: pd.DataFrame, judge_long: pd.DataFrame, weeks: list[int]) -> dict[tuple[int, int], int]:
    withdrew_people = df[df["results"].apply(is_withdrew)][["season", "contestant_id"]]
    withdrew_set = set(map(tuple, withdrew_people.values.tolist()))
    if not withdrew_set:
        return {}

    pivot = judge_long.pivot_table(
        index=["season", "contestant_id"],
        columns="week",
        values="judge_total",
        aggfunc="first",
    )

    out = {}
    for key in withdrew_set:
        if key not in pivot.index:
            continue
        series = pivot.loc[key]
        active_weeks = [w for w in weeks if (w in series.index and pd.notna(series[w]) and series[w] > 0)]
        if active_weeks:
            out[key] = max(active_weeks)
    return out


# ---------- 6) 事件对象 ----------
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

    elim_map = (
        df2.dropna(subset=["elim_week"])
        .groupby(["season", "elim_week"])["contestant_id"]
        .apply(list)
        .to_dict()
    )

    withdrew_week = infer_withdrew_week(df2, judge_long, weeks)

    # roster：每赛季每周 active_ids（judge_total > 0）
    active_long = judge_long[(judge_long["judge_total"].notna()) & (judge_long["judge_total"] > 0)]
    roster_map = (
        active_long.groupby(["season", "week"])["contestant_id"]
        .apply(list)
        .to_dict()
    )

    jt_map = {
        (int(r.season), int(r.week), int(r.contestant_id)): float(r.judge_total)
        for r in judge_long.itertuples(index=False)
        if pd.notna(r.judge_total)
    }

    events: list[WeekEvent] = []
    seasons = sorted(df2["season"].unique().tolist())

    for s in seasons:
        for w in weeks:
            active = roster_map.get((s, w), [])
            if len(active) == 0:
                continue

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

            events.append(
                WeekEvent(
                    season=int(s),
                    week=int(w),
                    rule=rule,
                    active_ids=active,
                    J=J,
                    zJ=zJ,
                    j_percent=j_percent,
                    eliminated_ids=eliminated,
                    skip_likelihood=skip,
                    note=note,
                )
            )
    return events


# ========= 工具函数 =========
def logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m)) + 1e-300))


def log_softmax(a: np.ndarray) -> np.ndarray:
    return a - logsumexp(a)


def softmax(a: np.ndarray) -> np.ndarray:
    ls = log_softmax(a)
    return np.exp(ls)


def rank_desc(values: np.ndarray) -> np.ndarray:
    """
    返回名次（1=最好，n=最差），按 values 从大到小排序；ties 用平均名次。
    """
    order = np.argsort(-values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    uniq = {}
    for i, v in enumerate(values):
        uniq.setdefault(v, []).append(i)
    for v, idxs in uniq.items():
        if len(idxs) > 1:
            avg = float(np.mean(ranks[idxs]))
            ranks[idxs] = avg
    return ranks


# ========= 由 (mu, gamma) 得到 vote share =========
def vote_share(mu_vec: np.ndarray, zJ: np.ndarray, gamma: float) -> np.ndarray:
    """
    p_i = softmax(mu_i + gamma * zJ_i)
    """
    return softmax(mu_vec + gamma * zJ)


# ========= 两套规则的 hazard =========
def hazard_percent(j_percent: np.ndarray, p: np.ndarray, w: float = 0.5) -> np.ndarray:
    combined = w * j_percent + (1.0 - w) * p
    return -combined  # 越大越危险


def hazard_rank(J: np.ndarray, p: np.ndarray) -> np.ndarray:
    rj = rank_desc(J)  # J 越大越好
    rp = rank_desc(p)  # p 越大越好
    return 0.5 * (rj + rp)  # 越大越危险


def _bottom2_judgesave_logprob_one_elim(
    remaining_ids: List[int],
    event: WeekEvent,
    mu_s: np.ndarray,
    gamma: float,
    kappa_bottom2: float,
    kappa_save: float,
    w_percent: float,
    elim_id: int,
) -> float:
    """
    S28+ 机制（本周一次淘汰）：
    1) 用 combined hazard 选 bottom2（用 kappa_bottom2 控制“像不像硬规则”）
       - 用两步“选最危险、再选次危险”的 Plackett–Luce 形式（允许噪声）
    2) 在 bottom2 中 judges save：更低 judge_total 更可能被淘汰
       - 用 kappa_save 控制“裁判有多确定”

    返回 log P(elim_id | remaining)
    """
    if elim_id not in remaining_ids:
        return -np.inf

    # 在 event.active_ids 的位置映射
    active_map = {cid: i for i, cid in enumerate(event.active_ids)}
    sub = np.array([active_map[cid] for cid in remaining_ids], dtype=int)

    J = event.J[sub]
    zJ = event.zJ[sub]
    jperc = event.j_percent[sub]

    mu_vec = mu_s[np.array(remaining_ids, dtype=int)]
    p = vote_share(mu_vec, zJ, gamma)

    if event.rule == "percent":
        hz = hazard_percent(jperc, p, w=w_percent)
    elif event.rule == "rank":
        hz = hazard_rank(J, p)
    else:
        raise ValueError(f"Unknown rule: {event.rule}")

    # Step 1: 选 bottom2（两步 PL）
    # logp1[i] = log P(first=b1=i)
    logp1 = log_softmax(kappa_bottom2 * hz)

    # 对 elim 的总概率做 log-sum-exp 累积
    log_q_elim = -np.inf

    n = len(remaining_ids)
    idx_map = {cid: i for i, cid in enumerate(remaining_ids)}
    all_idx = np.arange(n)
    for i in range(n):
        b1_id = remaining_ids[i]

        # second 从剩下 n-1 个里选
        rem2_idx = np.delete(all_idx, i)
        hz2 = hz[rem2_idx]
        rem2_ids = [remaining_ids[k] for k in rem2_idx.tolist()]

        logp2 = log_softmax(kappa_bottom2 * hz2)

        for j, b2_id in enumerate(rem2_ids):
            # Step 2: judges save（淘汰发生在 {b1,b2} 内）
            # 用 judge_total 决定更可能被淘汰：分低更危险 => hazard_save = -J
            # 注意：这里取 pair 的 J
            k2 = idx_map[b2_id]
            J_pair = np.array([J[i], J[k2]], dtype=float)

            hz_save = -J_pair  # 分越低，hazard越大
            logp_save = log_softmax(kappa_save * hz_save)  # [P(elim=b1), P(elim=b2)] 的 log

            # elim_id 在 pair 里才有贡献
            if elim_id == b1_id:
                log_term = float(logp1[i] + logp2[j] + logp_save[0])
                log_q_elim = np.logaddexp(log_q_elim, log_term)
            elif elim_id == b2_id:
                log_term = float(logp1[i] + logp2[j] + logp_save[1])
                log_q_elim = np.logaddexp(log_q_elim, log_term)

    return float(log_q_elim)


# ========= Likelihood：两段赛制 =========
def elimination_loglik_for_event(
    event: WeekEvent,
    mu_s: np.ndarray,
    gamma: float,
    kappa_strict: float,
    kappa_fuzzy: float,
    w_percent: float = 0.5,
) -> float:
    """
    - S1-27：硬规则 -> 用较高 kappa_strict 的 Plackett–Luce（你原本逻辑）
    - S28+ ：Bottom2 + Judge Save -> 用 kappa_fuzzy 做 bottom2 选择强度 + save 强度（更“软”）
    """
    if getattr(event, "skip_likelihood", False):
        return 0.0

    eliminated_ids: List[int] = list(event.eliminated_ids)
    if len(eliminated_ids) == 0:
        return 0.0

    # --- S1-27: 原始 PL 淘汰 ---
    if event.season <= 27:
        remaining: List[int] = list(event.active_ids)
        ll = 0.0

        for e_id in eliminated_ids:
            if e_id not in remaining:
                return -np.inf

            # remaining 在 event.active_ids 的子集索引
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

            log_q = log_softmax(kappa_strict * hz)

            e_pos = remaining.index(e_id)
            ll += float(log_q[e_pos])

            remaining.remove(e_id)

        return float(ll)

    # --- S28+: Bottom2 + Judge Save ---
    # 允许多淘汰时：按顺序逐个淘汰（近似处理）
    remaining = list(event.active_ids)
    ll = 0.0
    for e_id in eliminated_ids:
        lp = _bottom2_judgesave_logprob_one_elim(
            remaining_ids=remaining,
            event=event,
            mu_s=mu_s,
            gamma=gamma,
            kappa_bottom2=kappa_fuzzy,
            kappa_save=kappa_fuzzy,
            w_percent=w_percent,
            elim_id=e_id,
        )
        if not np.isfinite(lp):
            return -np.inf
        ll += float(lp)
        remaining.remove(e_id)

    return float(ll)


# ========= 先验 + 总 log posterior =========
def center_mu(mu: np.ndarray) -> np.ndarray:
    return mu - np.mean(mu)


def log_prior(
    mu_by_season: Dict[int, np.ndarray],
    gamma: float,
    log_kappa_strict: float,
    log_kappa_fuzzy: float,
    sigma_mu: float = 4.0,  # ✅ 放宽基础人气方差（可改成 3/5）
    # ✅ strict 偏大（硬规则），fuzzy 偏小（软规则）
    kappa_strict_loc: float = np.log(15.0),
    kappa_strict_scale: float = 0.6,
    kappa_fuzzy_loc: float = np.log(2.5),
    kappa_fuzzy_scale: float = 0.7,
) -> float:
    """
    mu_si ~ N(0, sigma_mu^2)  (proposal 中强制每赛季 sum(mu)=0)
    gamma ~ N(0,1)

    log_kappa_strict ~ N(log(15), 0.6^2)   -> 常见会落到 10+
    log_kappa_fuzzy  ~ N(log(2.5), 0.7^2)  -> 常见落到 2-4
    """
    mu_all = np.concatenate([mu_by_season[s] for s in sorted(mu_by_season.keys())])
    lp = 0.0
    lp += -0.5 * np.sum((mu_all / sigma_mu) ** 2)
    lp += -0.5 * (gamma ** 2)

    lp += -0.5 * ((log_kappa_strict - kappa_strict_loc) / kappa_strict_scale) ** 2
    lp += -0.5 * ((log_kappa_fuzzy - kappa_fuzzy_loc) / kappa_fuzzy_scale) ** 2
    return float(lp)


def log_posterior(
    mu_by_season: Dict[int, np.ndarray],
    gamma: float,
    log_kappa_strict: float,
    log_kappa_fuzzy: float,
    events: list[WeekEvent],
    sigma_mu: float = 4.0,
    w_percent: float = 0.5,
) -> float:
    kappa_strict = float(np.exp(log_kappa_strict))
    kappa_fuzzy = float(np.exp(log_kappa_fuzzy))

    lp = log_prior(
        mu_by_season,
        gamma,
        log_kappa_strict,
        log_kappa_fuzzy,
        sigma_mu=sigma_mu,
    )

    ll = 0.0
    for ev in events:
        mu_s = mu_by_season[ev.season]
        ll += elimination_loglik_for_event(
            ev,
            mu_s=mu_s,
            gamma=gamma,
            kappa_strict=kappa_strict,
            kappa_fuzzy=kappa_fuzzy,
            w_percent=w_percent,
        )

    return float(lp + ll)
