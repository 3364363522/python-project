import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# 0) 配置区：改这里就能跑
# =========================
POSTERIOR_SUMMARY_CSV = "/Users/garytchois/Desktop/w_0p5_修改版1/posterior_vote_share_summary.csv"
# 你在本环境里也可以用：
# POSTERIOR_SUMMARY_CSV = "/mnt/data/posterior_vote_share_summary.csv"

N_SIMS = 300          # 后验回放次数（越大CI越稳，先用100试跑）
SEED = 42

MECHANISMS = ["rank_sum", "percent_sum", "judge_save", "proposed"]

# proposed 机制参数（可在报告里解释/网格搜索）
PARAMS = {
    # 粉丝票凹变换：G(F) = (F+eps)^gamma / sum
    "gamma": 0.80,
    "fan_eps": 1e-6,

    # 动态权重 alpha_w（评委权重），早期高，后期低
    "alpha_high": 0.70,
    "alpha_low": 0.50,
    "alpha_schedule": "linear",  # "linear" or "logistic"
    # logistic 需要的参数（不用可忽略）
    "w0": None,   # 转折周（None=赛季中点）
    "k": 1.2,     # 转折陡峭度

    # judge_save 机制中，bottom2 的基准分数怎么取
    # "percent" = J占比 + F占比; "proposed" = 你新机制的S
    "base": "percent",

    # judges override 的执行方式
    "judge_mode": "deterministic",  # "deterministic" or "logit"
    "kappa": 5.0,                   # logit 模式下的“强硬程度”
}

# judges save 限次（每季最多几次“override”）
JUDGE_SAVE_LIMIT_PER_SEASON = 1

# 是否做后验采样（True=用q025/q975扰动；False=直接用p_mean）
SAMPLE_FAN = True

# 输出文件
OUT_SIM_CSV = "mechanism_eval_sims.csv"
OUT_SUMMARY_CSV = "mechanism_eval_summary.csv"


# ==================================
# 1) 数据结构与工具函数（高性能版本）
# ==================================
@dataclass
class WeekData:
    season: int
    week: int
    max_week: int
    contestant_id: np.ndarray
    judge_total: np.ndarray
    p_mean: np.ndarray
    p_q025: np.ndarray
    p_q975: np.ndarray


def rank_desc(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """返回名次：1=最好（值最大），并用id做确定性tie-break。"""
    order = np.lexsort((ids, -values))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def alpha_schedule(week: int, max_week: int,
                   alpha_high: float, alpha_low: float,
                   schedule: str = "linear",
                   w0: Optional[float] = None,
                   k: float = 1.0) -> float:
    """动态权重 alpha_w：早期偏评委，后期偏粉丝。"""
    if max_week <= 1:
        return alpha_high

    t = (week - 1) / (max_week - 1)

    if schedule == "linear":
        return alpha_high - (alpha_high - alpha_low) * t

    if schedule == "logistic":
        if w0 is None:
            w0 = (max_week + 1) / 2
        x = k * (w0 - week)
        sig = 1 / (1 + math.exp(-x))
        return alpha_low + (alpha_high - alpha_low) * sig

    raise ValueError("alpha_schedule must be 'linear' or 'logistic'")


def fan_transform_power(f: np.ndarray, gamma: float, eps: float) -> np.ndarray:
    """G(F) = (F+eps)^gamma 并归一化；0<gamma<1 压缩头部差距，eps解决边界效应。"""
    x = np.power(f + eps, gamma)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def sample_fan_lognormal(p_mean: np.ndarray, q025: np.ndarray, q975: np.ndarray,
                         rng: np.random.Generator, eps: float = 1e-12) -> np.ndarray:
    """
    用(q025,q975)拟合一个“对数尺度扰动强度”，对p_mean做 lognormal noise，再归一化到和为1。
    这样更贴近“投票占比在simplex上”的结构。
    """
    sigma = np.log(q975 / q025) / (2 * 1.96)
    z = rng.normal(0.0, 1.0, size=len(p_mean))
    w = p_mean * np.exp(sigma * z)
    w = np.maximum(w, eps)
    return w / w.sum()


def compute_proposed_S(judge_total: np.ndarray, fan_raw: np.ndarray,
                       week: int, max_week: int, params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """S = alpha*J + (1-alpha)*G(F)"""
    J = judge_total / judge_total.sum()
    Ftr = fan_transform_power(fan_raw, gamma=params["gamma"], eps=params.get("fan_eps", 1e-6))
    a = alpha_schedule(week, max_week,
                       params["alpha_high"], params["alpha_low"],
                       schedule=params.get("alpha_schedule", "linear"),
                       w0=params.get("w0", None),
                       k=params.get("k", 1.0))
    S = a * J + (1 - a) * Ftr
    return S, J, Ftr, a


# ==================================
# 2) 四种机制：每周淘汰（支持多淘汰周）
# ==================================
def elim_rank_sum_one(ids, judge_total, fan_raw) -> Tuple[int, int]:
    rj = rank_desc(judge_total, ids)
    rf = rank_desc(fan_raw, ids)
    rs = rj + rf
    # worst = 最大rank_sum；tie: judge_total小 -> fan小 -> id小
    order = np.lexsort((ids, fan_raw, judge_total, -rs))
    idx = order[0]
    return int(ids[idx]), idx


def elim_percent_sum_one(ids, judge_total, fan_raw) -> Tuple[int, int]:
    J = judge_total / judge_total.sum()
    S = J + fan_raw
    # worst = 最小S；tie: judge_total小 -> fan小 -> id小
    order = np.lexsort((ids, fan_raw, judge_total, S))
    idx = order[0]
    return int(ids[idx]), idx


def elim_judge_save_one(ids, judge_total, fan_raw,
                        week: int, max_week: int, params: dict,
                        rng: np.random.Generator,
                        save_remaining: Optional[int] = None) -> Tuple[int, int, int]:
    """
    bottom2 + judges override:
    - 先按 base 分数找 bottom2
    - 默认淘汰：bottom2里 base分更差者
    - 若 judges 认为另一个更该走（judge_total更低），且有save额度，则override（算一次save）
    """
    base = params.get("base", "percent")
    if base == "percent":
        S = judge_total / judge_total.sum() + fan_raw
    elif base == "proposed":
        S, _, _, _ = compute_proposed_S(judge_total, fan_raw, week, max_week, params)
    else:
        raise ValueError("params['base'] must be 'percent' or 'proposed'")

    order = np.lexsort((ids, S))
    bottom2 = order[:2]
    if len(bottom2) < 2:
        return int(ids[bottom2[0]]), bottom2[0], 0

    default_idx = bottom2[0]
    default_id = int(ids[default_idx])

    # judge更倾向淘汰 judge_total更低者
    j_order = bottom2[np.lexsort((ids[bottom2], judge_total[bottom2]))]
    judge_idx = j_order[0]
    judge_id = int(ids[judge_idx])

    # 无save额度：不能override
    if save_remaining is not None and save_remaining <= 0:
        return default_id, default_idx, 0

    # 不需要override
    if judge_id == default_id:
        return default_id, default_idx, 0

    # 需要override：用一次save
    mode = params.get("judge_mode", "deterministic")
    if mode == "deterministic":
        return judge_id, judge_idx, 1

    # 概率型：gap越大越接近必救（可写进论文：容错/不确定性）
    kappa = float(params.get("kappa", 5.0))
    j_low = judge_total[judge_idx]
    j_high = judge_total[j_order[1]]
    p = 1 / (1 + math.exp(-kappa * (j_high - j_low)))
    if rng.random() < p:
        return judge_id, judge_idx, 1
    return default_id, default_idx, 1


def elim_proposed_one(ids, judge_total, fan_raw,
                      week: int, max_week: int, params: dict,
                      rng: np.random.Generator,
                      save_remaining: Optional[int] = None) -> Tuple[int, int, int]:
    """
    proposed 机制（更“节目化”）：
    1) 用 proposed S 选 bottom3
    2) bottom3 里：粉丝（G(F)）最高者先“免死”（fan save）
    3) 剩下2人：默认淘汰 S 更差者；若 judge_total 指向另一人更该走且有save额度，则override（算一次save）
    """
    S, _, Ftr, _ = compute_proposed_S(judge_total, fan_raw, week, max_week, params)

    n = len(ids)
    k = min(3, n)
    order = np.lexsort((ids, S))
    bottom = order[:k]
    if k == 1:
        return int(ids[bottom[0]]), bottom[0], 0

    # 粉丝先救：在bottom里 Ftr 最大者
    fan_order = bottom[np.lexsort((ids[bottom], -Ftr[bottom]))]
    fan_saved = fan_order[0]

    remain = bottom[bottom != fan_saved]
    if len(remain) == 1:
        return int(ids[remain[0]]), remain[0], 0

    # 默认淘汰：S更差者
    def_order = remain[np.lexsort((ids[remain], S[remain]))]
    default_idx = def_order[0]
    default_id = int(ids[default_idx])

    # judge更倾向淘汰 judge_total更低者
    j_order = remain[np.lexsort((ids[remain], judge_total[remain]))]
    judge_idx = j_order[0]
    judge_id = int(ids[judge_idx])

    # 无save额度：不能override
    if save_remaining is not None and save_remaining <= 0:
        return default_id, default_idx, 0

    # 不需要override
    if judge_id == default_id:
        return default_id, default_idx, 0

    # 需要override：用一次save
    mode = params.get("judge_mode", "deterministic")
    if mode == "deterministic":
        return judge_id, judge_idx, 1

    kappa = float(params.get("kappa", 5.0))
    j_low = judge_total[judge_idx]
    j_high = judge_total[j_order[1]]
    p = 1 / (1 + math.exp(-kappa * (j_high - j_low)))
    if rng.random() < p:
        return judge_id, judge_idx, 1
    return default_id, default_idx, 1


def eliminate_multi(ids: np.ndarray, judge_total: np.ndarray, fan_raw: np.ndarray,
                    week: int, max_week: int,
                    mechanism: str, params: dict,
                    rng: np.random.Generator,
                    m: int,
                    save_remaining: Optional[int]) -> Tuple[List[int], int]:
    """
    支持一周淘汰 m 人（双淘汰/三淘汰周）。
    用“顺序淘汰近似”：每淘汰1人就从集合移除并重新归一化fan份额。
    """
    ids = ids.copy()
    judge_total = judge_total.copy()
    fan_raw = fan_raw.copy()

    eliminated = []
    saves_used = 0

    for _ in range(m):
        if len(ids) == 0:
            break

        if mechanism == "rank_sum":
            eid, idx = elim_rank_sum_one(ids, judge_total, fan_raw)
            used = 0
        elif mechanism == "percent_sum":
            eid, idx = elim_percent_sum_one(ids, judge_total, fan_raw)
            used = 0
        elif mechanism == "judge_save":
            eid, idx, used = elim_judge_save_one(ids, judge_total, fan_raw, week, max_week, params, rng, save_remaining)
        elif mechanism == "proposed":
            eid, idx, used = elim_proposed_one(ids, judge_total, fan_raw, week, max_week, params, rng, save_remaining)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        eliminated.append(eid)

        # remove
        ids = np.delete(ids, idx)
        judge_total = np.delete(judge_total, idx)
        fan_raw = np.delete(fan_raw, idx)

        # renormalize fan shares
        s = fan_raw.sum()
        if s > 0:
            fan_raw = fan_raw / s

        if save_remaining is not None:
            save_remaining -= used
        saves_used += used

    return eliminated, saves_used


# ==================================
# 3) 评估指标（公平性/节目性/对照历史）
# ==================================
def metrics_week(ids: np.ndarray, judge_total: np.ndarray, fan_raw: np.ndarray,
                 eliminated_ids: List[int],
                 mechanism: str,
                 week: int, max_week: int,
                 params: dict) -> Dict[str, float]:
    """
    - gap: bottom区域“分差”（越小通常越刺激）
    - upset_rate: 被淘汰者不是“最差合成分”的比例（有save/复杂机制才会>0）
    - merit_loss_rate: 被淘汰者在评委排名前半的比例（越低越公平）
    - controversy_rate: (评委前25%且粉丝后25%) 或 (粉丝前25%且评委后25%) 的比例
    """
    n = len(ids)
    rj = rank_desc(judge_total, ids)
    rf = rank_desc(fan_raw, ids)

    # “good”越大越好，用于gap与worst判定（这里只求相对即可）
    if mechanism == "rank_sum":
        good = -(rj + rf).astype(float)
    elif mechanism == "percent_sum":
        good = judge_total / judge_total.sum() + fan_raw
    elif mechanism == "judge_save":
        base = params.get("base", "percent")
        if base == "percent":
            good = judge_total / judge_total.sum() + fan_raw
        else:
            good, _, _, _ = compute_proposed_S(judge_total, fan_raw, week, max_week, params)
    else:  # proposed
        good, _, _, _ = compute_proposed_S(judge_total, fan_raw, week, max_week, params)

    gsort = np.sort(good)
    if n >= 3:
        gap = float(gsort[2] - gsort[0])   # third-worst - worst
    elif n >= 2:
        gap = float(gsort[1] - gsort[0])
    else:
        gap = float("nan")

    worst_id = int(ids[np.argmin(good)])
    upset_rate = float(np.mean([1 if eid != worst_id else 0 for eid in eliminated_ids])) if eliminated_ids else float("nan")

    id2i = {int(i): k for k, i in enumerate(ids)}
    top_half = math.ceil(n / 2)
    top_q = max(1, math.ceil(0.25 * n))
    bot_q = math.floor(0.75 * n) + 1

    merit = []
    contro = []
    for eid in eliminated_ids:
        i = id2i[eid]
        merit.append(1 if rj[i] <= top_half else 0)

        c = 0
        if rj[i] <= top_q and rf[i] >= bot_q:
            c = 1
        if rf[i] <= top_q and rj[i] >= bot_q:
            c = 1
        contro.append(c)

    return {
        "gap": gap,
        "upset_rate": upset_rate,
        "merit_loss_rate": float(np.mean(merit)) if merit else float("nan"),
        "controversy_rate": float(np.mean(contro)) if contro else float("nan"),
    }


# ==================================
# 4) 读取数据 + 构造“实际淘汰集合”（用于precision/recall）
# ==================================
def build_week_data(df: pd.DataFrame) -> Tuple[List[WeekData], Dict[int, int]]:
    df = df[["season", "week", "contestant_id", "judge_total", "p_mean", "p_q025", "p_q975"]].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["contestant_id"] = df["contestant_id"].astype(int)

    max_week = df.groupby("season")["week"].max().astype(int).to_dict()

    weeks: List[WeekData] = []
    for (s, w), g in df.groupby(["season", "week"]):
        weeks.append(
            WeekData(
                season=int(s),
                week=int(w),
                max_week=int(max_week[int(s)]),
                contestant_id=g["contestant_id"].to_numpy(int),
                judge_total=g["judge_total"].to_numpy(float),
                p_mean=g["p_mean"].to_numpy(float),
                p_q025=g["p_q025"].to_numpy(float),
                p_q975=g["p_q975"].to_numpy(float),
            )
        )

    weeks.sort(key=lambda x: (x.season, x.week))
    return weeks, max_week


def build_actual_elim_sets(df: pd.DataFrame) -> Dict[int, Dict[int, set]]:
    """
    用“某选手最后一次出现的week”近似其真实淘汰周：
    last_week < season_max_week => 该周被淘汰
    """
    elim: Dict[int, Dict[int, set]] = {}
    for season, sdf in df.groupby("season"):
        season = int(season)
        max_w = int(sdf["week"].max())
        last_week = sdf.groupby("contestant_id")["week"].max().astype(int)
        wk_map = last_week.reset_index().groupby("week")["contestant_id"].apply(list).to_dict()
        elim[season] = {w: set(ids) for w, ids in wk_map.items() if w < max_w}
    return elim


# ==================================
# 5) 主评估：后验回放 + CI + 输出
# ==================================
def run_evaluation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(POSTERIOR_SUMMARY_CSV)

    # 清理掉你文件里多余的 Unnamed 列
    keep = ["season", "week", "contestant_id", "judge_total", "p_mean", "p_q025", "p_q975"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["contestant_id"] = df["contestant_id"].astype(int)

    weeks, max_week_by_season = build_week_data(df)
    actual_elims = build_actual_elim_sets(df)

    # 只评估“真实有淘汰”的周（含双淘汰/三淘汰）
    elim_weeks = []
    for wd in weeks:
        actual_set = actual_elims.get(wd.season, {}).get(wd.week, set())
        if len(actual_set) > 0:
            elim_weeks.append(wd)

    rng0 = np.random.default_rng(SEED)
    sim_rows = []

    for sim in range(N_SIMS):
        rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))

        # 每季剩余save次数（只对 judge_save/proposed 有效）
        save_remaining = {m: {s: JUDGE_SAVE_LIMIT_PER_SEASON for s in max_week_by_season} for m in MECHANISMS}

        # 每个机制累计指标（按周平均）
        acc = {
            m: {"gap": [], "upset_rate": [], "merit_loss_rate": [], "controversy_rate": [],
                "precision": [], "recall": []}
            for m in MECHANISMS
        }

        for wd in elim_weeks:
            # 采样 fan share（同一份采样用于所有机制，保证公平对照）
            if SAMPLE_FAN:
                fan = sample_fan_lognormal(wd.p_mean, wd.p_q025, wd.p_q975, rng)
            else:
                fan = wd.p_mean / wd.p_mean.sum()

            actual_set = actual_elims[wd.season].get(wd.week, set())
            mcount = len(actual_set)

            for mech in MECHANISMS:
                sr = save_remaining[mech][wd.season] if mech in ("judge_save", "proposed") else None

                elim_ids, used = eliminate_multi(
                    wd.contestant_id, wd.judge_total, fan,
                    wd.week, wd.max_week,
                    mech, PARAMS, rng,
                    m=mcount,
                    save_remaining=sr
                )

                if mech in ("judge_save", "proposed"):
                    save_remaining[mech][wd.season] -= used

                met = metrics_week(
                    wd.contestant_id, wd.judge_total, fan,
                    elim_ids, mech, wd.week, wd.max_week, PARAMS
                )

                for k in ["gap", "upset_rate", "merit_loss_rate", "controversy_rate"]:
                    acc[mech][k].append(met[k])

                # accuracy vs actual (set-wise)
                pred_set = set(elim_ids)
                inter = len(pred_set & actual_set)
                precision = inter / len(pred_set) if len(pred_set) else float("nan")
                recall = inter / len(actual_set) if len(actual_set) else float("nan")

                acc[mech]["precision"].append(precision)
                acc[mech]["recall"].append(recall)

        # 汇总本次sim
        for mech in MECHANISMS:
            row = {"sim": sim, "mechanism": mech}
            for metric, arr in acc[mech].items():
                row[metric] = float(np.nanmean(arr)) if len(arr) else float("nan")
            sim_rows.append(row)

    sim_df = pd.DataFrame(sim_rows)

    # 跨sim给出 mean + 95% CI
    summary = []
    metrics = ["gap", "upset_rate", "merit_loss_rate", "controversy_rate", "precision", "recall"]
    for mech, sdf in sim_df.groupby("mechanism"):
        for metric in metrics:
            vals = sdf[metric].to_numpy(float)
            summary.append({
                "mechanism": mech,
                "metric": metric,
                "mean": float(np.nanmean(vals)),
                "q025": float(np.nanquantile(vals, 0.025)),
                "q975": float(np.nanquantile(vals, 0.975)),
            })

    summary_df = pd.DataFrame(summary)

    # 保存
    sim_df.to_csv(OUT_SIM_CSV, index=False)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    return summary_df, sim_df


if __name__ == "__main__":
    summary_df, _ = run_evaluation()
    print("\n=== DONE ===")
    print(summary_df.sort_values(["metric", "mechanism"]).to_string(index=False))
    print(f"\nSaved:\n- {OUT_SIM_CSV}\n- {OUT_SUMMARY_CSV}")
