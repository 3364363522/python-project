# dwts_mcmc_fan_votes_singlefile.py
# ------------------------------------------------------------
# Single-file MCMC for inferring latent fan vote shares (and scaled votes)
# with:
#   - Seasons 1-27: "strict" elimination (direct lowest combined) with kappa_strict
#   - Seasons 28+: "bottom2 + judge save" with kappa_fuzzy (bottom2 uncertainty) + (optional) kappa_judge
#   - Relaxed prior on popularity mu via learnable sigma_mu (log_sigma_mu) with wide prior
#
# Outputs:
#   mcmc_outputs_single/
#       posterior_global_params.csv
#       posterior_popularity_summary.csv
#       posterior_vote_share_summary.csv
#       weekly_elimination_check.csv
#       method_comparison_weekly.csv          <-- (NEW) Q2: rank vs percent comparison + bias + disagreement
#       mcmc_diagnostics.json
#
# Notes:
# - This infers vote *shares* p_{s,w,i}. To get "fan votes", we also output a scaled vote estimate:
#       votes_est = p_mean * TOTAL_FAN_VOTES_PER_WEEK
#   You can change TOTAL_FAN_VOTES_PER_WEEK to any convenient scale (e.g., 10_000_000).
# - For Q2 analysis, we DO NOT sample from CI independently. We use the posterior draws directly:
#   each saved MCMC draw produces one p-vector per (season, week); we compute both methods and
#   build distributions of correlations and disagreement probabilities across draws.
# ------------------------------------------------------------

import os
import re
import json
import math
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============== User settings ==============
# Prefer env var DWTS_CSV; fallback to your local path.
CSV_PATH = os.environ.get(
    "DWTS_CSV",
    "/Users/garytchois/Desktop/vs/2026_MCM_Problem_C_Data_with_week_stats1.csv",
)

# Output dir
BASE_OUTDIR = Path(__file__).resolve().parent / "mcmc_outputs_single"
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

# Scale vote share to pseudo "fan votes"
TOTAL_FAN_VOTES_PER_WEEK = float(os.environ.get("DWTS_TOTAL_VOTES", "10000000"))  # 10 million default

# Assumption about method switch:
# S1-2: rank, S3-27: percent, S28+: back to rank + bottom2/judge-save (reasonable assumption)
SEASON_JUDGESAVE_START = 28  # you can change if needed


# ---------------- 0) season rule ----------------
def season_rule(season: int) -> str:
    if season in (1, 2) or season >= SEASON_JUDGESAVE_START:
        return "rank"
    return "percent"


# ---------------- 1) load data ----------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure stable contestant_id within each season
    df = df.sort_values(["season", "celebrity_name"]).reset_index(drop=True)
    df["contestant_id"] = df.groupby("season").cumcount()
    return df


# ---------------- 2) infer weeks & compute judge totals ----------------
_WEEK_SUM_RE = re.compile(r"^week(\d+)_judge_score_sum$")
_WEEK_JUDGE_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")


def get_week_list(df: pd.DataFrame) -> List[int]:
    weeks = set()

    for c in df.columns:
        m1 = _WEEK_SUM_RE.fullmatch(c)
        if m1:
            weeks.add(int(m1.group(1)))
            continue
        m2 = _WEEK_JUDGE_RE.fullmatch(c)
        if m2:
            weeks.add(int(m2.group(1)))

    return sorted(weeks)


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    # Keep "N/A" as NaN, keep numeric strings, keep zeros
    return pd.to_numeric(s, errors="coerce")


def ensure_week_judge_score_sum(df: pd.DataFrame, weeks: List[int]) -> pd.DataFrame:
    """
    If week{k}_judge_score_sum doesn't exist, compute it from week{k}_judge{j}_score columns.

    Semantics:
    - If all judge scores for that week are missing (NaN after coercion), sum becomes NaN
      (represents show didn't run or contestant not present in that week).
    - If contestant eliminated, dataset often has 0 for remaining weeks; sum stays 0.
    """
    df = df.copy()

    for w in weeks:
        sum_col = f"week{w}_judge_score_sum"
        if sum_col in df.columns:
            df[sum_col] = _safe_to_numeric(df[sum_col])
            continue

        # Collect judge columns for this week
        week_cols = []
        for c in df.columns:
            m = _WEEK_JUDGE_RE.fullmatch(c)
            if m and int(m.group(1)) == w:
                week_cols.append(c)

        if not week_cols:
            continue

        numeric = df[week_cols].apply(_safe_to_numeric)
        all_nan = numeric.isna().all(axis=1)
        s = numeric.sum(axis=1, skipna=True)
        s[all_nan] = np.nan
        df[sum_col] = s

    return df


def build_judge_total_long(df: pd.DataFrame, weeks: List[int]) -> pd.DataFrame:
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


# ---------------- 3) parse results ----------------
def parse_elim_week(results: str) -> Optional[int]:
    m = re.search(r"Eliminated Week (\d+)", str(results))
    return int(m.group(1)) if m else None


def is_withdrew(results: str) -> bool:
    return str(results).strip().lower() == "withdrew"


def infer_withdrew_week(
    df: pd.DataFrame, judge_long: pd.DataFrame, weeks: List[int]
) -> Dict[Tuple[int, int], int]:
    """
    For contestants with results == "Withdrew", infer withdrew_week as last week with judge_total > 0.
    """
    withdrew_people = df[df["results"].apply(is_withdrew)][["season", "contestant_id"]]
    withdrew_set = set(map(tuple, withdrew_people.values.tolist()))
    if not withdrew_set:
        return {}

    pivot = judge_long.pivot_table(
        index=["season", "contestant_id"], columns="week", values="judge_total", aggfunc="first"
    )

    out = {}
    for key in withdrew_set:
        if key not in pivot.index:
            continue
        series = pivot.loc[key]
        active_weeks = [
            w
            for w in weeks
            if (w in series.index and pd.notna(series[w]) and series[w] > 0)
        ]
        if active_weeks:
            out[key] = max(active_weeks)
    return out


# ---------------- 4) events ----------------
@dataclass
class WeekEvent:
    season: int
    week: int
    rule: str                 # "percent" or "rank" (the show rule used for likelihood)
    mechanism: str            # "direct" or "bottom2_judgesave"
    active_ids: List[int]
    J: np.ndarray             # judge_total for active_ids
    zJ: np.ndarray            # z-score within active
    j_percent: np.ndarray     # J / sum(J)
    eliminated_ids: List[int] # observed eliminated (possibly multiple)
    skip_likelihood: bool     # withdrew week, etc.
    note: str


def build_events(df: pd.DataFrame, judge_long: pd.DataFrame, weeks: List[int]) -> List[WeekEvent]:
    df2 = df.copy()
    df2["elim_week"] = df2["results"].apply(parse_elim_week)

    elim_map = (
        df2.dropna(subset=["elim_week"])
        .groupby(["season", "elim_week"])["contestant_id"]
        .apply(list)
        .to_dict()
    )

    withdrew_week = infer_withdrew_week(df2, judge_long, weeks)

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

    events: List[WeekEvent] = []
    seasons = sorted(df2["season"].unique().tolist())

    for s in seasons:
        for w in weeks:
            active = roster_map.get((s, w), [])
            if not active:
                continue

            J = np.array([jt_map.get((s, w, i), np.nan) for i in active], dtype=float)

            m = np.nanmean(J)
            sd = np.nanstd(J)
            zJ = (J - m) / (sd + 1e-8)

            sumJ = np.nansum(J)
            j_percent = J / (sumJ + 1e-12)

            eliminated = elim_map.get((s, w), [])
            rule = season_rule(int(s))

            mechanism = "bottom2_judgesave" if int(s) >= SEASON_JUDGESAVE_START else "direct"

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
                    mechanism=mechanism,
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


# ---------------- 5) math utils ----------------
def logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return float(m + np.log(np.sum(np.exp(a - m)) + 1e-300))


def log_softmax(a: np.ndarray) -> np.ndarray:
    return a - logsumexp(a)


def softmax(a: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(a))


def rank_desc(values: np.ndarray) -> np.ndarray:
    """
    1 = best, n = worst, sorting by values descending.
    Ties get average rank (simple deterministic tie handling).
    """
    values = np.asarray(values, dtype=float)
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


def rank_asc(values: np.ndarray) -> np.ndarray:
    """
    1 = best, n = worst, sorting by values ascending.
    Ties get average rank.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (~np.isnan(x)) & (~np.isnan(y))
    if int(np.sum(m)) < 2:
        return float("nan")
    x = x[m]
    y = y[m]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman_corr_from_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    # Spearman rho = Pearson corr of ranks
    return pearson_corr(rx, ry)


# ---------------- 6) model core: vote share & hazards ----------------
def vote_share(mu_vec: np.ndarray, zJ: np.ndarray, gamma: float) -> np.ndarray:
    # p_i = softmax(mu_i + gamma * zJ_i)
    return softmax(mu_vec + gamma * zJ)


def hazard_percent(j_percent: np.ndarray, p: np.ndarray, w: float = 0.5) -> np.ndarray:
    combined = w * j_percent + (1.0 - w) * p
    return -combined  # smaller combined -> more dangerous -> larger hazard


def hazard_rank(J: np.ndarray, p: np.ndarray) -> np.ndarray:
    rj = rank_desc(J)
    rp = rank_desc(p)
    return 0.5 * (rj + rp)  # larger rank -> more dangerous


# ---------------- 6.5) Q2: compare rank-method vs percent-method (independent of show rule) ----------------
def compare_two_methods_for_week(ev: WeekEvent, p: np.ndarray) -> Dict[str, float]:
    """
    For this (season, week) and this posterior draw p-vector, compute:
    - rank-method combined ranking (judge_rank + fan_rank)
    - percent-method combined ranking (judge_percent + fan_percent)
    - judge-only ranking (judge_rank)
    - fan-only ranking (fan_rank)
    Then compute correlations:
      r_rank_percent: corr(rank_method_rank, percent_method_rank)
      r_rank_judge   : corr(rank_method_rank, judge_rank)
      r_percent_judge: corr(percent_method_rank, judge_rank)
      r_rank_fan     : corr(rank_method_rank, fan_rank)
      r_percent_fan  : corr(percent_method_rank, fan_rank)
      bias_rank      : r_rank_fan - r_rank_judge
      bias_percent   : r_percent_fan - r_percent_judge
    Plus IDs:
      elim_rank_id, elim_percent_id (worst under each)
      top1_rank_id, top1_percent_id (best under each)
    """
    J = np.asarray(ev.J, dtype=float)
    p = np.asarray(p, dtype=float)

    judge_rank = rank_desc(J)      # 1 best
    fan_rank = rank_desc(p)        # 1 best

    # Method A: "Rank method" (sum of ranks; smaller sum is better)
    sum_rank = judge_rank + fan_rank
    rank_method_rank = rank_asc(sum_rank)  # 1 best, n worst

    # Method B: "Percent method" (sum of percentages; larger is better)
    percent_score = np.asarray(ev.j_percent, dtype=float) + p
    percent_method_rank = rank_desc(percent_score)  # 1 best, n worst

    # Correlations on rank vectors (both are 1=best)
    r_rank_percent = spearman_corr_from_ranks(rank_method_rank, percent_method_rank)
    r_rank_judge = spearman_corr_from_ranks(rank_method_rank, judge_rank)
    r_percent_judge = spearman_corr_from_ranks(percent_method_rank, judge_rank)
    r_rank_fan = spearman_corr_from_ranks(rank_method_rank, fan_rank)
    r_percent_fan = spearman_corr_from_ranks(percent_method_rank, fan_rank)

    bias_rank = r_rank_fan - r_rank_judge if (not np.isnan(r_rank_fan) and not np.isnan(r_rank_judge)) else float("nan")
    bias_percent = r_percent_fan - r_percent_judge if (not np.isnan(r_percent_fan) and not np.isnan(r_percent_judge)) else float("nan")

    # Elimination under each method = worst (largest rank number)
    elim_rank_pos = int(np.argmax(rank_method_rank))
    elim_percent_pos = int(np.argmax(percent_method_rank))

    top1_rank_pos = int(np.argmin(rank_method_rank))
    top1_percent_pos = int(np.argmin(percent_method_rank))

    elim_rank_id = int(ev.active_ids[elim_rank_pos])
    elim_percent_id = int(ev.active_ids[elim_percent_pos])

    top1_rank_id = int(ev.active_ids[top1_rank_pos])
    top1_percent_id = int(ev.active_ids[top1_percent_pos])

    return {
        "r_rank_percent": float(r_rank_percent),
        "r_rank_judge": float(r_rank_judge),
        "r_percent_judge": float(r_percent_judge),
        "r_rank_fan": float(r_rank_fan),
        "r_percent_fan": float(r_percent_fan),
        "bias_rank": float(bias_rank),
        "bias_percent": float(bias_percent),
        "elim_rank_id": float(elim_rank_id),         # store as float for easy np.array; cast later
        "elim_percent_id": float(elim_percent_id),
        "top1_rank_id": float(top1_rank_id),
        "top1_percent_id": float(top1_percent_id),
    }


# ---------------- 7) likelihoods ----------------
def _hazard_for_event_subset(ev: WeekEvent, mu_s: np.ndarray, gamma: float, remaining_ids: List[int], w_percent: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build hazard on subset "remaining_ids".
    Returns: (hazard, p_sub, J_sub)
    """
    active_pos = {cid: i for i, cid in enumerate(ev.active_ids)}
    sub_idx = np.array([active_pos[cid] for cid in remaining_ids], dtype=int)

    mu_vec = mu_s[np.array(remaining_ids, dtype=int)]
    zJ_sub = ev.zJ[sub_idx]
    J_sub = ev.J[sub_idx]
    jperc_sub = ev.j_percent[sub_idx]

    p_sub = vote_share(mu_vec, zJ_sub, gamma)

    if ev.rule == "percent":
        hz = hazard_percent(jperc_sub, p_sub, w=w_percent)
    elif ev.rule == "rank":
        hz = hazard_rank(J_sub, p_sub)
    else:
        raise ValueError(f"Unknown rule: {ev.rule}")

    return hz, p_sub, J_sub


def direct_elimination_loglik_ordered(ev: WeekEvent, mu_s: np.ndarray, gamma: float, kappa_strict: float, w_percent: float, elim_order: List[int]) -> float:
    """
    Direct elimination: sequential PL on hazard with strength kappa_strict.
    elim_order is an ordered list.
    """
    if ev.skip_likelihood:
        return 0.0
    if not elim_order:
        return 0.0

    remaining = list(ev.active_ids)
    ll = 0.0

    for e_id in elim_order:
        if e_id not in remaining:
            return -np.inf

        hz, _, _ = _hazard_for_event_subset(ev, mu_s, gamma, remaining, w_percent=w_percent)
        log_q = log_softmax(kappa_strict * hz)

        e_pos = remaining.index(e_id)
        ll += float(log_q[e_pos])
        remaining.remove(e_id)

    return ll


def bottom2_judgesave_loglik_ordered(
    ev: WeekEvent,
    mu_s: np.ndarray,
    gamma: float,
    kappa_fuzzy: float,
    kappa_judge: float,
    w_percent: float,
    elim_order: List[int],
) -> float:
    """
    Seasons 28+ mechanism:
    1) Bottom-2 are identified using combined (judge+fan) -> modeled as stochastic "draw 2 without replacement"
       with probabilities proportional to exp(kappa_fuzzy * hazard).
    2) Judges then choose who to eliminate from those two -> modeled as softmax(kappa_judge * (-J)).
       (kappa_judge high => almost deterministic: lower J more likely eliminated)

    We only observe the eliminated contestant, not the other bottom-2 member => we marginalize over the other member
    and over the (b1,b2) order.
    """
    if ev.skip_likelihood:
        return 0.0
    if not elim_order:
        return 0.0

    remaining = list(ev.active_ids)
    ll = 0.0

    for e_id in elim_order:
        if e_id not in remaining:
            return -np.inf
        if len(remaining) < 2:
            return -np.inf

        hz, _, J_sub = _hazard_for_event_subset(ev, mu_s, gamma, remaining, w_percent=w_percent)

        log_q1 = log_softmax(kappa_fuzzy * hz)
        pos = {cid: i for i, cid in enumerate(remaining)}

        epos = pos[e_id]
        terms = []

        def logp_judge_elim_a(a_idx: int, b_idx: int) -> float:
            logits = kappa_judge * np.array([-J_sub[a_idx], -J_sub[b_idx]], dtype=float)
            lsm = log_softmax(logits)
            return float(lsm[0])

        for b_id in remaining:
            if b_id == e_id:
                continue
            bpos = pos[b_id]

            # Order 1: (e first, b second)
            rem2_1 = [cid for cid in remaining if cid != e_id]
            hz2_1 = np.array([hz[pos[cid]] for cid in rem2_1], dtype=float)
            log_q2_1 = log_softmax(kappa_fuzzy * hz2_1)
            bpos2_1 = rem2_1.index(b_id)

            log_j_eb = logp_judge_elim_a(epos, bpos)
            terms.append(float(log_q1[epos] + log_q2_1[bpos2_1] + log_j_eb))

            # Order 2: (b first, e second)
            rem2_2 = [cid for cid in remaining if cid != b_id]
            hz2_2 = np.array([hz[pos[cid]] for cid in rem2_2], dtype=float)
            log_q2_2 = log_softmax(kappa_fuzzy * hz2_2)
            epos2_2 = rem2_2.index(e_id)

            terms.append(float(log_q1[bpos] + log_q2_2[epos2_2] + log_j_eb))

        ll += logsumexp(np.array(terms, dtype=float))
        remaining.remove(e_id)

    return ll


def elimination_loglik_event(
    ev: WeekEvent,
    mu_s: np.ndarray,
    gamma: float,
    kappa_strict: float,
    kappa_fuzzy: float,
    kappa_judge: float,
    w_percent: float,
    max_perm: int = 6,
) -> float:
    """
    Wrapper with:
    - mechanism-based likelihood
    - permutation-marginalization for multiple eliminations in a week (order may be unknown in data)
    """
    if ev.skip_likelihood:
        return 0.0
    if not ev.eliminated_ids:
        return 0.0

    elim_ids = list(ev.eliminated_ids)
    m = len(elim_ids)

    if m > 1:
        perms = list(itertools.permutations(elim_ids))
        if len(perms) > max_perm:
            perms = [tuple(elim_ids)]
    else:
        perms = [tuple(elim_ids)]

    lls = []
    for perm in perms:
        perm_list = list(perm)
        if ev.mechanism == "direct":
            lls.append(direct_elimination_loglik_ordered(ev, mu_s, gamma, kappa_strict, w_percent, perm_list))
        elif ev.mechanism == "bottom2_judgesave":
            lls.append(bottom2_judgesave_loglik_ordered(ev, mu_s, gamma, kappa_fuzzy, kappa_judge, w_percent, perm_list))
        else:
            raise ValueError(f"Unknown mechanism: {ev.mechanism}")

    return logsumexp(np.array(lls, dtype=float))


# ---------------- 8) priors & posterior ----------------
def center_mu(mu: np.ndarray) -> np.ndarray:
    return mu - float(np.mean(mu))


def log_normal(x: float, m: float, s: float) -> float:
    z = (x - m) / s
    return -0.5 * z * z - math.log(s + 1e-300)  # omit const


def log_prior(
    mu_by_season: Dict[int, np.ndarray],
    gamma: float,
    log_kappa_strict: float,
    log_kappa_fuzzy: float,
    log_sigma_mu: float,
    log_kappa_judge: float,
) -> float:
    """
    - mu_si ~ N(0, sigma_mu^2), with sigma_mu = exp(log_sigma_mu)
    - gamma ~ N(0, 1.5^2)
    - log_kappa_strict ~ N(log(12), 0.5^2)  (encourage ~10+)
    - log_kappa_fuzzy  ~ N(log(2.5), 0.5^2) (encourage ~2-3)
    - log_sigma_mu     ~ N(log(3.0), 1.0^2) (wide; allows big mu for "Bobby Bones"-style cases)
    - log_kappa_judge  ~ N(log(10), 0.5^2)  (judges usually more deterministic than bottom2 noise)
    """
    sigma_mu = float(np.exp(log_sigma_mu))

    mu_all = np.concatenate([mu_by_season[s] for s in sorted(mu_by_season.keys())])
    n_mu = mu_all.size

    lp = 0.0
    lp += -0.5 * float(np.sum((mu_all / sigma_mu) ** 2)) - n_mu * math.log(sigma_mu + 1e-300)

    lp += log_normal(gamma, 0.0, 1.5)

    lp += log_normal(log_kappa_strict, math.log(12.0), 0.5)
    lp += log_normal(log_kappa_fuzzy,  math.log(2.5), 0.5)
    lp += log_normal(log_kappa_judge,  math.log(10.0), 0.5)

    lp += log_normal(log_sigma_mu, math.log(3.0), 1.0)

    return float(lp)


def log_posterior(
    mu_by_season: Dict[int, np.ndarray],
    gamma: float,
    log_kappa_strict: float,
    log_kappa_fuzzy: float,
    log_sigma_mu: float,
    log_kappa_judge: float,
    events: List[WeekEvent],
    w_percent: float = 0.5,
) -> float:
    kappa_strict = float(np.exp(log_kappa_strict))
    kappa_fuzzy = float(np.exp(log_kappa_fuzzy))
    kappa_judge = float(np.exp(log_kappa_judge))

    lp = log_prior(mu_by_season, gamma, log_kappa_strict, log_kappa_fuzzy, log_sigma_mu, log_kappa_judge)

    ll = 0.0
    for ev in events:
        mu_s = mu_by_season[ev.season]
        ll += elimination_loglik_event(
            ev,
            mu_s=mu_s,
            gamma=gamma,
            kappa_strict=kappa_strict,
            kappa_fuzzy=kappa_fuzzy,
            kappa_judge=kappa_judge,
            w_percent=w_percent,
        )

    return float(lp + ll)


# ---------------- 9) MCMC ----------------
def summarize_array(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q025": float("nan"), "q975": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q025": float(np.quantile(a, 0.025)),
        "q975": float(np.quantile(a, 0.975)),
    }


def run_mcmc(
    events: List[WeekEvent],
    seasons: List[int],
    season_sizes: Dict[int, int],
    n_iter: int = 5000,
    burn: int = 1500,
    thin: int = 10,
    step_mu: float = 0.08,
    step_gamma: float = 0.06,
    step_logk_strict: float = 0.05,
    step_logk_fuzzy: float = 0.05,
    step_log_sigma: float = 0.05,
    step_logk_judge: float = 0.05,
    adapt: bool = True,
    target_acc: float = 0.30,
    adapt_window: int = 250,
    w_percent: float = 0.5,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # init
    mu_by_season = {s: center_mu(rng.normal(0, 0.5, size=season_sizes[s])) for s in seasons}
    gamma = 0.0

    log_kappa_strict = math.log(12.0)
    log_kappa_fuzzy = math.log(2.5)
    log_kappa_judge = math.log(10.0)
    log_sigma_mu = math.log(3.0)

    cur_lp = log_posterior(
        mu_by_season, gamma, log_kappa_strict, log_kappa_fuzzy, log_sigma_mu, log_kappa_judge, events, w_percent=w_percent
    )

    draws = {
        "gamma": [],
        "log_kappa_strict": [],
        "log_kappa_fuzzy": [],
        "log_kappa_judge": [],
        "log_sigma_mu": [],
        "mu_by_season": [],
    }

    acc = defaultdict(int)
    prop = defaultdict(int)
    win_acc = defaultdict(int)
    win_prop = defaultdict(int)

    def adapt_step(step: float, acc_rate: float) -> float:
        if acc_rate > target_acc + 0.10:
            return step * 1.25
        if acc_rate < target_acc - 0.10:
            return step * 0.80
        return step

    for it in range(n_iter):
        # --- mu blocks ---
        for s in seasons:
            prop["mu"] += 1
            win_prop["mu"] += 1

            mu_new = {k: v.copy() for k, v in mu_by_season.items()}
            mu_new[s] = center_mu(mu_new[s] + rng.normal(0, step_mu, size=mu_new[s].shape))

            cand_lp = log_posterior(
                mu_new, gamma, log_kappa_strict, log_kappa_fuzzy, log_sigma_mu, log_kappa_judge, events, w_percent=w_percent
            )
            if math.log(rng.random()) < cand_lp - cur_lp:
                mu_by_season = mu_new
                cur_lp = cand_lp
                acc["mu"] += 1
                win_acc["mu"] += 1

        # --- gamma ---
        prop["gamma"] += 1
        win_prop["gamma"] += 1

        gamma_new = gamma + rng.normal(0, step_gamma)
        cand_lp = log_posterior(
            mu_by_season, gamma_new, log_kappa_strict, log_kappa_fuzzy, log_sigma_mu, log_kappa_judge, events, w_percent=w_percent
        )
        if math.log(rng.random()) < cand_lp - cur_lp:
            gamma = gamma_new
            cur_lp = cand_lp
            acc["gamma"] += 1
            win_acc["gamma"] += 1

        # --- log_kappa_strict ---
        prop["logk_strict"] += 1
        win_prop["logk_strict"] += 1

        lks_new = log_kappa_strict + rng.normal(0, step_logk_strict)
        cand_lp = log_posterior(
            mu_by_season, gamma, lks_new, log_kappa_fuzzy, log_sigma_mu, log_kappa_judge, events, w_percent=w_percent
        )
        if math.log(rng.random()) < cand_lp - cur_lp:
            log_kappa_strict = lks_new
            cur_lp = cand_lp
            acc["logk_strict"] += 1
            win_acc["logk_strict"] += 1

        # --- log_kappa_fuzzy ---
        prop["logk_fuzzy"] += 1
        win_prop["logk_fuzzy"] += 1

        lkf_new = log_kappa_fuzzy + rng.normal(0, step_logk_fuzzy)
        cand_lp = log_posterior(
            mu_by_season, gamma, log_kappa_strict, lkf_new, log_sigma_mu, log_kappa_judge, events, w_percent=w_percent
        )
        if math.log(rng.random()) < cand_lp - cur_lp:
            log_kappa_fuzzy = lkf_new
            cur_lp = cand_lp
            acc["logk_fuzzy"] += 1
            win_acc["logk_fuzzy"] += 1

        # --- log_kappa_judge ---
        prop["logk_judge"] += 1
        win_prop["logk_judge"] += 1

        lkj_new = log_kappa_judge + rng.normal(0, step_logk_judge)
        cand_lp = log_posterior(
            mu_by_season, gamma, log_kappa_strict, log_kappa_fuzzy, log_sigma_mu, lkj_new, events, w_percent=w_percent
        )
        if math.log(rng.random()) < cand_lp - cur_lp:
            log_kappa_judge = lkj_new
            cur_lp = cand_lp
            acc["logk_judge"] += 1
            win_acc["logk_judge"] += 1

        # --- log_sigma_mu ---
        prop["log_sigma"] += 1
        win_prop["log_sigma"] += 1

        lsm_new = log_sigma_mu + rng.normal(0, step_log_sigma)
        cand_lp = log_posterior(
            mu_by_season, gamma, log_kappa_strict, log_kappa_fuzzy, lsm_new, log_kappa_judge, events, w_percent=w_percent
        )
        if math.log(rng.random()) < cand_lp - cur_lp:
            log_sigma_mu = lsm_new
            cur_lp = cand_lp
            acc["log_sigma"] += 1
            win_acc["log_sigma"] += 1

        # --- store ---
        if it >= burn and ((it - burn) % thin == 0):
            draws["gamma"].append(float(gamma))
            draws["log_kappa_strict"].append(float(log_kappa_strict))
            draws["log_kappa_fuzzy"].append(float(log_kappa_fuzzy))
            draws["log_kappa_judge"].append(float(log_kappa_judge))
            draws["log_sigma_mu"].append(float(log_sigma_mu))
            draws["mu_by_season"].append({s: mu_by_season[s].copy() for s in seasons})

        # --- adapt ---
        if adapt and it < burn and (it + 1) % adapt_window == 0:
            for key, step_name in [
                ("mu", "step_mu"),
                ("gamma", "step_gamma"),
                ("logk_strict", "step_logk_strict"),
                ("logk_fuzzy", "step_logk_fuzzy"),
                ("logk_judge", "step_logk_judge"),
                ("log_sigma", "step_log_sigma"),
            ]:
                rate = win_acc[key] / max(1, win_prop[key])
                if step_name == "step_mu":
                    step_mu = adapt_step(step_mu, rate)
                elif step_name == "step_gamma":
                    step_gamma = adapt_step(step_gamma, rate)
                elif step_name == "step_logk_strict":
                    step_logk_strict = adapt_step(step_logk_strict, rate)
                elif step_name == "step_logk_fuzzy":
                    step_logk_fuzzy = adapt_step(step_logk_fuzzy, rate)
                elif step_name == "step_logk_judge":
                    step_logk_judge = adapt_step(step_logk_judge, rate)
                elif step_name == "step_log_sigma":
                    step_log_sigma = adapt_step(step_log_sigma, rate)

            win_acc = defaultdict(int)
            win_prop = defaultdict(int)

        if (it + 1) % 100 == 0:
            print(
                f"[w={w_percent}] iter {it+1}/{n_iter} lp={cur_lp:.2f} "
                f"gamma={gamma:.3f} "
                f"k_strict={math.exp(log_kappa_strict):.2f} k_fuzzy={math.exp(log_kappa_fuzzy):.2f} "
                f"k_judge={math.exp(log_kappa_judge):.2f} sigma_mu={math.exp(log_sigma_mu):.2f} "
                f"steps=({step_mu:.3f},{step_gamma:.3f},{step_logk_strict:.3f},{step_logk_fuzzy:.3f},{step_logk_judge:.3f},{step_log_sigma:.3f})",
                flush=True,
            )

    accept_rates = {k: acc[k] / max(1, prop[k]) for k in prop}
    return draws, accept_rates


# ---------------- 10) posterior predictive checks ----------------
def weekly_elim_probs(ev: WeekEvent, mu_s: np.ndarray, gamma: float, kappa_strict: float, kappa_fuzzy: float, kappa_judge: float, w_percent: float) -> np.ndarray:
    """
    Return a vector q over ev.active_ids representing (approx) marginal probability of being eliminated in that week,
    under the same mechanism as the likelihood (single-elimination approximation).

    For "direct": q ∝ exp(kappa_strict * hazard)
    For "bottom2_judgesave": q(i) = sum_{b!=i} P(bottom2 contains {i,b}) * P(judges eliminate i | {i,b})
        where bottom2 is a 2-draw w/o replacement with probs ∝ exp(kappa_fuzzy * hazard).
    """
    if ev.skip_likelihood:
        return np.full(len(ev.active_ids), np.nan, dtype=float)

    active = list(ev.active_ids)
    hz, _, J_sub = _hazard_for_event_subset(ev, mu_s, gamma, active, w_percent=w_percent)

    if ev.mechanism == "direct":
        logits = kappa_strict * hz
        return softmax(logits)

    # bottom2 + judge-save marginal
    n = len(active)
    if n < 2:
        return np.full(n, np.nan, dtype=float)

    log_q1 = log_softmax(kappa_fuzzy * hz)
    q1 = np.exp(log_q1)

    out = np.zeros(n, dtype=float)

    for a in range(n):
        rem_idx = [j for j in range(n) if j != a]
        hz2 = hz[rem_idx]
        q2 = np.exp(log_softmax(kappa_fuzzy * hz2))

        for t, b in enumerate(rem_idx):
            p_bottom2_order = q1[a] * q2[t]

            logits_j = kappa_judge * np.array([-J_sub[a], -J_sub[b]], dtype=float)
            pj = np.exp(log_softmax(logits_j))

            out[a] += p_bottom2_order * pj[0]
            out[b] += p_bottom2_order * pj[1]

    s = float(out.sum())
    if s <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    return out / s


# ---------------- 11) export ----------------
def export_all(df: pd.DataFrame, judge_long: pd.DataFrame, events: List[WeekEvent], draws: dict, accept_rates: dict, outdir: Path, w_percent: float):
    if len(draws["gamma"]) == 0:
        raise ValueError("draws 为空：检查 n_iter/burn/thin 或 MCMC 是否提前中断。")
    outdir.mkdir(exist_ok=True)

    seasons = sorted({e.season for e in events})
    name_map = {
        (int(r.season), int(r.contestant_id)): str(r.celebrity_name)
        for r in df[["season", "contestant_id", "celebrity_name"]].itertuples(index=False)
    }

    gamma_arr = np.array(draws["gamma"], dtype=float)
    k_strict_arr = np.exp(np.array(draws["log_kappa_strict"], dtype=float))
    k_fuzzy_arr = np.exp(np.array(draws["log_kappa_fuzzy"], dtype=float))
    k_judge_arr = np.exp(np.array(draws["log_kappa_judge"], dtype=float))
    sigma_mu_arr = np.exp(np.array(draws["log_sigma_mu"], dtype=float))

    # Global params
    global_rows = [
        {"param_name": "gamma", **summarize_array(gamma_arr)},
        {"param_name": "kappa_strict", **summarize_array(k_strict_arr)},
        {"param_name": "kappa_fuzzy", **summarize_array(k_fuzzy_arr)},
        {"param_name": "kappa_judge", **summarize_array(k_judge_arr)},
        {"param_name": "sigma_mu", **summarize_array(sigma_mu_arr)},
    ]
    pd.DataFrame(global_rows).to_csv(outdir / "posterior_global_params.csv", index=False)

    # Popularity mu
    mu_rows = []
    for s in seasons:
        mu_stack = np.stack([d[s] for d in draws["mu_by_season"]], axis=0)
        for i in range(mu_stack.shape[1]):
            stats = summarize_array(mu_stack[:, i])
            mu_rows.append(
                {
                    "season": s,
                    "contestant_id": i,
                    "celebrity_name": name_map.get((s, i), ""),
                    "mu_mean": stats["mean"],
                    "mu_median": stats["median"],
                    "mu_q025": stats["q025"],
                    "mu_q975": stats["q975"],
                }
            )

    mu_df = pd.DataFrame(mu_rows)
    mu_df["mu_rank"] = mu_df.groupby("season")["mu_mean"].rank(ascending=False, method="min").astype(int)
    mu_df.to_csv(outdir / "posterior_popularity_summary.csv", index=False)

    # Vote share summary + Q2 method comparison stores
    p_store = defaultdict(list)

    # Q2: per-(season,week) metric draws
    q2_store = defaultdict(lambda: defaultdict(list))  # key -> metric_name -> list
    q2_elim_rank = defaultdict(list)
    q2_elim_percent = defaultdict(list)
    q2_top1_rank = defaultdict(list)
    q2_top1_percent = defaultdict(list)

    # For easier metadata in output
    ev_by_key = {(ev.season, ev.week): ev for ev in events}

    for d in range(len(draws["gamma"])):
        gamma = float(draws["gamma"][d])
        mu_by_season = draws["mu_by_season"][d]

        for ev in events:
            mu_s = mu_by_season[ev.season]
            active = np.array(ev.active_ids, dtype=int)
            mu_vec = mu_s[active]
            p = vote_share(mu_vec, ev.zJ, gamma)

            # store p for Q1 summaries
            for cid, pi in zip(ev.active_ids, p):
                p_store[(ev.season, ev.week, int(cid))].append(float(pi))

            # Q2 metrics (skip withdrew weeks to avoid mixing with non-rule weeks)
            if ev.skip_likelihood:
                continue

            m = compare_two_methods_for_week(ev, p)
            key = (ev.season, ev.week)

            # correlations & bias
            for name in ["r_rank_percent", "r_rank_judge", "r_percent_judge", "r_rank_fan", "r_percent_fan", "bias_rank", "bias_percent"]:
                q2_store[key][name].append(float(m[name]))

            # elimination/top1 ids
            er = int(m["elim_rank_id"])
            ep = int(m["elim_percent_id"])
            tr = int(m["top1_rank_id"])
            tp = int(m["top1_percent_id"])

            q2_elim_rank[key].append(er)
            q2_elim_percent[key].append(ep)
            q2_top1_rank[key].append(tr)
            q2_top1_percent[key].append(tp)

    vote_rows = []
    for (s, w, cid), vals in p_store.items():
        arr = np.array(vals, dtype=float)
        stats = summarize_array(arr)

        jt = judge_long.loc[
            (judge_long["season"] == s) & (judge_long["week"] == w) & (judge_long["contestant_id"] == cid),
            "judge_total",
        ]
        judge_total = float(jt.iloc[0]) if len(jt) else np.nan

        p_mean = stats["mean"]
        vote_rows.append(
            {
                "season": s,
                "week": w,
                "contestant_id": cid,
                "celebrity_name": name_map.get((s, cid), ""),
                "judge_total": judge_total,
                "p_mean": p_mean,
                "p_median": stats["median"],
                "p_q025": stats["q025"],
                "p_q975": stats["q975"],
                "votes_est_mean": p_mean * TOTAL_FAN_VOTES_PER_WEEK,
                "votes_est_q025": stats["q025"] * TOTAL_FAN_VOTES_PER_WEEK,
                "votes_est_q975": stats["q975"] * TOTAL_FAN_VOTES_PER_WEEK,
            }
        )

    pd.DataFrame(vote_rows).to_csv(outdir / "posterior_vote_share_summary.csv", index=False)

    # Weekly elimination check (your existing PPC)
    week_prob_draws = defaultdict(list)
    for d in range(len(draws["gamma"])):
        gamma = float(draws["gamma"][d])
        k_strict = float(np.exp(draws["log_kappa_strict"][d]))
        k_fuzzy = float(np.exp(draws["log_kappa_fuzzy"][d]))
        k_judge = float(np.exp(draws["log_kappa_judge"][d]))
        mu_by_season = draws["mu_by_season"][d]
        for ev in events:
            if ev.skip_likelihood:
                continue
            q = weekly_elim_probs(ev, mu_by_season[ev.season], gamma, k_strict, k_fuzzy, k_judge, w_percent=w_percent)
            week_prob_draws[(ev.season, ev.week)].append(q)

    check_rows = []
    for ev in events:
        if ev.skip_likelihood:
            continue
        key = (ev.season, ev.week)
        if key not in week_prob_draws:
            continue

        q_stack = np.stack(week_prob_draws[key], axis=0)
        q_mean = np.mean(q_stack, axis=0)

        order = np.argsort(-q_mean)
        topk = min(3, len(order))
        top_idxs = list(order[:topk])
        top_ids = [ev.active_ids[i] for i in top_idxs]
        top_probs = [float(q_mean[i]) for i in top_idxs]

        while len(top_ids) < 3:
            top_ids.append(None)
            top_probs.append(np.nan)

        observed_name = ""
        observed_prob = np.nan
        is_in_top2 = False
        obs = None
        if len(ev.eliminated_ids) >= 1:
            obs = ev.eliminated_ids[0]
            observed_name = name_map.get((ev.season, obs), "")
            if obs in ev.active_ids:
                pos = ev.active_ids.index(obs)
                observed_prob = float(q_mean[pos])
            is_in_top2 = (obs == top_ids[0]) or (obs == top_ids[1])

        check_rows.append(
            {
                "season": ev.season,
                "week": ev.week,
                "mechanism": ev.mechanism,
                "rule": ev.rule,
                "observed_eliminated_name": observed_name,
                "pred_elim_prob_observed": observed_prob,
                "top1_pred_name": name_map.get((ev.season, top_ids[0]), "") if top_ids[0] is not None else "",
                "top1_prob": top_probs[0],
                "top2_pred_name": name_map.get((ev.season, top_ids[1]), "") if top_ids[1] is not None else "",
                "top2_prob": top_probs[1],
                "top3_pred_name": name_map.get((ev.season, top_ids[2]), "") if top_ids[2] is not None else "",
                "top3_prob": top_probs[2],
                "is_observed_in_top2": bool(is_in_top2),
                "notes": ev.note,
            }
        )

    pd.DataFrame(check_rows).to_csv(outdir / "weekly_elimination_check.csv", index=False)

    # ---------------- Q2: method comparison weekly summary (NEW) ----------------
    q2_rows = []
    for key in sorted(q2_store.keys()):
        s, w = key
        ev = ev_by_key.get(key, None)
        if ev is None:
            continue

        row = {
            "season": int(s),
            "week": int(w),
            "n_active": int(len(ev.active_ids)),
            "mechanism": ev.mechanism,
            "rule_used_in_likelihood": ev.rule,
            "notes": ev.note,
        }

        # summarize correlations/bias distributions
        for metric in ["r_rank_percent", "r_rank_judge", "r_percent_judge", "r_rank_fan", "r_percent_fan", "bias_rank", "bias_percent"]:
            arr = np.array(q2_store[key][metric], dtype=float)
            st = summarize_array(arr)
            row[f"{metric}_mean"] = st["mean"]
            row[f"{metric}_median"] = st["median"]
            row[f"{metric}_q025"] = st["q025"]
            row[f"{metric}_q975"] = st["q975"]

        # elimination disagreement probability (your requested "淘汰分歧率")
        er = np.array(q2_elim_rank[key], dtype=int)
        ep = np.array(q2_elim_percent[key], dtype=int)
        if er.size > 0 and ep.size == er.size:
            row["elim_disagree_prob"] = float(np.mean(er != ep))
        else:
            row["elim_disagree_prob"] = float("nan")

        # optional: top1 disagreement probability (rank-1 winner differs)
        tr = np.array(q2_top1_rank[key], dtype=int)
        tp = np.array(q2_top1_percent[key], dtype=int)
        if tr.size > 0 and tp.size == tr.size:
            row["top1_disagree_prob"] = float(np.mean(tr != tp))
        else:
            row["top1_disagree_prob"] = float("nan")

        # most frequent eliminated under each method (helps interpretation)
        if er.size > 0:
            c_er = Counter(er.tolist())
            cid, cnt = c_er.most_common(1)[0]
            row["most_likely_elim_rank_id"] = int(cid)
            row["most_likely_elim_rank_name"] = name_map.get((s, int(cid)), "")
            row["most_likely_elim_rank_prob"] = float(cnt / er.size)
        else:
            row["most_likely_elim_rank_id"] = np.nan
            row["most_likely_elim_rank_name"] = ""
            row["most_likely_elim_rank_prob"] = np.nan

        if ep.size > 0:
            c_ep = Counter(ep.tolist())
            cid, cnt = c_ep.most_common(1)[0]
            row["most_likely_elim_percent_id"] = int(cid)
            row["most_likely_elim_percent_name"] = name_map.get((s, int(cid)), "")
            row["most_likely_elim_percent_prob"] = float(cnt / ep.size)
        else:
            row["most_likely_elim_percent_id"] = np.nan
            row["most_likely_elim_percent_name"] = ""
            row["most_likely_elim_percent_prob"] = np.nan

        # probability each method matches the observed eliminated (if available)
        if len(ev.eliminated_ids) >= 1:
            obs = int(ev.eliminated_ids[0])
            row["observed_eliminated_id"] = obs
            row["observed_eliminated_name"] = name_map.get((s, obs), "")
            if er.size > 0:
                row["prob_rank_method_matches_observed_elim"] = float(np.mean(er == obs))
            else:
                row["prob_rank_method_matches_observed_elim"] = np.nan
            if ep.size > 0:
                row["prob_percent_method_matches_observed_elim"] = float(np.mean(ep == obs))
            else:
                row["prob_percent_method_matches_observed_elim"] = np.nan
        else:
            row["observed_eliminated_id"] = np.nan
            row["observed_eliminated_name"] = ""
            row["prob_rank_method_matches_observed_elim"] = np.nan
            row["prob_percent_method_matches_observed_elim"] = np.nan

        q2_rows.append(row)

    pd.DataFrame(q2_rows).to_csv(outdir / "method_comparison_weekly.csv", index=False)

    # Diagnostics
    diag = {
        "w_percent": w_percent,
        "n_draws": len(draws["gamma"]),
        "accept_rates": accept_rates,
        "gamma_summary": summarize_array(gamma_arr),
        "kappa_strict_summary": summarize_array(k_strict_arr),
        "kappa_fuzzy_summary": summarize_array(k_fuzzy_arr),
        "kappa_judge_summary": summarize_array(k_judge_arr),
        "sigma_mu_summary": summarize_array(sigma_mu_arr),
        "TOTAL_FAN_VOTES_PER_WEEK": TOTAL_FAN_VOTES_PER_WEEK,
        "SEASON_JUDGESAVE_START": SEASON_JUDGESAVE_START,
        "q2_outputs": {
            "method_comparison_weekly_csv": str(outdir / "method_comparison_weekly.csv"),
            "q2_metrics": [
                "r_rank_percent", "r_rank_judge", "r_percent_judge",
                "r_rank_fan", "r_percent_fan", "bias_rank", "bias_percent",
                "elim_disagree_prob", "top1_disagree_prob"
            ],
        },
    }
    with open(outdir / "mcmc_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)


# ---------------- 12) main ----------------
def main(
    w_list: Tuple[float, ...] = (0.5,),
    n_iter: int = 5000,
    burn: int = 1500,
    thin: int = 10,
    seed: int = 0,
):
    df = load_data(CSV_PATH)
    weeks = get_week_list(df)

    # Ensure week*_judge_score_sum exists (compute if needed)
    df = ensure_week_judge_score_sum(df, weeks)

    judge_long = build_judge_total_long(df, weeks)
    events = build_events(df, judge_long, weeks)

    if not events:
        raise ValueError("events 为空：请检查 CSV 是否读取成功、week 列是否存在。")

    seasons = sorted({e.season for e in events})
    season_sizes = {s: int(df.loc[df["season"] == s, "contestant_id"].max()) + 1 for s in seasons}

    print(f"Loaded seasons={len(seasons)} events={len(events)} weeks={len(weeks)}")
    print(f"Judge-save mechanism from season >= {SEASON_JUDGESAVE_START}")

    for w in w_list:
        outdir = BASE_OUTDIR / f"w_{str(w).replace('.','p')}"
        outdir.mkdir(parents=True, exist_ok=True)

        (outdir / "run_started.txt").write_text("run started\n", encoding="utf-8")
        print("\n" + "=" * 90)
        print(f"Running MCMC with w_percent={w}, outputs -> {outdir}")

        draws, accept_rates = run_mcmc(
            events,
            seasons,
            season_sizes,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            step_mu=0.08,
            step_gamma=0.06,
            step_logk_strict=0.05,
            step_logk_fuzzy=0.05,
            step_logk_judge=0.05,
            step_log_sigma=0.05,
            w_percent=w,
            seed=seed,
        )

        (outdir / "after_mcmc.txt").write_text(f"n_draws={len(draws['gamma'])}\n", encoding="utf-8")
        export_all(df, judge_long, events, draws, accept_rates, outdir, w_percent=w)
        (outdir / "run_finished.txt").write_text("run finished\n", encoding="utf-8")

        print(f"Done w={w}. Accept rates={accept_rates}")


if __name__ == "__main__":
    # Example: only run w=0.5 with moderately long chain.
    # Increase n_iter/burn for more stable posterior intervals.
    main(w_list=(0.5,), n_iter=5000, burn=1500, thin=10, seed=0)
