# -*- coding: utf-8 -*-
"""
DWTS (MCM 2026 Problem C) — Rank method elimination prediction + accuracy

Inputs (put in the same folder as this script, or adjust paths below):
  1) 2026_MCM_Problem_C_Data.csv                (official judges scores; 0 indicates eliminated in later weeks)
  2) posterior_vote_share_summary.csv           (your inferred fan votes summary; uses p_mean)

What this script does:
  - For seasons using the RANK-combination method (default: 1, 2, 28–34),
    for each evaluable week w (i.e., week w and week w+1 both exist in that season):
        * compute judge_rank (by judge_total) and fan_rank (by p_mean), both descending
        * sum the ranks
        * eliminate the worst k contestants where k = (# contestants who truly drop to 0 in week w+1)
        * compare predicted eliminated set vs actual eliminated set => week_correct (True/False)
  - Outputs:
        rank_method_contestant_week_detail.csv
        rank_method_week_summary.csv
        rank_method_season_summary.csv
        rank_method_predictions.xlsx   (week summary sheet has season summary appended at bottom)

Dependencies:
  - pandas, numpy, openpyxl
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd


# ----------------------------
# Config (edit if you want)
# ----------------------------
RAW_DATA_CSV = "/Users/garytchois/Desktop/美赛/2026_MCM_Problem_C_Data.csv"
POSTERIOR_SUMMARY_CSV = "/Users/garytchois/Desktop/mcmc_outputs_12000/w_0p5/posterior_vote_share_summary.csv"

# Seasons assumed to use RANK-combination (Problem statement: seasons 1, 2, and ~28–34).
RANK_SEASONS = [1, 2] + list(range(28, 35))

# Max weeks in the official dataset (columns week1..week11 exist in the provided file).
MAX_WEEK = 11

# Tie-breaking when choosing who gets eliminated among equal sum_rank:
#  - worse sum_rank first (bigger is worse)
#  - then lower p_mean first (worse fan support)
#  - then lower judge_total first (worse judges)
#  - then name ascending (stable)
TIEBREAK_SORT = dict(
    by=["sum_rank", "p_mean", "judge_total", "celebrity_name"],
    ascending=[False, True, True, True],
)


# ----------------------------
# Helpers
# ----------------------------
def build_official_week_totals(raw_df: pd.DataFrame, max_week: int = MAX_WEEK) -> pd.DataFrame:
    """
    From wide official judge score columns -> long table:
        season, celebrity_name, week, judge_total_official, next_judge_total_official, actual_eliminated
    """
    df = raw_df.copy()
    df["season"] = df["season"].astype(int)
    df["celebrity_name"] = df["celebrity_name"].astype(str).str.strip()

    # Compute per-week judge totals (ignore N/A -> NaN; sum across available judges)
    totals = {}
    for w in range(1, max_week + 1):
        cols = [f"week{w}_judge{i}_score" for i in range(1, 5)]
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        totals[w] = sub.sum(axis=1, min_count=1)  # NaN if all NaN (week not run)

    wide_totals = pd.DataFrame({f"week{w}_total": totals[w] for w in range(1, max_week + 1)})
    wide = pd.concat([df[["season", "celebrity_name"]], wide_totals], axis=1)

    long = wide.melt(
        id_vars=["season", "celebrity_name"],
        var_name="week",
        value_name="judge_total_official",
    )
    long["week"] = long["week"].str.extract(r"week(\d+)_total").astype(int)
    long = long.sort_values(["season", "celebrity_name", "week"]).reset_index(drop=True)

    long["next_judge_total_official"] = long.groupby(["season", "celebrity_name"])[
        "judge_total_official"
    ].shift(-1)

    # Actual eliminated at end of week w: has positive score in week w, then becomes 0 in week w+1
    long["actual_eliminated"] = (long["judge_total_official"] > 0) & (
        long["next_judge_total_official"] == 0
    )
    return long


def build_season_week_eval_mask(official_long: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which (season, week) are evaluable:
      - week w has at least one non-NaN judge total (show ran / week exists)
      - week w+1 also exists (so we can detect who turns to 0 next week)
    """
    sw = (
        official_long.groupby(["season", "week"], as_index=False)
        .agg(ran=("judge_total_official", lambda s: s.notna().any()))
        .sort_values(["season", "week"])
    )
    sw["next_ran"] = sw.groupby("season")["ran"].shift(-1).fillna(False).astype(bool)
    sw["eval_week"] = sw["ran"] & sw["next_ran"]
    return sw[["season", "week", "eval_week"]]


def compute_week_prediction(group: pd.DataFrame) -> pd.DataFrame:
    """
    Given one season-week group with columns: judge_total, p_mean, actual_eliminated,
    add ranks and predicted_eliminated, then stamp week-level correctness.
    """
    g = group.copy()
    # Some pandas versions drop group key columns; recover them from group.name or index.
    if "season" not in g.columns or "week" not in g.columns:
        if hasattr(group, "name") and isinstance(group.name, tuple) and len(group.name) == 2:
            g["season"] = group.name[0]
            g["week"] = group.name[1]
        else:
            g = g.reset_index()

    # ranks: 1 = best (highest score / highest votes)
    g["judge_rank"] = g["judge_total"].rank(method="min", ascending=False).astype(int)
    g["fan_rank"] = g["p_mean"].rank(method="min", ascending=False).astype(int)
    g["sum_rank"] = g["judge_rank"] + g["fan_rank"]

    k = int(g["actual_eliminated"].sum())
    g["k_actual_elim"] = k

    if k <= 0:
        g["predicted_eliminated"] = False
        pred_set: Set[str] = set()
    else:
        g_sorted = g.sort_values(**TIEBREAK_SORT)
        pred_set = set(g_sorted.head(k)["celebrity_name"].tolist())
        g["predicted_eliminated"] = g["celebrity_name"].isin(pred_set)

    actual_set: Set[str] = set(g.loc[g["actual_eliminated"], "celebrity_name"].tolist())
    g["week_correct"] = (pred_set == actual_set)

    g["predicted_elim_names"] = "; ".join(sorted(pred_set))
    g["actual_elim_names"] = "; ".join(sorted(actual_set))
    g["num_contestants"] = len(g)

    return g


def main() -> None:
    raw_path = Path(RAW_DATA_CSV)
    post_path = Path(POSTERIOR_SUMMARY_CSV)

    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot find {raw_path.resolve()}")
    if not post_path.exists():
        raise FileNotFoundError(f"Cannot find {post_path.resolve()}")

    # Load
    raw_df = pd.read_csv(raw_path)
    post_df = pd.read_csv(post_path)

    # Clean types
    post_df["season"] = post_df["season"].astype(int)
    post_df["week"] = post_df["week"].astype(int)
    post_df["celebrity_name"] = post_df["celebrity_name"].astype(str).str.strip()

    # Official elimination labels from "next week == 0"
    official_long = build_official_week_totals(raw_df, max_week=MAX_WEEK)
    sw_eval = build_season_week_eval_mask(official_long)

    actual_info = official_long[["season", "week", "celebrity_name", "actual_eliminated"]].copy()

    # Merge actual elimination + evaluable-week mask into posterior summary
    merged = (
        post_df.merge(actual_info, on=["season", "week", "celebrity_name"], how="left")
        .merge(sw_eval, on=["season", "week"], how="left")
    )
    merged["actual_eliminated"] = merged["actual_eliminated"].fillna(False).astype(bool)
    merged["eval_week"] = merged["eval_week"].fillna(False).astype(bool)

    # Keep only rank-method seasons + evaluable weeks
    rank_eval = merged[merged["season"].isin(RANK_SEASONS) & merged["eval_week"]].copy()

    if rank_eval.empty:
        raise RuntimeError(
            "No rows left after filtering by RANK_SEASONS and eval_week. "
            "Check your input files / season list."
        )

    # Compute predictions per season-week
    pred_detail = (
        rank_eval.groupby(["season", "week"], group_keys=False)
        .apply(compute_week_prediction)
        .reset_index(drop=True)
    )

    # Week-level summary
    week_summary = (
        pred_detail.groupby(["season", "week"], as_index=False)
        .agg(
            num_contestants=("num_contestants", "first"),
            k_actual_elim=("k_actual_elim", "first"),
            predicted_elim_names=("predicted_elim_names", "first"),
            actual_elim_names=("actual_elim_names", "first"),
            week_correct=("week_correct", "first"),
        )
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )

    # Season-level summary
    season_summary = (
        week_summary.groupby("season", as_index=False)
        .agg(correct_weeks=("week_correct", "sum"), total_weeks=("week_correct", "count"))
        .sort_values("season")
        .reset_index(drop=True)
    )
    season_summary["accuracy"] = season_summary["correct_weeks"] / season_summary["total_weeks"]

    # Save outputs
    pred_detail.to_csv("rank_method_contestant_week_detail.csv", index=False, encoding="utf-8-sig")
    week_summary.to_csv("rank_method_week_summary.csv", index=False, encoding="utf-8-sig")
    season_summary.to_csv("rank_method_season_summary.csv", index=False, encoding="utf-8-sig")

    # Excel: append season_summary below week_summary in the same sheet
    out_xlsx = "rank_method_predictions.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pred_detail.to_excel(writer, sheet_name="contestant_week_detail", index=False)
        week_summary.to_excel(writer, sheet_name="week_summary", index=False)

        # Append season summary below
        start_row = len(week_summary) + 3
        season_summary.to_excel(writer, sheet_name="week_summary", startrow=start_row, index=False)

        # Also separate sheet
        season_summary.to_excel(writer, sheet_name="season_summary", index=False)

    # Console brief
    overall_acc = week_summary["week_correct"].mean()
    print("Done.")
    print(f"Evaluated seasons (rank method): {sorted(RANK_SEASONS)}")
    print(f"Evaluated weeks: {len(week_summary)}")
    print(f"Overall week-level accuracy: {overall_acc:.3f}")
    print(
        "Outputs written:\n"
        f"  - {out_xlsx}\n"
        "  - rank_method_contestant_week_detail.csv\n"
        "  - rank_method_week_summary.csv\n"
        "  - rank_method_season_summary.csv"
    )


if __name__ == "__main__":
    main()
