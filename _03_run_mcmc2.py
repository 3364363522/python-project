# _03_run_mcmc.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from _02_model_likelihood2 import (
    load_data, get_week_list, build_judge_total_long, build_events,
    log_posterior, center_mu, vote_share
)

CSV_PATH = "/Users/garytchois/Desktop/vs/2026_MCM_Problem_C_Data_with_week_stats1.csv"
BASE_OUTDIR = (Path(__file__).resolve().parent / "mcmc_outputs")
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)


def summarize_array(a: np.ndarray):
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q025": float(np.quantile(a, 0.025)),
        "q975": float(np.quantile(a, 0.975)),
    }


def run_mcmc(
    events, seasons, season_sizes,
    n_iter=6000, burn=2000, thin=5,
    step_mu=0.10, step_gamma=0.10, step_logk_strict=0.08, step_logk_fuzzy=0.08,
    adapt=True, target_acc=0.30, adapt_window=200,
    w_percent=0.5, sigma_mu=3.0, seed=0
):
    rng = np.random.default_rng(seed)

    mu_by_season = {s: center_mu(rng.normal(0, 0.3, size=season_sizes[s])) for s in seasons}
    gamma = 0.0

    # 以你希望的量级初始化（strict大，fuzzy小）
    log_kappa_strict = float(np.log(12.0))
    log_kappa_fuzzy = float(np.log(2.5))

    cur_lp = log_posterior(
        mu_by_season, gamma, log_kappa_strict, log_kappa_fuzzy, events,
        sigma_mu=sigma_mu, w_percent=w_percent
    )

    draws = {"gamma": [], "log_kappa_strict": [], "log_kappa_fuzzy": [], "mu_by_season": []}
    acc = {"mu": 0, "gamma": 0, "logk_strict": 0, "logk_fuzzy": 0}
    prop = {"mu": 0, "gamma": 0, "logk_strict": 0, "logk_fuzzy": 0}

    win_acc = {"mu": 0, "gamma": 0, "logk_strict": 0, "logk_fuzzy": 0}
    win_prop = {"mu": 0, "gamma": 0, "logk_strict": 0, "logk_fuzzy": 0}

    def adapt_step(step: float, acc_rate: float) -> float:
        if acc_rate > target_acc + 0.10:
            return step * 1.25
        if acc_rate < target_acc - 0.10:
            return step * 0.80
        return step

    for it in range(n_iter):
        if (it + 1) % 50 == 0:
            print(f"[progress] iter {it + 1}/{n_iter}")
        # --- mu (season blocks) ---
        for s in seasons:
            prop["mu"] += 1
            win_prop["mu"] += 1
            mu_new = {k: v.copy() for k, v in mu_by_season.items()}
            mu_new[s] = center_mu(mu_new[s] + rng.normal(0, step_mu, size=mu_new[s].shape))

            cand_lp = log_posterior(
                mu_new, gamma, log_kappa_strict, log_kappa_fuzzy, events,
                sigma_mu=sigma_mu, w_percent=w_percent
            )
            if np.log(rng.random()) < cand_lp - cur_lp:
                mu_by_season = mu_new
                cur_lp = cand_lp
                acc["mu"] += 1
                win_acc["mu"] += 1

        # --- gamma ---
        prop["gamma"] += 1
        win_prop["gamma"] += 1
        gamma_new = gamma + rng.normal(0, step_gamma)
        cand_lp = log_posterior(
            mu_by_season, gamma_new, log_kappa_strict, log_kappa_fuzzy, events,
            sigma_mu=sigma_mu, w_percent=w_percent
        )
        if np.log(rng.random()) < cand_lp - cur_lp:
            gamma = gamma_new
            cur_lp = cand_lp
            acc["gamma"] += 1
            win_acc["gamma"] += 1

        # --- log_kappa_strict ---
        prop["logk_strict"] += 1
        win_prop["logk_strict"] += 1
        logk_s_new = log_kappa_strict + rng.normal(0, step_logk_strict)
        cand_lp = log_posterior(
            mu_by_season, gamma, logk_s_new, log_kappa_fuzzy, events,
            sigma_mu=sigma_mu, w_percent=w_percent
        )
        if np.log(rng.random()) < cand_lp - cur_lp:
            log_kappa_strict = logk_s_new
            cur_lp = cand_lp
            acc["logk_strict"] += 1
            win_acc["logk_strict"] += 1

        # --- log_kappa_fuzzy ---
        prop["logk_fuzzy"] += 1
        win_prop["logk_fuzzy"] += 1
        logk_f_new = log_kappa_fuzzy + rng.normal(0, step_logk_fuzzy)
        cand_lp = log_posterior(
            mu_by_season, gamma, log_kappa_strict, logk_f_new, events,
            sigma_mu=sigma_mu, w_percent=w_percent
        )
        if np.log(rng.random()) < cand_lp - cur_lp:
            log_kappa_fuzzy = logk_f_new
            cur_lp = cand_lp
            acc["logk_fuzzy"] += 1
            win_acc["logk_fuzzy"] += 1

        # --- store ---
        if it >= burn and ((it - burn) % thin == 0):
            draws["gamma"].append(float(gamma))
            draws["log_kappa_strict"].append(float(log_kappa_strict))
            draws["log_kappa_fuzzy"].append(float(log_kappa_fuzzy))
            draws["mu_by_season"].append({s: mu_by_season[s].copy() for s in seasons})

        # --- adapt ---
        if adapt and it < burn and (it + 1) % adapt_window == 0:
            for key, step_name in [
                ("mu", "step_mu"),
                ("gamma", "step_gamma"),
                ("logk_strict", "step_logk_strict"),
                ("logk_fuzzy", "step_logk_fuzzy"),
            ]:
                rate = win_acc[key] / max(1, win_prop[key])
                if key == "mu":
                    step_mu = adapt_step(step_mu, rate)
                elif key == "gamma":
                    step_gamma = adapt_step(step_gamma, rate)
                elif key == "logk_strict":
                    step_logk_strict = adapt_step(step_logk_strict, rate)
                else:
                    step_logk_fuzzy = adapt_step(step_logk_fuzzy, rate)

            win_acc = {k: 0 for k in win_acc}
            win_prop = {k: 0 for k in win_prop}

        if (it + 1) % 1000 == 0:
            print(
                f"[w={w_percent}] iter {it+1}/{n_iter} lp={cur_lp:.2f} "
                f"gamma={gamma:.3f} "
                f"k_strict={np.exp(log_kappa_strict):.3f} "
                f"k_fuzzy={np.exp(log_kappa_fuzzy):.3f} "
                f"steps=({step_mu:.3f},{step_gamma:.3f},{step_logk_strict:.3f},{step_logk_fuzzy:.3f})",
                flush=True,
            )

    accept_rates = {k: acc[k] / max(1, prop[k]) for k in acc}
    return draws, accept_rates


def weekly_elim_probs(ev, mu_s, gamma, kappa_strict, kappa_fuzzy, w_percent):
    """
    后验预测用：q ∝ exp(kappa * hazard)
    kappa 按 season 切换
    """
    kappa = float(kappa_strict if ev.season <= 27 else kappa_fuzzy)

    active = np.array(ev.active_ids, dtype=int)
    mu_vec = mu_s[active]
    p = vote_share(mu_vec, ev.zJ, gamma)

    if ev.rule == "percent":
        combined = w_percent * ev.j_percent + (1.0 - w_percent) * p
        hazard = -combined
    else:
        orderJ = np.argsort(-ev.J)
        rj = np.empty(len(ev.J), dtype=float)
        rj[orderJ] = np.arange(1, len(ev.J) + 1, dtype=float)

        orderP = np.argsort(-p)
        rp = np.empty(len(p), dtype=float)
        rp[orderP] = np.arange(1, len(p) + 1, dtype=float)

        hazard = 0.5 * (rj + rp)

    logits = kappa * hazard
    m = np.max(logits)
    q = np.exp(logits - (m + np.log(np.sum(np.exp(logits - m)) + 1e-300)))
    return q


def export_all(df, judge_long, events, draws, accept_rates, outdir: Path, w_percent: float):
    if len(draws["gamma"]) == 0:
        raise ValueError("draws 为空：检查 n_iter/burn/thin 或 MCMC 是否提前中断。")
    outdir.mkdir(exist_ok=True)

    seasons = sorted({e.season for e in events})
    name_map = {(int(r.season), int(r.contestant_id)): str(r.celebrity_name)
                for r in df[["season", "contestant_id", "celebrity_name"]].itertuples(index=False)}

    gamma_arr = np.array(draws["gamma"])
    k_strict_arr = np.exp(np.array(draws["log_kappa_strict"]))
    k_fuzzy_arr = np.exp(np.array(draws["log_kappa_fuzzy"]))

    # ---- global params
    global_rows = [
        {"param_name": "gamma", **summarize_array(gamma_arr)},
        {"param_name": "kappa_strict", **summarize_array(k_strict_arr)},
        {"param_name": "kappa_fuzzy", **summarize_array(k_fuzzy_arr)},
    ]
    pd.DataFrame(global_rows).to_csv(outdir / "posterior_global_params.csv", index=False)

    # ---- mu popularity summary
    mu_rows = []
    for s in seasons:
        mu_stack = np.stack([d[s] for d in draws["mu_by_season"]], axis=0)
        for i in range(mu_stack.shape[1]):
            stats = summarize_array(mu_stack[:, i])
            mu_rows.append({
                "season": s,
                "contestant_id": i,
                "celebrity_name": name_map.get((s, i), ""),
                "mu_mean": stats["mean"],
                "mu_median": stats["median"],
                "mu_q025": stats["q025"],
                "mu_q975": stats["q975"],
            })

    mu_df = pd.DataFrame(mu_rows)
    mu_df["mu_rank"] = mu_df.groupby("season")["mu_mean"].rank(ascending=False, method="min").astype(int)
    mu_df.to_csv(outdir / "posterior_popularity_summary.csv", index=False)

    # ---- vote share summary（仍按 events 的 active 输出；你之前若改了“补齐周数”的版本，这里可继续用那套）
    p_store = defaultdict(list)
    for d in range(len(draws["gamma"])):
        gamma = draws["gamma"][d]
        mu_by_season = draws["mu_by_season"][d]
        for ev in events:
            mu_s = mu_by_season[ev.season]
            mu_vec = mu_s[np.array(ev.active_ids, dtype=int)]
            p = vote_share(mu_vec, ev.zJ, gamma)
            for cid, pi in zip(ev.active_ids, p):
                p_store[(ev.season, ev.week, int(cid))].append(float(pi))

    vote_rows = []
    for (s, w, cid), vals in p_store.items():
        arr = np.array(vals)
        stats = summarize_array(arr)
        jt = judge_long.loc[
            (judge_long["season"] == s) & (judge_long["week"] == w) & (judge_long["contestant_id"] == cid),
            "judge_total"
        ]
        judge_total = float(jt.iloc[0]) if len(jt) else np.nan

        vote_rows.append({
            "season": s,
            "week": w,
            "contestant_id": cid,
            "celebrity_name": name_map.get((s, cid), ""),
            "judge_total": judge_total,
            "p_mean": stats["mean"],
            "p_median": stats["median"],
            "p_q025": stats["q025"],
            "p_q975": stats["q975"],
        })

    pd.DataFrame(vote_rows).to_csv(outdir / "posterior_vote_share_summary.csv", index=False)

    # ---- weekly elimination check
    week_prob_draws = defaultdict(list)
    for d in range(len(draws["gamma"])):
        gamma = float(draws["gamma"][d])
        k_strict = float(np.exp(draws["log_kappa_strict"][d]))
        k_fuzzy = float(np.exp(draws["log_kappa_fuzzy"][d]))
        mu_by_season = draws["mu_by_season"][d]

        for ev in events:
            if ev.skip_likelihood:
                continue
            q = weekly_elim_probs(ev, mu_by_season[ev.season], gamma, k_strict, k_fuzzy, w_percent=w_percent)
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
            is_in_top2 = obs in top_ids[:2]

        check_rows.append({
            "season": ev.season,
            "week": ev.week,
            "observed_eliminated_name": observed_name,
            "pred_elim_prob_observed": observed_prob,
            "top1_pred_name": name_map.get((ev.season, top_ids[0]), "") if top_ids[0] is not None else "",
            "top1_prob": top_probs[0],
            "top2_pred_name": name_map.get((ev.season, top_ids[1]), "") if top_ids[1] is not None else "",
            "top2_prob": top_probs[1],
            "top3_pred_name": name_map.get((ev.season, top_ids[2]), "") if top_ids[2] is not None else "",
            "top3_prob": top_probs[2],
            "is_observed_in_top2": bool(is_in_top2),
            "notes": ""
        })

    pd.DataFrame(check_rows).to_csv(outdir / "weekly_elimination_check.csv", index=False)

    # ---- diagnostics
    diag = {
        "w_percent": w_percent,
        "n_draws": len(draws["gamma"]),
        "accept_rates": accept_rates,
        "gamma_summary": summarize_array(gamma_arr),
        "kappa_strict_summary": summarize_array(k_strict_arr),
        "kappa_fuzzy_summary": summarize_array(k_fuzzy_arr),
    }
    with open(outdir / "mcmc_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)


def main(
    w_list=(0.5,),
    n_iter=50000, burn=15000, thin=5,
    seed=0,
    sigma_mu=3.0
):
    df = load_data(CSV_PATH)
    weeks = get_week_list(df)
    judge_long = build_judge_total_long(df, weeks)
    events = build_events(df, judge_long, weeks)
    if len(events) == 0:
        raise ValueError("events 为空：请检查 CSV 是否读取成功、week*_judge_score_sum 列是否存在。")

    seasons = sorted({e.season for e in events})
    season_sizes = {s: int(df.loc[df["season"] == s, "contestant_id"].max()) + 1 for s in seasons}

    for w in w_list:
        outdir = BASE_OUTDIR / f"w_{str(w).replace('.','p')}_sigmu_{str(sigma_mu).replace('.','p')}"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "run_started.txt").write_text("run started\n", encoding="utf-8")

        print("\n" + "="*80)
        print(f"Running MCMC w_percent={w}, sigma_mu={sigma_mu}, outputs -> {outdir}")

        draws, accept_rates = run_mcmc(
            events, seasons, season_sizes,
            n_iter=n_iter, burn=burn, thin=thin,
            step_mu=0.06, step_gamma=0.06, step_logk_strict=0.05, step_logk_fuzzy=0.05,
            w_percent=w, sigma_mu=sigma_mu, seed=seed
        )
        export_all(df, judge_long, events, draws, accept_rates, outdir, w_percent=w)
        (outdir / "run_finished.txt").write_text("run finished\n", encoding="utf-8")
        print(f"Done w={w}. Accept rates={accept_rates}")


if __name__ == "__main__":
    main(w_list=(0.5,), n_iter=300, burn=200, thin=5, seed=0, sigma_mu=3.0)
