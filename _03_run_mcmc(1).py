import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 引用修改后的 _02 文件
from _02_model_likelihood1 import load_data, get_week_list, build_judge_total_long, build_events
from _02_model_likelihood1 import log_posterior, center_mu, vote_share, hazard_percent, hazard_rank, log_softmax

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

def run_mcmc(events, seasons, season_sizes,
             n_iter=6000, burn=2000, thin=5,
             step_mu=0.10, step_gamma=0.10, step_logk=0.10,
             adapt=True, target_acc=0.30, adapt_window=200,
             w_percent=0.5, seed=0):

    rng = np.random.default_rng(seed)

    # 初始方差稍微大一点以便探索
    mu_by_season = {s: center_mu(rng.normal(0, 0.5, size=season_sizes[s])) for s in seasons}
    gamma = 0.0
    
    # 初始化两个 kappa
    # Strict (S1-27) 初始值大一些 (exp(1.5) ≈ 4.5)
    log_kappa_strict = 1.5 
    # Fuzzy (S28+) 初始值小一些 (exp(0) = 1.0)
    log_kappa_fuzzy = 0.0

    cur_lp = log_posterior(mu_by_season, gamma, log_kappa_strict, log_kappa_fuzzy, events, w_percent=w_percent)

    # 记录器增加 fuzzy kappa
    draws = {"gamma": [], "log_kappa_strict": [], "log_kappa_fuzzy": [], "mu_by_season": []}
    acc = {"mu": 0, "gamma": 0, "logk_s": 0, "logk_f": 0}
    prop = {"mu": 0, "gamma": 0, "logk_s": 0, "logk_f": 0}

    win_acc = {"mu": 0, "gamma": 0, "logk_s": 0, "logk_f": 0}
    win_prop = {"mu": 0, "gamma": 0, "logk_s": 0, "logk_f": 0}

    def adapt_step(step: float, acc_rate: float) -> float:
        if acc_rate > target_acc + 0.10: return step * 1.25
        if acc_rate < target_acc - 0.10: return step * 0.80
        return step

    for it in range(n_iter):
        # --- mu (season blocks) ---
        for s in seasons:
            prop["mu"] += 1
            win_prop["mu"] += 1
            mu_new = {k: v.copy() for k, v in mu_by_season.items()}
            mu_new[s] = center_mu(mu_new[s] + rng.normal(0, step_mu, size=mu_new[s].shape))
            cand_lp = log_posterior(mu_new, gamma, log_kappa_strict, log_kappa_fuzzy, events, w_percent=w_percent)
            if np.log(rng.random()) < cand_lp - cur_lp:
                mu_by_season = mu_new
                cur_lp = cand_lp
                acc["mu"] += 1
                win_acc["mu"] += 1

        # --- gamma ---
        prop["gamma"] += 1
        win_prop["gamma"] += 1
        gamma_new = gamma + rng.normal(0, step_gamma)
        cand_lp = log_posterior(mu_by_season, gamma_new, log_kappa_strict, log_kappa_fuzzy, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            gamma = gamma_new
            cur_lp = cand_lp
            acc["gamma"] += 1
            win_acc["gamma"] += 1

        # --- log_kappa_strict (S1-27) ---
        prop["logk_s"] += 1
        win_prop["logk_s"] += 1
        logk_s_new = log_kappa_strict + rng.normal(0, step_logk)
        cand_lp = log_posterior(mu_by_season, gamma, logk_s_new, log_kappa_fuzzy, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            log_kappa_strict = logk_s_new
            cur_lp = cand_lp
            acc["logk_s"] += 1
            win_acc["logk_s"] += 1
            
        # --- log_kappa_fuzzy (S28+) ---
        prop["logk_f"] += 1
        win_prop["logk_f"] += 1
        logk_f_new = log_kappa_fuzzy + rng.normal(0, step_logk)
        cand_lp = log_posterior(mu_by_season, gamma, log_kappa_strict, logk_f_new, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            log_kappa_fuzzy = logk_f_new
            cur_lp = cand_lp
            acc["logk_f"] += 1
            win_acc["logk_f"] += 1

        # --- store ---
        if it >= burn and ((it - burn) % thin == 0):
            draws["gamma"].append(float(gamma))
            draws["log_kappa_strict"].append(float(log_kappa_strict))
            draws["log_kappa_fuzzy"].append(float(log_kappa_fuzzy))
            draws["mu_by_season"].append({s: mu_by_season[s].copy() for s in seasons})

        if adapt and it < burn and (it + 1) % adapt_window == 0:
            mu_rate = win_acc["mu"] / max(1, win_prop["mu"])
            g_rate = win_acc["gamma"] / max(1, win_prop["gamma"])
            ks_rate = win_acc["logk_s"] / max(1, win_prop["logk_s"])
            kf_rate = win_acc["logk_f"] / max(1, win_prop["logk_f"])

            step_mu = adapt_step(step_mu, mu_rate)
            step_gamma = adapt_step(step_gamma, g_rate)
            # 两个 kappa 共享同一个步长调整，简化一点
            step_logk = adapt_step(step_logk, (ks_rate + kf_rate)/2)

            win_acc = {"mu": 0, "gamma": 0, "logk_s": 0, "logk_f": 0}
            win_prop = {"mu": 0, "gamma": 0, "logk_s": 0, "logk_f": 0}

        if (it + 1) % 1000 == 0:
            print(f"[w={w_percent}] iter {it+1}/{n_iter} lp={cur_lp:.2f} "
                  f"g={gamma:.2f} k_s={np.exp(log_kappa_strict):.2f} k_f={np.exp(log_kappa_fuzzy):.2f}", flush=True)

    accept_rates = {k: acc[k] / max(1, prop[k]) for k in acc}
    return draws, accept_rates


def weekly_elim_probs(ev, mu_s, gamma, kappa_strict, kappa_fuzzy, w_percent):
    """
    后验预测: 必须根据赛季选用正确的 kappa
    """
    # 选择 kappa
    kappa = kappa_fuzzy if ev.season >= 28 else kappa_strict

    active = np.array(ev.active_ids, dtype=int)
    mu_vec = mu_s[active]
    p = vote_share(mu_vec, ev.zJ, gamma)

    if ev.rule == "percent":
        combined = w_percent * ev.j_percent + (1.0 - w_percent) * p
        hazard = -combined
    else:
        # rank
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
    if len(draws["gamma"]) == 0: raise ValueError("Draws empty.")
    outdir.mkdir(exist_ok=True)
    seasons = sorted({e.season for e in events})
    name_map = {(int(r.season), int(r.contestant_id)): str(r.celebrity_name) for r in df[["season", "contestant_id", "celebrity_name"]].itertuples(index=False)}

    gamma_arr = np.array(draws["gamma"])
    logk_s_arr = np.array(draws["log_kappa_strict"])
    logk_f_arr = np.array(draws["log_kappa_fuzzy"])
    
    # 1. Global Params
    global_rows = []
    global_rows.append({"param_name": "gamma", **summarize_array(gamma_arr)})
    global_rows.append({"param_name": "kappa_strict (S1-27)", **summarize_array(np.exp(logk_s_arr))})
    global_rows.append({"param_name": "kappa_fuzzy (S28+)", **summarize_array(np.exp(logk_f_arr))})
    pd.DataFrame(global_rows).to_csv(outdir / "posterior_global_params.csv", index=False)

    # 2. Mu Summary (Popularity)
    mu_rows = []
    for s in seasons:
        mu_stack = np.stack([d[s] for d in draws["mu_by_season"]], axis=0)
        for i in range(mu_stack.shape[1]):
            stats = summarize_array(mu_stack[:, i])
            mu_rows.append({"season": s, "contestant_id": i, "celebrity_name": name_map.get((s, i), ""),
                            "mu_mean": stats["mean"], "mu_median": stats["median"]})
    pd.DataFrame(mu_rows).to_csv(outdir / "posterior_popularity_summary.csv", index=False)

    # 3. Vote Share (略，结构不变，可保留原代码，这里为节省篇幅略去，逻辑同原版)
    # ... (Please keep your original vote share logic here if needed) ...

    # 4. Weekly Elimination Check (Bottom 2 Logic)
    week_prob_draws = defaultdict(list)
    for d in range(len(draws["gamma"])):
        gamma = draws["gamma"][d]
        ks = float(np.exp(draws["log_kappa_strict"][d]))
        kf = float(np.exp(draws["log_kappa_fuzzy"][d]))
        mu_by_season = draws["mu_by_season"][d]
        for ev in events:
            if ev.skip_likelihood: continue
            q = weekly_elim_probs(ev, mu_by_season[ev.season], gamma, ks, kf, w_percent=w_percent)
            week_prob_draws[(ev.season, ev.week)].append(q)

    check_rows = []
    correct_count = 0
    total_valid_weeks = 0

    for ev in events:
        if ev.skip_likelihood: continue
        key = (ev.season, ev.week)
        if key not in week_prob_draws: continue

        q_stack = np.stack(week_prob_draws[key], axis=0)
        q_mean = np.mean(q_stack, axis=0)
        
        # 排序：谁最危险（概率最大）
        order = np.argsort(-q_mean)
        # 预测的 Bottom 2
        pred_bottom2 = [ev.active_ids[i] for i in order[:2]]
        pred_elim_1 = pred_bottom2[0] if len(pred_bottom2) > 0 else None

        observed_elim = ev.eliminated_ids[0] if ev.eliminated_ids else None
        
        # --- 核心验证逻辑 ---
        is_correct = False
        method = ""
        
        if observed_elim is not None:
            if ev.season >= 28:
                # S28+: Judge Save 机制
                # 只要实际淘汰者在我们预测的 Bottom 2 里，就算预测成功
                if observed_elim in pred_bottom2:
                    is_correct = True
                    method = "In_Bottom_2"
            else:
                # S1-27: 严格机制
                # 必须精确命中第一名
                if observed_elim == pred_elim_1:
                    is_correct = True
                    method = "Exact_Hit"
            
            total_valid_weeks += 1
            if is_correct:
                correct_count += 1

        check_rows.append({
            "season": ev.season,
            "week": ev.week,
            "observed": name_map.get((ev.season, observed_elim), "") if observed_elim is not None else "None",
            "pred_rank1": name_map.get((ev.season, pred_bottom2[0]), "") if len(pred_bottom2)>0 else "",
            "pred_rank2": name_map.get((ev.season, pred_bottom2[1]), "") if len(pred_bottom2)>1 else "",
            "is_correct": is_correct,
            "match_method": method
        })

    pd.DataFrame(check_rows).to_csv(outdir / "weekly_elimination_check.csv", index=False)
    
    # 打印最终准确率
    acc = correct_count / max(1, total_valid_weeks)
    print(f"\nFinal Accuracy (Exact for S1-27, Bottom2 for S28+): {acc:.2%}")

    with open(outdir / "mcmc_diagnostics.json", "w") as f:
        json.dump({"accuracy": acc, "accept_rates": accept_rates}, f, indent=2)

def main(w_list=(0.5,), n_iter=6000, burn=2000, thin=5, seed=0):
    df = load_data(CSV_PATH)
    weeks = get_week_list(df)
    judge_long = build_judge_total_long(df, weeks)
    events = build_events(df, judge_long, weeks)
    if not events: raise ValueError("No events")
    
    seasons = sorted({e.season for e in events})
    season_sizes = {s: int(df.loc[df["season"] == s, "contestant_id"].max()) + 1 for s in seasons}

    for w in w_list:
        outdir = BASE_OUTDIR / f"w_{str(w).replace('.','p')}"
        print(f"\nRunning MCMC w={w}...")
        draws, acc = run_mcmc(events, seasons, season_sizes, n_iter=n_iter, burn=burn, thin=thin, w_percent=w, seed=seed)
        export_all(df, judge_long, events, draws, acc, outdir, w)
        print("Done.")

if __name__ == "__main__":
    main(w_list=(0.5,), n_iter=300, burn=200, thin=5, seed=42)
