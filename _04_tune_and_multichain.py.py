import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from _02_model_likelihood import (
    load_data, get_week_list, build_judge_total_long, build_events,
    log_posterior, center_mu
)

# ====== 配置 ======
CSV_PATH = "/Users/garytchois/Desktop/vs/2026_MCM_Problem_C_Data_with_week_stats1.csv"
OUTDIR = Path("mcmc_outputs/04_multichain")
OUTDIR.mkdir(parents=True, exist_ok=True)

# 目标接受率（Metropolis RW 常用）
TARGET_ACC = 0.30

# 自适应窗口大小（每隔多少次 proposal 更新一次 step）
ADAPT_WINDOW = 200

# 多链设置
SEEDS = [0, 1, 2]


# ====== 统计工具：R-hat & ESS（简化版，足够论文） ======
def split_rhat(chains: np.ndarray) -> float:
    """
    chains: shape (n_chain, n_draw)
    split-Rhat（Gelman-Rubin）: 把每条链对半分成 2*n_chain 条链计算
    """
    m, n = chains.shape
    if n < 20:
        return float("nan")

    # split
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2*half]], axis=0)  # (2m, half)
    m2, n2 = split.shape

    chain_means = np.mean(split, axis=1)
    chain_vars = np.var(split, axis=1, ddof=1)

    W = np.mean(chain_vars)
    B = n2 * np.var(chain_means, ddof=1)

    var_hat = (n2 - 1) / n2 * W + B / n2
    return float(np.sqrt(var_hat / (W + 1e-12)))

def ess_bulk_approx(x: np.ndarray, max_lag: int = 200) -> float:
    """
    近似 ESS（单链），基于自相关和（非常简化但够用）
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 50:
        return float("nan")
    x = x - np.mean(x)
    var = np.dot(x, x) / n
    if var <= 0:
        return float("nan")

    def autocorr(lag):
        return float(np.dot(x[:-lag], x[lag:]) / ((n - lag) * var + 1e-12))

    rho_sum = 0.0
    for lag in range(1, min(max_lag, n // 2)):
        rho = autocorr(lag)
        if rho < 0:  # Geyer 初步截断思想：负相关后停止
            break
        rho_sum += rho

    tau = 1 + 2 * rho_sum
    return float(n / (tau + 1e-12))

def multi_ess(chains: np.ndarray) -> float:
    """
    把多链拼起来做一个保守 ESS：取各链 ESS 的和
    """
    return float(np.sum([ess_bulk_approx(chains[i]) for i in range(chains.shape[0])]))


# ====== 自适应 MCMC（只在 burn-in 内调 step） ======
def run_chain_adaptive(events, seasons, season_sizes,
                       n_iter=12000, burn=6000, thin=5,
                       step_mu=0.06, step_gamma=0.06, step_logk=0.06,
                       w_percent=0.5, seed=0) -> Dict[str, Any]:

    rng = np.random.default_rng(seed)

    mu_by_season = {s: center_mu(rng.normal(0, 0.1, size=season_sizes[s])) for s in seasons}
    gamma = 0.0
    log_kappa = 0.0
    cur_lp = log_posterior(mu_by_season, gamma, log_kappa, events, w_percent=w_percent)

    draws = {"gamma": [], "log_kappa": []}
    acc = {"mu": 0, "gamma": 0, "logk": 0}
    prop = {"mu": 0, "gamma": 0, "logk": 0}

    # 适应统计（窗口内）
    win_acc = {"mu": 0, "gamma": 0, "logk": 0}
    win_prop = {"mu": 0, "gamma": 0, "logk": 0}

    def adapt_step(step, a_rate):
        # 简单乘法调节：太高 => 放大步长；太低 => 缩小步长
        if a_rate > TARGET_ACC + 0.10:
            return step * 1.25
        if a_rate < TARGET_ACC - 0.10:
            return step * 0.80
        return step

    for it in range(n_iter):
        # --- mu block ---
        for s in seasons:
            prop["mu"] += 1
            win_prop["mu"] += 1

            mu_new = {k: v.copy() for k, v in mu_by_season.items()}
            mu_new[s] = center_mu(mu_new[s] + rng.normal(0, step_mu, size=mu_new[s].shape))

            cand_lp = log_posterior(mu_new, gamma, log_kappa, events, w_percent=w_percent)
            if np.log(rng.random()) < cand_lp - cur_lp:
                mu_by_season = mu_new
                cur_lp = cand_lp
                acc["mu"] += 1
                win_acc["mu"] += 1

        # --- gamma ---
        prop["gamma"] += 1
        win_prop["gamma"] += 1
        gamma_new = gamma + rng.normal(0, step_gamma)
        cand_lp = log_posterior(mu_by_season, gamma_new, log_kappa, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            gamma = gamma_new
            cur_lp = cand_lp
            acc["gamma"] += 1
            win_acc["gamma"] += 1

        # --- logk ---
        prop["logk"] += 1
        win_prop["logk"] += 1
        logk_new = log_kappa + rng.normal(0, step_logk)
        cand_lp = log_posterior(mu_by_season, gamma, logk_new, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            log_kappa = logk_new
            cur_lp = cand_lp
            acc["logk"] += 1
            win_acc["logk"] += 1

        # --- adapt (only during burn-in) ---
        if it < burn and (it + 1) % ADAPT_WINDOW == 0:
            mu_rate = win_acc["mu"] / max(1, win_prop["mu"])
            g_rate = win_acc["gamma"] / max(1, win_prop["gamma"])
            k_rate = win_acc["logk"] / max(1, win_prop["logk"])

            step_mu = adapt_step(step_mu, mu_rate)
            step_gamma = adapt_step(step_gamma, g_rate)
            step_logk = adapt_step(step_logk, k_rate)

            win_acc = {"mu": 0, "gamma": 0, "logk": 0}
            win_prop = {"mu": 0, "gamma": 0, "logk": 0}

        # --- store ---
        if it >= burn and ((it - burn) % thin == 0):
            draws["gamma"].append(float(gamma))
            draws["log_kappa"].append(float(log_kappa))

        if (it + 1) % 2000 == 0:
            print(f"[seed={seed}] it {it+1}/{n_iter} lp={cur_lp:.2f} "
                  f"gamma={gamma:.3f} kappa={np.exp(log_kappa):.3f} "
                  f"steps(mu,g,k)=({step_mu:.3f},{step_gamma:.3f},{step_logk:.3f})",
                  flush=True)

    accept_rates = {k: acc[k] / max(1, prop[k]) for k in acc}

    return {
        "seed": seed,
        "w_percent": w_percent,
        "n_iter": n_iter,
        "burn": burn,
        "thin": thin,
        "final_steps": {"mu": step_mu, "gamma": step_gamma, "logk": step_logk},
        "accept_rates": accept_rates,
        "draws": draws
    }


def plot_traces(chains_gamma, chains_kappa, outdir: Path):
    # gamma
    plt.figure()
    for i in range(chains_gamma.shape[0]):
        plt.plot(chains_gamma[i], linewidth=0.8)
    plt.title("Trace: gamma (multi-chain)")
    plt.xlabel("draw")
    plt.ylabel("gamma")
    plt.tight_layout()
    plt.savefig(outdir / "trace_gamma.png", dpi=160)
    plt.close()

    # kappa
    plt.figure()
    for i in range(chains_kappa.shape[0]):
        plt.plot(chains_kappa[i], linewidth=0.8)
    plt.title("Trace: kappa (multi-chain)")
    plt.xlabel("draw")
    plt.ylabel("kappa")
    plt.tight_layout()
    plt.savefig(outdir / "trace_kappa.png", dpi=160)
    plt.close()


def main(w_percent=0.5):
    df = load_data(CSV_PATH)
    weeks = get_week_list(df)
    judge_long = build_judge_total_long(df, weeks)
    events = build_events(df, judge_long, weeks)

    seasons = sorted({e.season for e in events})
    season_sizes = {s: int(df.loc[df["season"] == s, "contestant_id"].max()) + 1 for s in seasons}

    chain_results = []
    for seed in SEEDS:
        res = run_chain_adaptive(
            events, seasons, season_sizes,
            n_iter=12000, burn=6000, thin=5,
            step_mu=0.06, step_gamma=0.06, step_logk=0.06,
            w_percent=w_percent, seed=seed
        )
        chain_results.append(res)
        with open(OUTDIR / f"diagnostics_chain_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump({
                k: res[k] for k in ["seed", "w_percent", "n_iter", "burn", "thin", "final_steps", "accept_rates"]
            }, f, ensure_ascii=False, indent=2)

    # stack chains
    chains_gamma = np.stack([np.array(r["draws"]["gamma"]) for r in chain_results], axis=0)
    chains_logk = np.stack([np.array(r["draws"]["log_kappa"]) for r in chain_results], axis=0)
    chains_kappa = np.exp(chains_logk)

    # diagnostics
    diag = {
        "w_percent": w_percent,
        "n_chain": len(SEEDS),
        "n_draws_each": int(chains_gamma.shape[1]),
        "split_rhat_gamma": split_rhat(chains_gamma),
        "split_rhat_kappa": split_rhat(chains_kappa),
        "ess_gamma_total": multi_ess(chains_gamma),
        "ess_kappa_total": multi_ess(chains_kappa),
        "accept_rates": {f"seed{r['seed']}": r["accept_rates"] for r in chain_results},
        "final_steps": {f"seed{r['seed']}": r["final_steps"] for r in chain_results},
    }

    with open(OUTDIR / "combined_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    plot_traces(chains_gamma, chains_kappa, OUTDIR)
    print("Saved:", OUTDIR / "combined_diagnostics.json")
    print("Saved:", OUTDIR / "trace_gamma.png")
    print("Saved:", OUTDIR / "trace_kappa.png")


if __name__ == "__main__":
    main(w_percent=0.5)
