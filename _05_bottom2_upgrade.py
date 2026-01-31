import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from _02_model_likelihood import (
    load_data, get_week_list, build_judge_total_long, build_events,
    log_softmax, vote_share, hazard_percent, hazard_rank, center_mu
)

CSV_PATH = "/Users/garytchois/Desktop/vs/2026_MCM_Problem_C_Data_with_week_stats1.csv"
OUTDIR = Path("mcmc_outputs/05_bottom2")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------- bottom2 + judge-save likelihood (S28-34) ----------
def sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = np.exp(-x)
        return float(1 / (1 + z))
    else:
        z = np.exp(x)
        return float(z / (1 + z))

def bottom2_pair_prob(h: np.ndarray, i: int, j: int) -> float:
    """
    hazard h: higher => more dangerous.
    Use Plackett-Luce without replacement:
      P(i then j) = softmax(h)[i] * softmax(h without i)[j]
    unordered {i,j} = P(i then j) + P(j then i)
    """
    # first pick
    logp1 = log_softmax(h)
    # i then j
    pi = np.exp(logp1[i])
    # remove i
    mask_i = np.ones(len(h), dtype=bool)
    mask_i[i] = False
    h2 = h[mask_i]
    logp2 = log_softmax(h2)
    # map j to index in h2
    j2 = j - 1 if j > i else j
    pij = pi * np.exp(logp2[j2])

    # j then i
    pj = np.exp(logp1[j])
    mask_j = np.ones(len(h), dtype=bool)
    mask_j[j] = False
    h3 = h[mask_j]
    logp3 = log_softmax(h3)
    i3 = i - 1 if i > j else i
    pji = pj * np.exp(logp3[i3])

    return float(pij + pji)

def elim_prob_given_bottom2(J_i: float, J_j: float, rho: float, eliminate_i: bool) -> float:
    """
    rho>0: judges more likely eliminate lower-J person.
    P(elim i | {i,j}) = sigmoid( rho * (J_j - J_i) )
      - if J_i << J_j => J_j-J_i positive large => P(elim i) ~ 1
      - if J_i >> J_j => negative => ~0
    """
    p_elim_i = sigmoid(rho * (J_j - J_i))
    return float(p_elim_i if eliminate_i else (1 - p_elim_i))

def elimination_loglik_event_bottom2(ev, mu_s: np.ndarray, gamma: float,
                                     kappa: float, rho: float, w_percent: float) -> float:
    """
    For seasons>=28 (bottom2 + judge-save).
    Only handles single elimination cleanly; for multi-elim weeks we fallback to old rule.
    """
    if getattr(ev, "skip_likelihood", False):
        return 0.0
    if len(ev.eliminated_ids) == 0:
        return 0.0

    # 如果一周淘汰多人，这套 bottom2 结构不够信息识别；先退回旧的“直接淘汰”结构更稳
    if len(ev.eliminated_ids) > 1:
        return np.nan  # caller decides fallback

    active = ev.active_ids
    n = len(active)
    if n < 3:
        return 0.0

    # compute p
    mu_vec = mu_s[np.array(active, dtype=int)]
    p = vote_share(mu_vec, ev.zJ, gamma)

    # hazard base
    if ev.rule == "percent":
        h = hazard_percent(ev.j_percent, p, w=w_percent)  # higher more dangerous (already -combined)
    else:
        h = hazard_rank(ev.J, p)  # higher rank => more dangerous

    # sharpen selection of bottom2
    h = kappa * h

    # observed eliminated id -> index
    e_id = ev.eliminated_ids[0]
    if e_id not in active:
        return -np.inf
    e = active.index(e_id)

    # Sum over partner b != e
    prob = 0.0
    for b in range(n):
        if b == e:
            continue
        pair_p = bottom2_pair_prob(h, e, b)
        # judge-save elimination probability within pair
        J_e = float(ev.J[e])
        J_b = float(ev.J[b])
        prob_elim_e = elim_prob_given_bottom2(J_i=J_e, J_j=J_b, rho=rho, eliminate_i=True)
        prob += pair_p * prob_elim_e

    return float(np.log(prob + 1e-300))


# ---------- old direct elimination likelihood (S1-27 or fallback) ----------
def elimination_loglik_event_direct(ev, mu_s: np.ndarray, gamma: float, kappa: float, w_percent: float) -> float:
    if getattr(ev, "skip_likelihood", False):
        return 0.0
    if len(ev.eliminated_ids) == 0:
        return 0.0

    remaining = list(ev.active_ids)
    ll = 0.0

    for e_id in list(ev.eliminated_ids):
        if e_id not in remaining:
            return -np.inf

        # subset arrays in remaining order
        active_map = {cid: i for i, cid in enumerate(ev.active_ids)}
        sub = np.array([active_map[cid] for cid in remaining], dtype=int)

        mu_vec = mu_s[np.array(remaining, dtype=int)]
        J_sub = ev.J[sub]
        zJ_sub = ev.zJ[sub]
        jperc_sub = ev.j_percent[sub]

        p_sub = vote_share(mu_vec, zJ_sub, gamma)

        if ev.rule == "percent":
            hz = hazard_percent(jperc_sub, p_sub, w=w_percent)
        else:
            hz = hazard_rank(J_sub, p_sub)

        log_q = log_softmax(kappa * hz)
        e_pos = remaining.index(e_id)
        ll += float(log_q[e_pos])
        remaining.remove(e_id)

    return float(ll)


# ---------- priors + posterior (now includes rho) ----------
def log_prior(mu_by_season, gamma, log_kappa, log_rho, sigma_mu=1.0):
    mu_all = np.concatenate([mu_by_season[s] for s in sorted(mu_by_season.keys())])
    lp = 0.0
    lp += -0.5 * np.sum((mu_all / sigma_mu) ** 2)
    lp += -0.5 * (gamma ** 2)
    lp += -0.5 * (log_kappa ** 2)
    lp += -0.5 * (log_rho ** 2)  # rho>0 via exp(log_rho)
    return float(lp)

def log_posterior_bottom2(mu_by_season, gamma, log_kappa, log_rho, events, w_percent=0.5):
    kappa = float(np.exp(log_kappa))
    rho = float(np.exp(log_rho))
    lp = log_prior(mu_by_season, gamma, log_kappa, log_rho)

    ll = 0.0
    for ev in events:
        mu_s = mu_by_season[ev.season]

        if ev.season >= 28:
            l1 = elimination_loglik_event_bottom2(ev, mu_s, gamma, kappa, rho, w_percent=w_percent)
            if np.isnan(l1):
                # multi-elim fallback
                ll += elimination_loglik_event_direct(ev, mu_s, gamma, kappa, w_percent=w_percent)
            else:
                ll += l1
        else:
            ll += elimination_loglik_event_direct(ev, mu_s, gamma, kappa, w_percent=w_percent)

    return float(lp + ll)


# ---------- MCMC runner (adds log_rho update) ----------
def run_mcmc(events, seasons, season_sizes,
             n_iter=12000, burn=6000, thin=5,
             step_mu=0.15, step_gamma=0.12, step_logk=0.12, step_logrho=0.12,
             w_percent=0.5, seed=0):

    rng = np.random.default_rng(seed)

    mu_by_season = {s: center_mu(rng.normal(0, 0.1, size=season_sizes[s])) for s in seasons}
    gamma = 0.0
    log_kappa = 0.0
    log_rho = 0.0

    cur_lp = log_posterior_bottom2(mu_by_season, gamma, log_kappa, log_rho, events, w_percent=w_percent)

    draws = {"gamma": [], "log_kappa": [], "log_rho": []}
    acc = {"mu": 0, "gamma": 0, "logk": 0, "logrho": 0}
    prop = {"mu": 0, "gamma": 0, "logk": 0, "logrho": 0}

    for it in range(n_iter):
        # mu blocks
        for s in seasons:
            prop["mu"] += 1
            mu_new = {k: v.copy() for k, v in mu_by_season.items()}
            mu_new[s] = center_mu(mu_new[s] + rng.normal(0, step_mu, size=mu_new[s].shape))
            cand_lp = log_posterior_bottom2(mu_new, gamma, log_kappa, log_rho, events, w_percent=w_percent)
            if np.log(rng.random()) < cand_lp - cur_lp:
                mu_by_season = mu_new
                cur_lp = cand_lp
                acc["mu"] += 1

        # gamma
        prop["gamma"] += 1
        gamma_new = gamma + rng.normal(0, step_gamma)
        cand_lp = log_posterior_bottom2(mu_by_season, gamma_new, log_kappa, log_rho, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            gamma = gamma_new
            cur_lp = cand_lp
            acc["gamma"] += 1

        # logk
        prop["logk"] += 1
        logk_new = log_kappa + rng.normal(0, step_logk)
        cand_lp = log_posterior_bottom2(mu_by_season, gamma, logk_new, log_rho, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            log_kappa = logk_new
            cur_lp = cand_lp
            acc["logk"] += 1

        # logrho
        prop["logrho"] += 1
        logrho_new = log_rho + rng.normal(0, step_logrho)
        cand_lp = log_posterior_bottom2(mu_by_season, gamma, log_kappa, logrho_new, events, w_percent=w_percent)
        if np.log(rng.random()) < cand_lp - cur_lp:
            log_rho = logrho_new
            cur_lp = cand_lp
            acc["logrho"] += 1

        if it >= burn and ((it - burn) % thin == 0):
            draws["gamma"].append(float(gamma))
            draws["log_kappa"].append(float(log_kappa))
            draws["log_rho"].append(float(log_rho))

        if (it + 1) % 2000 == 0:
            print(f"it {it+1}/{n_iter} lp={cur_lp:.2f} gamma={gamma:.3f} "
                  f"kappa={np.exp(log_kappa):.3f} rho={np.exp(log_rho):.3f}", flush=True)

    accept_rates = {k: acc[k] / max(1, prop[k]) for k in acc}
    return draws, accept_rates


def summarize_array(a):
    a = np.asarray(a, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q025": float(np.quantile(a, 0.025)),
        "q975": float(np.quantile(a, 0.975)),
    }

def main():
    df = load_data(CSV_PATH)
    weeks = get_week_list(df)
    judge_long = build_judge_total_long(df, weeks)
    events = build_events(df, judge_long, weeks)

    seasons = sorted({e.season for e in events})
    season_sizes = {s: int(df.loc[df["season"] == s, "contestant_id"].max()) + 1 for s in seasons}

    draws, accept = run_mcmc(
        events, seasons, season_sizes,
        n_iter=12000, burn=6000, thin=5,
        step_mu=0.15, step_gamma=0.12, step_logk=0.12, step_logrho=0.12,
        w_percent=0.5, seed=0
    )

    gamma = np.array(draws["gamma"])
    kappa = np.exp(np.array(draws["log_kappa"]))
    rho = np.exp(np.array(draws["log_rho"]))

    diag = {
        "w_percent": 0.5,
        "n_draws": len(gamma),
        "accept_rates": accept,
        "gamma_summary": summarize_array(gamma),
        "kappa_summary": summarize_array(kappa),
        "rho_summary": summarize_array(rho),
    }
    with open(OUTDIR / "mcmc_diagnostics_bottom2.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    print("Saved:", OUTDIR / "mcmc_diagnostics_bottom2.json")


if __name__ == "__main__":
    main()
