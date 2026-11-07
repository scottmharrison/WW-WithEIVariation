"""
12_fit_accuracy_mixtures.py

Purpose
    The mixture fits of 07_fit_mixtures.py redone with choices: the model must now produce each
    contrast's change in accuracy and in error RTs as well as in correct RTs, and two further
    ingredients are available, evidence and the top-down control signal of Lo, Wang & Wang (2015).

    Model: model/wwsim_ext.py (the extended Wong-Wang model plus evidence e, the control signal
    through 4_output/bsi_table.csv, and choices).

    Step 1, calibrate A from the WW-CarefulFitting fit named in 2_data_processing/baselines.csv by
    changing six parameters: drive mean u0 and spread u0_sd, log instances logN, non-decision mean
    ter_mu, evidence e (Hz; pools at u0 +- e/2), and the instance route's noise sigL (log factor q), so
    that the route's accuracy is set by the data rather than left at its RT-only value
    (process/output/check_06.txt). Loss: sum over the seven correct-RT percentiles of the squared log
    error, plus 7 x (model accuracy - observed accuracy)^2, so that a 1-point accuracy miss costs about
    as much as a 1% miss at one percentile. Contrasts without accuracy (P3, S1) keep e = 0 and sigL.

    Step 2, fit B from calibrated A. Changes:
        s  skill: u0_sd x exp(s), u0 fixed          p  practice: logN + p
        d  drive: u0 and u0_sd x exp(d)             e  evidence: e + de Hz (kept >= 0)
        cS, cR control signal strength and ratio    t  non-decision: ter_mu + t ms
    Models: S, P, SP, SP+drive, SP+evidence, SP+control, SP+ndt, SP+control+ndt, SP+drive+ndt.
    Each starts from the best solution of the models nested in it (no change for S and P) and, for the
    control signal, also from (cS, cR) = (0.3, 1.1) and (0.3, 1.3), on either side of the critical ratio.

    Criterion: chi-square over the terms the contrast has: percent change at the 7 correct-RT
    percentiles; change in accuracy; percent change at the 5 error-RT percentiles where A and B each
    have >= 100 pooled errors (4_output/pooled_accuracy_profiles.csv). SE = data bootstrap SE combined
    with the simulation's SE at the solution (4 further seeds). Where the model has fewer than 30 errors
    in A or B, its error-RT terms cannot be computed and each costs 9 (a 3-SE miss). df = terms - changes;
    adequate when p > .05. Common random numbers; 5,000 simulated trials per condition.

Inputs
    2_data_processing/baselines.csv, 4_output/pooled_accuracy_profiles.csv, 4_output/bsi_table.csv

Outputs
    4_output/acc_mixtures/<contrast>.json — calibration and every fit (collected by 13)

Depends on
    11_pooled_accuracy_profiles.py, 10_build_bsi_table.py, model/wwsim_ext.py

Seed
    One per contrast (1000 + sum of the contrast id's character codes), written to its file.

Last substantive change
    2026-11-07 — first version.

Run from the project root. Hours of CPU; contrasts run in parallel; a contrast whose file exists is
skipped (delete it to refit).
"""
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import wwsim_ext  # noqa: E402

PCTS = [5, 10, 25, 50, 75, 90, 99]
ERR_PCTS = [10, 30, 50, 70, 90]
N_TRIALS, T_MAX = 5000, 10000
NO_ACCURACY = {"P3", "S1"}
KEYS = ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]
BOUNDS = {"s": (-3, 2), "p": (-5, 6), "d": (-1.5, 1.5), "e": (-20, 30), "cS": (0, 2), "cR": (1.0, 1.4), "t": (-300, 600),
          "e_abs": (0, 30), "q": (-3, 2)}
STEP = {"s": 0.5, "p": 1.0, "d": 0.3, "e": 2.0, "cS": 0.3, "cR": 0.1, "t": 50.0, "e_abs": 2.0, "q": 0.5}
MODELS = {"S": ["s"], "P": ["p"], "SP": ["s", "p"], "SP+drive": ["s", "p", "d"], "SP+evidence": ["s", "p", "e"],
          "SP+control": ["s", "p", "cS", "cR"], "SP+ndt": ["s", "p", "t"], "SP+control+ndt": ["s", "p", "cS", "cR", "t"],
          "SP+drive+ndt": ["s", "p", "d", "t"]}
NESTED_IN = {"S": [], "P": [], "SP": ["S", "P"], "SP+drive": ["SP"], "SP+evidence": ["SP"], "SP+control": ["SP"],
             "SP+ndt": ["SP"], "SP+control+ndt": ["SP+control", "SP+ndt"], "SP+drive+ndt": ["SP+drive", "SP+ndt"]}
OUT = Path("4_output/acc_mixtures")


def clipped(names, x):
    return {n: float(np.clip(v, *BOUNDS[n])) for n, v in zip(names, x)}


def apply(theta, ch):
    th = dict(theta)
    th["u0_sd"] = theta["u0_sd"] * np.exp(ch.get("s", 0.0) + ch.get("d", 0.0))
    th["u0"] = theta["u0"] * np.exp(ch.get("d", 0.0))
    th["logN"] = theta["logN"] + ch.get("p", 0.0)
    th["ter_mu"] = theta["ter_mu"] + ch.get("t", 0.0)
    th["e"] = max(theta.get("e", 0.0) + ch.get("e", 0.0), 0.0)
    if "e_abs" in ch:
        th["e"] = ch["e_abs"]
    if "q" in ch:
        th["sigL"] = theta["sigL"] * np.exp(ch["q"])
    if "cS" in ch:
        th["bsi_S"], th["bsi_R"] = ch["cS"], ch["cR"]
    return th


def summarise(theta, seed):
    """Correct-RT percentiles, accuracy, error-RT percentiles (None if < 30 errors) of a simulation.
    Trials with no commitment by T_MAX count as slow correct-RT trials at T_MAX + 1000 ms."""
    sim = wwsim_ext.simulate_condition(theta, N_TRIALS, T_MAX, seed)
    rt, ch = sim["rt"], sim["choice"]
    rt_c = np.where(np.isfinite(rt[ch != 0]), rt[ch != 0], T_MAX + 1000.0)
    decided = ch >= 0
    acc = float(np.mean(ch[decided] == 1)) if decided.any() else np.nan
    err = rt[ch == 0]
    return np.percentile(rt_c, PCTS), acc, (np.percentile(err, ERR_PCTS) if err.size >= 30 else None)


def run(cid):
    path = OUT / f"{cid}.json"
    if path.exists():
        return cid, "skipped"
    seed = 1000 + sum(ord(c) for c in cid)
    base = {k: float(v) for k, v in next(r for r in csv.DictReader(open("2_data_processing/baselines.csv")) if r["contrast"] == cid).items() if k in KEYS}
    prof = [r for r in csv.DictReader(open("4_output/pooled_accuracy_profiles.csv")) if r["contrast"] == cid]
    q_obs = np.array([float(r["change"]) for r in prof if r["measure"] == "correct_rt_pct"])
    q_se = np.array([float(r["se"]) for r in prof if r["measure"] == "correct_rt_pct"])
    qA_obs = np.array([float(r["value_A"]) for r in prof if r["measure"] == "correct_rt_pct"])
    acc_row = [r for r in prof if r["measure"] == "accuracy"]
    has_acc = bool(acc_row) and cid not in NO_ACCURACY
    accA_obs = float(acc_row[0]["value_A"]) if has_acc else np.nan
    dacc_obs, dacc_se = (float(acc_row[0]["change"]), float(acc_row[0]["se"])) if has_acc else (np.nan, np.nan)
    err_rows = [r for r in prof if r["measure"] == "error_rt_pct"]
    e_obs = np.array([float(r["change"]) for r in err_rows]); e_se = np.array([float(r["se"]) for r in err_rows])

    # Step 1: calibrate A.
    cal = ["d", "s", "p", "t"] + (["e_abs", "q"] if has_acc else [])
    start = {"d": 0.0, "s": 0.0, "p": 0.0, "t": 0.0, "e_abs": 3.0, "q": 0.0}

    def cal_loss(x):
        q, acc, _ = summarise(apply(base, clipped(cal, x)), seed)
        loss = float(np.sum((np.log(q) - np.log(qA_obs)) ** 2))
        if has_acc:
            loss += 7 * (acc - accA_obs) ** 2
        return loss

    x0 = np.array([start[n] for n in cal])
    simplex = np.vstack([x0] + [x0 + np.eye(len(cal))[i] * STEP[n] for i, n in enumerate(cal)])
    res = minimize(cal_loss, x0, method="Nelder-Mead", options=dict(maxfev=60 * len(cal), initial_simplex=simplex))
    theta_a = apply(base, clipped(cal, res.x))
    qA_mod, accA_mod, _ = summarise(theta_a, seed)

    cache = {}

    def A_at(s):
        if s not in cache:
            cache[s] = summarise(theta_a, s)
        return cache[s]

    def predict(ch, s=seed):
        qa, aa, ea = A_at(s)
        qb, ab, eb = summarise(apply(theta_a, ch), s)
        err = 100 * (eb / ea - 1) if (ea is not None and eb is not None) else None
        return 100 * (qb / qa - 1), ab - aa, err

    def chi(pred, se_q, se_a, se_e):
        dq, da, de = pred
        total = float(np.sum(((dq - q_obs) / se_q) ** 2))
        n = len(PCTS)
        if has_acc:
            total += float(((da - dacc_obs) / se_a) ** 2); n += 1
        if len(e_obs):
            n += len(e_obs)
            total += float(np.sum(((de - e_obs) / se_e) ** 2)) if de is not None else 9.0 * len(e_obs)
        return total, n

    floor_q, floor_a, floor_e = np.maximum(q_se, 0.25), max(dacc_se, 0.002) if has_acc else 1.0, np.maximum(e_se, 0.5)
    fits = {}
    for name, pars in MODELS.items():
        k = len(pars)
        # Starts: the best solution among the models nested in this one (no change for S and P), and, for
        # the control signal, two starts on either side of the critical ratio.
        if NESTED_IN[name]:
            m = min(NESTED_IN[name], key=lambda m: fits[m]["chi2"])
            x = np.array([fits[m]["changes"].get(n, 0.0) for n in pars])
            if "cS" in pars and "cS" not in fits[m]["changes"]:
                x[pars.index("cR")] = 1.2
            starts = [x]
        else:
            starts = [np.zeros(k)]
        if "cS" in pars:
            for cs, cr in [(0.3, 1.1), (0.3, 1.3)]:
                x = starts[0].copy(); x[pars.index("cS")] = cs; x[pars.index("cR")] = cr
                starts.append(x)
        best = None
        for x0 in starts:
            simplex = np.vstack([x0] + [x0 + np.eye(k)[i] * STEP[n] for i, n in enumerate(pars)])
            r = minimize(lambda x: chi(predict(clipped(pars, x)), floor_q, floor_a, floor_e)[0], x0, method="Nelder-Mead",
                         options=dict(maxfev=30 + 40 * k, initial_simplex=simplex))
            if best is None or r.fun < best.fun:
                best = r
        ch = clipped(pars, best.x)
        pred = predict(ch)
        reps = [predict(ch, seed + 17 * (j + 1)) for j in range(4)]
        mc_q = np.std([r[0] for r in reps], axis=0, ddof=1)
        mc_a = float(np.std([r[1] for r in reps], ddof=1))
        e_reps = [r[2] for r in reps if r[2] is not None]
        mc_e = np.std(e_reps, axis=0, ddof=1) if len(e_reps) >= 2 else np.zeros(len(e_obs))
        c2, n_terms = chi(pred, np.sqrt(floor_q**2 + mc_q**2), np.sqrt(floor_a**2 + mc_a**2),
                          np.sqrt(floor_e**2 + (mc_e if len(e_obs) else 0) ** 2) if len(e_obs) else floor_e)
        df = n_terms - k
        fits[name] = dict(changes=ch, predicted_correct_pct=np.round(pred[0], 3).tolist(),
                          predicted_accuracy_change=None if np.isnan(pred[1]) else round(float(pred[1]), 5),
                          predicted_error_pct=None if pred[2] is None else np.round(pred[2], 3).tolist(),
                          chi2=round(c2, 3), n_terms=n_terms, df=df, p=float(stats.chi2.sf(c2, df)) if df > 0 else np.nan,
                          at_bound=[n for n, v in ch.items() if min(abs(v - BOUNDS[n][0]), abs(v - BOUNDS[n][1])) < 1e-6])
    out = dict(contrast=cid, seed=seed, has_accuracy=has_acc, n_error_terms=len(e_obs),
               observed=dict(correct_pct=q_obs.tolist(), accuracy_change=None if not has_acc else dacc_obs,
                             error_pct=e_obs.tolist(), accuracy_A=None if not has_acc else accA_obs),
               calibration=dict(changes=clipped(cal, res.x), q_A_model_ms=np.round(qA_mod, 2).tolist(),
                                q_A_observed_ms=qA_obs.tolist(), accuracy_A_model=accA_mod,
                                rms_log=float(np.sqrt(np.mean((np.log(qA_mod) - np.log(qA_obs)) ** 2)))),
               theta_A=theta_a, fits=fits)
    path.write_text(json.dumps(out, indent=1, default=float))
    return cid, "fitted"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cids = sys.argv[1:] or [r["contrast"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))]
    with Pool(min(len(cids), os.cpu_count())) as pool:
        for cid, status in pool.imap_unordered(run, cids):
            print(cid, status, flush=True)
