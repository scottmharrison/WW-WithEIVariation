"""
15_fit_mechanism_selection.py

Purpose
    Which mechanisms, added to skill and practice, account for each contrast's change in correct RTs,
    accuracy and error RTs, and which combination does so best? Eight candidate mechanisms are compared
    against each other on all 16 contrasts:

        drive       d        u0 and u0_sd x exp(d) (common drive; precision u0_sd/u0 unchanged)
        evidence    e        difference between the pools' drive, + e Hz
        control     cS, cR   balanced E/I control signal, strength and ratio (Lo, Wang & Wang, 2015)
        threshold   h, hz    circuit commitment rate 15 x exp(h) Hz and instance-route threshold 1 x exp(hz)
        urgency     urg      a current growing through the trial, urg nA per second, on both pools
        lapse       pL, gL   a fraction pL of trials are lapses: drive x gL, no evidence (WW-NSGA2's lapse)
        ndt         t        non-decision time + t ms
        retrieval   b        instance route's bias towards the correct answer, 0.9 + b (its accuracy)
        (skill s and practice p are always present, as in 12_fit_accuracy_mixtures.py)

    Model: model/wwsim_ext.py. Condition A of each contrast is the calibrated A of
    12_fit_accuracy_mixtures.py (4_output/acc_mixtures/<contrast>.json), reused unchanged.

    Procedure per contrast (forward selection; Burnham & Anderson, 2002, Model Selection and Multimodel
    Inference):
      1. Fit skill + practice, from no change and from the stage-12 solution.
      2. Try adding each mechanism not yet in the model; keep the one with the lowest AIC. Repeat while
         AIC falls by more than 2, up to 5 added mechanisms. Every fit of step 1 is kept, so each
         mechanism's effect on its own is recorded for every contrast.
      3. Fit the model with all eight mechanisms, started from the selected model: the ceiling for that
         contrast. If it is not adequate, no combination of these mechanisms is.
    Each fit starts from the current best solution with the new mechanism at its neutral value, plus one
    or two other starts for mechanisms whose neutral value is at an edge (control, lapse); when the current
    best has a parameter at its search limit, also from no change at all.
    Criterion and adequacy as in stage 12, but over 12 correct-RT percentiles (5th to 99th), accuracy
    change and 5 error-RT percentiles, data and simulation SE; p > .05; AIC = chi-square + 2k).
    Simulation: 4,000 trials per condition, common random numbers within a contrast.

Inputs
    4_output/acc_mixtures/<contrast>.json, 4_output/pooled_accuracy_profiles_fine.csv, 4_output/bsi_table.csv

Outputs
    4_output/mech_selection/<contrast>.json — every fit, the selection path and the ceiling

Depends on
    12_fit_accuracy_mixtures.py, model/wwsim_ext.py

Seed
    Each contrast's own seed (as stage 12), written to its file.

Last substantive change
    2026-11-07 — restart from no change when the current best is at a search limit; fit_one for stage 16.

Run from the project root. Several hours; contrasts in parallel; a contrast whose file exists is skipped.
"""
import csv
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

spec = importlib.util.spec_from_file_location("fa", Path(__file__).parent / "12_fit_accuracy_mixtures.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)
fa.N_TRIALS = 4000
# Twelve correct-RT percentiles (11_pooled_accuracy_profiles.py --fine), so that models with many changes
# keep degrees of freedom; fa.summarise reads this module-level list.
fa.PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

MECHS = {"drive": ["d"], "evidence": ["e"], "control": ["cS", "cR"], "threshold": ["h", "hz"], "urgency": ["urg"],
         "lapse": ["pL", "gL"], "ndt": ["t"], "retrieval": ["b"]}
BOUNDS = dict(fa.BOUNDS, h=(-0.5, 1.0), hz=(-1.0, 1.5), urg=(-0.03, 0.1), pL=(0.0, 0.3), gL=(0.05, 1.0), b=(-0.9, 0.1))
STEP = dict(fa.STEP, h=0.15, hz=0.3, urg=0.01, pL=0.05, gL=0.2, b=0.1)
NEUTRAL = {"s": 0.0, "p": 0.0, "d": 0.0, "e": 0.0, "cS": 0.0, "cR": 1.2, "h": 0.0, "hz": 0.0, "urg": 0.0, "pL": 0.0, "gL": 0.5, "t": 0.0, "b": 0.0}
EXTRA_STARTS = {"control": [{"cS": 0.3, "cR": 1.1}, {"cS": 0.3, "cR": 1.3}], "lapse": [{"pL": 0.05, "gL": 0.5}, {"pL": 0.1, "gL": 0.25}],
                "threshold": [{"h": 0.3, "hz": 0.0}]}
MAX_ADDED = 5
OUT = Path("4_output/mech_selection")


def apply(theta, ch):
    th = fa.apply(theta, ch)
    if "h" in ch:
        th["thr"] = 15.0 * np.exp(ch["h"]); th["lca_z"] = 1.0 * np.exp(ch["hz"])
    if "urg" in ch:
        th["urg"] = ch["urg"]
    if "pL" in ch:
        th["p_lapse"], th["gain_lapse"] = ch["pL"], ch["gL"]
    if "b" in ch:
        th["lca_bias"] = float(np.clip(0.9 + ch["b"], 0.0, 1.0))
    return th


def clipped(names, x):
    return {n: float(np.clip(v, *BOUNDS[n])) for n, v in zip(names, x)}


def make_fitter(cid):
    """The fit function for one contrast (its observations, calibrated A and seed), and its context."""
    prev = json.loads((Path("4_output/acc_mixtures") / f"{cid}.json").read_text())
    seed, theta_a = prev["seed"], prev["theta_A"]
    prof = [r for r in csv.DictReader(open("4_output/pooled_accuracy_profiles_fine.csv")) if r["contrast"] == cid]
    q_obs = np.array([float(r["change"]) for r in prof if r["measure"] == "correct_rt_pct"])
    q_se = np.maximum([float(r["se"]) for r in prof if r["measure"] == "correct_rt_pct"], 0.25)
    acc = [r for r in prof if r["measure"] == "accuracy"]
    has_acc = bool(acc) and cid not in fa.NO_ACCURACY
    a_obs, a_se = (float(acc[0]["change"]), max(float(acc[0]["se"]), 0.002)) if has_acc else (np.nan, 1.0)
    er = [r for r in prof if r["measure"] == "error_rt_pct"]
    e_obs = np.array([float(r["change"]) for r in er]); e_se = np.maximum([float(r["se"]) for r in er], 0.5) if er else np.array([])

    cache = {}

    def predict(ch, s=seed):
        if s not in cache:
            cache[s] = fa.summarise(theta_a, s)
        qa, aa, ea = cache[s]
        qb, ab, eb = fa.summarise(apply(theta_a, ch), s)
        return 100 * (qb / qa - 1), ab - aa, (100 * (eb / ea - 1) if ea is not None and eb is not None else None)

    def chi(pred, sq, sa, se):
        dq, da, de = pred
        total, n = float(np.sum(((dq - q_obs) / sq) ** 2)), len(q_obs)
        if has_acc:
            total += float(((da - a_obs) / sa) ** 2); n += 1
        if len(e_obs):
            n += len(e_obs)
            total += float(np.sum(((de - e_obs) / se) ** 2)) if de is not None else 9.0 * len(e_obs)
        return total, n

    def fit(pars, starts):
        """Nelder-Mead from each start (dicts of parameter values); returns the record of the best fit."""
        k = len(pars)
        best = None
        for st in starts:
            x0 = np.array([st.get(n, NEUTRAL[n]) for n in pars])
            simplex = np.vstack([x0] + [x0 + np.eye(k)[i] * STEP[n] for i, n in enumerate(pars)])
            r = minimize(lambda x: chi(predict(clipped(pars, x)), q_se, a_se, e_se)[0], x0, method="Nelder-Mead",
                         options=dict(maxfev=25 + 30 * k, initial_simplex=simplex))
            if best is None or r.fun < best.fun:
                best = r
        ch = clipped(pars, best.x)
        pred = predict(ch)
        reps = [predict(ch, seed + 17 * (j + 1)) for j in range(4)]
        sq = np.sqrt(q_se**2 + np.std([r[0] for r in reps], axis=0, ddof=1) ** 2)
        sa = np.sqrt(a_se**2 + np.std([r[1] for r in reps], ddof=1) ** 2) if has_acc else a_se
        ereps = [r[2] for r in reps if r[2] is not None]
        se = np.sqrt(e_se**2 + (np.std(ereps, axis=0, ddof=1) ** 2 if len(ereps) >= 2 else 0)) if len(e_obs) else e_se
        c2, n_terms = chi(pred, sq, sa, se)
        df = n_terms - k
        return dict(params=pars, changes=ch, chi2=round(c2, 3), n_terms=n_terms, df=df, aic=round(c2 + 2 * k, 3),
                    p=float(stats.chi2.sf(c2, df)) if df > 0 else float("nan"),
                    predicted_correct_pct=np.round(pred[0], 3).tolist(),
                    predicted_accuracy_change=None if np.isnan(pred[1]) else round(float(pred[1]), 5),
                    predicted_error_pct=None if pred[2] is None else np.round(pred[2], 3).tolist(),
                    at_bound=[n for n, v in ch.items() if min(abs(v - BOUNDS[n][0]), abs(v - BOUNDS[n][1])) < 1e-6])

    return fit, prev, seed, has_acc


def fit_one(cid, pars, starts):
    """One fit of the parameters pars for contrast cid from the given starts (used by stage 16)."""
    return make_fitter(cid)[0](pars, starts)


def run(cid):
    path = OUT / f"{cid}.json"
    if path.exists():
        return cid, "skipped"
    fit, prev, seed, has_acc = make_fitter(cid)

    def model_params(mechs):
        return ["s", "p"] + [q for m in mechs for q in MECHS[m]]

    fits = {}
    prev_sp = prev["fits"]["SP"]["changes"]
    fits["SP"] = fit(["s", "p"], [{}, prev_sp])
    current, path_log = [], ["SP"]
    best = fits["SP"]
    for step in range(1, MAX_ADDED + 1):
        trials = {}
        for m in MECHS:
            if m in current:
                continue
            mechs = current + [m]
            name = "SP+" + "+".join(mechs)
            base = dict(best["changes"])
            starts = [base] + [dict(base, **x) for x in EXTRA_STARTS.get(m, [])]
            # A current best with a parameter at its search limit may be a corner the search cannot leave
            # by adding one mechanism; then the new model is also started from no change at all.
            if best["at_bound"]:
                starts += [{}] + [dict(x) for x in EXTRA_STARTS.get(m, [])]
            fits[name] = trials[m] = fit(model_params(mechs), starts)
        m_best = min(trials, key=lambda m: trials[m]["aic"])
        if trials[m_best]["aic"] < best["aic"] - 2:
            current.append(m_best)
            best = trials[m_best]
            path_log.append("SP+" + "+".join(current))
        else:
            break
    full_name = "SP+all"
    base = dict(best["changes"])
    fits[full_name] = fit(model_params(list(MECHS)), [base, dict(base, cS=0.3, cR=1.2, pL=0.05, gL=0.5)])
    out = dict(contrast=cid, seed=seed, has_accuracy=has_acc, n_terms=fits["SP"]["n_terms"], selected=path_log[-1],
               path=path_log, fits=fits, mechanisms=MECHS)
    path.write_text(json.dumps(out, indent=1, default=float))
    return cid, f"selected {path_log[-1]} (p {fits[path_log[-1]]['p']:.3g}); all mechanisms p {fits[full_name]['p']:.3g}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cids = sys.argv[1:] or [r["contrast"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))]
    with Pool(min(len(cids), os.cpu_count())) as pool:
        for cid, status in pool.imap_unordered(run, cids):
            print(cid, status, flush=True)
