"""
07_fit_mixtures.py

Purpose
    Ask, for each contrast, whether the two mechanisms of SKILL_vs_PRACTICE_explained.pdf, in any
    combination, can turn the model's condition A into condition B, and which single addition helps
    where they cannot.

    Model: the extended Wong-Wang model (two-pool Wong & Wang, 2006, circuit; drive redrawn on every
    trial, "roaming"; a racing instance route, a leaky competing accumulator started at a retrieval
    onset that shortens as stored instances N grow, after Logan, 1988; RT = faster route + non-decision
    time). Simulated by WW-RTShapeAnatomy/model/wwsim.py, a numpy port of the WW-CarefulFitting kernel.

    Step 1, calibrate A. From the WW-CarefulFitting fit named in 2_data_processing/baselines.csv, four
    parameters are adjusted (drive mean u0, drive spread u0_sd, log instances logN, non-decision mean
    ter_mu) to match A's pooled percentiles, by least squares on log RT. The rest stay as fitted.

    Step 2, fit B from A, changing only (each a change from calibrated A):
        skill     s: u0_sd x exp(s), mean drive fixed (the document's drive precision; s < 0 = more precise)
        practice  p: logN + p (stored instances; p > 0 = more instances)
        drive     d: u0 and u0_sd both x exp(d), so the precision u0_sd/u0 is unchanged
        ndt       t: ter_mu + t ms (non-decision time: encoding and motor execution)
        retrieval r: s_W x exp(r) (retrieval onset scale; r < 0 = faster retrieval at any N)
    Models: S, P, SP, SP+drive, SP+ndt, SP+retrieval, and SP+drive+ndt (two additions together). Each
    starts from no change and from the best solution of every model nested in it. The observed change is the percent change at the
    seven percentiles (4_output/pooled_profiles.csv); the predicted change is the model's percent change
    from A to B, so a misfit of A's shape is not counted against the mechanisms.

    Fit criterion: chi-square = sum over percentiles of ((predicted - observed) / SE)^2, SE combining
    the data's bootstrap SE and the simulation's own SE (SD of the predicted change over 4 further
    seeds at the solution). df = 7 - number of changed parameters; adequate when p > .05.
    Optimiser: Nelder-Mead from no change, with common random numbers (one seed per contrast for every
    evaluation, so differences between parameter values are not simulation noise).

    Step 3, the reachable region: a 7 x 7 grid of skill and practice changes from calibrated A, with
    the predicted change at the 10th, 50th and 99th percentiles, to show every outcome any combination
    of the two mechanisms can produce.

Inputs
    2_data_processing/baselines.csv, 4_output/pooled_profiles.csv,
    <WW_MODEL_DIR, default ../WW-RTShapeAnatomy/model>/wwsim.py

Outputs
    4_output/mixtures/<contrast>.json — calibration, every model fit, the grid; collected by 08

Depends on
    06_pooled_profiles.py, 2_data_processing/02_select_baselines.py

Last substantive change
    2026-11-06 — first version.

Run from the project root. Several hours of CPU; contrasts run in parallel, one per core, and a contrast
whose file holds every model is skipped; a saved calibration is reused (delete the file to recalibrate).
"""
import csv
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

MODEL_DIR = Path(os.environ.get("WW_MODEL_DIR", "../WW-RTShapeAnatomy/model"))
sys.path.insert(0, str(MODEL_DIR))
import wwsim  # noqa: E402

PCTS = [5, 10, 25, 50, 75, 90, 99]
N_TRIALS, T_MAX = 6000, 10000
KEYS = ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]
# Parameter: (lower bound, upper bound) of the change, on the scale the optimiser uses.
BOUNDS = {"s": (-3.0, 2.0), "p": (-5.0, 6.0), "d": (-1.5, 1.5), "t": (-300.0, 600.0), "r": (-3.0, 3.0)}
STEP = {"s": 0.5, "p": 1.0, "d": 0.3, "t": 50.0, "r": 0.5}
MODELS = {"S": ["s"], "P": ["p"], "SP": ["s", "p"], "SP+drive": ["s", "p", "d"], "SP+ndt": ["s", "p", "t"],
          "SP+retrieval": ["s", "p", "r"], "SP+drive+ndt": ["s", "p", "d", "t"]}
# Each model is started from the best solution of the models it contains (and from no change), so a
# larger model can never end with a worse fit than one nested in it.
NESTED_IN = {"S": [], "P": [], "SP": ["S", "P"], "SP+drive": ["SP"], "SP+ndt": ["SP"], "SP+retrieval": ["SP"],
             "SP+drive+ndt": ["SP+drive", "SP+ndt"]}
OUT = Path("4_output/mixtures")


def apply(theta, ch):
    """Parameters after the changes in ch (a dict of s, p, d, t, r)."""
    th = dict(theta)
    th["u0_sd"] = theta["u0_sd"] * np.exp(ch.get("s", 0.0) + ch.get("d", 0.0))
    th["u0"] = theta["u0"] * np.exp(ch.get("d", 0.0))
    th["logN"] = theta["logN"] + ch.get("p", 0.0)
    th["ter_mu"] = theta["ter_mu"] + ch.get("t", 0.0)
    th["s_W"] = theta["s_W"] * np.exp(ch.get("r", 0.0))
    return th


def percentiles(theta, seed):
    """Model RT percentiles (ms). A trial with no commitment by T_MAX is placed above every committed
    trial (at T_MAX + 1000 ms), so it counts in the upper percentiles as a slow trial, not dropped."""
    rt = wwsim.simulate_condition(theta, N_TRIALS, T_MAX, seed)["rt"]
    rt = np.where(np.isfinite(rt), rt, T_MAX + 1000.0)
    return np.percentile(rt, PCTS)


def clipped(names, x):
    return {n: float(np.clip(v, *BOUNDS[n])) for n, v in zip(names, x)}


def run(cid):
    path = OUT / f"{cid}.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and set(previous["fits"]) == set(MODELS) and previous.get("nested_starts"):
        return cid, "skipped"
    seed = 1000 + sum(ord(ch) for ch in cid)
    base = {k: float(v) for k, v in next(r for r in csv.DictReader(open("2_data_processing/baselines.csv")) if r["contrast"] == cid).items() if k in KEYS}
    prof = [r for r in csv.DictReader(open("4_output/pooled_profiles.csv")) if r["contrast"] == cid]
    qa_obs = np.array([float(r["q_A_ms"]) for r in prof])
    obs = np.array([float(r["change_pct"]) for r in prof])
    se_data = np.array([float(r["se_pct"]) for r in prof])

    # Step 1: calibrate A (log-RT least squares over the seven percentiles).
    cal_names = ["d", "s", "p", "t"]

    def cal_loss(x):
        q = percentiles(apply(base, clipped(cal_names, x)), seed)
        return float(np.sum((np.log(q) - np.log(qa_obs)) ** 2))

    if previous is not None:
        # The calibration is deterministic (fixed seed), so a saved one is reused rather than repeated.
        theta_a = previous["theta_A"]
    else:
        res = minimize(cal_loss, np.zeros(4), method="Nelder-Mead",
                       options=dict(maxfev=220, initial_simplex=np.vstack([np.zeros(4)] + [np.eye(4)[i] * STEP[n] for i, n in enumerate(cal_names)])))
        theta_a = apply(base, clipped(cal_names, res.x))
    qa_mod = percentiles(theta_a, seed)

    qa_by_seed = {}  # A's model percentiles per seed, computed once (common random numbers)

    def predicted(ch, s=seed):
        if s not in qa_by_seed:
            qa_by_seed[s] = percentiles(theta_a, s)
        return 100 * (percentiles(apply(theta_a, ch), s) / qa_by_seed[s] - 1)

    fits = {}
    for name, pars in MODELS.items():
        def chi(x):
            return float(np.sum(((predicted(clipped(pars, x)) - obs) / np.maximum(se_data, 0.25)) ** 2))
        k = len(pars)
        starts = [np.zeros(k)] + [np.array([fits[m]["changes"].get(n, 0.0) for n in pars]) for m in NESTED_IN[name]]
        best = None
        for x0 in starts:
            simplex = np.vstack([x0] + [x0 + np.eye(k)[i] * STEP[n] for i, n in enumerate(pars)])
            r = minimize(chi, x0, method="Nelder-Mead", options=dict(maxfev=40 + 50 * k, initial_simplex=simplex))
            if best is None or r.fun < best.fun:
                best = r
        ch = clipped(pars, best.x)
        pred = predicted(ch)
        # Simulation error of the predicted change at the solution, from 4 further seeds.
        reps = np.array([predicted(ch, seed + 17 * (j + 1)) for j in range(4)])
        se_mc = reps.std(axis=0, ddof=1)
        se = np.sqrt(se_data**2 + se_mc**2)
        chi2 = float(np.sum(((pred - obs) / se) ** 2))
        df = len(PCTS) - k
        fits[name] = dict(changes=ch, predicted_pct=pred.round(3).tolist(), se_mc_pct=se_mc.round(3).tolist(),
                          chi2=round(chi2, 3), df=df, p=float(stats.chi2.sf(chi2, df)),
                          rms_pct=round(float(np.sqrt(np.mean((pred - obs) ** 2))), 3),
                          at_bound=[n for n, v in ch.items() if min(abs(v - BOUNDS[n][0]), abs(v - BOUNDS[n][1])) < 1e-6])

    # Step 3: reachable region of skill x practice.
    grid_s = np.linspace(-2.0, 1.0, 7)
    grid_p = np.linspace(-3.0, 5.0, 7)
    grid = [dict(s=float(s), p=float(p), predicted_pct=predicted({"s": s, "p": p}).round(3).tolist()) for s in grid_s for p in grid_p]

    # Share of trials won by the instance route in calibrated A: practice acts only through this route.
    sim = wwsim.simulate_condition(theta_a, N_TRIALS, T_MAX, seed)
    out = dict(contrast=cid, seed=seed, percentiles=PCTS, observed_pct=obs.tolist(), se_data_pct=se_data.tolist(),
               q_A_observed_ms=qa_obs.tolist(), q_A_model_ms=qa_mod.round(2).tolist(),
               calibration_rms_log=float(np.sqrt(np.mean((np.log(qa_mod) - np.log(qa_obs)) ** 2))),
               theta_A=theta_a, instance_share_A=float(np.mean(sim["T_lca"] < sim["T_ww"])), fits=fits, grid=grid,
               nested_starts=True, wwsim_sha256=hashlib.sha256((MODEL_DIR / "wwsim.py").read_bytes()).hexdigest())
    path.write_text(json.dumps(out, indent=1))
    return cid, "fitted"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cids = [r["contrast"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))]
    if len(sys.argv) > 1:
        cids = sys.argv[1:]
    with Pool(min(len(cids), os.cpu_count())) as pool:
        for cid, status in pool.imap_unordered(run, cids):
            print(cid, status, flush=True)
