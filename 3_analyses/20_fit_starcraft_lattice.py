"""
20_fit_starcraft_lattice.py

Purpose
    Refit StarCraft 2 by league (contrast S1, Bronze to Grandmaster games) to separate two reasons why
    stage 15 found no adequate model for it:
      (1) the league difference lies in parameters stage 15 holds fixed. Two mechanisms are added:
            circuit     w        W_self and W_inhib x exp(w) (stronger recurrent excitation and
                                 mutual inhibition together)
            retrieval   rr, ro   instance route: retrieval rate rho_max x exp(rr) (u* scales the same
                        strength  rate, so a change of u* is a change of rr here) and retrieval onset
                                 scale s_W x exp(-ro) (larger ro = earlier retrieval)
      (2) StarCraft 2 RTs are recorded on a lattice: every S1 RT is a whole multiple of 1000/22.125 =
          45.198 ms, and several steps are empty, so the observed percentiles form a staircase that
          a smooth model is scored against step by step.

    Lattice-aware percentiles. Each RT on the lattice is treated as spread evenly over its recording
    step (from half a step below to half a step above), and percentiles are read from the resulting
    piecewise-linear cumulative distribution (the mid-distribution quantile for discrete data; Parzen,
    2004, Statistical Science 19:652). The model's RTs are first rounded to the nearest step, then
    treated the same way, so model and data pass through the same recording grid.

    The same forward selection as stage 15 (15_fit_mechanism_selection.py), in a 2 x 2 design that
    answers which reason matters:
        raw_8         raw percentiles, stage 15's eight mechanisms: stage 15's own S1 result
                      (4_output/mech_selection/S1.json), not refitted here
        raw_10        raw percentiles, ten mechanisms
        lattice_8     lattice-aware percentiles, eight mechanisms
        lattice_10    lattice-aware percentiles, ten mechanisms
    The all-mechanism ceiling of stage 15 is not fitted: with ten mechanisms it has more parameters
    (16) than S1 has observations (12 percentiles; no accuracy).

    Condition A is stage 12's calibrated Bronze condition, unchanged.

Inputs
    2_data_processing/contrast_trials_acc.csv, 4_output/acc_mixtures/S1.json, 4_output/bsi_table.csv

Outputs
    4_output/s1_lattice_profile.csv (observations; dictionary beside it, from 05_write_dictionaries.py),
    4_output/s1_lattice/<variant>.json (every fit and the selection path); <variant>.partial.json while running

Depends on
    11_pooled_accuracy_profiles.py (bootstrap design), 12_fit_accuracy_mixtures.py,
    15_fit_mechanism_selection.py, model/wwsim_ext.py

Seed
    BOOT_SEED for the bootstrap; S1's stage-12 seed for every simulation (common random numbers).

Last substantive change
    2026-11-08 — progress saved after every selection step.

Run from the project root: python 3_analyses/20_fit_starcraft_lattice.py <variant> (one variant per
process; candidate fits of each step run in parallel).
"""
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

spec = importlib.util.spec_from_file_location("ms", Path(__file__).parent / "15_fit_mechanism_selection.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)
fa = ms.fa
wwsim_ext = fa.wwsim_ext

CID = "S1"
STEP_MS = 1000.0 / 22.125  # every S1 RT is a whole multiple of this (checked to 1e-12 ms)
PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
N_TRIALS = 4000
BOOT_SEED, N_BOOT = 61108, 500
VARIANTS = {"raw_10": (False, 10), "lattice_8": (True, 8), "lattice_10": (True, 10)}

MECHS = dict(ms.MECHS, circuit=["w"], retrieval_strength=["rr", "ro"])
MECHS8 = list(ms.MECHS)
BOUNDS = dict(ms.BOUNDS, w=(-0.15, 0.3), rr=(-2.0, 3.5), ro=(-2.0, 3.0))
STEP = dict(ms.STEP, w=0.05, rr=0.5, ro=0.5)
NEUTRAL = dict(ms.NEUTRAL, w=0.0, rr=0.0, ro=0.0)
OUT = Path("4_output/s1_lattice")


def lattice_percentiles(rt_ms, pcts):
    """Percentiles of RTs on the recording lattice, each RT spread evenly over its step.

    k = step number of each RT. The cumulative distribution rises linearly across each occupied step
    (from k - 0.5 to k + 0.5) and is flat across empty steps; np.interp reads it backwards."""
    k, counts = np.unique(np.round(np.asarray(rt_ms) / STEP_MS).astype(np.int64), return_counts=True)
    cdf = np.cumsum(counts) / counts.sum()
    x = np.repeat(k, 2) + np.tile([-0.5, 0.5], k.size)
    f = np.repeat(np.concatenate([[0.0], cdf[:-1]]), 2)
    f[1::2] = cdf
    return np.interp(np.asarray(pcts) / 100.0, f, x) * STEP_MS


def percentiles(rt_ms, lattice):
    return lattice_percentiles(rt_ms, PCTS) if lattice else np.percentile(rt_ms, PCTS)


def apply(theta, ch):
    th = ms.apply(theta, ch)
    if "w" in ch:
        th["W_self"], th["W_inhib"] = theta["W_self"] * np.exp(ch["w"]), theta["W_inhib"] * np.exp(ch["w"])
    if "rr" in ch:
        th["rho_max"] = theta["rho_max"] * np.exp(ch["rr"])
        th["s_W"] = theta["s_W"] * np.exp(-ch["ro"])
    return th


def observations(lattice):
    """Observed percent change, Bronze to Grandmaster, at PCTS, with cluster-bootstrap SE over games
    (games resampled within each league; 11_pooled_accuracy_profiles.py's design)."""
    data = defaultdict(list)
    for r in csv.DictReader(open("2_data_processing/contrast_trials_acc.csv", newline="")):
        if r["contrast"] == CID:
            data[(r["period"], r["unit"])].append(float(r["rt_s"]) * 1000)
    units = {p: sorted(u for (q, u) in data if q == p) for p in "AB"}

    def pooled(p, us):
        return np.concatenate([data[(p, u)] for u in us])

    qa, qb = percentiles(pooled("A", units["A"]), lattice), percentiles(pooled("B", units["B"]), lattice)
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(N_BOOT):
        pa = [units["A"][j] for j in rng.integers(0, len(units["A"]), len(units["A"]))]
        pb = [units["B"][j] for j in rng.integers(0, len(units["B"]), len(units["B"]))]
        boot.append(100 * (percentiles(pooled("B", pb), lattice) / percentiles(pooled("A", pa), lattice) - 1))
    return qa, qb, 100 * (qb / qa - 1), np.std(boot, axis=0)


def setup(lattice):
    global Q_OBS, Q_SE, THETA_A, SEED, LATTICE, CACHE
    prev = json.loads(Path(f"4_output/acc_mixtures/{CID}.json").read_text())
    SEED, THETA_A, LATTICE, CACHE = prev["seed"], prev["theta_A"], lattice, {}
    prof = [r for r in csv.DictReader(open("4_output/s1_lattice_profile.csv")) if r["percentiles"] == ("lattice" if lattice else "raw")]
    Q_OBS = np.array([float(r["change"]) for r in prof])
    Q_SE = np.maximum([float(r["se"]) for r in prof], 0.25)


def simulate_percentiles(theta, seed):
    """Percentiles of 'correct' RTs as in fa.summarise: StarCraft 2 has no correct/error label, so the
    model chooses at chance (e = 0) and the trials it labels correct stand for all trials; no
    commitment by T_MAX counts as T_MAX + 1000 ms."""
    sim = wwsim_ext.simulate_condition(theta, N_TRIALS, fa.T_MAX, seed)
    rt, ch = sim["rt"], sim["choice"]
    rt_c = np.where(np.isfinite(rt[ch != 0]), rt[ch != 0], fa.T_MAX + 1000.0)
    return percentiles(rt_c, LATTICE)


def predict(ch, seed):
    """Percent change from A at PCTS; A and B simulated with the same seed (common random numbers)."""
    if seed not in CACHE:
        CACHE[seed] = simulate_percentiles(THETA_A, seed)
    return 100 * (simulate_percentiles(apply(THETA_A, ch), seed) / CACHE[seed] - 1)


def clipped(names, x):
    return {n: float(np.clip(v, *BOUNDS[n])) for n, v in zip(names, x)}


def fit(task):
    """Nelder-Mead from each start; chi-square over the 12 percentiles with data and simulation SE
    (stage 15's criterion without the accuracy and error-RT terms, which S1 does not have)."""
    lattice, pars, starts = task
    setup(lattice)
    k, best = len(pars), None
    for st in starts:
        x0 = np.array([st.get(n, NEUTRAL[n]) for n in pars])
        simplex = np.vstack([x0] + [x0 + np.eye(k)[i] * STEP[n] for i, n in enumerate(pars)])
        r = minimize(lambda x: float(np.sum(((predict(clipped(pars, x), SEED) - Q_OBS) / Q_SE) ** 2)), x0, method="Nelder-Mead",
                     options=dict(maxfev=25 + 30 * k, initial_simplex=simplex))
        if best is None or r.fun < best.fun:
            best = r
    ch = clipped(pars, best.x)
    pred = predict(ch, SEED)
    reps = [predict(ch, SEED + 17 * (j + 1)) for j in range(4)]
    sq = np.sqrt(Q_SE**2 + np.std(reps, axis=0, ddof=1) ** 2)
    c2 = float(np.sum(((pred - Q_OBS) / sq) ** 2))
    df = len(Q_OBS) - k
    return dict(params=pars, changes=ch, chi2=round(c2, 3), n_terms=len(Q_OBS), df=df, aic=round(c2 + 2 * k, 3),
                p=float(stats.chi2.sf(c2, df)) if df > 0 else float("nan"), predicted_pct=np.round(pred, 3).tolist(),
                at_bound=[n for n, v in ch.items() if min(abs(v - BOUNDS[n][0]), abs(v - BOUNDS[n][1])) < 1e-6])


def extra_starts(m):
    return ms.EXTRA_STARTS.get(m, []) + {"circuit": [{"w": 0.15}], "retrieval_strength": [{"rr": 1.5, "ro": 1.5}]}.get(m, [])


def run(variant):
    lattice, n_mech = VARIANTS[variant]
    mechs_all = MECHS8 if n_mech == 8 else list(MECHS)
    # Progress is saved after every selection step, so a run that is stopped resumes at the step it was on.
    partial = OUT / f"{variant}.partial.json"
    state = json.loads(partial.read_text()) if partial.exists() else None

    def save(next_step):
        partial.write_text(json.dumps(dict(fits=fits, current=current, path_log=path_log, best=best, next_step=next_step), indent=1, default=float))

    with Pool(min(len(mechs_all), 13)) as pool:
        if state:
            fits, current, path_log, best, first_step = state["fits"], state["current"], state["path_log"], state["best"], state["next_step"]
        else:
            fits, current, path_log, first_step = {}, [], ["SP"], 1
            prev_sp = json.loads(Path(f"4_output/mech_selection/{CID}.json").read_text())["fits"]["SP"]["changes"]
            fits["SP"] = best = pool.map(fit, [(lattice, ["s", "p"], [{}, prev_sp])])[0]
            save(1)
        for step in range(first_step, ms.MAX_ADDED + 1):
            cands = [m for m in mechs_all if m not in current]
            tasks = []
            for m in cands:
                base = dict(best["changes"])
                starts = [base] + [dict(base, **x) for x in extra_starts(m)]
                if best["at_bound"]:
                    starts += [{}] + [dict(x) for x in extra_starts(m)]
                tasks.append((lattice, ["s", "p"] + [q for c in current + [m] for q in MECHS[c]], starts))
            trials = dict(zip(cands, pool.map(fit, tasks)))
            for m in cands:
                fits["SP+" + "+".join(current + [m])] = trials[m]
            m_best = min(trials, key=lambda m: trials[m]["aic"])
            print(variant, "step", step, {m: trials[m]["aic"] for m in cands}, flush=True)
            if trials[m_best]["aic"] < best["aic"] - 2:
                current.append(m_best)
                best = trials[m_best]
                path_log.append("SP+" + "+".join(current))
                save(step + 1)
            else:
                break
    seed = json.loads(Path(f"4_output/acc_mixtures/{CID}.json").read_text())["seed"]
    out = dict(contrast=CID, variant=variant, lattice=lattice, mechanisms={m: MECHS[m] for m in mechs_all}, seed=seed,
               selected=path_log[-1], path=path_log, fits=fits)
    (OUT / f"{variant}.json").write_text(json.dumps(out, indent=1, default=float))
    partial.unlink()
    print(variant, "selected", path_log[-1], "p", fits[path_log[-1]]["p"], flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "observations":
        rows = []
        for lattice in (False, True):
            qa, qb, change, se = observations(lattice)
            rows += [dict(percentiles="lattice" if lattice else "raw", level=p, value_A=round(a, 3), value_B=round(b, 3),
                          change=round(c, 4), se=round(s, 4)) for p, a, b, c, s in zip(PCTS, qa, qb, change, se)]
        with open("4_output/s1_lattice_profile.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("seed", BOOT_SEED, "; wrote 4_output/s1_lattice_profile.csv")
    else:
        run(sys.argv[1])
