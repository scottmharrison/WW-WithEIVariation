"""
check_06_bsi_circuit_validation.py (process code; not run by the pipeline)

Purpose
    Validate model/wwsim_ext.py before any fit uses it.
    (0) With e = 0 and S = 0 it reproduces WW-RTShapeAnatomy/model/wwsim.py trial for trial.
    (1)-(4) Against the circuit results of Lo, Wang & Wang (2015), in the reduced model started from the
        WW-CarefulFitting fit of monkey random-dot motion at 6.4% coherence (roitman2002__12), with the
        evidence e set so that accuracy is about 80% without control:
        (1) at S = 0.5, mean decision time rises for R >= 1.2 and falls for R <= 1.156, by 90% or more
            (paper, Fig. 2D);
        (2) ex-Gaussian tau/sigma of decision time rises with R (paper: 1.02 at I<E, 4.00 at I>E, Fig. 3);
        (3) accuracy rises where decisions slow (speed-accuracy trade-off, Figs. 4-5);
        (4) the change of sign lies near R = 1.156 (Wang et al., 2013).
    Two settings: "circuit only" (no roaming, no instance route; closest to the paper's network) and
    "full model".

Inputs
    1_raw/ww_conditions.csv; 4_output/bsi_table.csv

Outputs
    Printed text; saved by the caller to process/output/check_06.txt

Depends on
    3_analyses/10_build_bsi_table.py, model/wwsim_ext.py

Seed
    SEED below; every simulation uses it (common random numbers).

Last substantive change
    2026-11-07 — first version.
"""
import csv
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "model")
sys.path.insert(0, "../WW-RTShapeAnatomy/model")
import exgaussian  # noqa: E402
import wwsim  # noqa: E402
import wwsim_ext  # noqa: E402

SEED = 20261107
KEYS = ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]
conds = {r["cond_id"]: {k: float(r[k]) for k in KEYS} for r in csv.DictReader(open("1_raw/ww_conditions.csv"))}
N, TMAX = 4000, 8000


def run(args):
    th, roam, lca = args
    sim = wwsim_ext.simulate_condition(th, N, TMAX, SEED, roam=roam, lca=lca)
    ok = np.isfinite(sim["rt"])
    dt = sim["rt"][ok]
    # Fewer than 50 committed trials: no distribution to describe; reported as such.
    if dt.size < 50:
        return dict(mean_rt=np.nan, acc=np.nan, none=float(1 - ok.mean()), tau_sigma=np.nan, q10=np.nan, q99=np.nan)
    mu, sigma, tau = exgaussian.fit(dt / 1000.0)
    return dict(mean_rt=float(dt.mean()), acc=float(np.mean(sim["choice"][ok] == 1)), none=float(1 - ok.mean()),
                tau_sigma=float(tau / max(sigma, 1e-6)), q10=float(np.percentile(dt, 10)), q99=float(np.percentile(dt, 99)))


if __name__ == "__main__":
    print("seed", SEED)
    # (0) exactness against the original simulator
    for cid in ["blp__03", "roitman2002__12", "sc2__21"]:
        a = wwsim.simulate_condition(conds[cid], 1500, 6000, SEED)["rt"]
        b = wwsim_ext.simulate_condition(conds[cid], 1500, 6000, SEED)["rt"]
        same = np.array_equal(np.nan_to_num(a, posinf=-1), np.nan_to_num(b, posinf=-1))
        print(f"(0) {cid}: identical RTs to wwsim.py: {same}")

    base = dict(conds["roitman2002__12"])
    for roam, lca, label in [(False, False, "circuit only"), (True, True, "full model")]:
        # Evidence giving about 80% correct without control.
        es = [0.5, 1, 2, 3, 4, 6, 8, 12]
        with Pool() as pool:
            accs = pool.map(run, [(dict(base, e=e), roam, lca) for e in es])
        e = es[int(np.argmin([abs(r["acc"] - 0.8) for r in accs]))]
        print(f"\n== {label}: evidence e = {e} Hz (accuracy without control {accs[es.index(e)]['acc']:.3f}); "
              f"accuracy by e: {[round(r['acc'], 3) for r in accs]}")
        grid = [(S, R) for S in [0.0, 0.25, 0.5, 1.0] for R in ([1.2] if S == 0 else [1.0, 1.1, 1.156, 1.2, 1.247, 1.297, 1.4])]
        with Pool() as pool:
            out = pool.map(run, [(dict(base, e=e, bsi_S=S, bsi_R=R), roam, lca) for S, R in grid])
        ref = out[0]
        print(f"   no control: mean RT {ref['mean_rt']:.0f} ms, accuracy {ref['acc']:.3f}, tau/sigma {ref['tau_sigma']:.2f}, "
              f"10th/99th {ref['q10']:.0f}/{ref['q99']:.0f} ms, no commitment {100 * ref['none']:.1f}%")
        for (S, R), r in zip(grid[1:], out[1:]):
            print(f"   S {S:<4} R {R:<5}: mean RT {100 * (r['mean_rt'] / ref['mean_rt'] - 1):+7.1f}%, accuracy {r['acc']:.3f}, "
                  f"tau/sigma {r['tau_sigma']:6.2f}, 10th/99th {r['q10']:.0f}/{r['q99']:.0f} ms, no commitment {100 * r['none']:.1f}%")
