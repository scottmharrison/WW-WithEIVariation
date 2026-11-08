"""
check_07_new_mechanisms.py (process code; not run by the pipeline)

Purpose
    Check model/wwsim_ext.py after adding free thresholds, urgency and lapses.
    (0) At every default it still reproduces WW-RTShapeAnatomy/model/wwsim.py trial for trial.
    (1) What one change of each new mechanism does, from the calibrated A of contrast A2 (lexical
        decision, speed instruction): change in the 10th, 50th and 99th percentiles of correct RT,
        accuracy, and the share of trials with no commitment. In particular, whether a higher circuit
        threshold lengthens the slow tail or shifts the whole distribution.

Inputs
    1_raw/ww_conditions.csv; 4_output/acc_mixtures/A2.json

Outputs
    Printed text; saved by the caller to process/output/check_07.txt

Depends on
    model/wwsim_ext.py; 3_analyses/12_fit_accuracy_mixtures.py (for the calibrated A)

Seed
    SEED below; every simulation uses it (common random numbers).

Last substantive change
    2026-11-07 — first version.
"""
import csv
import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "model")
sys.path.insert(0, "../WW-RTShapeAnatomy/model")
import wwsim  # noqa: E402
import wwsim_ext  # noqa: E402

SEED = 20261107
KEYS = ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]


def stats_of(th):
    sim = wwsim_ext.simulate_condition(th, 4000, 10000, SEED)
    rt, ch = sim["rt"], sim["choice"]
    ok = ch >= 0
    c = rt[ch == 1]
    return np.percentile(c, [10, 50, 99]), float(np.mean(ch[ok] == 1)), float(1 - ok.mean())


if __name__ == "__main__":
    print("seed", SEED)
    conds = {r["cond_id"]: {k: float(r[k]) for k in KEYS} for r in csv.DictReader(open("1_raw/ww_conditions.csv"))}
    for cid in ["blp__03", "roitman2002__12", "sc2__21"]:
        a = wwsim.simulate_condition(conds[cid], 1500, 6000, SEED)["rt"]
        b = wwsim_ext.simulate_condition(conds[cid], 1500, 6000, SEED)["rt"]
        print(f"(0) {cid}: identical to wwsim.py: {np.array_equal(np.nan_to_num(a, posinf=-1), np.nan_to_num(b, posinf=-1))}")
    A = json.load(open("4_output/acc_mixtures/A2.json"))["theta_A"]
    changes = [("none", {})] + [(f"thr {v} Hz", {"thr": v}) for v in [18, 22, 26, 30, 35]] \
        + [(f"lca_z {v}", {"lca_z": v}) for v in [1.5, 2.0, 3.0]] \
        + [(f"urg {v} nA/s", {"urg": v}) for v in [-0.01, 0.01, 0.03, 0.06]] \
        + [(f"lapse p {p} gain {g}", {"p_lapse": p, "gain_lapse": g}) for p, g in [(0.05, 0.5), (0.1, 0.5), (0.1, 0.25)]] \
        + [(f"lca_bias {v}", {"lca_bias": v}) for v in [0.95, 0.99]]
    with Pool() as pool:
        out = pool.map(stats_of, [dict(A, **ch) for _, ch in changes])
    q0, a0, _ = out[0]
    print(f"\n(1) From calibrated A of A2: 10th/50th/99th {q0.round(0)} ms, accuracy {a0:.3f}")
    for (name, _), (q, acc, none) in zip(changes[1:], out[1:]):
        d = 100 * (q / q0 - 1)
        print(f"   {name:22s}: 10th {d[0]:+7.1f}%  median {d[1]:+7.1f}%  99th {d[2]:+7.1f}%  (ms shift at 10th/99th "
              f"{q[0] - q0[0]:+.0f}/{q[2] - q0[2]:+.0f})  accuracy {acc - a0:+.3f}  no commitment {100 * none:.1f}%")
