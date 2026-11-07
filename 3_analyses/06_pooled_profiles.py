"""
06_pooled_profiles.py

Purpose
    The change from A to B in every contrast, measured on trials pooled over units: percentiles of
    the pooled RT distributions, their percent change, and a bootstrap over units for its uncertainty.

    Pooling uses every trial, so a percentile is never estimated from a handful of trials; it is how
    WW-CarefulFitting fitted the model to each condition, so the model and the data describe the same
    distribution. The cost is that differences between units are mixed into the pooled shape; the
    unit-level results (02-03) remain alongside.

    Uncertainty: cluster bootstrap. Units are resampled with replacement, keeping all of a unit's
    trials (Field & Welsh, 2007, J R Stat Soc B 69:369); for within-unit contrasts a resampled unit
    brings its A and its B trials; for between-group contrasts (S1, S2) units are resampled within each
    group. 500 replicates; standard error = SD over replicates.

Inputs
    2_data_processing/contrast_rts.csv, 2_data_processing/contrasts.csv

Outputs
    4_output/pooled_profiles.csv — one row per contrast and percentile (dictionary beside it)

Depends on
    2_data_processing/01_build_contrasts.py

Last substantive change
    2026-11-06 — first version.

Run from the project root.
"""
import csv
from collections import defaultdict

import numpy as np

SEED = 61107
N_BOOT = 500
PCTS = [5, 10, 25, 50, 75, 90, 99]
rng = np.random.default_rng(SEED)
print("seed", SEED)

contrasts = list(csv.DictReader(open("2_data_processing/contrasts.csv")))
trials = defaultdict(list)
for r in csv.DictReader(open("2_data_processing/contrast_rts.csv", newline="")):
    trials[(r["contrast"], r["period"], r["unit"])].append(float(r["rt_s"]) * 1000)

rows = []
for c in contrasts:
    cid = c["contrast"]
    by = {p: {u: np.array(v) for (cc, pp, u), v in trials.items() if cc == cid and pp == p} for p in ("A", "B")}
    qa = np.percentile(np.concatenate(list(by["A"].values())), PCTS)
    qb = np.percentile(np.concatenate(list(by["B"].values())), PCTS)
    boot = np.empty((N_BOOT, len(PCTS)))
    if cid.startswith("S"):
        ua, ub = list(by["A"]), list(by["B"])
        for i in range(N_BOOT):
            a = np.concatenate([by["A"][ua[j]] for j in rng.integers(0, len(ua), len(ua))])
            b = np.concatenate([by["B"][ub[j]] for j in rng.integers(0, len(ub), len(ub))])
            boot[i] = 100 * (np.percentile(b, PCTS) / np.percentile(a, PCTS) - 1)
    else:
        units = sorted(set(by["A"]) & set(by["B"]))
        for i in range(N_BOOT):
            pick = [units[j] for j in rng.integers(0, len(units), len(units))]
            a = np.concatenate([by["A"][u] for u in pick])
            b = np.concatenate([by["B"][u] for u in pick])
            boot[i] = 100 * (np.percentile(b, PCTS) / np.percentile(a, PCTS) - 1)
    for k, p in enumerate(PCTS):
        rows.append(dict(contrast=cid, axis=c["axis"], percentile=p, n_A=sum(len(v) for v in by["A"].values()),
                         n_B=sum(len(v) for v in by["B"].values()), q_A_ms=round(qa[k], 3), q_B_ms=round(qb[k], 3),
                         change_pct=round(100 * (qb[k] / qa[k] - 1), 3), se_pct=round(float(np.std(boot[:, k])), 3),
                         low_pct=round(float(np.percentile(boot[:, k], 2.5)), 3),
                         high_pct=round(float(np.percentile(boot[:, k], 97.5)), 3)))
    print(cid, "change at 10/50/99:", [r["change_pct"] for r in rows[-7:] if r["percentile"] in (10, 50, 99)])

with open("4_output/pooled_profiles.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to 4_output/pooled_profiles.csv")
