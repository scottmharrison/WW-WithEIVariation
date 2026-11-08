"""
11_pooled_accuracy_profiles.py

Purpose
    The change from A to B in every contrast, on pooled correct and error trials: accuracy, the
    percentiles of correct RTs, and the percentiles of error RTs, with a cluster bootstrap over units.
    These are the observations the accuracy-aware model fits (12_fit_accuracy_mixtures.py) must match.

    Accuracy is the proportion correct; its change is B - A in proportion (not logit, because some
    conditions have no errors, e.g. monkeys at 51.2% coherence). Correct RT: percent change at the 5th
    to 99th percentiles. Error RT: percent change at the 10th, 30th, 50th, 70th and 90th, only where A
    and B each have at least 100 pooled errors. Contrasts without accuracy (British Lexicon Project:
    error trials have no RT in DataPrep; StarCraft 2: no correct/error label) have correct RTs only.

    Cluster bootstrap: units resampled with all their trials, within groups for S1 and S2;
    500 replicates; SE = SD over replicates.

Inputs
    2_data_processing/contrast_trials_acc.csv, 2_data_processing/contrasts.csv

Outputs
    4_output/pooled_accuracy_profiles.csv — one row per contrast and measure (dictionary beside it)
    with --fine: 4_output/pooled_accuracy_profiles_fine.csv, the same with 12 correct-RT percentiles

Depends on
    2_data_processing/01_build_contrasts.py --with-errors

Seed
    SEED below.

Last substantive change
    2026-11-07 — --fine option (12 percentiles) for stage 15.

Run from the project root.
"""
import csv
from collections import defaultdict

import numpy as np

SEED = 61108
N_BOOT = 500
import sys
# --fine: 12 correct-RT percentiles, for the mechanism comparison of stage 15, which fits models with up
# to 13 changed parameters and so needs more observations than the 7 percentiles of stages 12-14.
FINE = "--fine" in sys.argv
PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99] if FINE else [5, 10, 25, 50, 75, 90, 99]
OUT_FILE = "4_output/pooled_accuracy_profiles_fine.csv" if FINE else "4_output/pooled_accuracy_profiles.csv"
ERR_PCTS = [10, 30, 50, 70, 90]
MIN_ERRORS = 100
# The British Lexicon Project table keeps only trials with an RT, and its error trials have none in
# DataPrep, so every kept trial is correct; its accuracy is not observed and is not reported.
NO_ACCURACY = {"P3"}
rng = np.random.default_rng(SEED)
print("seed", SEED)

contrasts = [r["contrast"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))]
data = defaultdict(lambda: ([], []))
for r in csv.DictReader(open("2_data_processing/contrast_trials_acc.csv", newline="")):
    rts, cor = data[(r["contrast"], r["period"], r["unit"])]
    rts.append(float(r["rt_s"]) * 1000)
    cor.append(-1 if r["correct"] == "" else int(r["correct"]))


def measures(trials):
    """Accuracy, correct-RT percentiles, error-RT percentiles (or None) of pooled (rt, correct) arrays."""
    rt, c = trials
    has_acc = np.any(c >= 0)
    correct_rt = rt[(c == 1) | (c == -1)]
    acc = float(np.mean(c[c >= 0] == 1)) if has_acc else np.nan
    err = rt[c == 0]
    return acc, np.percentile(correct_rt, PCTS), (np.percentile(err, ERR_PCTS) if err.size >= MIN_ERRORS else None), err.size


def pooled(units, period, cid):
    rts = np.concatenate([np.array(data[(cid, period, u)][0]) for u in units])
    cor = np.concatenate([np.array(data[(cid, period, u)][1]) for u in units])
    return rts, cor


rows = []
for cid in contrasts:
    ua = sorted({u for (c, p, u) in data if c == cid and p == "A"})
    ub = sorted({u for (c, p, u) in data if c == cid and p == "B"})
    within = not cid.startswith("S")
    if within:
        ua = ub = sorted(set(ua) & set(ub))
    mA, mB = measures(pooled(ua, "A", cid)), measures(pooled(ub, "B", cid))
    use_err = mA[2] is not None and mB[2] is not None
    boot_acc, boot_q, boot_e = [], [], []
    for _ in range(N_BOOT):
        if within:
            pick = [ua[j] for j in rng.integers(0, len(ua), len(ua))]
            pa = pb = pick
        else:
            pa = [ua[j] for j in rng.integers(0, len(ua), len(ua))]
            pb = [ub[j] for j in rng.integers(0, len(ub), len(ub))]
        a, b = measures(pooled(pa, "A", cid)), measures(pooled(pb, "B", cid))
        boot_acc.append(b[0] - a[0])
        boot_q.append(100 * (b[1] / a[1] - 1))
        if use_err and a[2] is not None and b[2] is not None:
            boot_e.append(100 * (b[2] / a[2] - 1))
    boot_q = np.array(boot_q)

    def add(measure, level, va, vb, change, boot):
        rows.append(dict(contrast=cid, measure=measure, level=level, value_A=round(va, 5), value_B=round(vb, 5),
                         change=round(change, 5), se=round(float(np.nanstd(boot)), 5)))

    if not np.isnan(mA[0]) and cid not in NO_ACCURACY:
        add("accuracy", "", mA[0], mB[0], mB[0] - mA[0], np.array(boot_acc))
    for k, p in enumerate(PCTS):
        add("correct_rt_pct", p, mA[1][k], mB[1][k], 100 * (mB[1][k] / mA[1][k] - 1), boot_q[:, k])
    if use_err:
        boot_e = np.array(boot_e)
        for k, p in enumerate(ERR_PCTS):
            add("error_rt_pct", p, mA[2][k], mB[2][k], 100 * (mB[2][k] / mA[2][k] - 1), boot_e[:, k])
    print(cid, f"accuracy {mA[0]:.3f} -> {mB[0]:.3f}; errors {mA[3]} / {mB[3]}; error RTs used: {use_err}")

with open(OUT_FILE, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {OUT_FILE}")
