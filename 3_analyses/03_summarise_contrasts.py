"""
03_summarise_contrasts.py

Purpose
    Summarise each contrast as the percent change from A to B in each percentile and ex-Gaussian
    parameter, with 95% bootstrap intervals, and classify where the change lives.

    Within-unit contrasts (P, F, D). For each unit and cell, change = 100 (B / A - 1); a unit's change
    is the mean over its cells; the contrast's change is the median over units. Between-unit contrasts
    (S): the group value is the median over units, and change = 100 (group B / group A - 1).

    Front minus tail = change at the 10th percentile minus change at the 99th, in percentage points.
    The document predicts it negative for practice (the front moves, the tail does not: -44 - (-2) =
    -42 in its simulation) and positive for skill (-6 - (-56) = +50). Front-led: 95% interval below 0;
    tail-led: above 0; uniform: interval includes 0.

    Intervals. Percentiles: two-stage (hierarchical) bootstrap, resampling units and then trials within
    each unit, cell and period, so that both the spread between units and the sampling error within a
    unit count (Davison & Hinkley, 1997, Bootstrap Methods, sect. 3.8). Ex-Gaussian parameters: units
    resampled, the fits not repeated. Percentile intervals (2.5th and 97.5th of 1,000 replicates).
    With two or three units (D1, P1) the interval mainly reflects trials within units.

Inputs
    2_data_processing/contrast_rts.csv, 2_data_processing/contrasts.csv, 4_output/unit_measures.csv

Outputs
    4_output/unit_changes.csv — per unit percent changes, within-unit contrasts (dictionary beside it)
    4_output/contrast_summary.csv — one row per contrast (dictionary beside it)

Depends on
    02_measure_units.py

Last substantive change
    2026-11-06 — first version.

Run from the project root.
"""
import csv
from collections import defaultdict

import numpy as np

SEED = 61106
N_BOOT = 1000
PCTS = [5, 10, 25, 50, 75, 90, 99]
PARAMS = ["mu", "sigma", "tau"]
rng = np.random.default_rng(SEED)
print("seed", SEED)

contrasts = list(csv.DictReader(open("2_data_processing/contrasts.csv")))
measures = list(csv.DictReader(open("4_output/unit_measures.csv")))
trials = defaultdict(list)
for r in csv.DictReader(open("2_data_processing/contrast_rts.csv", newline="")):
    trials[(r["contrast"], r["unit"], r["cell"], r["period"])].append(float(r["rt_s"]))
trials = {k: np.array(v) for k, v in trials.items()}


def pct_change(a, b):
    return 100 * (b / a - 1)


def interval(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def direction(lo, hi):
    return "falls" if hi < 0 else "rises" if lo > 0 else "no clear change"


summary, unit_rows = [], []
for c in contrasts:
    cid = c["contrast"]
    rows = [m for m in measures if m["contrast"] == cid]
    within = not cid.startswith("S")
    row = dict(c)
    if within:
        # Cells and units of this contrast; a unit enters only where it has both periods in a cell.
        cells = defaultdict(list)
        for m in rows:
            cells[(m["unit"], m["cell"])].append(m)
        units = sorted({u for (u, cell), ms in cells.items() if len(ms) == 2})
        unit_cells = {u: [cell for (uu, cell), ms in cells.items() if uu == u and len(ms) == 2] for u in units}

        def unit_change(u, stat):
            vals = []
            for cell in unit_cells[u]:
                a, b = sorted(cells[(u, cell)], key=lambda m: m["period"])
                # A fit whose sigma or tau in period A is under 2 ms has no defined percent change in
                # that parameter (the denominator is about 0: the fit has collapsed to a pure
                # exponential or a pure normal); that cell gives no value, and the unit has none if no
                # cell does. Counts are written to the summary (n_units_without_sigma, _tau).
                if stat in ("sigma_ms", "tau_ms") and float(a[stat]) < 2:
                    vals.append(np.nan)
                    continue
                vals.append(pct_change(float(a[stat]), float(b[stat])))
            return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan

        point = {s: np.nanmedian([unit_change(u, s) for u in units]) for s in [f"q{p:02d}_ms" for p in PCTS] + [f"{p}_ms" for p in PARAMS]}
        row["n_units_without_sigma"] = int(np.sum(np.isnan([unit_change(u, "sigma_ms") for u in units])))
        row["n_units_without_tau"] = int(np.sum(np.isnan([unit_change(u, "tau_ms") for u in units])))
        for u in units:
            unit_rows.append(dict(contrast=cid, unit=u, n_cells=len(unit_cells[u]),
                                  **{s.replace("_ms", "_change_pct"): round(unit_change(u, s), 3)
                                     for s in [f"q{p:02d}_ms" for p in PCTS] + [f"{p}_ms" for p in PARAMS]}))
        # Bootstrap of percentiles: units, then trials within each unit, cell and period.
        boot_q = np.empty((N_BOOT, len(PCTS)))
        boot_p = np.empty((N_BOOT, len(PARAMS)))
        unit_par = np.array([[unit_change(u, f"{p}_ms") for p in PARAMS] for u in units])
        for i in range(N_BOOT):
            pick = rng.integers(0, len(units), len(units))
            per_unit = []
            for j in pick:
                u = units[j]
                ch = []
                for cell in unit_cells[u]:
                    qa, qb = (np.percentile(rng.choice(trials[(cid, u, cell, p)], len(trials[(cid, u, cell, p)])), PCTS)
                              for p in ("A", "B"))
                    ch.append(pct_change(qa, qb))
                per_unit.append(np.mean(ch, axis=0))
            boot_q[i] = np.median(per_unit, axis=0)
            boot_p[i] = np.nanmedian(unit_par[pick], axis=0)
        row["n_units"] = len(units)
    else:
        groups = {p: [m for m in rows if m["period"] == p] for p in ("A", "B")}
        point = {}
        for s in [f"q{p:02d}_ms" for p in PCTS] + [f"{p}_ms" for p in PARAMS]:
            point[s] = pct_change(np.median([float(m[s]) for m in groups["A"]]), np.median([float(m[s]) for m in groups["B"]]))
        boot_q = np.empty((N_BOOT, len(PCTS)))
        boot_p = np.empty((N_BOOT, len(PARAMS)))
        for i in range(N_BOOT):
            gq, gp = {}, {}
            for p in ("A", "B"):
                ms = groups[p]
                pick = rng.integers(0, len(ms), len(ms))
                gq[p] = np.median([np.percentile(rng.choice(x, len(x)), PCTS)
                                   for x in (trials[(cid, ms[j]["unit"], ms[j]["cell"], p)] for j in pick)], axis=0)
                gp[p] = np.median([[float(ms[j][f"{q}_ms"]) for q in PARAMS] for j in pick], axis=0)
            boot_q[i] = pct_change(gq["A"], gq["B"])
            boot_p[i] = pct_change(gp["A"], gp["B"])
        row["n_units"] = f"{len(groups['A'])} / {len(groups['B'])}"
        row["n_units_without_sigma"] = 0  # group medians are used, so no unit is left out
        row["n_units_without_tau"] = 0

    for k, p in enumerate(PCTS):
        lo, hi = interval(boot_q[:, k])
        row.update({f"q{p:02d}_change_pct": round(point[f"q{p:02d}_ms"], 2), f"q{p:02d}_low": round(lo, 2), f"q{p:02d}_high": round(hi, 2)})
    for k, p in enumerate(PARAMS):
        lo, hi = interval(boot_p[:, k])
        row.update({f"{p}_change_pct": round(point[f"{p}_ms"], 2), f"{p}_low": round(lo, 2), f"{p}_high": round(hi, 2),
                    f"{p}_direction": direction(lo, hi)})
    ft = boot_q[:, PCTS.index(10)] - boot_q[:, PCTS.index(99)]
    lo, hi = interval(ft)
    row.update(front_minus_tail=round(point["q10_ms"] - point["q99_ms"], 2), front_minus_tail_low=round(lo, 2),
               front_minus_tail_high=round(hi, 2),
               location="front-led" if hi < 0 else "tail-led" if lo > 0 else "uniform")
    summary.append(row)
    print(cid, row["location"], row["front_minus_tail"], (row["front_minus_tail_low"], row["front_minus_tail_high"]),
          "mu", row["mu_direction"], "tau", row["tau_direction"])

for path, rows in [("4_output/contrast_summary.csv", summary), ("4_output/unit_changes.csv", unit_rows)]:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
print("wrote 4_output/contrast_summary.csv and 4_output/unit_changes.csv")
