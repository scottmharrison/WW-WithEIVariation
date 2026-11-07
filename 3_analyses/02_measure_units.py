"""
02_measure_units.py

Purpose
    For every contrast, unit, cell and period (A or B), measure the RT distribution at the points
    where SKILL_vs_PRACTICE_explained.pdf says skill and practice act differently: the 5th, 10th,
    25th, 50th, 75th, 90th and 99th percentiles, and the ex-Gaussian mu, sigma and tau.

    Percentiles are numpy's default (linear interpolation between order statistics). With n trials
    the 99th percentile rests on the slowest n/100 or so trials; n is written beside every row.

Inputs
    2_data_processing/contrast_rts.csv

Outputs
    4_output/unit_measures.csv — one row per contrast, unit, cell and period
        (dictionary: unit_measures_dictionary.csv)

Depends on
    2_data_processing/01_build_contrasts.py; model/exgaussian.py

Last substantive change
    2026-11-06 — first version.

Run from the project root.
"""
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import exgaussian  # noqa: E402

PERCENTILES = [5, 10, 25, 50, 75, 90, 99]


def measure(item):
    """Worker: percentiles and ex-Gaussian of one group of RTs (run in a separate process)."""
    key, x = item
    x = np.array(x)
    mu, sigma, tau = exgaussian.fit(x)
    return key, len(x), np.percentile(x, PERCENTILES), (mu, sigma, tau)


# On Windows each worker process re-imports this file, so the work sits under this guard.
if __name__ == "__main__":
    groups = defaultdict(list)
    for r in csv.DictReader(open("2_data_processing/contrast_rts.csv", newline="")):
        groups[(r["contrast"], r["unit"], r["cell"], r["period"])].append(float(r["rt_s"]))

    with Pool() as pool:
        results = pool.map(measure, sorted(groups.items()), chunksize=8)

    with open("4_output/unit_measures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "unit", "cell", "period", "n"] + [f"q{p:02d}_ms" for p in PERCENTILES]
                   + ["mu_ms", "sigma_ms", "tau_ms"])
        for (contrast, unit, cell, period), n, q, (mu, sigma, tau) in results:
            w.writerow([contrast, unit, cell, period, n] + [round(v * 1000, 3) for v in q]
                       + [round(mu * 1000, 3), round(sigma * 1000, 3), round(tau * 1000, 3)])
    print(f"wrote {len(results):,} rows to 4_output/unit_measures.csv")
