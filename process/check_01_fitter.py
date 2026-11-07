"""
check_01_fitter.py (process code; not run by the pipeline)

Purpose
    Recovery of known ex-Gaussian parameters by model/exgaussian.py at the sample sizes of these
    contrasts (100, 500, 5,000 RTs), in seconds, and with values rounded to an 11.3 ms grid as in the
    StarCraft 2 table.

Inputs
    None.

Outputs
    Printed text; saved by the caller to process/output/check_01.txt

Depends on
    model/exgaussian.py

Last substantive change
    2026-11-06 — first version.
"""
import sys
import numpy as np
sys.path.insert(0, "model")
import exgaussian

SEED = 20261106
rng = np.random.default_rng(SEED)
print("seed", SEED)
for mu, sigma, tau in [(0.45, 0.05, 0.15), (0.6, 0.08, 0.3), (0.3, 0.04, 0.5)]:
    for n in [100, 500, 5000]:
        for grid in [None, 1 / 88.5]:
            est = []
            for _ in range(30):
                x = rng.normal(mu, sigma, n) + rng.exponential(tau, n)
                if grid:
                    x = np.floor(x / grid) * grid
                est.append(exgaussian.fit(x))
            e = np.array(est)
            print(f"true {mu},{sigma},{tau} n {n} grid {'11.3 ms' if grid else 'none'}: mean {e.mean(0).round(4)} SD {e.std(0).round(4)}")
