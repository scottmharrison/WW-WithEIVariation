"""
check_03_bsi_transfer.py (process code; not run by the pipeline)

Purpose
    Validate model/bsi_transfer.py before it is used: (1) the baseline noise sigma0 at which the
    derived transfer function comes closest to the reduced model's phi (a = 270, b = 108, d = 0.154);
    (2) the strength scale KAPPA at which S = 0.5 changes the slope below 10 Hz by 12.5% (the middle of
    the paper's 10-15%, Lo, Wang & Wang 2015, Fig. 2C) at R = 1.247; (3) the sign check: the slope
    falls for R >= 1.2 (I > E) and rises for R <= 1.156 (I < E), as the paper reports (Fig. 2A).

Inputs
    None.

Outputs
    Printed text; saved by the caller to process/output/check_03.txt

Depends on
    model/bsi_transfer.py

Last substantive change
    2026-11-07 — first version.
"""
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, "model")
import bsi_transfer as bt

I = np.linspace(0.2, 0.8, 61)
target = bt.abbott_chance(I, 270, 108, 0.154)
m = target <= 60


def rms_to_reduced(s0):
    nu = np.array([bt.rate(i, s0) for i in I])
    return float(np.sqrt(np.mean((nu[m] - target[m]) ** 2)))


best = minimize_scalar(rms_to_reduced, bounds=(1.5, 6.0), method="bounded")
sigma0 = best.x
p0, _ = bt.fit_abbott_chance(I, np.array([bt.rate(i, sigma0) for i in I]))
print(f"(1) sigma0 = {sigma0:.3f} mV: RMS to reduced phi {best.fun:.2f} Hz over 0-60 Hz; derived (a,b,d) = {p0.round(3)} vs (270, 108, 0.154)")

Ifine = np.linspace(0.25, 0.6, 141)


def low_slope(S, R, kappa):
    """Slope (Hz/nA) of a straight line through the f-I curve where the rate is 1 to 10 Hz."""
    nu = np.array([bt.rate(i, sigma0, S, R, kappa) for i in Ifine])
    k = (nu >= 1) & (nu <= 10)
    return np.polyfit(Ifine[k], nu[k], 1)[0]


base = low_slope(0.0, 1.2, 1.0)
for kappa in [1, 3, 10, 30]:
    print(f"   kappa {kappa}: slope change at S=0.5, R=1.247: {100 * (low_slope(0.5, 1.247, kappa) / base - 1):+.1f}%; "
          f"R=1.114: {100 * (low_slope(0.5, 1.114, kappa) / base - 1):+.1f}%")
kappa = brentq(lambda k: low_slope(0.5, 1.247, k) / base - 1 + 0.125, 0.5, 60, xtol=1e-3)
print(f"(2) KAPPA = {kappa:.3f} nS per unit S gives -12.5% at S=0.5, R=1.247")
for R in [1.0, 1.114, 1.156, 1.2, 1.247, 1.297, 1.35, 1.4]:
    vb = (2.5 * R * bt.VI) / (1 + 2.5 * R)
    print(f"(3) R {R:.3f} (V_B {vb:.1f} mV): slope change at S=0.5 {100 * (low_slope(0.5, R, kappa) / base - 1):+.1f}%, "
          f"at S=1.0 {100 * (low_slope(1.0, R, kappa) / base - 1):+.1f}%")
