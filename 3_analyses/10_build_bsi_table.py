"""
10_build_bsi_table.py

Purpose
    The f-I table that carries the balanced-synaptic-input (BSI) control signal of Lo, Wang & Wang
    (2015, J Neurophysiol, doi:10.1152/jn.00845.2013) into the reduced Wong-Wang model.

    For each strength S and ratio R on a grid, the paper's excitatory neuron is simulated directly: a
    conductance-based leaky integrate-and-fire cell (C = 0.5 nF, gL = 25 nS, VL = -70 mV, threshold
    -50 mV, reset -55 mV, refractory 2 ms) with 2,400 Hz background AMPA input at 2.1 nS, an injected
    current I (the reduced model's input axis), and the control signal as Poisson spikes from two
    populations of 500 neurons (Wang, Lee, Wang & Lo, 2013, PLoS ONE 8:e62379, Table 1): excitatory
    AMPA at 0.1 nS, inhibitory GABA_A at R x 0.1 nS, both firing at r = S / (0.3 x 0.1) Hz per neuron, so
    that S = 0.3 r g_AMPA and R = (r_i g_GABA) / (r_e g_AMPA) as the papers define them.

    Each rate curve is fitted to the Abbott-Chance form phi(I) = (aI - b) / (1 - exp(-d (aI - b))) over
    0-60 Hz. The reduced model receives the control signal as changes relative to the curve without it,
    so a fitted condition is unchanged at S = 0:
        gain      a(S,R) / a(0)            multiplies a = 270
        shift     b(S,R)/a(S,R) - b(0)/a(0) nA added to the current at which the curve starts (b/a)
        curvature d(S,R) / d(0)            multiplies d = 0.154
    Simulation: Euler, dt = 0.05 ms, 10 s per point after 0.2 s, the same seed in every condition
    (common random numbers), conditions in parallel.

Inputs
    None.

Outputs
    4_output/bsi_table.csv — one row per (S, R) (dictionary: bsi_table_dictionary.csv)

Depends on
    model/bsi_transfer.py (Abbott-Chance form and its fit)

Seed
    SEED below; every condition uses a fresh generator with it.

Last substantive change
    2026-11-07 — first version.

Run from the project root.
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from bsi_transfer import fit_abbott_chance  # noqa: E402

SEED = 20261107
C, GL, VL, VTH, VR, TREF = 0.5, 25.0, -70.0, -50.0, -55.0, 2.0
DT, T, T0 = 0.05, 10000.0, 200.0
G_EXT, R_BG, N_CTR, G_A, TA, TG = 2.1, 2.4, 500, 0.1, 2.0, 5.0
CURRENTS = np.linspace(-0.4, 0.35, 76)          # nA
S_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
R_GRID = [1.0, 1.05, 1.1, 1.156, 1.2, 1.247, 1.297, 1.35, 1.4]


def rates(args):
    """Worker: rates (Hz) at every current for one (S, R)."""
    S, R = args
    rng = np.random.default_rng(SEED)
    n = len(CURRENTS)
    r_ctl = S / (0.3 * G_A) * N_CTR / 1000.0      # kHz of control spikes arriving per population
    V = np.full(n, VL); s_x = np.zeros(n); s_e = np.zeros(n); s_i = np.zeros(n)
    refr = np.zeros(n); count = np.zeros(n)
    I = CURRENTS * 1000.0                         # pA
    for k in range(int((T + T0) / DT)):
        s_x += -s_x * DT / TA + rng.poisson(R_BG * DT, n)
        if r_ctl > 0:
            s_e += -s_e * DT / TA + rng.poisson(r_ctl * DT, n)
            s_i += -s_i * DT / TG + rng.poisson(r_ctl * DT, n)
        # Synaptic currents in pA (nS x mV); dV in mV: pA / nF = mV/s, times dt in ms / 1000.
        Isyn = -G_EXT * s_x * V - G_A * s_e * V - R * G_A * s_i * (V + 70.0)
        V = np.where(refr <= 0, V + (-GL * (V - VL) + Isyn + I) / C * DT / 1000.0, VR)
        refr -= DT
        spike = V >= VTH
        if k * DT >= T0:
            count += spike
        V[spike] = VR
        refr[spike] = TREF
    return S, R, count / (T / 1000.0)


if __name__ == "__main__":
    grid = [(0.0, 1.2)] + [(S, R) for S in S_GRID if S > 0 for R in R_GRID]
    with Pool() as pool:
        results = pool.map(rates, grid)
    fits = {}
    for S, R, nu in results:
        p, rms = fit_abbott_chance(CURRENTS, nu)
        fits[(S, R)] = (p, rms)
    a0, b0, d0 = fits[(0.0, 1.2)][0]
    rows = []
    for (S, R), ((a, b, d), rms) in fits.items():
        for RR in (R_GRID if S == 0 else [R]):   # S = 0 has no ratio; written for every R
            rows.append(dict(S=S, R=RR, a_fit=round(a, 4), b_fit=round(b, 4), d_fit=round(d, 5), fit_rms_hz=round(rms, 3),
                             gain=round(a / a0, 5), shift_nA=round(b / a - b0 / a0, 5), curvature=round(d / d0, 5)))
    rows.sort(key=lambda r: (r["S"], r["R"]))
    with open("4_output/bsi_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to 4_output/bsi_table.csv; baseline fit (a, b, d) = ({a0:.1f}, {b0:.2f}, {d0:.4f})")
