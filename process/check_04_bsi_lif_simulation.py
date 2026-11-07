"""
check_04_bsi_lif_simulation.py (process code; not run by the pipeline)

Purpose
    The f-I curve of the paper's own excitatory neuron, simulated directly (no mean-field step), with
    and without the balanced-input control signal, to test two things before any fit uses it:
    (1) whether the slope below 10 Hz falls for R >= 1.2 and rises for R <= 1.156 at S = 0.5, and by
        10-15%, as Lo, Wang & Wang (2015) report (Figs. 2A, 2C);
    (2) the assumed units: S = 0.3 r_e g_AMPA with r_e in kHz and g_AMPA = 0.1 nS, so the mean
        excitatory control conductance is tau_AMPA r_e g = (2/0.3) S nS.

    Neuron: conductance-based leaky integrate-and-fire (paper's Methods): C = 0.5 nF, gL = 25 nS,
    VL = -70 mV, threshold -50 mV, reset -55 mV, refractory 2 ms; background Poisson AMPA input
    2,400 Hz at 2.1 nS (tau 2 ms); control: Poisson AMPA (reversal 0 mV, tau 2 ms) and GABA_A
    (reversal -70 mV, tau 5 ms) at 0.1 nS per spike; injected current I (nA) as the input axis.
    Euler integration, dt = 0.05 ms, 10 s per point after 0.2 s settling. Every condition uses the same
    random-number seed (common random numbers), so differences between conditions are not sampling noise.
    Gain is read from a fit of the Abbott-Chance form to the whole curve, not from a few points.

Inputs
    None.

Outputs
    Printed text; saved by the caller to process/output/check_04.txt

Depends on
    Nothing.

Seed
    SEED below, set once; every draw comes from that generator, passed into fI().

Last substantive change
    2026-11-07 — first version.
"""
import numpy as np

SEED = 20261107
C, GL, VL, VTH, VR, TREF = 0.5, 25.0, -70.0, -50.0, -55.0, 2.0
DT, T, T0 = 0.05, 10000.0, 200.0
G_BG, R_BG, G_CTL = 2.1, 2.4, 0.1          # nS, kHz, nS
TA, TG = 2.0, 5.0


def fI(currents, S, R, rng):
    """Rates (Hz) at each injected current (nA) for control strength S and ratio R."""
    n = len(currents)
    re = S / 0.3 / G_CTL                       # kHz of excitatory control spikes (S = 0.3 r_e g)
    ri = R * re * G_CTL / G_CTL                # R = (r_i g_GABA) / (r_e g_AMPA), same g
    V = np.full(n, VL); s_bg = np.zeros(n); s_e = np.zeros(n); s_i = np.zeros(n)
    refr = np.zeros(n); count = np.zeros(n)
    I = np.asarray(currents) * 1000.0          # pA
    steps = int((T + T0) / DT)
    for k in range(steps):
        s_bg += -s_bg * DT / TA + rng.poisson(R_BG * DT, n)
        s_e += -s_e * DT / TA + (rng.poisson(re * DT, n) if re > 0 else 0)
        s_i += -s_i * DT / TG + (rng.poisson(ri * DT, n) if ri > 0 else 0)
        # Currents in pA: nS * mV = pA.
        Isyn = -G_BG * s_bg * (V - 0.0) - G_CTL * s_e * (V - 0.0) - G_CTL * s_i * (V + 70.0)
        dV = (-GL * (V - VL) + Isyn + I) / C * DT / 1000.0      # pA / nF * ms = mV * 1e-3 ... (pA/nF = mV/s)
        active = refr <= 0
        V = np.where(active, V + dV, VR)
        refr -= DT
        spike = V >= VTH
        if k * DT >= T0:
            count += spike
        V[spike] = VR
        refr[spike] = TREF
    return count / (T / 1000.0)


import sys
sys.path.insert(0, "model")
from bsi_transfer import abbott_chance, fit_abbott_chance  # noqa: E402

print("seed", SEED)
currents = np.linspace(-0.35, 0.35, 71)
fits = {}
for S, R in [(0.0, 1.2), (0.25, 1.114), (0.5, 1.114), (0.25, 1.156), (0.5, 1.156), (0.25, 1.247), (0.5, 1.247),
             (0.25, 1.297), (0.5, 1.297), (1.0, 1.114), (1.0, 1.297)]:
    nu = fI(currents, S, R, np.random.default_rng(SEED))
    p, rms = fit_abbott_chance(currents, nu)
    # Slope at 5 Hz and 20 Hz from the fitted curve (finite difference on a fine grid).
    fine = np.linspace(-0.4, 0.4, 4001)
    f = abbott_chance(fine, *p)
    slopes = [np.gradient(f, fine)[np.argmin(np.abs(f - r))] for r in (5.0, 20.0)]
    fits[(S, R)] = (p, slopes)
    print(f"S {S}, R {R}: (a, b, d) = {p.round(3)}, fit RMS {rms:.2f} Hz; slope at 5 Hz {slopes[0]:.0f} Hz/nA, at 20 Hz {slopes[1]:.0f} Hz/nA")
b5, b20 = fits[(0.0, 1.2)][1]
print("Slope change from no control signal (%), at 5 Hz / 20 Hz:")
for (S, R), (p, sl) in fits.items():
    if S > 0:
        print(f"   S {S}, R {R}: {100 * (sl[0] / b5 - 1):+.1f} / {100 * (sl[1] / b20 - 1):+.1f}")
