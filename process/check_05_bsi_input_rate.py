"""
check_05_bsi_input_rate.py (process code; not run by the pipeline)

Purpose
    Repeat check_04 with the strength units given by Wang, Lee, Wang & Lo (2013, PLoS ONE 8:e62379,
    Table 1): 500 control neurons per population, 0.1 nS AMPA synapses, S = 0.3 r_e g with r_e the
    rate (Hz) of one control neuron, so mean excitatory control conductance = 500 x 0.1 nS x 2 ms x r_e
    = 3.33 S nS. The inhibitory population fires at the same rate with conductance R x 0.1 nS, so that
    R = (r_i g_GABA) / (r_e g_AMPA). The input axis is the rate of an extra excitatory Poisson input
    through the external synapse (2.1 nS, AMPA), as stimulus input enters the network (Wang, 2002),
    rather than an injected current; a constant -0.3 nA current places the curve's onset in range.

    Question: does the slope of the rate-input curve rise for R <= 1.156 and fall for R >= 1.2, as Lo,
    Wang & Wang (2015, Fig. 2A) report?

Inputs
    None.

Outputs
    Printed text; saved by the caller to process/output/check_05.txt

Depends on
    Nothing.

Seed
    SEED below; each condition uses a fresh generator with that seed (common random numbers).

Last substantive change
    2026-11-07 — first version.
"""
import numpy as np

SEED = 20261107
C, GL, VL, VTH, VR, TREF = 0.5, 25.0, -70.0, -50.0, -55.0, 2.0
DT, T, T0 = 0.05, 10000.0, 200.0
G_EXT, R_BG = 2.1, 2.4              # nS, kHz (background, Wang 2002 / Lo et al. 2015)
N_CTR, G_A = 500, 0.1               # control neurons per population, nS
TA, TG = 2.0, 5.0
I_OFFSET = -300.0                   # pA


def curve(inputs_khz, S, R, rng):
    n = len(inputs_khz)
    r_ctl = S / (0.3 * G_A) / 1000.0 * N_CTR      # kHz of control spikes arriving per population
    g_i = R * G_A                                 # same rate, inhibitory conductance R x 0.1 nS
    V = np.full(n, VL); s_x = np.zeros(n); s_e = np.zeros(n); s_i = np.zeros(n)
    refr = np.zeros(n); count = np.zeros(n)
    for k in range(int((T + T0) / DT)):
        s_x += -s_x * DT / TA + rng.poisson((R_BG + inputs_khz) * DT)
        if r_ctl > 0:
            s_e += -s_e * DT / TA + rng.poisson(r_ctl * DT, n)
            s_i += -s_i * DT / TG + rng.poisson(r_ctl * DT, n)
        Isyn = -G_EXT * s_x * V - G_A * s_e * V - g_i * s_i * (V + 70.0)      # pA
        V = np.where(refr <= 0, V + (-GL * (V - VL) + Isyn + I_OFFSET) / C * DT / 1000.0, VR)
        refr -= DT
        spike = V >= VTH
        if k * DT >= T0:
            count += spike
        V[spike] = VR
        refr[spike] = TREF
    return count / (T / 1000.0)


import sys  # noqa: E402
sys.path.insert(0, "model")
from bsi_transfer import abbott_chance, fit_abbott_chance  # noqa: E402

print("seed", SEED)
inputs = np.linspace(0.0, 3.0, 61)        # kHz of extra excitatory input
base = curve(inputs, 0.0, 1.2, np.random.default_rng(SEED))
print("baseline rates at 0, 0.5, ... 3 kHz:", base[::10].round(1))
fine = np.linspace(0, 3, 3001)


def slope_at(nu, target):
    p, _ = fit_abbott_chance(inputs, nu)
    f = abbott_chance(fine, *p)
    return np.gradient(f, fine)[np.argmin(np.abs(f - target))]


b5, b20 = slope_at(base, 5), slope_at(base, 20)
for S in [0.5, 1.0, 2.0]:
    for R in [1.0, 1.114, 1.156, 1.247, 1.297, 1.4]:
        nu = curve(inputs, S, R, np.random.default_rng(SEED))
        print(f"S {S}, R {R}: slope change at 5 Hz {100 * (slope_at(nu, 5) / b5 - 1):+.1f}%, at 20 Hz {100 * (slope_at(nu, 20) / b20 - 1):+.1f}%; "
              f"rate at 1.5 kHz {nu[30]:.1f} (baseline {base[30]:.1f})")
