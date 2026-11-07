"""
bsi_transfer.py

Purpose
    The transfer function (input current -> firing rate) of an excitatory neuron of the Wong-Wang
    circuit, with and without a top-down control signal of balanced excitation and inhibition (BSI;
    Lo, Wang & Wang, 2015, J Neurophysiol, doi:10.1152/jn.00845.2013), and its fit to the
    Abbott-Chance form phi(I) = (aI - b) / (1 - exp(-d (aI - b))) that the reduced two-variable model
    uses (Wong & Wang, 2006, J Neurosci 26:1314).

    Neuron: leaky integrate-and-fire, the paper's excitatory cell: C = 0.5 nF, gL = 25 nS,
    VL = -70 mV, threshold -50 mV, reset -55 mV, refractory 2 ms.

    Rate: the Siegert (Ricciardi) first-passage formula for a leaky integrator driven by Gaussian
    noise (Brunel & Hakim, 1999, Neural Comput 11:1621):
        1/nu = t_ref + tau * sqrt(pi) * integral from (Vr - mu)/sigma to (theta - mu)/sigma of
               exp(u^2) (1 + erf u) du,
    with mu the mean free membrane potential and sigma its SD. exp(u^2)(1 + erf u) = erfcx(-u),
    which scipy computes without overflow.

    Control signal (BSI): an excitatory conductance Ge (reversal 0 mV) and an inhibitory conductance
    Gi (reversal -70 mV) onto the selective neurons. In the mean-field limit they add Ge + Gi to the
    leak, which shortens the membrane time constant, and pull the membrane toward the balance
    potential V_B = (Ge * 0 + Gi * (-70)) / (Ge + Gi). The paper's ratio R = (r_i g_GABA) / (r_e g_AMPA);
    mean conductance = g * tau_syn * r, so Gi / Ge = R * tau_GABA / tau_AMPA = 2.5 R, which gives
    R = 0.4 (0 - V_B) / (V_B + 70), the paper's equation 9 (R = 1.156 <-> V_B = -52 mV). Poisson
    arrival also adds membrane-potential variance (shot noise in the diffusion limit, filtered by
    the synaptic decay; Tuckwell, 1988, Introduction to Theoretical Neurobiology, vol. 2, ch. 9).

    Strength: Ge = KAPPA * S nS. The paper defines S = 0.3 r_e g_AMPA without the size of the control
    population, so KAPPA is not given; it is set (process/check_03) so that S = 0.5 changes the slope
    of the f-I curve below 10 Hz by the paper's reported 10-15% (its Fig. 2C).

Inputs
    None (functions only).

Outputs
    None (functions only).

Depends on
    Nothing.

Last substantive change
    2026-11-07 — first version.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import erfcx

C, GL, VL, VTH, VR, TREF = 0.5, 25.0, -70.0, -50.0, -55.0, 2.0   # nF, nS, mV, mV, mV, ms
VE, VI, TAU_AMPA, TAU_GABA = 0.0, -70.0, 2.0, 5.0                 # mV, mV, ms, ms
G_SYN = 0.1                                                      # nS per control synapse (paper)
V_OPERATING = -53.0                                              # mV, membrane level where noise is evaluated


def siegert(mu, sigma, tau):
    """Firing rate (Hz) of the LIF neuron with mean free potential mu (mV), SD sigma (mV), time
    constant tau (ms)."""
    lo, hi = (VR - mu) / sigma, (VTH - mu) / sigma
    integral = quad(lambda u: erfcx(-u), lo, hi, limit=200)[0]
    return 1000.0 / (TREF + tau * np.sqrt(np.pi) * integral)


def rate(I, sigma0, S=0.0, R=1.2, kappa=1.0):
    """Rate (Hz) at input current I (nA) with baseline membrane noise sigma0 (mV) and BSI (S, R)."""
    ge = kappa * S
    gi = 2.5 * R * ge
    g_tot = GL + ge + gi                                     # nS
    tau = C / g_tot * 1000.0                                 # ms (nF / nS = s)
    v_rest = (GL * VL + ge * VE + gi * VI) / g_tot           # mV
    mu = v_rest + I / g_tot * 1000.0                         # mV (nA / nS = V)
    # Baseline noise is scaled with the membrane time constant (a current noise passed through the
    # membrane: SD of V proportional to sqrt(tau) / g_tot ... here as sigma0 * (GL / g_tot) sqrt(tau/tau0)).
    tau0 = C / GL * 1000.0
    var = (sigma0 * GL / g_tot) ** 2 * (tau / tau0)
    # BSI shot noise: rate r = G / (g tau_syn); current variance per type g^2 (V - Vrev)^2 r tau_syn / 2
    # (in nA^2 after unit conversion); membrane variance = current variance / g_tot^2 * tau_syn / (tau_syn + tau).
    for G, vrev, ts in ((ge, VE, TAU_AMPA), (gi, VI, TAU_GABA)):
        if G > 0:
            r = G / (G_SYN * ts * 1e-3)                      # Hz
            var_I = (G_SYN * 1e-3 * (V_OPERATING - vrev)) ** 2 * r * ts * 1e-3 / 2  # nA^2 ... (nS*mV = pA)
            var += var_I / (g_tot * 1e-3) ** 2 * ts / (ts + tau)
    return siegert(mu, np.sqrt(var), tau)


def abbott_chance(I, a, b, d):
    x = a * I - b
    return np.where(np.abs(x) < 1e-9, 1.0 / d, x / (1.0 - np.exp(-d * x)))


def fit_abbott_chance(I, nu):
    """Least-squares fit of (a, b, d) to rates nu at currents I, over rates up to 60 Hz."""
    m = nu <= 60.0
    res = least_squares(lambda p: abbott_chance(I[m], *p) - nu[m], x0=[270.0, 108.0, 0.154],
                        bounds=([10, -500, 0.001], [2000, 2000, 5]))
    return res.x, float(np.sqrt(np.mean(res.fun ** 2)))
