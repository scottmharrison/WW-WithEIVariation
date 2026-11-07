"""
exgaussian.py

Purpose
    Maximum-likelihood fit of the ex-Gaussian distribution to RTs, used by 3_analyses/02_measure_units.py.

    The ex-Gaussian is the distribution of the sum of a normal variable (mean mu, SD sigma) and an
    independent exponential variable (mean tau); mu and sigma describe the location and spread of the
    bulk of the distribution, tau the weight of its slow tail (Ratcliff, 1979, Psychol Bull 86:446;
    Heathcote, Popiel & Mewhort, 1991, Psychol Bull 109:340). The parameters describe shape; they are
    not measures of particular processes (Matzke & Wagenmakers, 2009, Psychon Bull Rev 16:798).

    These datasets record RT to the millisecond (StarCraft 2 to 11.3 ms), so the continuous density is
    used; process/output/check_01.txt shows the fit recovering known parameters and flags fits whose
    sigma ends near 0.

Inputs
    None (functions only).

Outputs
    None (functions only).

Depends on
    Nothing.

Last substantive change
    2026-11-06 — first version.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr


def log_density(x, mu, sigma, tau):
    """log f(x) = -log tau + (mu - x)/tau + sigma^2/(2 tau^2) + log Phi((x - mu)/sigma - sigma/tau).
    log_ndtr gives log Phi accurately even where Phi is far below the smallest representable number."""
    return -np.log(tau) + (mu - x) / tau + sigma**2 / (2 * tau**2) + log_ndtr((x - mu) / sigma - sigma / tau)


def fit(x):
    """Return (mu, sigma, tau) in the units of x.

    The optimiser works on mu and the logs of sigma and tau, scaled by the sample SD, so sigma and tau
    stay positive and one tolerance suits every dataset. Nelder-Mead moves a triangle of parameter
    points uphill using likelihood values only. It starts from the method-of-moments estimate (third
    central moment = 2 tau^3) and from two other points, and keeps the highest likelihood.
    """
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std()
    skew = np.mean((x - m) ** 3)
    tau0 = min((skew / 2) ** (1 / 3), 0.9 * s) if skew > 0 else 0.5 * s
    sigma0 = np.sqrt(max(s**2 - tau0**2, (0.2 * s) ** 2))

    def nll(theta):
        return -np.mean(log_density(x, theta[0] * s, np.exp(theta[1]) * s, np.exp(theta[2]) * s))

    best = None
    for mu_s, sg, tu in [(m - tau0, sigma0, tau0), (np.median(x) - 0.3 * s, 0.3 * s, 0.6 * s), (np.percentile(x, 10), 0.1 * s, 0.9 * s)]:
        r = minimize(nll, [mu_s / s, np.log(sg / s), np.log(tu / s)], method="Nelder-Mead",
                     options=dict(maxiter=3000, xatol=1e-5, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    return best.x[0] * s, np.exp(best.x[1]) * s, np.exp(best.x[2]) * s
