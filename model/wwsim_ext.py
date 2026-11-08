"""
wwsim_ext.py

Purpose
    The extended Wong-Wang + roaming + instance-LCA simulator of WW-RTShapeAnatomy/model/wwsim.py
    (sha256 714df22b...), with three additions, so that the model predicts choices as well as RTs and
    can carry a top-down control signal:

    1. Evidence. Pool 1 receives drive u0 + e/2 and pool 2 u0 - e/2 (Hz; e = 0 is the original model,
       where both pools receive the same drive). A trial is correct when pool 1 commits first, or, on
       the instance route, when retrieval unit 1 does (that route is biased towards the correct answer
       by BIAS = 0.9, unchanged).
    2. Control signal (balanced synaptic input; Lo, Wang & Wang, 2015). Strength S and ratio R change
       the transfer function of both selective pools, through the table of 3_analyses/10_build_bsi_table.py:
       a = 270 x gain(S,R), b = a x (108/270 + shift(S,R)), d = 0.154 x curvature(S,R). Values between
       grid points are interpolated bilinearly. At S = 0 the transfer function is the original.
    3. Choice is returned for every trial (1 = correct, 0 = error, -1 = no commitment).

    Everything else is copied from wwsim.py unchanged: constants, the Euler-Maruyama dynamics at
    dt = 1 ms, roaming of u0, the instance route and the non-decision time.

Inputs
    4_output/bsi_table.csv (read once, on first use of the control signal)

Outputs
    None (functions only).

Depends on
    3_analyses/10_build_bsi_table.py

Last substantive change
    2026-11-07 — added free thresholds (circuit and instance route), urgency and lapses.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

A_PHI, B_PHI, D_PHI = 270.0, 108.0, 0.154
GAMMA, TAU_S = 0.641, 100.0
I_BG, J_EXT = 0.3255, 0.00052
NOISE_SD, NOISE_TAU, THR = 0.02, 2.0, 15.0
BIAS, N_HALF, C_W = 0.9, 200.0, 2.0
LCA_K, LCA_BETA, LCA_TAU, LCA_Z, LCA_DEADLINE = 0.2, 0.3, 20.0, 1.0, 2000.0
U0_LO, U0_HI = 0.5, 60.0
NDT_FLOOR = 38.0

TABLE_PATH = Path(__file__).resolve().parents[1] / "4_output" / "bsi_table.csv"
_TABLE = None


def bsi_params(S, R):
    """(a, b, d) of the transfer function under control strength S and ratio R (bilinear in the table)."""
    global _TABLE
    if S <= 0:
        return A_PHI, B_PHI, D_PHI
    if _TABLE is None:
        rows = list(csv.DictReader(open(TABLE_PATH)))
        Ss = sorted({float(r["S"]) for r in rows}); Rs = sorted({float(r["R"]) for r in rows})
        grid = {k: np.zeros((len(Ss), len(Rs))) for k in ("gain", "shift_nA", "curvature")}
        for r in rows:
            i, j = Ss.index(float(r["S"])), Rs.index(float(r["R"]))
            for k in grid:
                grid[k][i, j] = float(r[k])
        _TABLE = (np.array(Ss), np.array(Rs), grid)
    Ss, Rs, grid = _TABLE
    S = float(np.clip(S, Ss[0], Ss[-1])); R = float(np.clip(R, Rs[0], Rs[-1]))
    i = min(np.searchsorted(Ss, S, side="right") - 1, len(Ss) - 2); j = min(np.searchsorted(Rs, R, side="right") - 1, len(Rs) - 2)
    ts, tr = (S - Ss[i]) / (Ss[i + 1] - Ss[i]), (R - Rs[j]) / (Rs[j + 1] - Rs[j])

    def interp(m):
        return (m[i, j] * (1 - ts) * (1 - tr) + m[i + 1, j] * ts * (1 - tr) + m[i, j + 1] * (1 - ts) * tr + m[i + 1, j + 1] * ts * tr)

    a = A_PHI * interp(grid["gain"])
    b = a * (B_PHI / A_PHI + interp(grid["shift_nA"]))
    d = D_PHI * interp(grid["curvature"])
    return a, b, d


def phi(I, a=A_PHI, b=B_PHI, d=D_PHI):
    """Abbott-Chance f-I curve: rate (Hz) for input current I (nA)."""
    x = a * np.asarray(I, float) - b
    small = np.abs(x) < 1e-8
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 / d, xs / (1.0 - np.exp(-d * xs)))


def simulate_ww(u0, e, ws, wl, t_max, rng, abd, thr=THR, urg=0.0):
    """Decision time (ms; inf = none) and winning pool (1 or 2; 0 = none) of the circuit route.
    e: evidence (Hz), one value or one per trial. thr: commitment rate (Hz). urg: urgency, a current
    (nA per second of the trial) added to both pools, so the circuit is pushed harder the longer the
    trial lasts (Churchland, Kiani & Shadlen, 2008, Nat Neurosci 11:693; Lo, Wang & Wang, 2015)."""
    u0 = np.asarray(u0, float)
    n = u0.size
    T = np.full(n, np.inf); W = np.zeros(n, dtype=np.int8)
    idx = np.arange(n)
    S1, S2, n1, n2 = (np.zeros(n) for _ in range(4))
    e = np.broadcast_to(np.asarray(e, float), u0.shape)
    d1, d2 = J_EXT * (u0 + e / 2), J_EXT * (u0 - e / 2)
    dec, sc = 1.0 - 1.0 / NOISE_TAU, np.sqrt(1.0 / NOISE_TAU)
    for t in range(1, int(t_max) + 1):
        g = rng.standard_normal((2, idx.size))
        n1 = n1 * dec + NOISE_SD * sc * g[0]
        n2 = n2 * dec + NOISE_SD * sc * g[1]
        u_t = urg * t / 1000.0
        r1 = phi(I_BG + d1 + u_t + ws * S1 + wl * S2 + n1, *abd)
        r2 = phi(I_BG + d2 + u_t + ws * S2 + wl * S1 + n2, *abd)
        S1 = S1 + (-S1 / TAU_S + GAMMA * (1 - S1) * r1 / 1000.0)
        S2 = S2 + (-S2 / TAU_S + GAMMA * (1 - S2) * r2 / 1000.0)
        done = np.maximum(r1, r2) >= thr
        if done.any():
            T[idx[done]] = t
            W[idx[done]] = np.where(r1[done] >= r2[done], 1, 2)
            keep = ~done
            idx, S1, S2, n1, n2, d1, d2 = idx[keep], S1[keep], S2[keep], n1[keep], n2[keep], d1[keep], d2[keep]
            if idx.size == 0:
                break
    return T, W


def simulate_lca(rho1, rho2, onset, sigL, t_max, rng, z=LCA_Z):
    """Decision time (ms; inf = none) and winning unit (1, 2; 0 = none) of the instance route; z is its
    threshold."""
    n = onset.size
    T = np.full(n, np.inf); W = np.zeros(n, dtype=np.int8)
    t0 = np.maximum(np.ceil(onset), 1.0)
    idx = np.where(np.isfinite(onset) & (t0 <= t_max))[0]
    x1, x2 = np.zeros(idx.size), np.zeros(idx.size)
    r1, r2, s0, on = rho1[idx], rho2[idx], t0[idx], onset[idx]
    a, sq = 1.0 / LCA_TAU, np.sqrt(1.0 / LCA_TAU)
    for t in range(int(s0.min()) if idx.size else 1, int(t_max) + 1):
        if idx.size == 0:
            break
        live = s0 <= t
        g = rng.standard_normal((2, idx.size))
        nx1 = x1 + a * (r1 - LCA_K * x1 - LCA_BETA * x2) + sigL * sq * g[0]
        nx2 = x2 + a * (r2 - LCA_K * x2 - LCA_BETA * x1) + sigL * sq * g[1]
        x1 = np.where(live, np.maximum(nx1, 0.0), x1)
        x2 = np.where(live, np.maximum(nx2, 0.0), x2)
        hit = live & (np.maximum(x1, x2) >= z)
        out = live & ~hit & (t >= on + LCA_DEADLINE)
        T[idx[hit]] = t
        W[idx[hit]] = np.where(x1[hit] >= x2[hit], 1, 2)
        drop = hit | out
        if drop.any():
            k = ~drop
            idx, x1, x2, r1, r2, s0, on = idx[k], x1[k], x2[k], r1[k], r2[k], s0[k], on[k]
    return T, W


def simulate_condition(th, n, t_max, seed, roam=True, lca=True):
    """n trials of one condition. th as in wwsim.py, plus optional 'e' (evidence, Hz), 'bsi_S', 'bsi_R',
    'lca_bias' (instance route's input bias towards the correct answer; default BIAS), 'thr' (circuit
    commitment rate, Hz; default THR), 'lca_z' (instance route threshold; default LCA_Z), 'urg' (urgency,
    nA/s; default 0), 'p_lapse' and 'gain_lapse' (lapses: on a fraction p_lapse of trials, drawn
    independently, the drive is u0 x gain_lapse and there is no evidence, so the choice is at chance;
    after WW-NSGA2's lapse, whose noise scaling is not used here). A setting at its default draws no
    extra random numbers, so results at the defaults are identical to before.
    Returns per-trial rt (ms; inf = none), choice (1 correct, 0 error, -1 none), route (1 instance)."""
    rng = np.random.default_rng(seed)
    u0 = np.full(n, th['u0'])
    if roam:
        u0 = np.clip(th['u0'] + th['u0_sd'] * rng.standard_normal(n), U0_LO, U0_HI)
    e = np.full(n, th.get('e', 0.0))
    if th.get('p_lapse', 0.0) > 0:
        lapse = rng.random(n) < th['p_lapse']
        u0 = np.where(lapse, u0 * th.get('gain_lapse', 0.5), u0)
        e = np.where(lapse, 0.0, e)
    if lca:
        N = np.exp(th['logN'] + th['N_logsd'] * rng.standard_normal(n))
        rho_m = th['rho_max'] * N / (N + N_HALF)
        g = np.clip(u0 / th['u_star'], 0, 3)
        bias = th.get('lca_bias', BIAS)
        rho1, rho2 = g * rho_m * (1 + bias) / 2, g * rho_m * (1 - bias) / 2
        onset = th['s_W'] * N ** (-1.0 / C_W) * rng.weibull(C_W, n)
        T_lca, W_lca = simulate_lca(rho1, rho2, onset, th['sigL'], t_max, rng, th.get('lca_z', LCA_Z))
    else:
        T_lca, W_lca = np.full(n, np.inf), np.zeros(n, dtype=np.int8)
    abd = bsi_params(th.get('bsi_S', 0.0), th.get('bsi_R', 1.2))
    T_ww, W_ww = simulate_ww(u0, e, th['W_self'], th['W_inhib'], t_max, rng, abd, th.get('thr', THR), th.get('urg', 0.0))
    a = (0.0 - th['ter_mu']) / th['ter_sd']
    ndt = NDT_FLOOR + stats.truncnorm.rvs(a, np.inf, loc=th['ter_mu'], scale=th['ter_sd'], size=n, random_state=rng)
    inst = T_lca < T_ww
    T_dec = np.minimum(T_ww, T_lca)
    winner = np.where(inst, W_lca, W_ww)
    choice = np.where(~np.isfinite(T_dec), -1, np.where(winner == 1, 1, 0))
    return dict(rt=T_dec + ndt, choice=choice, route=inst.astype(np.int8), u0=u0)
