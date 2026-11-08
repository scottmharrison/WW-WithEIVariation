"""
18_mechanism_atlas.py

Purpose
    What each process in the extended Wong-Wang model does to the RT distribution, on its own: each is
    moved through a range of values, all else held, from three calibrated starting points, and the
    resulting distributions are summarised. Also, for every contrast, the chosen combination of stage 15
    taken apart: each of its mechanisms applied alone at its fitted value, to show how much of the fitted
    change each one carries.

    Processes (14): skill (drive spread), practice (stored instances), common drive, evidence, E/I control
    toward excitation (ratio 1.05) and toward inhibition (ratio 1.3), circuit threshold, instance-route
    threshold, urgency, lapse, non-decision time, retrieval accuracy (instance-route bias), and two that
    differ between StarCraft 2 leagues in WW-CarefulFitting's fits but are not among stage 15's
    mechanisms: circuit strength (W_self and W_inhib scaled together) and retrieval strength (rho_max
    scaled up and s_W, the retrieval onset scale, scaled down by the same factor).

    Starting points: the calibrated condition A (4_output/acc_mixtures/) of P9 (flanker; instance route
    inactive), A2 (lexical decision, speed instruction) and S1 (StarCraft 2 Bronze; instance route
    active).

    Summaries per simulated condition (correct RTs unless stated): percentiles 1-99, a density on 60
    log-spaced bins, accuracy, median error RT, ex-Gaussian mu, sigma, tau (model/exgaussian.py), skew
    (third standardised moment), Hill tail index of the slowest 5% (Hill, 1975, Ann Stat 3:1163; larger
    = lighter tail), share of trials won by the instance route, share with no commitment by 10 s.

Inputs
    4_output/acc_mixtures/<contrast>.json, 4_output/mech_selection/<contrast>.json, 4_output/bsi_table.csv

Outputs
    4_output/atlas/sweeps.json, 4_output/atlas/decomposition.json

Depends on
    15_fit_mechanism_selection.py, model/wwsim_ext.py, model/exgaussian.py

Seed
    SEED below for every simulation (common random numbers across levels and processes).

Last substantive change
    2026-11-07 — first version.

Run from the project root.
"""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import exgaussian  # noqa: E402
import wwsim_ext  # noqa: E402

spec = importlib.util.spec_from_file_location("ms", Path(__file__).parent / "15_fit_mechanism_selection.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

SEED = 20261107
N, T_MAX = 10000, 10000
BASES = {"P9": "flanker (ANT)", "A2": "lexical decision, speed", "S1": "StarCraft 2, Bronze"}
DENSITY_EDGES = np.geomspace(100, 12000, 61)


def circuit(th, f):
    return dict(th, W_self=th["W_self"] * f, W_inhib=th["W_inhib"] * f)


def retrieval_strength(th, f):
    return dict(th, rho_max=th["rho_max"] * f, s_W=th["s_W"] / f)


# Each process: (label, list of (level label, function of theta)). Levels run from less to more of it.
PROCESSES = {
    "skill": ("skill: drive spread u0_sd (lower = more skill)", [(f"x{v}", lambda th, v=v: dict(th, u0_sd=th["u0_sd"] * v)) for v in [2.0, 1.5, 1.0, 0.6, 0.35, 0.2]]),
    "practice": ("practice: stored instances logN", [(f"{v:+}", lambda th, v=v: dict(th, logN=th["logN"] + v)) for v in [-2, -1, 0, 1, 2, 4]]),
    "drive": ("common drive u0 (u0_sd scaled with it)", [(f"x{v}", lambda th, v=v: dict(th, u0=th["u0"] * v, u0_sd=th["u0_sd"] * v)) for v in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]),
    "evidence": ("evidence: pool difference e", [(f"+{v} Hz", lambda th, v=v: dict(th, e=th.get("e", 0.0) + v)) for v in [0, 2, 4, 6, 8]]),
    "control_exc": ("E/I control, toward excitation (R 1.05)", [(f"S {v}", lambda th, v=v: dict(th, bsi_S=v, bsi_R=1.05)) for v in [0.0, 0.25, 0.5, 0.75, 1.0]]),
    "control_inh": ("E/I control, toward inhibition (R 1.3)", [(f"S {v}", lambda th, v=v: dict(th, bsi_S=v, bsi_R=1.3)) for v in [0.0, 0.15, 0.3, 0.45, 0.6]]),
    "threshold_ww": ("circuit threshold (commitment rate)", [(f"{v} Hz", lambda th, v=v: dict(th, thr=v)) for v in [12, 15, 18, 22, 26, 30]]),
    "threshold_lca": ("instance-route threshold", [(f"x{v}", lambda th, v=v: dict(th, lca_z=v)) for v in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]]),
    "urgency": ("urgency (input growing in the trial)", [(f"{v} nA/s", lambda th, v=v: dict(th, urg=v)) for v in [-0.01, 0.0, 0.01, 0.03, 0.06]]),
    "lapse": ("lapse (fraction of disengaged trials, gain 0.5)", [(f"{v}", lambda th, v=v: dict(th, p_lapse=v, gain_lapse=0.5)) for v in [0.0, 0.03, 0.06, 0.1, 0.15]]),
    "ndt": ("non-decision time ter_mu", [(f"{v:+} ms", lambda th, v=v: dict(th, ter_mu=th["ter_mu"] + v)) for v in [-60, -30, 0, 30, 60, 120]]),
    "retrieval_acc": ("retrieval accuracy (instance-route bias)", [(f"{v}", lambda th, v=v: dict(th, lca_bias=v)) for v in [0.5, 0.7, 0.9, 0.97, 1.0]]),
    "circuit": ("circuit strength (W_self, W_inhib scaled)", [(f"x{v}", lambda th, v=v: circuit(th, v)) for v in [0.9, 0.95, 1.0, 1.05, 1.1, 1.18]]),
    "retrieval_str": ("retrieval strength (rho_max up, s_W down)", [(f"x{v}", lambda th, v=v: retrieval_strength(th, v)) for v in [0.5, 0.75, 1.0, 1.5, 2.5, 4.4]]),
}


def hill(x, frac=0.05):
    x = np.sort(x)[::-1]
    k = max(int(frac * x.size), 10)
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def summarise(th):
    sim = wwsim_ext.simulate_condition(th, N, T_MAX, SEED)
    rt, ch, route = sim["rt"], sim["choice"], sim["route"]
    c = rt[ch == 1]
    decided = ch >= 0
    out = dict(n_correct=int(c.size), no_commit=float(1 - decided.mean()),
               accuracy=float(np.mean(ch[decided] == 1)) if decided.any() else float("nan"),
               instance_share=float(np.mean(route[decided] == 1)) if decided.any() else float("nan"))
    if c.size < 100:
        return out
    out["q"] = np.percentile(c, np.arange(1, 100)).round(2).tolist()
    out["density"] = (np.histogram(c, DENSITY_EDGES)[0] / c.size / np.diff(DENSITY_EDGES) * 1000).tolist()  # per second
    err = rt[ch == 0]
    out["error_median"] = float(np.median(err)) if err.size >= 30 else float("nan")
    mu, sigma, tau = exgaussian.fit(c / 1000.0)
    out.update(mu=mu * 1000, sigma=sigma * 1000, tau=tau * 1000)
    z = (c - c.mean()) / c.std()
    out.update(skew=float(np.mean(z ** 3)), hill=hill(c))
    return out


def sweep_task(item):
    base, proc, i = item
    th0 = json.loads(Path(f"4_output/acc_mixtures/{base}.json").read_text())["theta_A"]
    label, fn = PROCESSES[proc][1][i]
    return base, proc, i, label, summarise(fn(th0))


def decomp_task(item):
    cid, part, ch = item
    th0 = json.loads(Path(f"4_output/acc_mixtures/{cid}.json").read_text())["theta_A"]
    return cid, part, summarise(ms.apply(th0, ch) if ch else th0)


if __name__ == "__main__":
    print("seed", SEED)
    Path("4_output/atlas").mkdir(parents=True, exist_ok=True)
    tasks = [(b, p, i) for b in BASES for p in PROCESSES for i in range(len(PROCESSES[p][1]))]
    with Pool() as pool:
        res = pool.map(sweep_task, tasks)
    sweeps = {b: {p: dict(label=PROCESSES[p][0], levels=[]) for p in PROCESSES} for b in BASES}
    for b, p, i, lab, s in sorted(res, key=lambda r: (r[0], r[1], r[2])):
        sweeps[b][p]["levels"].append(dict(level=lab, **s))
    Path("4_output/atlas/sweeps.json").write_text(json.dumps(dict(bases=BASES, density_edges=DENSITY_EDGES.tolist(), sweeps=sweeps), default=float))
    # Decomposition of each contrast's chosen combination.
    dtasks = []
    for f in sorted(Path("4_output/mech_selection").glob("*.json")):
        r = json.loads(f.read_text())
        ch = r["fits"][r["selected"]]["changes"]
        mechs = [] if r["selected"] == "SP" else r["selected"].split("+")[1:]
        dtasks += [(r["contrast"], "A", None), (r["contrast"], "combined", ch)]
        dtasks.append((r["contrast"], "skill + practice", {k: ch[k] for k in ("s", "p")}))
        for m in mechs:
            dtasks.append((r["contrast"], m, {k: ch[k] for k in ms.MECHS[m]}))
    with Pool() as pool:
        dres = pool.map(decomp_task, dtasks)
    decomp = {}
    for cid, part, s in dres:
        decomp.setdefault(cid, {})[part] = s
    Path("4_output/atlas/decomposition.json").write_text(json.dumps(decomp, default=float))
    print(f"wrote 4_output/atlas/sweeps.json ({len(res)} conditions) and decomposition.json ({len(dres)} conditions)")
