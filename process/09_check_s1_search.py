"""Process check for 20_fit_starcraft_lattice.py. Seed: S1's stage-12 seed (simulations).
(a) Did Nelder-Mead move? Objective on a grid of skill s and practice p around the reported lattice SP solution.
(b) WW-CarefulFitting's own Bronze (sc2__21) and Grandmaster (sc2__27) parameters, simulated, scored on the
    12-percentile chi-square, raw and lattice-aware.
The joint fit of circuit and retrieval strength is in 10_check_s1_joint.py."""
import csv, importlib.util, json
from multiprocessing import Pool
import numpy as np

spec = importlib.util.spec_from_file_location("lf", "3_analyses/20_fit_starcraft_lattice.py")
lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)


def grid_point(a):
    s, p = a
    lf.setup(True)
    return s, p, float(np.sum(((lf.predict(dict(s=s, p=p), lf.SEED) - lf.Q_OBS) / lf.Q_SE) ** 2))


def direct(lattice):
    lf.setup(lattice)
    conds = {r["cond_id"]: r for r in csv.DictReader(open("1_raw/ww_conditions.csv"))}
    keys = ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]
    a = {k: float(conds["sc2__21"][k]) for k in keys}; b = {k: float(conds["sc2__27"][k]) for k in keys}
    a["e"] = b["e"] = 0.0
    qa, qb = lf.simulate_percentiles(a, lf.SEED), lf.simulate_percentiles(b, lf.SEED)
    pred = 100 * (qb / qa - 1)
    return lattice, pred.round(1).tolist(), lf.Q_OBS.round(1).tolist(), round(float(np.sum(((pred - lf.Q_OBS) / lf.Q_SE) ** 2)), 1)


if __name__ == "__main__":
    d = json.load(open("4_output/s1_lattice/lattice_8.json"))["fits"]["SP"]
    print("(a) lattice SP changes, full digits:", d["changes"], "chi2 (with simulation SE)", d["chi2"])
    s0, p0 = d["changes"]["s"], d["changes"]["p"]
    with Pool(13) as pool:
        g = pool.map(grid_point, [(s0 + ds, p0 + dp) for ds in (-0.6, -0.3, -0.1, 0, 0.1, 0.3) for dp in (-0.6, -0.3, -0.1, 0, 0.1, 0.3)])
        print("    objective at reported solution:", [round(x[2], 1) for x in g if x[0] == s0 and x[1] == p0])
        print("    lowest 3 on grid:", [(round(s, 3), round(p, 3), round(o, 1)) for s, p, o in sorted(g, key=lambda x: x[2])[:3]])
        for r in pool.map(direct, [False, True]):
            print("(b) lattice" if r[0] else "(b) raw", "predicted", r[1], "\n    observed", r[2], "\n    chi2 (data SE only, 12 terms)", r[3])
