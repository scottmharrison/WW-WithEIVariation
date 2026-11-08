"""Process check for 20_fit_starcraft_lattice.py. Seed: S1's stage-12 seed (simulations).
(a) Absolute percentiles of WW-CarefulFitting's Bronze (sc2__21) and Grandmaster (sc2__27) parameters against
    S1's observed percentiles (raw), to see whether the two sets of fits describe the same data.
(b) Joint fit, lattice-aware: skill + practice + circuit + retrieval strength together, from no change and from
    WW-CarefulFitting's ratios. Forward selection adds one mechanism at a time and cannot make this joint move."""
import csv, importlib.util
from multiprocessing import Pool
import numpy as np

spec = importlib.util.spec_from_file_location("lf", "3_analyses/20_fit_starcraft_lattice.py")
lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)


def joint(task):
    return lf.fit(task)


if __name__ == "__main__":
    lf.setup(False)
    conds = {r["cond_id"]: r for r in csv.DictReader(open("1_raw/ww_conditions.csv"))}
    keys = ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]
    prof = [r for r in csv.DictReader(open("4_output/s1_lattice_profile.csv")) if r["percentiles"] == "raw"]
    print("percentiles", lf.PCTS)
    for cid, col, name in [("sc2__21", "value_A", "Bronze"), ("sc2__27", "value_B", "Grandmaster")]:
        th = dict({k: float(conds[cid][k]) for k in keys}, e=0.0)
        print(name, "CarefulFitting parameters:", lf.simulate_percentiles(th, lf.SEED).round(0).tolist())
        print(name, "observed S1 games        :", [round(float(r[col])) for r in prof], flush=True)
    pars = ["s", "p", "w", "rr", "ro"]
    cf = dict(s=0.0, p=0.0, w=float(np.log(1.17)), rr=float(np.log(4.4 * 24.0 / 9.1)), ro=float(np.log(1 / 0.17)))
    with Pool(2) as pool:
        res = pool.map(joint, [(True, pars, [{}]), (True, pars, [cf])])
    for name, r in zip(["from no change", "from CarefulFitting ratios"], res):
        print("(b)", name, {k: round(v, 3) for k, v in r["changes"].items()}, "chi2", r["chi2"], "df", r["df"], "p", r["p"], "aic", r["aic"], "at bound", r["at_bound"], flush=True)
