"""
13_collect_accuracy_mixtures.py

Purpose
    Collect the accuracy-aware fits of 12_fit_accuracy_mixtures.py into tables, classify each contrast,
    and simulate one step of each ingredient from every calibrated A, giving its effect on accuracy and
    on the median and 99th percentile of correct RTs (the speed-accuracy signature of each ingredient).

    Per contrast: whether skill + practice is adequate (p > .05); where it is not, the single addition
    (drive, evidence, control, ndt) with the lowest AIC among adequate ones; and the lowest-AIC model
    overall. AIC = chi-square + 2k.

Inputs
    4_output/acc_mixtures/<contrast>.json, 2_data_processing/contrasts.csv

Outputs
    4_output/acc_mixture_fits.csv, acc_mixture_summary.csv, acc_ingredient_steps.csv (dictionaries
    beside them)

Depends on
    12_fit_accuracy_mixtures.py

Seed
    Each contrast's own seed, from its file.

Last substantive change
    2026-11-07 — first version.

Run from the project root.
"""
import csv
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("fa", Path(__file__).parent / "12_fit_accuracy_mixtures.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)

STEPS = {"skill (u0_sd x 0.5)": {"s": np.log(0.5)}, "practice (logN + 2)": {"p": 2.0},
         "drive (u0, u0_sd x 1.3)": {"d": np.log(1.3)}, "evidence (+4 Hz)": {"e": 4.0},
         "control, excitatory (S 0.5, R 1.05)": {"cS": 0.5, "cR": 1.05},
         "control, inhibitory (S 0.5, R 1.3)": {"cS": 0.5, "cR": 1.3}, "non-decision (-50 ms)": {"t": -50.0}}
SINGLE_ADDITIONS = ["SP+drive", "SP+evidence", "SP+control", "SP+ndt"]


def steps(item):
    cid, theta, seed = item
    qa, aa, _ = fa.summarise(theta, seed)
    out = {}
    for name, ch in STEPS.items():
        qb, ab, _ = fa.summarise(fa.apply(theta, ch), seed)
        out[name] = (100 * (qb[3] / qa[3] - 1), 100 * (qb[6] / qa[6] - 1), ab - aa)
    return cid, out


if __name__ == "__main__":
    contrasts = list(csv.DictReader(open("2_data_processing/contrasts.csv")))
    res = {c["contrast"]: json.loads((fa.OUT / f"{c['contrast']}.json").read_text()) for c in contrasts}
    fit_rows, summ = [], []
    for c in contrasts:
        cid, r = c["contrast"], res[c["contrast"]]
        aic = {}
        for m, f in r["fits"].items():
            k = len(fa.MODELS[m]); aic[m] = f["chi2"] + 2 * k
            fit_rows.append(dict(contrast=cid, model=m, k=k, **{f"change_{n}": round(f["changes"].get(n, 0.0), 4) for n in ["s", "p", "d", "e", "cS", "cR", "t"]},
                                 chi2=f["chi2"], n_terms=f["n_terms"], df=f["df"], p=round(f["p"], 5), aic=round(aic[m], 3),
                                 adequate=f["p"] > 0.05, predicted_accuracy_change=f["predicted_accuracy_change"],
                                 at_bound=";".join(f["at_bound"])))
        ok = {m: f["p"] > 0.05 for m, f in r["fits"].items()}
        if ok["SP"]:
            verdict = "either" if ok["S"] and ok["P"] else "skill" if ok["S"] else "practice" if ok["P"] else "both"
            addition = ""
        else:
            verdict = "not adequate"
            adds = [m for m in SINGLE_ADDITIONS if ok[m]]
            addition = min(adds, key=aic.get) if adds else "none adequate"
        summ.append(dict(contrast=cid, axis=c["axis"], has_accuracy=r["has_accuracy"], n_error_terms=r["n_error_terms"],
                         calibration_rms_log=round(r["calibration"]["rms_log"], 4),
                         accuracy_A_observed=r["observed"]["accuracy_A"], accuracy_A_model=round(r["calibration"]["accuracy_A_model"], 4),
                         sp_adequate=ok["SP"], needed=verdict, best_single_addition=addition,
                         lowest_aic_model=min(aic, key=aic.get),
                         adequate_models=";".join(m for m in fa.MODELS if ok[m])))
        print(cid, verdict, addition, "lowest AIC:", min(aic, key=aic.get))
    for path, rows in [("4_output/acc_mixture_fits.csv", fit_rows), ("4_output/acc_mixture_summary.csv", summ)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with Pool() as pool:
        out = pool.map(steps, [(cid, res[cid]["theta_A"], res[cid]["seed"]) for cid in res])
    with open("4_output/acc_ingredient_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "ingredient", "median_change_pct", "q99_change_pct", "accuracy_change"])
        for cid, d in out:
            for name, (m, q99, a) in d.items():
                w.writerow([cid, name, round(m, 3), round(q99, 3), round(a, 5)])
    print("wrote 4_output/acc_mixture_fits.csv, acc_mixture_summary.csv, acc_ingredient_steps.csv")
