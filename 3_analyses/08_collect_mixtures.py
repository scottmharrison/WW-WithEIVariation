"""
08_collect_mixtures.py

Purpose
    Collect the model fits of 07_fit_mixtures.py into tables, classify each contrast, and simulate the
    effect of each ingredient alone (one step of each change) from every calibrated A, for the figure
    that shows what each ingredient does to the distribution.

    Per contrast:
        SP adequate?     the skill + practice fit has p > .05
        needed           where SP is adequate: "either" if skill alone and practice alone are each
                         adequate; "skill" or "practice" if only that one is; "both" if only SP is
        best addition    where SP is not adequate: the SP+X model with the lowest AIC among adequate
                         ones; "none adequate" if no SP+X is adequate
    AIC = chi-square + 2k, with k the number of changed parameters (Akaike, 1974); a lower value is a
    better balance of fit and number of changes.

Inputs
    4_output/mixtures/<contrast>.json, 2_data_processing/contrasts.csv

Outputs
    4_output/mixture_fits.csv — one row per contrast and model (dictionary beside it)
    4_output/mixture_summary.csv — one row per contrast (dictionary beside it)
    4_output/mechanism_steps.csv — one row per contrast, ingredient and percentile (dictionary beside it)

Depends on
    07_fit_mixtures.py

Last substantive change
    2026-11-06 — first version.

Run from the project root.
"""
import csv
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("fm", Path(__file__).parent / "07_fit_mixtures.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

# One step of each ingredient, in the direction usually meant by "improvement" or the named change.
STEPS = {"skill (u0_sd x 0.5)": {"s": np.log(0.5)}, "practice (logN + 2)": {"p": 2.0},
         "drive level (u0, u0_sd x 1.3)": {"d": np.log(1.3)}, "non-decision time (-50 ms)": {"t": -50.0},
         "retrieval speed (s_W x 0.5)": {"r": np.log(0.5)}}


def steps(item):
    cid, theta, seed = item
    qa = fm.percentiles(theta, seed)
    return cid, {name: (100 * (fm.percentiles(fm.apply(theta, ch), seed) / qa - 1)).tolist() for name, ch in STEPS.items()}


if __name__ == "__main__":
    cids = [r["contrast"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))]
    axis = {r["contrast"]: r["axis"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))}
    res = {c: json.loads((fm.OUT / f"{c}.json").read_text()) for c in cids}

    fit_rows, summary = [], []
    for c in cids:
        r = res[c]
        for name, fit in r["fits"].items():
            k = len(fm.MODELS[name])
            fit_rows.append(dict(contrast=c, model=name, k=k, **{f"change_{n}": round(fit["changes"].get(n, 0.0), 4) for n in "spdtr"},
                                 chi2=fit["chi2"], df=fit["df"], p=round(fit["p"], 5), aic=round(fit["chi2"] + 2 * k, 3),
                                 rms_pct=fit["rms_pct"], adequate=fit["p"] > 0.05, at_bound=";".join(fit["at_bound"])))
        ok = {m: f["p"] > 0.05 for m, f in r["fits"].items()}
        if ok["SP"]:
            needed = "either" if ok["S"] and ok["P"] else "skill" if ok["S"] else "practice" if ok["P"] else "both"
            addition = ""
        else:
            needed = "not adequate"
            adds = [m for m in ("SP+drive", "SP+ndt", "SP+retrieval") if ok[m]]
            addition = min(adds, key=lambda m: r["fits"][m]["chi2"] + 2 * len(fm.MODELS[m])) if adds else "none adequate"
        best = min(r["fits"], key=lambda m: r["fits"][m]["chi2"] + 2 * len(fm.MODELS[m]))
        n_units = len({row["unit"] for row in csv.DictReader(open("2_data_processing/contrast_rts.csv")) if row["contrast"] == c}) \
            if c in ("P1", "A1", "D1") else ""
        summary.append(dict(contrast=c, axis=axis[c], calibration_rms_log=round(r["calibration_rms_log"], 4),
                            two_additions_adequate=ok["SP+drive+ndt"], few_units=n_units,
                            instance_share_A=round(r["instance_share_A"], 4), sp_adequate=ok["SP"], sp_p=round(r["fits"]["SP"]["p"], 5),
                            needed=needed, best_addition=addition, lowest_aic_model=best,
                            sp_change_s=round(r["fits"]["SP"]["changes"]["s"], 4), sp_change_p=round(r["fits"]["SP"]["changes"]["p"], 4)))
        print(c, needed, addition, "lowest AIC:", best)

    for path, rows in [("4_output/mixture_fits.csv", fit_rows), ("4_output/mixture_summary.csv", summary)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    with Pool() as pool:
        out = pool.map(steps, [(c, res[c]["theta_A"], res[c]["seed"]) for c in cids])
    with open("4_output/mechanism_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "ingredient", "percentile", "change_pct"])
        for c, d in out:
            for name, vals in d.items():
                for p, v in zip(fm.PCTS, vals):
                    w.writerow([c, name, p, round(v, 3)])
    print("wrote 4_output/mixture_fits.csv, mixture_summary.csv, mechanism_steps.csv")
