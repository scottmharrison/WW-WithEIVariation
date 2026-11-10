"""Process summary of stage 15's fits (no simulation, no random numbers): for each mechanism added alone to skill + practice,
how much of the chi-square it removes in each part of the data. Data SE only (the fit's objective).
Regions: fast edge = 5th-20th percentile of correct RT; body = 30th-70th; slow tail = 80th-99th; accuracy; error RTs.
Reads 4_output/mech_selection/<contrast>.json, 4_output/pooled_accuracy_profiles_fine.csv. Writes 4_output is not touched;
output goes to process/output/check_11.txt and process/output/mechanism_regions.csv."""
import csv, json, glob
from collections import defaultdict
import numpy as np
prof = list(csv.DictReader(open("4_output/pooled_accuracy_profiles_fine.csv")))
PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
REG = {"fast": [0, 1, 2], "body": [3, 4, 5, 6, 7], "tail": [8, 9, 10, 11]}
rows = []; sel = defaultdict(list); first = defaultdict(list)
for f in sorted(glob.glob("4_output/mech_selection/*.json")):
    d = json.load(open(f)); c = d["contrast"]
    P = [r for r in prof if r["contrast"] == c]
    qo = np.array([float(r["change"]) for r in P if r["measure"] == "correct_rt_pct"]); qs = np.maximum([float(r["se"]) for r in P if r["measure"] == "correct_rt_pct"], .25)
    ac = [r for r in P if r["measure"] == "accuracy"]; has = d["has_accuracy"]
    ao, asd = (float(ac[0]["change"]), max(float(ac[0]["se"]), .002)) if has else (None, None)
    er = [r for r in P if r["measure"] == "error_rt_pct"]; eo = np.array([float(r["change"]) for r in er]); es = np.maximum([float(r["se"]) for r in er], .5) if er else None
    def parts(fit):
        z2 = ((np.array(fit["predicted_correct_pct"]) - qo) / qs) ** 2
        o = {k: float(z2[v].sum()) for k, v in REG.items()}
        o["accuracy"] = ((fit["predicted_accuracy_change"] - ao) / asd) ** 2 if has else 0.0
        o["error"] = 0.0
        if len(eo):
            o["error"] = float((((np.array(fit["predicted_error_pct"]) - eo) / es) ** 2).sum()) if fit["predicted_error_pct"] is not None else 9.0 * len(eo)
        return o
    sp = parts(d["fits"]["SP"])
    mechs = [] if d["selected"] == "SP" else d["selected"].split("+")[1:]
    for m in mechs: sel[m].append(c)
    best = None
    for m in d["mechanisms"]:
        fit = d["fits"]["SP+" + m]
        drop = d["fits"]["SP"]["aic"] - fit["aic"]
        p = parts(fit)
        rows.append(dict(contrast=c, mechanism=m, aic_drop=round(drop, 2), **{"fix_" + k: round(sp[k] - p[k], 2) for k in sp}, **{"sp_" + k: round(sp[k], 2) for k in sp}))
        if best is None or drop > best[1]: best = (m, drop)
    if best[1] > 2: first[best[0]].append(c)
with open("process/output/mechanism_regions.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("contrasts:", sorted({r["contrast"] for r in rows}))
print("mechanism | single AIC drop>2 (n) | best single | selected in | median fix among drop>2: fast body tail acc err")
for m in json.load(open("4_output/mech_selection/P1.json"))["mechanisms"]:
    R = [r for r in rows if r["mechanism"] == m and r["aic_drop"] > 2]
    med = {k: round(float(np.median([r["fix_" + k] for r in R])), 1) if R else None for k in ("fast", "body", "tail", "accuracy", "error")}
    print(m, "|", len(R), [r["contrast"] for r in R], "| best", first[m], "| selected", sel[m], "|", med)
print("\nchosen combination by contrast")
for f in sorted(glob.glob("4_output/mech_selection/*.json")):
    d = json.load(open(f)); x = d["fits"][d["selected"]]
    print(d["contrast"], d["selected"], "p", round(x["p"], 4), {k: round(v, 3) for k, v in x["changes"].items()})
