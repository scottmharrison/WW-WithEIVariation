"""
16_fit_shared_combinations.py

Purpose
    Does any one combination of mechanisms account for all 16 contrasts? Stage 15 chose a combination
    for each contrast separately. Here every combination chosen for any contrast, and the union of all
    of them, is fitted to every contrast, so each combination is judged on all 16 at once: how many it
    fits adequately, and its total AIC over contrasts (a lower total is a better balance of fit and
    number of changes across the datasets).

    Each fit starts from the best fit in that contrast's stage-15 file whose mechanisms are a subset of
    the combination (the remaining mechanisms at their neutral values), plus the ceiling's solution
    restricted to the combination; fitting rules as in stage 15.

Inputs
    4_output/mech_selection/<contrast>.json (and the inputs of stage 15)

Outputs
    4_output/shared_combinations/<contrast>.json — one fit per combination

Depends on
    15_fit_mechanism_selection.py

Seed
    Each contrast's own seed (as stage 12).

Last substantive change
    2026-11-07 — first version.

Run from the project root. A contrast whose file exists is skipped.
"""
import csv
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("ms", Path(__file__).parent / "15_fit_mechanism_selection.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)
OUT = Path("4_output/shared_combinations")


def mechs_of(name):
    return frozenset() if name == "SP" else frozenset(name.split("+")[1:])


def combinations():
    """Every combination selected in stage 15, and their union."""
    files = sorted(ms.OUT.glob("*.json"))
    chosen = {mechs_of(json.loads(f.read_text())["selected"]) for f in files}
    union = frozenset().union(*chosen)
    combos = sorted(chosen | {union}, key=lambda c: (len(c), sorted(c)))
    return [sorted(c, key=list(ms.MECHS).index) for c in combos]


def run(item):
    cid, combos = item
    path = OUT / f"{cid}.json"
    if path.exists():
        return cid, "skipped"
    prev = json.loads((ms.OUT / f"{cid}.json").read_text())
    # Re-use stage 15's machinery for this contrast by running a one-off fit function built the same way.
    results = {}
    for combo in combos:
        name = "SP" + "".join("+" + m for m in combo)
        if name in prev["fits"]:
            results[name] = prev["fits"][name]
            continue
        subsets = [f for n, f in prev["fits"].items() if n != "SP+all" and mechs_of(n) <= set(combo)]
        start = min(subsets, key=lambda f: f["aic"])["changes"]
        ceiling = {k: v for k, v in prev["fits"]["SP+all"]["changes"].items()
                   if k in ["s", "p"] + [q for m in combo for q in ms.MECHS[m]]}
        results[name] = ms.fit_one(cid, ["s", "p"] + [q for m in combo for q in ms.MECHS[m]], [start, ceiling])
    path.write_text(json.dumps(dict(contrast=cid, fits=results), indent=1, default=float))
    return cid, "fitted"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    combos = combinations()
    print("combinations:", ["SP" + "".join("+" + m for m in c) for c in combos])
    cids = [r["contrast"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))]
    with Pool(min(len(cids), os.cpu_count())) as pool:
        for cid, status in pool.imap_unordered(run, [(c, combos) for c in cids]):
            print(cid, status, flush=True)
