"""
check_02_condition_balance.py (process code; not run by the pipeline)

Purpose
    The time-on-task contrasts pool conditions that are randomised within blocks. Check that each
    such condition is about equally common in the first and last quarter of each subject's trials,
    reproducing the quarter split of 2_data_processing/01_build_contrasts.py.

Inputs
    1_raw/<key>.csv

Outputs
    Printed text; saved by the caller to process/output/check_02.txt

Depends on
    2_data_processing/01_build_contrasts.py

Last substantive change
    2026-11-06 — first version.
"""
import csv
from collections import Counter, defaultdict

SPEC = {  # dataset: (column pooled, order columns, cell column or None, test phase only)
    "ratcliff1998": ("strength", ["session", "block", "trial"], "condition", False),
    "keuleers2012": ("condition", ["order"], None, False),
    "cochrane2023": ("condition", ["trial"], None, False),
    "wolfe2010": ("set_size", ["trial"], "search_type", True),
    "rouder2008": ("set_size", ["block", "trial"], None, False),
    "pratte2010": ("condition", ["block", "trial"], "task", False),
    "enkavi2019_threebytwo": ("switch_type", ["trial_num"], None, True),
}
for key, (col, order, cell, test_only) in SPEC.items():
    groups = defaultdict(list)
    for r in csv.DictReader(open(f"1_raw/{key}.csv", newline="")):
        if r["exclude_reason"] or r["correct"] not in ("", "1") or (test_only and r["practice"] == "True"):
            continue
        if key == "pratte2010" and r["stop_trial"] == "True":
            continue
        groups[(r["subjects"], r[cell] if cell else "all")].append(r)
    early, late = Counter(), Counter()
    for rs in groups.values():
        rs.sort(key=lambda r: tuple(float(r[c]) for c in order))
        q = len(rs) // 4
        early.update(r[col] for r in rs[:q]); late.update(r[col] for r in rs[len(rs) - q:])
    ne, nl = sum(early.values()), sum(late.values())
    worst = max(abs(early[v] / ne - late[v] / nl) for v in set(early) | set(late))
    print(f"{key}: {col}, largest difference in share between first and last quarter = {100 * worst:.1f} points "
          f"({len(set(early) | set(late))} values)")
