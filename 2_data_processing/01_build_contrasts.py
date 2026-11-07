"""
01_build_contrasts.py

Purpose
    Turn the DataPrep tables into the contrasts that test the predictions of
    SKILL_vs_PRACTICE_explained.pdf, each one a comparison between a condition A and a condition B
    within each unit (a participant, a monkey, or for StarCraft 2 a league, whose games are the units):

    Axis "practice" (time on task). Each subject's trials, in the order they were run, are cut into
    quarters within each cell; A = first quarter, B = last quarter. Retrieval availability should rise
    with practice (Logan, 1988, Psychol Rev 95:492). A cell is a condition that alternates by block and
    changes RT a great deal (speed vs accuracy instructions; search type, run as separate files), so
    early and late trials are compared within it. Other conditions are randomised within blocks and are
    pooled.

    Axis "item practice". Word frequency: A = very low frequency words, B = high frequency words, within
    instruction. Frequent words have been met more often over a lifetime, which instance theory treats
    as more stored instances (Logan, 1988).

    Axis "skill" (expertise). A = the less expert group, B = the more expert group: StarCraft 2 games of
    Bronze against Grandmaster league; non-players against action video game players, on the first of
    four days only, so that training in the experiment is not counted.

    Axis "drive level" (stimulus strength). A = weak motion coherence, B = strong, within subject.

    Axis "instruction" (benchmark). A = speed instruction, B = accuracy instruction, within subject.

    Trials used: DataPrep's usable rows (exclude_reason empty, as dataprep.load() keeps), correct
    trials where accuracy is recorded, and only the test phase where a practice phase is marked. Pratte
    et al.'s stop trials are dropped, because their RT is a failure to stop, not a response to the go
    stimulus. No other trial is removed.

Inputs
    1_raw/<key>.csv for each key in 00_fetch_raw.py

Outputs
    (with --with-errors) 2_data_processing/contrast_trials_acc.csv — correct and error trials, with correct column
    2_data_processing/contrast_rts.csv — one row per trial used (dictionary: contrast_rts_dictionary.csv)
    2_data_processing/contrasts.csv — one row per contrast (dictionary: contrasts_dictionary.csv)

Depends on
    00_fetch_raw.py

Last substantive change
    2026-11-07 — --with-errors writes the same contrasts with error trials, for accuracy fits.

Run from the project root.
"""
import csv
import sys
from collections import defaultdict

# With --with-errors the same contrasts are built from correct and error trials (quarters then cut over
# all trials) and written to contrast_trials_acc.csv with the correct column, for the accuracy fits;
# without it, correct trials only, as before.
WITH_ERRORS = "--with-errors" in sys.argv


def usable(key, test_only=False):
    """DataPrep rows with a usable RT and, unless WITH_ERRORS, a correct response where accuracy exists."""
    for r in csv.DictReader(open(f"1_raw/{key}.csv", newline="")):
        if r["exclude_reason"] != "":
            continue
        if r["correct"] not in ("", "1") and not (WITH_ERRORS and r["correct"] == "0"):
            continue
        if test_only and r["practice"] == "True":
            continue
        yield r


CONTRASTS = []  # (contrast id, dataset, task kind, axis, A label, B label)
OUT = []        # (contrast id, unit, cell, period A/B, rt in seconds)


def quarters(contrast, rows, order_key, cell_key):
    """First vs last quarter of each subject's trials within each cell, in run order."""
    by_unit_cell = defaultdict(list)
    for r in rows:
        by_unit_cell[(r["subjects"], cell_key(r))].append(r)
    for (unit, cell), rs in by_unit_cell.items():
        rs.sort(key=order_key)
        q = len(rs) // 4
        for r in rs[:q]:
            OUT.append((contrast, unit, cell, "A", r["reaction_times"], r["correct"]))
        for r in rs[len(rs) - q:]:
            OUT.append((contrast, unit, cell, "B", r["reaction_times"], r["correct"]))


def split(contrast, rows, period_of, cell_key=lambda r: "all", unit_key=lambda r: r["subjects"]):
    """Rows whose period_of(row) is "A" or "B"; others are not in the contrast."""
    for r in rows:
        p = period_of(r)
        if p in ("A", "B"):
            OUT.append((contrast, unit_key(r), cell_key(r), p, r["reaction_times"], r["correct"]))


def num(*keys):
    return lambda r: tuple(float(r[k]) for k in keys)


# ---- Practice: time on task ----------------------------------------------------------------------
CONTRASTS.append(("P1", "ratcliff1998", "brightness discrimination", "practice", "first quarter of sessions", "last quarter"))
quarters("P1", usable("ratcliff1998"), num("session", "block", "trial"), lambda r: r["condition"])

CONTRASTS.append(("P2", "wagenmakers2008", "lexical decision", "practice", "first quarter of trials", "last quarter"))
# The source keeps trials within a block in run order; the row position (enumerate) breaks ties.
quarters("P2", [dict(r, _i=i) for i, r in enumerate(usable("wagenmakers2008"))],
         lambda r: (float(r["block"]), r["_i"]), lambda r: r["condition"])

CONTRASTS.append(("P3", "keuleers2012", "lexical decision (BLP)", "practice", "first quarter of trials", "last quarter"))
quarters("P3", usable("keuleers2012"), num("order"), lambda r: "all")

CONTRASTS.append(("P4", "cochrane2023", "motion discrimination", "practice", "day 1 (first quarter)", "day 4 (last quarter)"))
quarters("P4", usable("cochrane2023"), num("trial"), lambda r: "all")

CONTRASTS.append(("P5", "wolfe2010", "visual search", "practice", "first quarter of trials", "last quarter"))
quarters("P5", usable("wolfe2010", test_only=True), num("trial"), lambda r: r["search_type"])

CONTRASTS.append(("P6", "rouder2008", "visual working memory", "practice", "first quarter of trials", "last quarter"))
quarters("P6", usable("rouder2008"), num("block", "trial"), lambda r: "all")

CONTRASTS.append(("P7", "pratte2010", "Stroop and Simon", "practice", "first quarter of trials", "last quarter"))
quarters("P7", (r for r in usable("pratte2010") if r["stop_trial"] != "True"), num("block", "trial"), lambda r: r["task"])

CONTRASTS.append(("P8", "enkavi2019_threebytwo", "cued task switching", "practice", "first quarter of trials", "last quarter"))
quarters("P8", usable("enkavi2019_threebytwo", test_only=True), num("trial_num"), lambda r: "all")

CONTRASTS.append(("P9", "enkavi2019_ant", "attention network (flanker)", "practice", "first quarter of trials", "last quarter"))
quarters("P9", usable("enkavi2019_ant", test_only=True), num("trial_num"), lambda r: "all")

# ---- Instruction: speed vs accuracy (benchmarks whose cause is known) ------------------------------
# The instruction, not skill or practice, causes the change; these contrasts test whether skill and
# practice can stand in for a change of response caution, which the model has no parameter for.
CONTRASTS.append(("A1", "ratcliff1998", "brightness discrimination", "instruction", "speed instruction", "accuracy instruction"))
split("A1", usable("ratcliff1998"), lambda r: {"speed": "A", "accuracy": "B"}[r["condition"]])
CONTRASTS.append(("A2", "wagenmakers2008", "lexical decision", "instruction", "speed instruction", "accuracy instruction"))
split("A2", usable("wagenmakers2008"), lambda r: {"speed": "A", "accuracy": "B"}[r["condition"]])

# ---- Item practice: word frequency ----------------------------------------------------------------
CONTRASTS.append(("F1", "wagenmakers2008", "lexical decision", "item practice", "very low frequency words", "high frequency words"))
split("F1", usable("wagenmakers2008"), lambda r: {"very_low": "A", "high": "B"}.get(r["frequency"]), lambda r: r["condition"])

# ---- Skill: expertise -----------------------------------------------------------------------------
CONTRASTS.append(("S1", "sc2_action_type3", "StarCraft 2 by league", "skill", "Bronze games", "Grandmaster games"))
# Games are the units; each game belongs to one league, so A and B are different units. Unit "league".
split("S1", usable("sc2_action_type3"), lambda r: {"1": "A", "7": "B"}.get(r["skill_levels"]),
      unit_key=lambda r: r["subjects"])

CONTRASTS.append(("S2", "cochrane2023", "motion discrimination", "skill", "non-players, day 1", "action video game players, day 1"))
# Day 1 = trials 1 to 700 (dayTrials counts 1 to 700 within each of four days).
split("S2", (r for r in usable("cochrane2023") if float(r["trial"]) <= 700),
      lambda r: {"-0.5": "A", "0.5": "B"}[r["isVGP"]])

# ---- Drive level: stimulus strength ---------------------------------------------------------------
CONTRASTS.append(("D1", "roitman2002", "motion discrimination (monkeys)", "drive level", "coherence 0.032", "coherence 0.512"))
split("D1", usable("roitman2002"), lambda r: {"coh_0.032": "A", "coh_0.512": "B"}.get(r["condition"]))

CONTRASTS.append(("D2", "cochrane2023", "motion discrimination", "drive level", "coherence 0.064", "coherence 0.448"))
# Only the two positive coherence values at the ends of the range are used; the meaning of the
# negative values in the source is not documented in DataPrep (AntiLearner log: assumed).
split("D2", usable("cochrane2023"), lambda r: {"0.064": "A", "0.448": "B"}.get(r["condition"]))

if WITH_ERRORS:
    with open("2_data_processing/contrast_trials_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "unit", "cell", "period", "rt_s", "correct"])
        w.writerows(OUT)
    print(f"wrote {len(OUT):,} trials (correct and error) to 2_data_processing/contrast_trials_acc.csv")
else:
    with open("2_data_processing/contrast_rts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "unit", "cell", "period", "rt_s"])
        w.writerows(row[:5] for row in OUT)
    with open("2_data_processing/contrasts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "dataset", "task", "axis", "period_A", "period_B"])
        w.writerows(CONTRASTS)
    print(f"wrote {len(OUT):,} trials in {len(CONTRASTS)} contrasts to 2_data_processing/contrast_rts.csv")
