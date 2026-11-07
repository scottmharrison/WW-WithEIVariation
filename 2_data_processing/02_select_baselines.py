"""
02_select_baselines.py

Purpose
    For each contrast, the WW-CarefulFitting condition whose fitted parameters start the model fit of
    its condition A (3_analyses/07_fit_mixtures.py), chosen as the fitted condition closest to A in
    task and condition. The choice only sets the starting point: A is then calibrated to its own pooled
    percentiles. Visual search (Wolfe et al., 2010) has no fitted condition; it starts from the lexical
    decision fit, whose median RT is in the same range.

Inputs
    1_raw/ww_conditions.csv, 2_data_processing/contrasts.csv

Outputs
    2_data_processing/baselines.csv — one row per contrast with its start parameters
        (dictionary: baselines_dictionary.csv)

Depends on
    00_fetch_raw.py, 01_build_contrasts.py

Last substantive change
    2026-11-06 — first version.

Run from the project root.
"""
import csv

START = {  # contrast: WW-CarefulFitting condition id, and why
    "P1": ("rr98__20", "brightness, accuracy instruction (pooled A mixes both instructions)"),
    "P2": ("speed_acc__29", "lexical decision, accuracy instruction"),
    "P3": ("blp__03", "BLP words"),
    "P4": ("teddm__35", "human motion, middle coherence 0.192"),
    "P5": ("blp__03", "no fitted visual-search condition; lexical decision is in the same RT range"),
    "P6": ("rouder2008__17", "working memory, set size 5 (middle)"),
    "P7": ("pratte2010__07", "Stroop neutral"),
    "P8": ("threebytwo__37", "task switching, stay trials"),
    "P9": ("ant__01", "flanker, neutral"),
    "A1": ("rr98__19", "brightness, speed instruction"),
    "A2": ("speed_acc__28", "lexical decision, speed instruction"),
    "F1": ("speed_acc__29", "lexical decision, accuracy instruction"),
    "S1": ("sc2__21", "StarCraft 2, Bronze league"),
    "S2": ("teddm__35", "human motion, coherence 0.192"),
    "D1": ("roitman2002__11", "monkey motion, coherence 0.032"),
    "D2": ("teddm__34", "human motion, coherence 0.064"),
}
KEYS = ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]

fits = {r["cond_id"]: r for r in csv.DictReader(open("1_raw/ww_conditions.csv", newline=""))}
contrasts = [r["contrast"] for r in csv.DictReader(open("2_data_processing/contrasts.csv"))]
with open("2_data_processing/baselines.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["contrast", "start_condition", "reason"] + KEYS)
    for c in contrasts:
        cond, why = START[c]  # a contrast missing here raises KeyError and stops the pipeline
        w.writerow([c, cond, why] + [fits[cond][k] for k in KEYS])
print(f"wrote {len(contrasts)} rows to 2_data_processing/baselines.csv")
