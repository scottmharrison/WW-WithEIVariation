"""
05_write_dictionaries.py

Purpose
    Write a data dictionary beside each table this project produces: column name, description, units
    and valid values (read from the table).

Inputs
    2_data_processing/raw_manifest.csv, contrasts.csv, contrast_rts.csv;
    4_output/unit_measures.csv, unit_changes.csv, contrast_summary.csv

Outputs
    <table>_dictionary.csv beside each table

Depends on
    03_summarise_contrasts.py

Last substantive change
    2026-11-07 — added the mechanism-comparison tables.

Run from the project root.
"""
import csv

PCTS = [5, 10, 25, 50, 75, 90, 99]
CHANGE = "Percent change from A to B (see 03_summarise_contrasts.py)"

TABLES = {
    "2_data_processing/raw_manifest.csv": [("file", "File in 1_raw/.", ""), ("source", "Path copied from.", ""),
                                           ("bytes", "Size.", "bytes"), ("sha256", "SHA-256 checksum.", "")],
    "2_data_processing/contrasts.csv": [("contrast", "Contrast id.", ""), ("dataset", "DataPrep key.", ""),
                                        ("task", "Task.", ""), ("axis", "Axis of SKILL_vs_PRACTICE_explained.pdf moved by the contrast.", ""),
                                        ("period_A", "Condition A.", ""), ("period_B", "Condition B.", "")],
    "2_data_processing/contrast_rts.csv": [("contrast", "Contrast id.", ""), ("unit", "Participant, monkey, or StarCraft 2 game.", ""),
                                           ("cell", "Condition within which A and B are compared; all = pooled.", ""),
                                           ("period", "A or B.", ""), ("rt_s", "Reaction time.", "s")],
    "4_output/unit_measures.csv": [("contrast", "Contrast id.", ""), ("unit", "Unit.", ""), ("cell", "Cell.", ""),
                                   ("period", "A or B.", ""), ("n", "Trials.", "trials")]
        + [(f"q{p:02d}_ms", f"{p}th percentile of RT.", "ms") for p in PCTS]
        + [("mu_ms", "Ex-Gaussian mu.", "ms"), ("sigma_ms", "Ex-Gaussian sigma.", "ms"), ("tau_ms", "Ex-Gaussian tau.", "ms")],
    "4_output/unit_changes.csv": [("contrast", "Contrast id (within-unit contrasts only).", ""), ("unit", "Unit.", ""),
                                  ("n_cells", "Cells averaged.", "")]
        + [(f"q{p:02d}_change_pct", f"{CHANGE}, {p}th percentile, mean over cells.", "%") for p in PCTS]
        + [(f"{p}_change_pct", f"{CHANGE}, ex-Gaussian {p}, mean over cells; nan for sigma or tau when its period-A value is < 2 ms.", "%") for p in ["mu", "sigma", "tau"]],
    "4_output/contrast_summary.csv": [("contrast", "Contrast id.", ""), ("dataset", "DataPrep key.", ""), ("task", "Task.", ""),
                                      ("axis", "Axis moved.", ""), ("period_A", "Condition A.", ""), ("period_B", "Condition B.", ""),
                                      ("n_units", "Units (within) or units in A / B (between).", ""),
                                      ("n_units_without_sigma", "Units with no defined sigma change (period-A sigma < 2 ms).", "units"),
                                      ("n_units_without_tau", "Units with no defined tau change (period-A tau < 2 ms).", "units")]
        + [c for p in PCTS for c in [(f"q{p:02d}_change_pct", f"{CHANGE}, {p}th percentile, median over units.", "%"),
                                     (f"q{p:02d}_low", "Lower end of 95% bootstrap interval.", "%"), (f"q{p:02d}_high", "Upper end.", "%")]]
        + [c for p in ["mu", "sigma", "tau"] for c in [(f"{p}_change_pct", f"{CHANGE}, ex-Gaussian {p}.", "%"),
                                                        (f"{p}_low", "Lower end of 95% interval.", "%"), (f"{p}_high", "Upper end.", "%"),
                                                        (f"{p}_direction", "falls / no clear change / rises, by the interval.", "")]]
        + [("front_minus_tail", "Change at 10th minus change at 99th percentile.", "percentage points"),
           ("front_minus_tail_low", "Lower end of 95% interval.", "percentage points"),
           ("front_minus_tail_high", "Upper end.", "percentage points"),
           ("location", "front-led (interval below 0), tail-led (above 0), uniform.", "")],
    "2_data_processing/baselines.csv": [("contrast", "Contrast id.", ""), ("start_condition", "WW-CarefulFitting condition id that starts the fit.", ""),
                                        ("reason", "Why that condition.", "")] + [(k, f"Fitted parameter {k} of that condition (WW-CarefulFitting units).", "")
                                        for k in ["W_self", "W_inhib", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_mu", "ter_sd", "u_star"]],
    "4_output/pooled_profiles.csv": [("contrast", "Contrast id.", ""), ("axis", "Axis moved.", ""), ("percentile", "Percentile.", ""),
                                     ("n_A", "Trials in A, pooled over units.", "trials"), ("n_B", "Trials in B.", "trials"),
                                     ("q_A_ms", "Percentile of pooled A.", "ms"), ("q_B_ms", "Percentile of pooled B.", "ms"),
                                     ("change_pct", "100 (q_B / q_A - 1).", "%"), ("se_pct", "Cluster-bootstrap SE over units.", "%"),
                                     ("low_pct", "2.5th percentile of the bootstrap.", "%"), ("high_pct", "97.5th percentile.", "%")],
    "4_output/mixture_fits.csv": [("contrast", "Contrast id.", ""), ("model", "Changes allowed (07_fit_mixtures.py).", ""),
                                  ("k", "Number of changed parameters.", "")]
        + [(f"change_{n}", d, u) for n, d, u in [("s", "Skill: log factor on u0_sd.", "log"), ("p", "Practice: change in logN.", "log"),
                                                  ("d", "Drive level: log factor on u0 and u0_sd.", "log"), ("t", "Non-decision shift.", "ms"),
                                                  ("r", "Retrieval: log factor on s_W.", "log")]]
        + [("chi2", "Chi-square over 7 percentiles, data and simulation SE.", ""), ("df", "7 - k.", ""), ("p", "Chi-square p.", ""),
           ("aic", "chi2 + 2k.", ""), ("rms_pct", "RMS of predicted minus observed change.", "%"),
           ("adequate", "p > .05.", ""), ("at_bound", "Changes at a search bound.", "")],
    "4_output/mixture_summary.csv": [("contrast", "Contrast id.", ""), ("axis", "Axis moved.", ""),
                                     ("calibration_rms_log", "RMS log error of the calibrated model A against pooled A percentiles.", "log ms"),
                                     ("two_additions_adequate", "SP+drive+ndt adequate.", ""), ("few_units", "Units, where 2 or 3; else empty.", "units"),
                                     ("instance_share_A", "Share of calibrated A's simulated trials won by the instance route.", "proportion"),
                                     ("sp_adequate", "Skill + practice adequate.", ""), ("sp_p", "Its p.", ""),
                                     ("needed", "Where SP adequate: either / skill / practice / both; else not adequate.", ""),
                                     ("best_addition", "Where SP not adequate: adequate SP+X with lowest AIC, or none adequate.", ""),
                                     ("lowest_aic_model", "Model with the lowest AIC.", ""), ("sp_change_s", "SP fit: skill change.", "log"),
                                     ("sp_change_p", "SP fit: practice change.", "log")],
    "4_output/mechanism_steps.csv": [("contrast", "Starting point (calibrated A of this contrast).", ""), ("ingredient", "One step of one change.", ""),
                                     ("percentile", "Percentile.", ""), ("change_pct", "Simulated percent change.", "%")],
    "4_output/bsi_table.csv": [("S", "Control strength (Lo, Wang & Wang 2015 definition, 0.3 r_e g_AMPA).", ""),
                               ("R", "Control ratio (r_i g_GABA) / (r_e g_AMPA).", ""),
                               ("a_fit", "Abbott-Chance a fitted to the simulated neuron.", "Hz/nA"), ("b_fit", "Abbott-Chance b.", "Hz"),
                               ("d_fit", "Abbott-Chance d.", "s"), ("fit_rms_hz", "RMS error of that fit over 0-60 Hz.", "Hz"),
                               ("gain", "a / a at S = 0.", "ratio"), ("shift_nA", "b/a minus b/a at S = 0.", "nA"),
                               ("curvature", "d / d at S = 0.", "ratio")],
    "4_output/pooled_accuracy_profiles.csv": [("contrast", "Contrast id.", ""), ("measure", "accuracy, correct_rt_pct or error_rt_pct.", ""),
                                              ("level", "Percentile, for RT measures.", ""), ("value_A", "Value in A (proportion or ms).", ""),
                                              ("value_B", "Value in B.", ""), ("change", "Accuracy: B - A; RT: percent change.", "proportion or %"),
                                              ("se", "Cluster-bootstrap SE over units.", "proportion or %")],
    "2_data_processing/contrast_trials_acc.csv": [("contrast", "Contrast id.", ""), ("unit", "Unit.", ""), ("cell", "Cell.", ""),
                                                  ("period", "A or B.", ""), ("rt_s", "Reaction time.", "s"),
                                                  ("correct", "1 correct, 0 error, empty where accuracy is not recorded.", "")],
    "4_output/acc_mixture_fits.csv": [("contrast", "Contrast id.", ""), ("model", "Changes allowed (12_fit_accuracy_mixtures.py).", ""),
                                      ("k", "Number of changed parameters.", "")]
        + [(f"change_{n}", d, u) for n, d, u in [("s", "Skill: log factor on u0_sd.", "log"), ("p", "Practice: change in logN.", "log"),
                                                  ("d", "Common drive: log factor on u0 and u0_sd.", "log"), ("e", "Evidence change.", "Hz"),
                                                  ("cS", "Control strength S.", ""), ("cR", "Control ratio R.", ""), ("t", "Non-decision shift.", "ms")]]
        + [("chi2", "Chi-square over correct-RT percentiles, accuracy and error-RT percentiles.", ""), ("n_terms", "Terms in the chi-square.", ""),
           ("df", "n_terms - k.", ""), ("p", "Chi-square p.", ""), ("aic", "chi2 + 2k.", ""), ("adequate", "p > .05.", ""),
           ("predicted_accuracy_change", "Model accuracy B - A; empty without accuracy.", "proportion"), ("at_bound", "Changes at a search bound.", "")],
    "4_output/acc_mixture_summary.csv": [("contrast", "Contrast id.", ""), ("axis", "Axis moved.", ""), ("has_accuracy", "Accuracy observed.", ""),
                                         ("n_error_terms", "Error-RT percentiles in the chi-square.", ""),
                                         ("calibration_rms_log", "RMS log error of model A against observed A correct-RT percentiles.", "log ms"),
                                         ("accuracy_A_observed", "Observed accuracy in A.", "proportion"), ("accuracy_A_model", "Calibrated model accuracy in A.", "proportion"),
                                         ("sp_adequate", "Skill + practice adequate.", ""), ("needed", "Where SP adequate: either / skill / practice / both.", ""),
                                         ("best_single_addition", "Where SP not adequate: adequate single addition with lowest AIC, or none adequate.", ""),
                                         ("lowest_aic_model", "Model with the lowest AIC.", ""), ("adequate_models", "All adequate models.", "")],
    "4_output/acc_ingredient_steps.csv": [("contrast", "Starting point (calibrated A).", ""), ("ingredient", "One step of one change.", ""),
                                          ("median_change_pct", "Simulated change in median correct RT.", "%"),
                                          ("q99_change_pct", "Simulated change in 99th percentile.", "%"),
                                          ("accuracy_change", "Simulated change in accuracy.", "proportion")],
    "4_output/pooled_accuracy_profiles_fine.csv": [("contrast", "Contrast id.", ""), ("measure", "accuracy, correct_rt_pct or error_rt_pct.", ""),
                                                   ("level", "Percentile (12 for correct RTs), for RT measures.", ""), ("value_A", "Value in A.", ""),
                                                   ("value_B", "Value in B.", ""), ("change", "Accuracy: B - A; RT: percent change.", "proportion or %"),
                                                   ("se", "Cluster-bootstrap SE over units.", "proportion or %")],
    "4_output/mechanism_comparison.csv": [("contrast", "Contrast id.", ""), ("axis", "Axis moved.", ""),
                                          ("selected", "Combination chosen by forward selection (stage 15).", ""),
                                          ("selected_p", "Its chi-square p.", ""), ("selected_adequate", "p > .05.", ""),
                                          ("selected_k", "Changed parameters in it.", ""), ("sp_p", "Skill + practice p.", ""),
                                          ("sp_adequate", "Skill + practice p > .05.", ""), ("ceiling_chi2", "Lowest chi-square of any fit for the contrast.", ""),
                                          ("ceiling_model_p", "p of the all-mechanism model; empty where it has as many changes as terms.", ""),
                                          ("best_single", "Mechanism with the largest AIC drop when added alone.", "")]
        + [(f"aic_gain_{m}", f"AIC of skill + practice minus AIC with {m} added alone.", "") for m in ['drive', 'evidence', 'control', 'threshold', 'urgency', 'lapse', 'ndt', 'retrieval']],
    "4_output/shared_combinations.csv": [("combination", "Mechanisms added to skill + practice.", ""), ("n_mechanisms", "Number of mechanisms.", ""),
                                         ("adequate_in", "Contrasts where the combination is adequate.", "contrasts"),
                                         ("total_aic", "Sum of AIC over the 16 contrasts.", ""), ("failing", "Contrasts where it is not adequate.", "")],
}

for path, columns in TABLES.items():
    rows = list(csv.DictReader(open(path, newline="")))
    with open(path.replace(".csv", "_dictionary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "description", "units", "valid_values"])
        for name, description, units in columns:
            present = [r[name] for r in rows if r[name] not in ("", "nan")]
            # A column is numeric when every present value parses as a number (sign, one decimal point,
            # optional exponent); otherwise its distinct values are listed, or it is described as text.
            numeric = all(v.lstrip("-").replace(".", "", 1).replace("e-", "", 1).replace("e+", "", 1).isdigit() for v in present)
            if numeric and present:
                nums = [float(v) for v in present]
                valid = f"{min(nums):g} to {max(nums):g}"
            else:
                valid = "; ".join(sorted(set(present))) if len(set(present)) <= 8 else "text"
            w.writerow([name, description, units, f"{valid}; {len(rows) - len(present)} missing"])
    print(f"wrote {path.replace('.csv', '_dictionary.csv')}")
