"""
SkillPractice-AcrossTasks_pipeline.py

Purpose
    Run the whole pipeline, from DataPrep's prepared tables to the report PDF, and save a record of the
    Python environment beside the outputs.

Inputs
    <DATAPREP_DIR>/2_data_processing/prepared/ and <WW_ANATOMY_DIR> (see 2_data_processing/00_fetch_raw.py)

Outputs
    Everything in 2_data_processing/ and 4_output/; 4_output/environment.txt

Depends on
    Nothing in this project.

Last substantive change
    2026-11-07 — added the mechanism atlas (stages 18 and 19).

Run from the project root: python SkillPractice-AcrossTasks_pipeline.py
"""
import subprocess
import sys

STAGES = [
    "2_data_processing/00_fetch_raw.py",
    "2_data_processing/01_build_contrasts.py",
    "2_data_processing/01_build_contrasts.py --with-errors",
    "2_data_processing/02_select_baselines.py",
    "3_analyses/02_measure_units.py",
    "3_analyses/03_summarise_contrasts.py",
    "3_analyses/04_make_report.py",
    "3_analyses/06_pooled_profiles.py",
    "3_analyses/07_fit_mixtures.py",
    "3_analyses/08_collect_mixtures.py",
    "3_analyses/09_make_mixture_report.py",
    "3_analyses/10_build_bsi_table.py",
    "3_analyses/11_pooled_accuracy_profiles.py",
    "3_analyses/12_fit_accuracy_mixtures.py",
    "3_analyses/13_collect_accuracy_mixtures.py",
    "3_analyses/14_make_accuracy_report.py",
    "3_analyses/11_pooled_accuracy_profiles.py --fine",
    "3_analyses/15_fit_mechanism_selection.py",
    "3_analyses/16_fit_shared_combinations.py",
    "3_analyses/17_make_mechanism_report.py",
    "3_analyses/18_mechanism_atlas.py",
    "3_analyses/19_make_atlas_report.py",
    "3_analyses/05_write_dictionaries.py",
]

for stage in STAGES:
    print(f"== {stage}")
    # check=True stops the pipeline at the first stage that fails.
    subprocess.run([sys.executable, *stage.split()], check=True)

# Record of the environment beside the outputs (lab standards, Principle 7): Python version and pip freeze.
with open("4_output/environment.txt", "w") as f:
    f.write(sys.version + "\n")
    f.write(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout)
print("wrote 4_output/environment.txt")
