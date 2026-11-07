# SkillPractice-AcrossTasks

Tests whether two mechanisms of the extended Wong-Wang model (roaming drive plus a racing instance-LCA route) can produce the changes seen in eleven of the lab's reaction-time datasets and StarCraft 2: skill (drive precision) and practice (stored instances), in any combination. Where they cannot, it tests what must be added: a change of mean drive, evidence, a top-down control signal with balanced excitation and inhibition (Lo, Wang & Wang, 2015), or non-decision time. The model is fitted to correct RTs, accuracy and error RTs.

Repository: https://github.com/scottmharrison/WW-WithEIVariation

Questions are in [SkillPractice-AcrossTasks.md](SkillPractice-AcrossTasks.md); decisions and results in [SkillPractice-AcrossTasks_log.md](SkillPractice-AcrossTasks_log.md).

## Reports

| Report | Contents |
|---|---|
| `4_output/SkillPractice-AcrossTasks_report.pdf` | Per-participant percentile changes in 16 contrasts against the signatures predicted for skill and practice |
| `4_output/SkillPractice-Mixtures_report.pdf` | Pooled data; can skill and practice, mixed, produce each change in correct RTs, and which addition is needed |
| `4_output/SkillPractice-Accuracy_report.pdf` | The same with accuracy and error RTs, evidence and the control signal |

## Running it

Clone [DataPrep](https://github.com/scottmharrison/DataPrep) and WW-RTShapeAnatomy beside this repository (or set `DATAPREP_DIR` and `WW_ANATOMY_DIR`), and run DataPrep's pipeline first. Raw data is never stored here: `1_raw/` and the two large trial tables are rebuilt by the pipeline.

```bash
python -m pip install -r requirements.txt
```

```bash
python SkillPractice-AcrossTasks_pipeline.py
```

Stages 07 and 12 fit the model and take hours on 14 cores; a contrast whose result file exists is skipped.

## Method in brief

1. Sixteen contrasts, each changing one thing with the task held fixed: time on task (first vs last quarter of trials), word frequency, expertise, stimulus strength, and speed vs accuracy instruction (benchmarks with a known cause).
2. Trials pooled over units; percentile changes, accuracy change and error-RT changes, with a cluster bootstrap over units.
3. The model's condition A is calibrated from the nearest WW-CarefulFitting fit; B is fitted by skill, practice, both, and both plus one addition; adequacy by chi-square with data and simulation error.
4. The control signal comes from an f-I table derived by simulating the paper's conductance-based neuron (strength S, ratio R), validated against the paper's circuit results before use (`process/output/check_03` to `check_06`).

## Layout

| Path | Contents |
|---|---|
| `1_raw/` | DataPrep's prepared tables and WW-CarefulFitting's fitted conditions, copied; not in git; checksums in `2_data_processing/raw_manifest.csv` |
| `2_data_processing/` | Contrast definitions, trials, starting conditions |
| `3_analyses/` | Measures, bootstrap summaries, model fits, reports, dictionaries |
| `model/` | `wwsim_ext.py` (the extended simulator with evidence, choices and the control signal) and `bsi_transfer.py` |
| `4_output/` | Tables, figures, reports |
| `process/` | Checks; the pipeline never runs these |
