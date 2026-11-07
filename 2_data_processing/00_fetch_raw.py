"""
00_fetch_raw.py

Purpose
    First pipeline stage: bring the DataPrep prepared tables this project uses into 1_raw/, and record
    each file's source, size and checksum. DataPrep (../DataPrep, or the folder in DATAPREP_DIR) finds,
    downloads and standardises the public datasets; its own raw files and their checksums are in its
    2_data_processing/prepared/raw_manifest.csv, which is copied here as well.

Inputs
    <DATAPREP_DIR>/2_data_processing/prepared/<key>.csv for the keys in TABLES, and raw_manifest.csv

Outputs
    1_raw/<key>.csv — added only when absent; an existing file is never overwritten
    2_data_processing/raw_manifest.csv — source, size, SHA-256 of each file in 1_raw/

Depends on
    DataPrep_pipeline.py, run in DataPrep.

Last substantive change
    2026-11-06 — added the Attention Network Task for the pooled analysis.

Run from the project root.
"""
import csv
import hashlib
import os
import shutil
from pathlib import Path

SOURCE = Path(os.environ.get("DATAPREP_DIR", "../DataPrep")) / "2_data_processing" / "prepared"
TABLES = ["ratcliff1998", "wagenmakers2008", "keuleers2012", "cochrane2023", "wolfe2010", "rouder2008",
          "pratte2010", "enkavi2019_threebytwo", "enkavi2019_ant", "roitman2002", "sc2_action_type3"]
FILES = [f"{k}.csv" for k in TABLES] + [f"{k}_dictionary.csv" for k in TABLES] + ["raw_manifest.csv"]

Path("1_raw").mkdir(exist_ok=True)
rows = []
for name in FILES:
    target = Path("1_raw") / ("dataprep_raw_manifest.csv" if name == "raw_manifest.csv" else name)
    if not target.exists():
        # shutil.copyfile raises and stops the pipeline if the source is missing, naming the file.
        shutil.copyfile(SOURCE / name, target)
        print(f"added {target}")
    digest = hashlib.sha256()
    with open(target, "rb") as f:
        for piece in iter(lambda: f.read(64 * 2**20), b""):
            digest.update(piece)
    rows.append(dict(file=target.name, source=str((SOURCE / name).resolve()), bytes=target.stat().st_size,
                     sha256=digest.hexdigest()))

# The Wong-Wang + roaming + instance-LCA fits of WW-CarefulFitting, one row per condition, as
# WW-RTShapeAnatomy selected them (its 2_data_processing/conditions.csv); they start each model fit.
MODEL_SOURCE = Path(os.environ.get("WW_ANATOMY_DIR", "../WW-RTShapeAnatomy")) / "2_data_processing" / "conditions.csv"
target = Path("1_raw") / "ww_conditions.csv"
if not target.exists():
    shutil.copyfile(MODEL_SOURCE, target)
    print(f"added {target}")
rows.append(dict(file=target.name, source=str(MODEL_SOURCE.resolve()), bytes=target.stat().st_size,
                 sha256=hashlib.sha256(target.read_bytes()).hexdigest()))

with open("2_data_processing/raw_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to 2_data_processing/raw_manifest.csv")
