"""Process check for 20_fit_starcraft_lattice.py: (a) lattice percentiles of a smooth sample, rounded to the
45.198 ms grid, recover the sample's own percentiles; (b) the SP fit on raw percentiles reproduces stage 15's
S1 SP fit; (c) time per fit. Seed 8."""
import importlib.util, json, time
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("lf", "3_analyses/20_fit_starcraft_lattice.py")
lf = importlib.util.module_from_spec(spec); spec.loader.exec_module(lf)
rng = np.random.default_rng(8)
x = 200 + rng.gamma(2.0, 300.0, 200000)
print("(a) continuous:", np.round(np.percentile(x, lf.PCTS), 1))
print("    lattice   :", np.round(lf.lattice_percentiles(np.round(x / lf.STEP_MS) * lf.STEP_MS, lf.PCTS), 1))
print("    raw on grid:", np.round(np.percentile(np.round(x / lf.STEP_MS) * lf.STEP_MS, lf.PCTS), 1))
t = time.time()
r = lf.fit((False, ["s", "p"], [json.loads(Path("4_output/mech_selection/S1.json").read_text())["fits"]["SP"]["changes"]]))
old = json.loads(Path("4_output/mech_selection/S1.json").read_text())["fits"]["SP"]
print(f"(b) raw SP here chi2 {r['chi2']} {r['changes']}; stage 15 chi2 {old['chi2']} {old['changes']}")
print(f"(c) one 2-parameter fit, one start: {time.time() - t:.0f} s")
