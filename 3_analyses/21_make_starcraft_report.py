"""
21_make_starcraft_report.py

Purpose
    Report the StarCraft 2 refit (20_fit_starcraft_lattice.py): observed league differences on raw and
    lattice-aware percentiles; the selected model and its fit under each of the four settings
    (percentiles raw or lattice-aware x stage 15's eight mechanisms or those plus circuit strength and
    retrieval strength); and what each mechanism alone adds at the first selection step.

Inputs
    4_output/s1_lattice_profile.csv, 4_output/s1_lattice/<variant>.json (20_fit_starcraft_lattice.py),
    4_output/mech_selection/S1.json (stage 15: raw percentiles, eight mechanisms)

Outputs
    4_output/s1_refit_summary.csv (dictionary beside it, from 05_write_dictionaries.py);
    4_output/SkillPractice-StarCraftRefit_report.pdf; 4_output/figures/s1_*.png (300 dpi)

Depends on
    20_fit_starcraft_lattice.py, 15_fit_mechanism_selection.py

Last substantive change
    2026-11-08 — first version.

Run from the project root.
"""
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, with no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 6.5
out = Path("4_output")
PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
prof = list(csv.DictReader(open(out / "s1_lattice_profile.csv")))
OBS = {k: (np.array([float(r["change"]) for r in prof if r["percentiles"] == k]), np.array([float(r["se"]) for r in prof if r["percentiles"] == k]))
       for k in ("raw", "lattice")}

# The four settings. raw_8 is stage 15's own S1 fit, whose predictions are stored under another key.
s15 = json.loads((out / "mech_selection" / "S1.json").read_text())
for f in s15["fits"].values():
    f["predicted_pct"] = f["predicted_correct_pct"]
RUNS = {"raw_8": dict(s15, lattice=False, mechanisms={m: v for m, v in s15["mechanisms"].items()})}
for v in ("raw_10", "lattice_8", "lattice_10"):
    RUNS[v] = json.loads((out / "s1_lattice" / f"{v}.json").read_text())
LABEL = {"raw_8": "raw percentiles, 8 mechanisms", "raw_10": "raw percentiles, 10 mechanisms",
         "lattice_8": "lattice-aware, 8 mechanisms", "lattice_10": "lattice-aware, 10 mechanisms"}
COL = {"raw_8": "#999999", "raw_10": "#E69F00", "lattice_8": "#56B4E9", "lattice_10": "#0072B2"}


def pretty(name):
    return name.replace("+", " + ").replace("SP", "skill + practice").replace("_", " ")


rows = []
for v, r in RUNS.items():
    for name in (["SP"] if r["selected"] == "SP" else ["SP", r["selected"]]):
        f = r["fits"][name]
        rows.append(dict(setting=v, model=name, role="selected" if name == r["selected"] else "skill + practice",
                         k=len(f["params"]), chi2=f["chi2"], df=f["df"], p=f["p"], aic=f["aic"],
                         changes="; ".join(f"{k}={val:.3f}" for k, val in f["changes"].items())))
with open(out / "s1_refit_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

figures, captions = [], {}

# ---- Figure 1: observations, raw and lattice-aware -------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
va = {k: [float(r["value_A"]) for r in prof if r["percentiles"] == k] for k in ("raw", "lattice")}
vb = {k: [float(r["value_B"]) for r in prof if r["percentiles"] == k] for k in ("raw", "lattice")}
for k, mk, c in [("raw", "s", "#999999"), ("lattice", "o", "black")]:
    a1.plot(PCTS, va[k], mk + "--", ms=3, lw=0.7, color=c, label=f"Bronze, {k}")
    a1.plot(PCTS, vb[k], mk + "-", ms=3, lw=0.7, color=c, label=f"Grandmaster, {k}")
    a2.errorbar(PCTS, OBS[k][0], yerr=1.96 * OBS[k][1], fmt=mk + "-", ms=3, lw=0.7, capsize=1.5, color=c, label=k)
a1.set_yscale("log"); a1.set_xlabel("Percentile"); a1.set_ylabel("RT (ms, log)"); a1.legend(frameon=False, fontsize=6)
a2.set_xlabel("Percentile"); a2.set_ylabel("Change, Bronze to Grandmaster (%)"); a2.legend(frameon=False, title="percentiles", fontsize=6.5, title_fontsize=6.5)
a2.axhline(0, color="black", lw=0.4)
for ax in (a1, a2):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
a2.yaxis.set_minor_locator(AutoMinorLocator())
fig.tight_layout()
figures.append(("s1_fig1_observations", fig))
captions["s1_fig1_observations"] = ("Figure 1. StarCraft 2 RTs, Bronze (167 games) and Grandmaster (32 games), pooled. Every RT is a whole "
    "multiple of 45.198 ms. Raw percentiles (grey squares) sit on those steps; lattice-aware percentiles (black circles) spread each RT "
    "evenly over its step. Right: percent change with 95% intervals (1.96 SE, cluster bootstrap over games, 500 replicates). Source: "
    "4_output/s1_lattice_profile.csv.")

# ---- Figure 2: fit of the selected model in each setting ------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True)
for ax, (v, r) in zip(axes.ravel(), RUNS.items()):
    obs, se = OBS["lattice" if r["lattice"] else "raw"]
    sel, sp = r["fits"][r["selected"]], r["fits"]["SP"]
    ax.errorbar(PCTS, obs, yerr=1.96 * se, fmt="o", ms=3, capsize=1.5, color="black", label="observed")
    ax.plot(PCTS, sp["predicted_pct"], "s--", ms=2.5, lw=0.8, color="#999999", label=f"skill + practice (p = {sp['p']:.2g})")
    ax.plot(PCTS, sel["predicted_pct"], "^-", ms=3, lw=1.1, color=COL[v], label=f"selected (p = {sel['p']:.2g})")
    ax.set_title(f"{LABEL[v]}\nselected: {textwrap.shorten(pretty(r['selected']), 60, placeholder='…')}", fontsize=7)
    ax.axhline(0, color="black", lw=0.4); ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.yaxis.set_minor_locator(AutoMinorLocator()); ax.xaxis.set_minor_locator(AutoMinorLocator())
for ax in axes[1]:
    ax.set_xlabel("Percentile")
for ax in axes[:, 0]:
    ax.set_ylabel("Change, Bronze to Grandmaster (%)")
fig.tight_layout()
figures.append(("s1_fig2_fits", fig))
captions["s1_fig2_fits"] = ("Figure 2. Observed change (95% intervals) and model predictions in the four settings: percentiles raw or "
    "lattice-aware, and stage 15's eight mechanisms or those plus circuit strength and retrieval strength. Selected model by forward "
    "selection on AIC (stage 15's rule). p from a chi-square test over the 12 percentiles with data and simulation SE; a model is "
    "adequate at p > .05. Raw, 8 mechanisms is stage 15's S1 fit.")

# ---- Figure 3: AIC of each mechanism added alone to skill + practice ---------------------------------------
mechs = list(RUNS["lattice_10"]["mechanisms"])
fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
width = 0.2
for j, (v, r) in enumerate(RUNS.items()):
    base = r["fits"]["SP"]["aic"]
    drops = [base - r["fits"][f"SP+{m}"]["aic"] if f"SP+{m}" in r["fits"] else np.nan for m in mechs]
    ax.bar(np.arange(len(mechs)) + (j - 1.5) * width, drops, width, color=COL[v], label=LABEL[v])
ax.set_xticks(range(len(mechs))); ax.set_xticklabels([m.replace("_", " ") for m in mechs], rotation=30, ha="right")
ax.set_ylabel("AIC drop vs skill + practice"); ax.axhline(0, color="black", lw=0.4); ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(frameon=False, fontsize=6.3, ncol=2)
fig.tight_layout()
figures.append(("s1_fig3_single", fig))
captions["s1_fig3_single"] = ("Figure 3. AIC drop when each mechanism alone is added to skill + practice (first selection step), in "
    "each setting. Larger = more of the league difference accounted for. Circuit and retrieval strength are absent from the "
    "8-mechanism settings.")

# ---- Summary page -------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(WIDTH, 4.2))
fig.text(0, 0.97, "StarCraft 2, Bronze to Grandmaster: refit", fontsize=13, weight="bold", va="top")
heads, xs = ["Setting", "Selected model", "chi2 / df", "p", "AIC"], [0, 0.27, 0.73, 0.84, 0.92]
y = 0.86
for cx, h in zip(xs, heads):
    fig.text(cx, y, h, weight="bold", fontsize=7.5, va="top")
fig.add_artist(plt.Line2D([0, 1], [y - 0.05, y - 0.05], color="black", lw=0.6)); y -= 0.08
for v, r in RUNS.items():
    f = r["fits"][r["selected"]]
    vals = [LABEL[v], textwrap.fill(pretty(r["selected"]), 50), f"{f['chi2']:.1f} / {f['df']}", f"{f['p']:.2g}", f"{f['aic']:.1f}"]
    for cx, val in zip(xs, vals):
        fig.text(cx, y, val, fontsize=7.2, va="top")
    y -= 0.075 * (1 + vals[1].count("\n"))
fig.add_artist(plt.Line2D([0, 1], [y + 0.02, y + 0.02], color="black", lw=0.6))
note = ("Sources: 4_output/s1_refit_summary.csv; fits in 4_output/s1_lattice/ and, for raw percentiles with 8 mechanisms, "
        "4_output/mech_selection/S1.json. AIC values compare models within one setting only: raw and lattice-aware settings fit "
        "different observations.")
fig.text(0, y - 0.03, textwrap.fill(note, 125), fontsize=7, va="top", color="#333333")
figures.insert(0, ("s1_summary", fig))

with PdfPages(out / "SkillPractice-StarCraftRefit_report.pdf") as pdf:
    for name, fig in figures:
        fig.savefig(out / "figures" / f"{name}.png", dpi=300)
        if name in captions:
            hgt = fig.get_figheight()
            wrapped = textwrap.fill(captions[name], 118)
            extra = 0.25 + 0.16 * (wrapped.count("\n") + 1)
            page = plt.figure(figsize=(WIDTH, hgt + extra))
            ax = page.add_axes((0, extra / (hgt + extra), 1, hgt / (hgt + extra)))
            ax.imshow(plt.imread(out / "figures" / f"{name}.png")); ax.axis("off")
            page.text(0, (extra - 0.1) / (hgt + extra), wrapped, fontsize=7.8, va="top")
            pdf.savefig(page, dpi=300)
            plt.close(page)
        else:
            pdf.savefig(fig)
        plt.close(fig)
print("wrote 4_output/SkillPractice-StarCraftRefit_report.pdf and 4_output/s1_refit_summary.csv")
