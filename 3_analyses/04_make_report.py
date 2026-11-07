"""
04_make_report.py

Purpose
    Write the report PDF: the questions and answers, the design (how each dataset tests the skill and
    practice predictions of SKILL_vs_PRACTICE_explained.pdf), the data used, the method shown on one
    unit, the results for every contrast, and the checks behind them.

Inputs
    2_data_processing/contrasts.csv, contrast_rts.csv; 4_output/contrast_summary.csv,
    unit_changes.csv, unit_measures.csv

Outputs
    4_output/SkillPractice-AcrossTasks_report.pdf; 4_output/figures/*.png (300 dpi)

Depends on
    03_summarise_contrasts.py

Last substantive change
    2026-11-07 — Figure 8 no longer reads process output (lab standards: product code does not use process code).

Run from the project root.
"""
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, with no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

# Arial has Helvetica's letter widths and is the Helvetica substitute installed on this machine.
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 6.5  # inches, the APA maximum
out = Path("4_output")
(out / "figures").mkdir(exist_ok=True)

# The same colour means the same axis in every figure (Okabe-Ito colour-blind-safe palette).
AXES = ["practice", "instruction", "item practice", "skill", "drive level"]
AXIS_COLOUR = {"practice": "#E69F00", "instruction": "#CC79A7", "item practice": "#D55E00", "skill": "#0072B2", "drive level": "#009E73"}
AXIS_LABEL = {"practice": "Time on task", "instruction": "Speed vs accuracy", "item practice": "Word frequency", "skill": "Expertise", "drive level": "Stimulus strength"}
PCTS = [5, 10, 25, 50, 75, 90, 99]
# Percent changes reported in SKILL_vs_PRACTICE_explained.pdf, p. 4 (one simulation per mechanism).
PRED = {"practice": {"q10": -44, "q50": -27, "q99": -2, "mu": -73, "sigma": -94, "tau": 1},
        "skill": {"q10": -6, "q50": 2, "q99": -56, "mu": 88, "sigma": 75, "tau": -57}}

summary = list(csv.DictReader(open(out / "contrast_summary.csv")))
by_id = {s["contrast"]: s for s in summary}
unit_changes = list(csv.DictReader(open(out / "unit_changes.csv")))
measures = list(csv.DictReader(open(out / "unit_measures.csv")))
n_trials = defaultdict(int)
for m in measures:
    n_trials[m["contrast"]] += int(m["n"])


def f(s, k):
    return float(s[k])


def minor(ax, x=True, y=True):
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def category(k, n):
    # Answer categories (SkillPractice-AcrossTasks.md): none = 0; most = more than half; a few = between.
    return "none" if k == 0 else "most" if k > n / 2 else "a few"


def label(s):
    return f"{s['contrast']} {s['task']}"


practice = [s for s in summary if s["axis"] == "practice" and s["contrast"] != "P9"]
q1 = sum(s["location"] == "front-led" for s in practice)
q2 = sum(s["mu_direction"] == "falls" and s["tau_direction"] != "falls" for s in practice)
q3 = by_id["F1"]["location"]
skill = [s for s in summary if s["axis"] == "skill"]
q4 = sum(s["location"] == "tail-led" and s["mu_direction"] == "rises" for s in skill)
drive = [s for s in summary if s["axis"] == "drive level"]
q5 = sum(s["mu_direction"] == s["tau_direction"] and s["mu_direction"] != "no clear change" for s in drive)

figures, captions = [], {}


def text_page(name, title, subtitle, table, notes, col_x, row_h=0.05, height=9.3, fontsize=8.3):
    fig = plt.figure(figsize=(WIDTH, height))
    fig.text(0, 0.985, title, fontsize=13, weight="bold", va="top")
    fig.text(0, 0.955, subtitle, fontsize=8.5, va="top", color="#444444")
    y = 0.915
    fig.add_artist(plt.Line2D([0, 1], [y + 0.012, y + 0.012], color="black", lw=0.8))
    for i, row in enumerate(table):
        for c, v in zip(col_x, row):
            fig.text(c, y, v, va="top", weight="bold" if i == 0 else "normal", fontsize=fontsize)
        if i == 0:
            fig.add_artist(plt.Line2D([0, 1], [y - 0.022, y - 0.022], color="black", lw=0.8))
            y -= 0.035
        else:
            y -= row_h * (max(str(v).count("\n") for v in row) + 1) / 2 + 0.008
    fig.add_artist(plt.Line2D([0, 1], [y + 0.012, y + 0.012], color="black", lw=0.8))
    y -= 0.02
    for k, note in enumerate(notes, start=1):
        wrapped = textwrap.fill(f"{k}. {note}", 125)
        fig.text(0, y, wrapped, fontsize=7.3, va="top", color="#333333")
        y -= 0.0125 * (wrapped.count("\n") + 1) + 0.007
    figures.append((name, fig))


# ---- Page 1: questions and answers ------------------------------------------------------------------
text_page(
    "summary", "Skill and practice signatures across tasks",
    f"{len(summary)} contrasts in 10 datasets prepared by DataPrep; {sum(n_trials.values()):,} trials. Change = B relative to A.",
    [("Prediction tested (SKILL_vs_PRACTICE_explained.pdf, p. 5)", "Measured", "Answer"),
     ("1. Time on task is front-led (10th percentile falls\nmore than the 99th), as practice predicts", f"{q1} of {len(practice)} contrasts", category(q1, len(practice))),
     ("2. Time on task: mu falls and tau does not fall,\nas practice predicts", f"{q2} of {len(practice)} contrasts", category(q2, len(practice))),
     ("3. Word frequency (more stored instances) is\nfront-led", f"F1: {q3}", q3),
     ("4. Expertise is tail-led with mu rising, as skill\n(drive precision) predicts", f"{q4} of {len(skill)} contrasts", {0: "none", 1: "one", 2: "both"}[q4]),
     ("5. Stimulus strength moves mu and tau in the\nsame direction (the document's drive-level axis)", f"{q5} of {len(drive)} contrasts", {0: "none", 1: "one", 2: "both"}[q5])],
    ["Front minus tail = percent change at the 10th percentile minus that at the 99th; front-led: 95% interval below 0; "
     "tail-led: above 0; uniform: includes 0. The document's simulation gives -42 for practice and +50 for skill.",
     "Direction of mu, sigma, tau: 95% interval of the median percent change below 0 (falls), above 0 (rises), or "
     "including 0. Intervals: bootstrap over units and trials (percentiles) or over units (ex-Gaussian), 1,000 replicates, seed 61106.",
     "Trials: DataPrep's usable rows, correct trials where accuracy is recorded, test phase only. Answer categories: none = 0; "
     "most = more than half; a few = between.",
     "Sources: 4_output/contrast_summary.csv, unit_changes.csv, unit_measures.csv."],
    [0, 0.6, 0.86], row_h=0.03)

# ---- Page 2: design -----------------------------------------------------------------------------------
pred_text = {"practice": "front-led; mu, sigma fall; tau flat", "instruction": "benchmark: no prediction", "item practice": "front-led; mu, sigma fall; tau flat",
             "skill": "tail-led; mu rises; tau falls", "drive level": "mu and tau move together"}
design = [("Id", "Dataset (task)", "Axis", "A  vs  B", "Units", "Prediction")]
for s in summary:
    design.append((s["contrast"], f"{s['dataset']}\n({s['task']})", AXIS_LABEL[s["axis"]],
                   textwrap.fill(f"{s['period_A']} vs {s['period_B']}", 28), s["n_units"], textwrap.fill(pred_text[s["axis"]], 22)))
text_page(
    "design", "Step 1. How each dataset tests the hypothesis",
    "Each contrast moves one of the document's axes and holds the task fixed, so each is an independent test.",
    design,
    ["Time on task (P1-P8): each unit's trials in run order, first vs last quarter, within cells that alternate by block "
     "(speed/accuracy instruction, search type, Stroop/Simon). Other conditions are randomised within blocks; their share "
     "differs by at most 1.0 point between quarters (process/output/check_02.txt).",
     "Word frequency (F1): lifetime exposure; instance theory treats frequent words as more stored instances (Logan, 1988).",
     "Expertise (S1, S2): between groups. Action video game players are compared on day 1 only, before the experiment's "
     "training. StarCraft 2 units are games. The document holds that league moves both mechanisms at once.",
     "Stimulus strength (D1, D2): drive level, the document's third axis. D2 uses coherence 0.064 and 0.448 only, because "
     "the meaning of the source's negative coherence values is not documented.",
     "Not included: the Attention Network Task (144 trials per subject); practice in the Roitman monkeys (session order unknown)."],
    [0, 0.06, 0.33, 0.5, 0.72, 0.8], row_h=0.042, fontsize=7.3)

# ---- Figure 1: predicted fingerprints ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
names = ["q10", "q50", "q99", "mu", "sigma", "tau"]
xs = np.arange(len(names))
for dx, ax_name in [(-0.2, "skill"), (0.2, "practice")]:
    vals = [PRED[ax_name][n] for n in names]
    ax.bar(xs + dx, vals, 0.38, color=AXIS_COLOUR[ax_name],
           label={"skill": "Drive precision raised (skill)", "practice": "Instances raised (practice)"}[ax_name])
    for x, v in zip(xs + dx, vals):
        ax.text(x, v + (3 if v >= 0 else -3), f"{v:+d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels(["10th pct", "median", "99th pct", "mu", "sigma", "tau"])
ax.set_ylabel("Change in simulated statistic (%)"); ax.set_ylim(-110, 110)
minor(ax, x=False)
ax.legend(frameon=False, ncol=2, loc="upper right")
fig.tight_layout()
figures.append(("fig1_predictions", fig))
captions["fig1_predictions"] = ("Figure 1. Predictions as reported in SKILL_vs_PRACTICE_explained.pdf (p. 4): percent change in "
    "each statistic of the extended Wong-Wang model's RT distribution when drive precision (skill) or the number of stored "
    "instances (practice) is raised, 20,000 simulated trials each. Not recomputed here.")

# ---- Figure 2: data used ------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
for i, s in enumerate(summary):
    ns = [int(m["n"]) for m in measures if m["contrast"] == s["contrast"]]
    jitter = (np.random.default_rng(i).random(len(ns)) - 0.5) * 0.5
    ax.plot(np.array(ns), i + jitter, "o", ms=1.6, alpha=0.35, color=AXIS_COLOUR[s["axis"]])
    ax.plot(np.median(ns), i, "|", ms=12, mew=1.6, color="black")
    ax.text(1.6e5, i, f"{n_trials[s['contrast']]:,}", va="center", fontsize=7)
ax.set_xscale("log"); ax.set_xlim(10, 1e6)
ax.set_yticks(range(len(summary))); ax.set_yticklabels([label(s) for s in summary], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("Trials per unit, cell and period (log); black bar = median")
ax.text(1.6e5, -1, "total trials", fontsize=7, weight="bold")
ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(2, 10)))
fig.tight_layout()
figures.append(("fig2_data", fig))
captions["fig2_data"] = ("Figure 2. Step 2, the data. Trials behind each percentile estimate: one point per unit, cell and "
    "period. Below about 100 trials, the 99th percentile rests on one or two trials (P6, P8, and many StarCraft 2 games in S1).")

# ---- Figure 3: the method on one unit ---------------------------------------------------------------------
trials = defaultdict(list)
example_unit = sorted([u for u in unit_changes if u["contrast"] == "P3"], key=lambda u: f(u, "q50_change_pct"))[39]["unit"]
for r in csv.DictReader(open("2_data_processing/contrast_rts.csv", newline="")):
    if r["contrast"] == "P3" and r["unit"] == example_unit:
        trials[r["period"]].append(float(r["rt_s"]) * 1000)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
for p, ls, lab in [("A", "-", "first quarter"), ("B", "--", "last quarter")]:
    x = np.sort(trials[p])
    a1.step(x, np.arange(1, len(x) + 1) / len(x), where="post", ls=ls, color=AXIS_COLOUR["practice"], label=f"{lab} (n = {len(x):,})")
qa = np.percentile(trials["A"], PCTS); qb = np.percentile(trials["B"], PCTS)
for q, p in zip(qa, PCTS):
    a1.axhline(p / 100, color="#DDDDDD", lw=0.5, zorder=0)
a1.set_xlim(300, 1600); a1.set_xlabel("RT (ms)"); a1.set_ylabel("Cumulative proportion")
a1.legend(frameon=False, loc="lower right"); minor(a1)
a1.set_title(f"BLP participant {example_unit}: early and late trials", fontsize=8)
a2.plot(PCTS, 100 * (qb / qa - 1), "o-", color=AXIS_COLOUR["practice"])
a2.axhline(0, color="black", lw=0.6)
a2.set_xticks(PCTS); a2.set_xlabel("Percentile"); a2.set_ylabel("Change, last vs first quarter (%)")
a2.set_title("Percent change at each percentile", fontsize=8)
a2.yaxis.set_minor_locator(AutoMinorLocator())
fig.tight_layout()
figures.append(("fig3_method", fig))
captions["fig3_method"] = ("Figure 3. Step 3, the method, on one unit: the participant with the median change in median RT "
    "in the British Lexicon Project (P3). Left: distribution of correct RTs in the first and last quarter of trials. Right: "
    "percent change at each percentile. Practice predicts the largest fall at low percentiles and none at the 99th.")

# ---- Figure 4: change profiles ------------------------------------------------------------------------------
fig, axes = plt.subplots(5, 4, figsize=(WIDTH, 8.6), sharex=True)
axes = axes.ravel()
for ax, s in zip(axes, summary):
    y = np.array([f(s, f"q{p:02d}_change_pct") for p in PCTS])
    lo = np.array([f(s, f"q{p:02d}_low") for p in PCTS]); hi = np.array([f(s, f"q{p:02d}_high") for p in PCTS])
    ax.fill_between(range(7), lo, hi, color=AXIS_COLOUR[s["axis"]], alpha=0.25, lw=0)
    ax.plot(range(7), y, "o-", ms=2.5, color=AXIS_COLOUR[s["axis"]])
    pred_axis = {"practice": "practice", "item practice": "practice", "skill": "skill"}.get(s["axis"])
    if pred_axis:
        ax.plot([1, 3, 6], [PRED[pred_axis][k] for k in ["q10", "q50", "q99"]], "x", color="black", ms=4, mew=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{s['contrast']} {textwrap.shorten(s["task"], 32)}\n{s['location']}", fontsize=7)
    ax.set_xticks(range(7)); ax.set_xticklabels(PCTS, fontsize=6.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelsize=6.5)
for ax in axes[len(summary):]:
    ax.axis("off")
handles = [Rectangle((0, 0), 1, 1, color=AXIS_COLOUR[a]) for a in AXES] + [plt.Line2D([], [], marker="x", ls="", color="black")]
axes[len(summary)].legend(handles, [AXIS_LABEL[a] for a in AXES] + ["document's prediction"], frameon=False, loc="center", fontsize=7)
fig.supxlabel("Percentile of the RT distribution", fontsize=8)
fig.supylabel("Change, B relative to A (%), 95% interval", fontsize=8)
fig.tight_layout()
figures.append(("fig4_profiles", fig))
captions["fig4_profiles"] = ("Figure 4. Step 4, results. Percent change at each percentile for every contrast, with 95% "
    "bootstrap intervals (bands); the label under each title is the classification of front minus tail. Crosses: the "
    "document's simulated change at the 10th, 50th and 99th percentiles for the mechanism the contrast moves. Each panel has its own y scale.")

# ---- Figure 5: front minus tail -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
for i, s in enumerate(summary):
    v, lo, hi = f(s, "front_minus_tail"), f(s, "front_minus_tail_low"), f(s, "front_minus_tail_high")
    ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", ms=5, color=AXIS_COLOUR[s["axis"]], elinewidth=1.2, capsize=2)
ax.axvline(0, color="black", lw=0.6)
for val, name in [(-42, "practice"), (50, "skill")]:
    ax.axvline(val, color=AXIS_COLOUR[name], ls="--", lw=1)
    ax.text(val, -1.1, f"document's {name}\nprediction ({val:+d})", ha="center", fontsize=6.5, color=AXIS_COLOUR[name])
ax.set_yticks(range(len(summary))); ax.set_yticklabels([f"{label(s)}  [{s['location']}]" for s in summary], fontsize=7.5)
ax.invert_yaxis(); ax.set_ylim(len(summary) - 0.5, -1.6)
ax.set_xlabel("Change at 10th minus change at 99th percentile\n(percentage points), 95% interval")
minor(ax, y=False)
fig.tight_layout()
figures.append(("fig5_front_minus_tail", fig))
captions["fig5_front_minus_tail"] = ("Figure 5. Where the change lives, one row per contrast. Below 0: the fast front changed "
    "more than the slow tail (front-led, the practice signature); above 0: the tail changed more (tail-led, the skill "
    "signature). Dashed: the document's simulated values. Colour: axis, as in Figure 4.")

# ---- Figure 6: ex-Gaussian parameters -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 4.0), sharey=True)
for ax, p in zip(axes, ["mu", "sigma", "tau"]):
    for i, s in enumerate(summary):
        v, lo, hi = f(s, f"{p}_change_pct"), f(s, f"{p}_low"), f(s, f"{p}_high")
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", ms=4, color=AXIS_COLOUR[s["axis"]], elinewidth=1, capsize=1.5)
        pred_axis = {"practice": "practice", "item practice": "practice", "skill": "skill"}.get(s["axis"])
        if pred_axis:
            sign = np.sign(PRED[pred_axis][p]) if abs(PRED[pred_axis][p]) > 5 else 0
            ax.text(1.02, i, {1: "▲", -1: "▼", 0: "●"}[sign], transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7, color=AXIS_COLOUR[s["axis"]])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(p); ax.set_xlabel("Change (%), 95% interval")
    minor(ax, y=False)
axes[0].set_yticks(range(len(summary))); axes[0].set_yticklabels([label(s) for s in summary], fontsize=7.5)
axes[0].invert_yaxis()
fig.tight_layout()
figures.append(("fig6_exgaussian", fig))
captions["fig6_exgaussian"] = ("Figure 6. Median percent change in ex-Gaussian mu, sigma and tau, with 95% intervals over "
    "units. Symbols right of each panel: the direction the document predicts (▲ rises, ▼ falls, ● no change of more than 5%). With 2 or 3 units (D1, P1) the interval over units is not informative, and where a unit has no defined change (D1 tau, monkey 2) none is drawn.")

# ---- Figure 7: per-unit front against tail -------------------------------------------------------------------
within = [s for s in summary if not s["contrast"].startswith("S")]
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 7.2))
axes = axes.ravel()
for ax, s in zip(axes, within):
    us = [u for u in unit_changes if u["contrast"] == s["contrast"]]
    x = np.array([f(u, "q10_change_pct") for u in us]); y = np.array([f(u, "q99_change_pct") for u in us])
    ax.plot(x, y, "o", ms=2.5 if len(us) > 50 else 4, alpha=0.6, color=AXIS_COLOUR[s["axis"]], mec="none")
    lim = np.percentile(np.abs(np.r_[x, y]), 98) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.6, ls=":")
    ax.axhline(0, color="black", lw=0.4); ax.axvline(0, color="black", lw=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    below = np.sum(x < y)
    ax.set_title(f"{s['contrast']}, n = {len(us)}\nabove diagonal: {below}", fontsize=6.8)
    ax.tick_params(labelsize=6.5); minor(ax)
for ax in axes[len(within):]:
    ax.axis("off")
fig.supxlabel("Change at 10th percentile (%)", fontsize=8)
fig.supylabel("Change at 99th percentile (%)", fontsize=8)
fig.tight_layout()
figures.append(("fig7_units", fig))
captions["fig7_units"] = ("Figure 7. Each unit's change at the 10th against the 99th percentile, within-unit contrasts. "
    "Above the dotted diagonal (count in each title), the 10th percentile changed more (more negative) than the 99th. Axes are symmetric and set per panel; "
    "the most extreme 2% of values may fall outside.")

# ---- Figure 8: checks -----------------------------------------------------------------------------------------
fig, a2 = plt.subplots(figsize=(WIDTH, 2.9))
low_sigma = defaultdict(int); total = defaultdict(int)
for m in measures:
    total[m["contrast"]] += 1
    low_sigma[m["contrast"]] += float(m["sigma_ms"]) < 2
ids = [s["contrast"] for s in summary]
a2.barh(range(len(ids)), [100 * low_sigma[c] / total[c] for c in ids], color=[AXIS_COLOUR[s["axis"]] for s in summary])
a2.set_yticks(range(len(ids))); a2.set_yticklabels(ids, fontsize=7); a2.invert_yaxis()
a2.set_xlabel("% of ex-Gaussian fits with sigma < 2 ms")
a2.set_title("Ex-Gaussian fits with sigma < 2 ms", fontsize=8)
a2.xaxis.set_minor_locator(AutoMinorLocator())
fig.tight_layout()
figures.append(("fig8_checks", fig))
captions["fig8_checks"] = ("Figure 8. Share of ex-Gaussian fits whose sigma is under 2 ms, where mu sits at the fastest RTs "
    "and is not a separate location; these fits are kept. Checks run separately: pooled conditions differ by at most 1.0 point in "
    "share between first and last quarter (process/output/check_02.txt); fit recovery on simulated data (check_01.txt).")

with PdfPages(out / "SkillPractice-AcrossTasks_report.pdf") as pdf:
    for name, fig in figures:
        fig.savefig(out / "figures" / f"{name}.png", dpi=300 if name in captions else 150)
        if name in captions:
            # The caption goes under the figure on a page of its own; the PNG keeps no caption.
            h = fig.get_figheight()
            wrapped = textwrap.fill(captions[name], 118)
            extra = 0.25 + 0.16 * (wrapped.count("\n") + 1)
            page = plt.figure(figsize=(WIDTH, h + extra))
            ax = page.add_axes((0, extra / (h + extra), 1, h / (h + extra)))
            ax.imshow(plt.imread(out / "figures" / f"{name}.png")); ax.axis("off")
            page.text(0, (extra - 0.1) / (h + extra), wrapped, fontsize=7.8, va="top")
            pdf.savefig(page, dpi=300)
            plt.close(page)
        else:
            pdf.savefig(fig)
        plt.close(fig)
print("wrote 4_output/SkillPractice-AcrossTasks_report.pdf")
