"""
14_make_accuracy_report.py

Purpose
    Write the report of the accuracy-aware mixture fits: whether skill and practice, in any
    combination, produce each contrast's change in correct RTs, accuracy and error RTs, and which
    addition (common drive, evidence, the top-down control signal of Lo, Wang & Wang 2015, or
    non-decision time) is needed where they do not.

Inputs
    2_data_processing/contrasts.csv; 4_output/pooled_accuracy_profiles.csv, acc_mixture_fits.csv,
    acc_mixture_summary.csv, acc_ingredient_steps.csv, acc_mixtures/<contrast>.json

Outputs
    4_output/SkillPractice-Accuracy_report.pdf; 4_output/figures/acc_*.png (300 dpi)

Depends on
    13_collect_accuracy_mixtures.py

Last substantive change
    2026-11-07 — first version.

Run from the project root.
"""
import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, with no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

# Arial has Helvetica's letter widths and is the Helvetica substitute installed on this machine.
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 6.5
out = Path("4_output")
(out / "figures").mkdir(exist_ok=True)

AXIS_COLOUR = {"practice": "#E69F00", "instruction": "#CC79A7", "item practice": "#D55E00", "skill": "#0072B2", "drive level": "#009E73"}
AXIS_LABEL = {"practice": "Time on task", "instruction": "Speed vs accuracy instruction", "item practice": "Word frequency",
              "skill": "Expertise", "drive level": "Stimulus strength"}
MODELS = ["S", "P", "SP", "SP+drive", "SP+evidence", "SP+control", "SP+ndt", "SP+control+ndt", "SP+drive+ndt"]
LABEL = {"S": "skill only", "P": "practice only", "SP": "skill + practice", "SP+drive": "+ common drive", "SP+evidence": "+ evidence",
         "SP+control": "+ control signal", "SP+ndt": "+ non-decision time", "SP+control+ndt": "+ control + non-decision",
         "SP+drive+ndt": "+ drive + non-decision"}
SHORT = {"S": "skill", "P": "practice", "SP": "SP", "SP+drive": "SP + drive", "SP+evidence": "SP + evidence", "SP+control": "SP + control",
         "SP+ndt": "SP + ndt", "SP+control+ndt": "SP + control + ndt", "SP+drive+ndt": "SP + drive + ndt"}
COL = {"S": "#0072B2", "P": "#E69F00", "SP": "#000000", "SP+drive": "#009E73", "SP+evidence": "#56B4E9", "SP+control": "#D55E00",
       "SP+ndt": "#CC79A7", "SP+control+ndt": "#882255", "SP+drive+ndt": "#117733"}
ING_COL = {"skill (u0_sd x 0.5)": "#0072B2", "practice (logN + 2)": "#E69F00", "drive (u0, u0_sd x 1.3)": "#009E73",
           "evidence (+4 Hz)": "#56B4E9", "control, excitatory (S 0.5, R 1.05)": "#F0A07A", "control, inhibitory (S 0.5, R 1.3)": "#D55E00",
           "non-decision (-50 ms)": "#CC79A7"}
PCTS = [5, 10, 25, 50, 75, 90, 99]

contrasts = list(csv.DictReader(open("2_data_processing/contrasts.csv")))
cids = [c["contrast"] for c in contrasts]
info = {c["contrast"]: c for c in contrasts}
prof = list(csv.DictReader(open(out / "pooled_accuracy_profiles.csv")))
fits = {(f["contrast"], f["model"]): f for f in csv.DictReader(open(out / "acc_mixture_fits.csv"))}
summ = {s["contrast"]: s for s in csv.DictReader(open(out / "acc_mixture_summary.csv"))}
steps = list(csv.DictReader(open(out / "acc_ingredient_steps.csv")))
res = {c: json.loads((out / "acc_mixtures" / f"{c}.json").read_text()) for c in cids}


def obs(c, measure):
    rows = [r for r in prof if r["contrast"] == c and r["measure"] == measure]
    return np.array([float(r["change"]) for r in rows]), np.array([float(r["se"]) for r in rows])


def title(c):
    return f"{c} {textwrap.shorten(info[c]['task'], 26, placeholder='…')}"


def minor(ax, x=True, y=True):
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def category(k, n):
    return "none" if k == 0 else "most" if k > n / 2 else "a few"


figures, captions = [], {}

# ---- Summary ----------------------------------------------------------------------------------------
q_sp = sum(summ[c]["sp_adequate"] == "True" for c in cids)
fails = [c for c in cids if summ[c]["sp_adequate"] != "True"]
best = Counter(summ[c]["best_single_addition"] for c in fails)
adequate_each = {m: sum(fits[(c, m)]["adequate"] == "True" for c in fails) for m in ["SP+drive", "SP+evidence", "SP+control", "SP+ndt", "SP+control+ndt", "SP+drive+ndt"]}
with_acc = [c for c in cids if summ[c]["has_accuracy"] == "True"]
fig = plt.figure(figsize=(WIDTH, 9.3))
fig.text(0, 0.985, "Skill, practice and what else: fits with accuracy", fontsize=13, weight="bold", va="top")
fig.text(0, 0.957, f"{len(cids)} contrasts; {len(with_acc)} with accuracy and error RTs. Model: extended Wong-Wang with evidence, "
         "choices and the control signal of Lo, Wang & Wang (2015).", fontsize=8.2, va="top", color="#444444")
rows = [("Question", "Measured", ""),
        ("Skill + practice produce correct RTs, accuracy\nand error RTs adequately (p > .05)", f"{q_sp} of {len(cids)} contrasts", category(q_sp, len(cids))),
        ("Where they do not: lowest-AIC adequate single\naddition", "; ".join(f"{SHORT[k].replace('SP + ', '')}: {v}" if k in SHORT else f"{k}: {v}" for k, v in sorted(best.items())), ""),
        ("Where they do not: adequate with each addition", "; ".join(f"{SHORT[m].replace('SP + ', '')}: {v}" for m, v in adequate_each.items()), f"of {len(fails)}")]
y = 0.915
fig.add_artist(plt.Line2D([0, 1], [y + 0.012, y + 0.012], color="black", lw=0.8))
for i, row in enumerate(rows):
    for cx, v in zip([0, 0.47, 0.9], row):
        fig.text(cx, y, textwrap.fill(v, 60) if cx == 0.47 else v, va="top", fontsize=8.0, weight="bold" if i == 0 else "normal")
    if i == 0:
        fig.add_artist(plt.Line2D([0, 1], [y - 0.022, y - 0.022], color="black", lw=0.8))
    y -= 0.035 if i == 0 else 0.052
fig.add_artist(plt.Line2D([0, 1], [y + 0.018, y + 0.018], color="black", lw=0.8))
y -= 0.01
fig.text(0, y, "Per contrast", weight="bold", va="top")
y -= 0.028
heads = ["Id", "Task", "Change", "Accuracy A → B", "Skill + practice", "Lowest AIC"]
xs = [0, 0.06, 0.33, 0.52, 0.68, 0.84]
for cx, h in zip(xs, heads):
    fig.text(cx, y, h, weight="bold", fontsize=7.3, va="top")
fig.add_artist(plt.Line2D([0, 1], [y - 0.016, y - 0.016], color="black", lw=0.6))
y -= 0.026
for c in cids:
    s = summ[c]
    acc = [r for r in prof if r["contrast"] == c and r["measure"] == "accuracy"]
    acc_txt = f"{float(acc[0]['value_A']):.3f} → {float(acc[0]['value_B']):.3f}" if acc else "not observed"
    verdict = s["needed"] if s["sp_adequate"] == "True" else f"no; {SHORT.get(s['best_single_addition'], s['best_single_addition']).replace('SP + ', '+ ')}"
    vals = [c, textwrap.shorten(info[c]["task"], 26, placeholder="…"), AXIS_LABEL[info[c]["axis"]].replace("Speed vs accuracy instruction", "Instruction"),
            acc_txt, verdict, SHORT[s["lowest_aic_model"]]]
    for cx, v in zip(xs, vals):
        fig.text(cx, y, v, fontsize=7.0, va="top")
    y -= 0.0185
fig.add_artist(plt.Line2D([0, 1], [y + 0.004, y + 0.004], color="black", lw=0.6))
notes = ["Skill = drive spread u0_sd (u0 fixed); practice = stored instances logN; common drive = u0 and u0_sd scaled together; "
         "evidence = difference between the pools' drive (Hz); control signal = balanced synaptic input, strength S and ratio R, from the "
         "f-I table derived from the paper's neuron; non-decision time = ter_mu.",
         "Each contrast's A is calibrated to its pooled correct RTs and accuracy, with evidence and the instance route's noise free. "
         "Chi-square over correct-RT percentiles, accuracy change and error-RT percentiles (where A and B each have >= 100 errors), "
         "with data and simulation error; AIC = chi-square + 2 x changes.",
         "No accuracy: P3 (BLP errors have no RT in DataPrep) and S1 (StarCraft 2); these are fitted on correct RTs only.",
         "Sources: 4_output/acc_mixture_summary.csv, acc_mixture_fits.csv, pooled_accuracy_profiles.csv."]
y -= 0.015
for k, n in enumerate(notes, 1):
    wrapped = textwrap.fill(f"{k}. {n}", 128)
    fig.text(0, y, wrapped, fontsize=7, va="top", color="#333333")
    y -= 0.0115 * (wrapped.count("\n") + 1) + 0.006
figures.append(("acc_summary", fig))

# ---- Figure 1: observed accuracy and RT changes -------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 4.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
ys = np.arange(len(with_acc))
for i, c in enumerate(with_acc):
    v, se = obs(c, "accuracy")
    a1.errorbar(100 * v[0], i, xerr=196 * se[0], fmt="o", color=AXIS_COLOUR[info[c]["axis"]], ms=5, capsize=2)
a1.axvline(0, color="black", lw=0.5)
a1.set_yticks(ys); a1.set_yticklabels([title(c) for c in with_acc], fontsize=7); a1.invert_yaxis()
a1.set_xlabel("Change in accuracy, B - A (points), 95% interval"); minor(a1, y=False)
for c in with_acc:
    q, _ = obs(c, "correct_rt_pct"); v, _ = obs(c, "accuracy")
    a2.plot(q[3], 100 * v[0], "o", ms=6, color=AXIS_COLOUR[info[c]["axis"]])
    a2.annotate(c, (q[3], 100 * v[0]), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
a2.axhline(0, color="black", lw=0.5); a2.axvline(0, color="black", lw=0.5)
a2.set_xlabel("Change in median correct RT (%)"); a2.set_ylabel("Change in accuracy (points)"); minor(a2)
handles = [Rectangle((0, 0), 1, 1, color=v) for v in AXIS_COLOUR.values()]
a2.legend(handles, [AXIS_LABEL[k] for k in AXIS_COLOUR], frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout()
figures.append(("acc_fig1_observed", fig))
captions["acc_fig1_observed"] = ("Figure 1. The data. Left: change in accuracy from A to B, pooled over units, with 95% "
    "cluster-bootstrap intervals. Right: change in accuracy against change in median correct RT. Upper left = faster and more "
    "accurate; lower left = faster and less accurate; upper right = slower and more accurate.")

# ---- Figure 2: what each ingredient does to speed and accuracy --------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.6))
for ing, col in ING_COL.items():
    pts = [(float(s["median_change_pct"]), 100 * float(s["accuracy_change"])) for s in steps if s["ingredient"] == ing and s["contrast"] in with_acc]
    pts = np.array(pts)
    a1.scatter(pts[:, 0], pts[:, 1], s=8, color=col, alpha=0.5)
    a1.plot(np.median(pts[:, 0]), np.median(pts[:, 1]), "o", ms=8, color=col, mec="black", label=ing)
    q99 = [float(s["q99_change_pct"]) for s in steps if s["ingredient"] == ing]
    med = [float(s["median_change_pct"]) for s in steps if s["ingredient"] == ing]
    a2.plot(np.median(med), np.median(q99), "o", ms=8, color=col, mec="black")
    a2.plot(med, q99, ".", ms=3, color=col, alpha=0.5)
for ax in (a1, a2):
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5); minor(ax)
a1.set_xlabel("Change in median correct RT (%)"); a1.set_ylabel("Change in accuracy (points)")
a2.set_xlabel("Change in median correct RT (%)"); a2.set_ylabel("Change in 99th percentile (%)")
a1.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3)
fig.tight_layout(rect=(0, 0.12, 1, 1))
figures.append(("acc_fig2_ingredients", fig))
captions["acc_fig2_ingredients"] = ("Figure 2. What each ingredient does. One step of each change from every contrast's "
    "calibrated A (small points; large = median over contrasts). Left: speed against accuracy, where ingredients that make the same "
    "RT change separate by their effect on accuracy. Right: the median against the 99th percentile.")

# ---- Figure 3: fits ---------------------------------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 7.4), sharex=True)
for ax, c in zip(axes.ravel(), cids):
    q, se = obs(c, "correct_rt_pct")
    ax.fill_between(range(7), q - 1.96 * se, q + 1.96 * se, color="#DDDDDD", lw=0)
    ax.plot(range(7), q, "o", ms=3, color="black")
    best_add = min(["SP+drive", "SP+evidence", "SP+control", "SP+ndt"], key=lambda m: float(fits[(c, m)]["aic"]))
    for m, ls in [("SP", "-"), (best_add, "-")]:
        ax.plot(range(7), res[c]["fits"][m]["predicted_correct_pct"], ls, lw=1.1, color=COL[m])
    acc_txt = ""
    if summ[c]["has_accuracy"] == "True":
        v, _ = obs(c, "accuracy")
        acc_txt = (f"\nacc {100 * v[0]:+.1f}; SP {100 * res[c]['fits']['SP']['predicted_accuracy_change']:+.1f}; "
                   f"{SHORT[best_add].replace('SP + ', '')} {100 * res[c]['fits'][best_add]['predicted_accuracy_change']:+.1f}")
    ax.set_title(f"{title(c)}\nSP p = {float(fits[(c, 'SP')]['p']):.2g}; {SHORT[best_add].replace('SP + ', '+')} p = {float(fits[(c, best_add)]['p']):.2g}{acc_txt}",
                 fontsize=5.8)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(range(7)); ax.set_xticklabels(PCTS, fontsize=6); ax.tick_params(labelsize=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
h = [plt.Line2D([], [], marker="o", ls="", color="black")] + [plt.Line2D([], [], color=COL[m]) for m in ["SP", "SP+drive", "SP+evidence", "SP+control", "SP+ndt"]]
fig.legend(h, ["observed"] + [LABEL[m] for m in ["SP", "SP+drive", "SP+evidence", "SP+control", "SP+ndt"]], loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.supxlabel("Percentile of correct RTs", fontsize=8, y=0.065); fig.supylabel("Change, B relative to A (%)", fontsize=8)
fig.tight_layout(rect=(0, 0.075, 1, 1))
figures.append(("acc_fig3_fits", fig))
captions["acc_fig3_fits"] = ("Figure 3. The fits. Observed change in correct-RT percentiles (points; band = 95% interval), "
    "skill + practice (black) and the single addition with the lowest AIC (coloured). Titles: chi-square p of each, and the observed "
    "and predicted change in accuracy (points).")

# ---- Figure 4: adequacy matrix ------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 4.8))
P = np.array([[float(fits[(c, m)]["p"]) for m in MODELS] for c in cids])
L = -np.log10(np.clip(np.nan_to_num(P, nan=1e-12), 1e-12, 1))
ax.imshow(L, cmap="Greys", aspect="auto", vmin=0, vmax=12)
for i in range(len(cids)):
    for j in range(len(MODELS)):
        ok = P[i, j] > 0.05
        ax.text(j, i, "ok" if ok else (f"{P[i, j]:.0e}" if P[i, j] > 1e-12 else "<1e-12"), ha="center", va="center",
                fontsize=6.3 if not ok else 8, color="white" if L[i, j] > 6 else "black", weight="bold" if ok else "normal")
        if ok:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#009E73", lw=1.5))
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=30, ha="right", fontsize=7)
ax.set_yticks(range(len(cids))); ax.set_yticklabels([title(c) for c in cids], fontsize=7)
fig.tight_layout()
figures.append(("acc_fig4_adequacy", fig))
captions["acc_fig4_adequacy"] = ("Figure 4. Which models are adequate (ok in a green box: chi-square p > .05). Shading: -log10 p, "
    "capped at 12. Contrasts with accuracy are judged on correct RTs, accuracy and error RTs together.")

# ---- Figure 5: accuracy predicted vs observed -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
for k, m in enumerate(["SP", "SP+drive", "SP+evidence", "SP+control", "SP+ndt"]):
    for i, c in enumerate(with_acc):
        v, _ = obs(c, "accuracy")
        pr = res[c]["fits"][m]["predicted_accuracy_change"]
        ax.plot(i + (k - 2) * 0.13, 100 * (pr - v[0]), "o", ms=4, color=COL[m], label=LABEL[m] if i == 0 else None)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(range(len(with_acc))); ax.set_xticklabels(with_acc)
ax.set_ylabel("Predicted minus observed\nchange in accuracy (points)"); minor(ax, x=False)
ax.legend(frameon=False, ncol=3, fontsize=6.8, loc="best")
fig.tight_layout()
figures.append(("acc_fig5_accuracy_errors", fig))
captions["acc_fig5_accuracy_errors"] = ("Figure 5. How well each fit reproduces the change in accuracy: predicted minus observed, "
    "in percentage points, for skill + practice and each single addition.")

# ---- Figure 6: the control signal fitted --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
for c in cids:
    f = fits[(c, "SP+control")]
    ok = f["adequate"] == "True"
    ax.plot(float(f["change_cR"]), float(f["change_cS"]), "o" if ok else "X", ms=8, color=AXIS_COLOUR[info[c]["axis"]], mec="black" if ok else "none")
    ax.annotate(c, (float(f["change_cR"]), float(f["change_cS"])), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.axvspan(1.156, 1.2, color="#EEEEEE", zorder=0)
ax.text(1.178, 2.05, "sign change\n(validation)", ha="center", fontsize=6.5)
ax.set_xlabel("Control ratio R (inhibition / excitation)"); ax.set_ylabel("Control strength S")
ax.set_xlim(0.98, 1.42); ax.set_ylim(-0.05, 2.25); minor(ax)
fig.tight_layout()
figures.append(("acc_fig6_control", fig))
captions["acc_fig6_control"] = ("Figure 6. The control signal in the skill + practice + control fit of each contrast. Left of the "
    "grey band the signal speeds decisions (net excitation); right of it, slows them (net inhibition) (process/output/check_06.txt). "
    "Circle = adequate fit; cross = not. Colours: kind of contrast, as in Figure 1.")

# ---- Figure 7: AIC improvement -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
xs = np.arange(len(cids))
adds = ["SP+drive", "SP+evidence", "SP+control", "SP+ndt"]
for k, m in enumerate(adds):
    gain = [float(fits[(c, "SP")]["aic"]) - float(fits[(c, m)]["aic"]) for c in cids]
    ax.bar(xs + (k - 1.5) * 0.2, np.clip(gain, 0.1, None), 0.2, color=COL[m], label=LABEL[m])
ax.set_yscale("log"); ax.axhline(2, color="black", ls="--", lw=0.7)
ax.set_xticks(xs); ax.set_xticklabels(cids)
ax.set_ylabel("AIC of skill + practice minus AIC\nwith the addition (log; < 0.1 drawn at 0.1)")
ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper left")
fig.tight_layout()
figures.append(("acc_fig7_aic", fig))
captions["acc_fig7_aic"] = ("Figure 7. How much each single addition improves on skill + practice (drop in AIC; dashed line = 2).")

with PdfPages(out / "SkillPractice-Accuracy_report.pdf") as pdf:
    for name, fig in figures:
        fig.savefig(out / "figures" / f"{name}.png", dpi=300 if name in captions else 150)
        if name in captions:
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
print("wrote 4_output/SkillPractice-Accuracy_report.pdf")
