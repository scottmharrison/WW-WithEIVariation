"""
09_make_mixture_report.py

Purpose
    Write the report on whether skill (drive precision) and practice (stored instances), in any
    combination, can produce each contrast's observed change in the extended Wong-Wang model, and what
    single addition is needed where they cannot. Every dataset is shown, on pooled trials.

Inputs
    2_data_processing/contrasts.csv, baselines.csv; 4_output/pooled_profiles.csv, mixture_fits.csv,
    mixture_summary.csv, mechanism_steps.csv, mixtures/<contrast>.json

Outputs
    4_output/SkillPractice-Mixtures_report.pdf; 4_output/figures/mix_*.png (300 dpi)

Depends on
    08_collect_mixtures.py

Last substantive change
    2026-11-06 — first version.

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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

# Arial has Helvetica's letter widths and is the Helvetica substitute installed on this machine.
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 6.5  # inches, the APA maximum
out = Path("4_output")
(out / "figures").mkdir(exist_ok=True)

# The same colour means the same thing in every figure (Okabe-Ito colour-blind-safe palette).
AXIS_COLOUR = {"practice": "#E69F00", "instruction": "#CC79A7", "item practice": "#D55E00", "skill": "#0072B2", "drive level": "#009E73"}
AXIS_LABEL = {"practice": "Time on task", "instruction": "Speed vs accuracy instruction", "item practice": "Word frequency",
              "skill": "Expertise", "drive level": "Stimulus strength"}
MODEL_COLOUR = {"S": "#0072B2", "P": "#E69F00", "SP": "#000000", "SP+drive": "#009E73", "SP+ndt": "#CC79A7", "SP+retrieval": "#D55E00",
                "SP+drive+ndt": "#56B4E9"}
MODEL_LABEL = {"S": "skill only", "P": "practice only", "SP": "skill + practice", "SP+drive": "+ drive level",
               "SP+ndt": "+ non-decision time", "SP+retrieval": "+ retrieval speed",
               "SP+drive+ndt": "+ drive level + non-decision time"}
MODELS = list(MODEL_LABEL)
PCTS = [5, 10, 25, 50, 75, 90, 99]

contrasts = list(csv.DictReader(open("2_data_processing/contrasts.csv")))
cids = [c["contrast"] for c in contrasts]
info = {c["contrast"]: c for c in contrasts}
base = {b["contrast"]: b for b in csv.DictReader(open("2_data_processing/baselines.csv"))}
prof = list(csv.DictReader(open(out / "pooled_profiles.csv")))
fits = list(csv.DictReader(open(out / "mixture_fits.csv")))
summ = {s["contrast"]: s for s in csv.DictReader(open(out / "mixture_summary.csv"))}
steps = list(csv.DictReader(open(out / "mechanism_steps.csv")))
res = {c: json.loads((out / "mixtures" / f"{c}.json").read_text()) for c in cids}
fit = {(f["contrast"], f["model"]): f for f in fits}


def profile(c, key):
    return np.array([float(r[key]) for r in prof if r["contrast"] == c])


def minor(ax, x=True, y=True):
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def title(c):
    return f"{c} {textwrap.shorten(info[c]['task'], 26, placeholder='…')}"


def category(k, n):
    return "none" if k == 0 else "most" if k > n / 2 else "a few"


figures, captions = [], {}

# ---- Answers ----------------------------------------------------------------------------------------
q6 = sum(summ[c]["sp_adequate"] == "True" for c in cids)
needed = Counter(summ[c]["needed"] for c in cids if summ[c]["sp_adequate"] == "True")
fails = [c for c in cids if summ[c]["sp_adequate"] != "True"]
adds = Counter(summ[c]["best_addition"] for c in fails)
bench = [c for c in cids if info[c]["axis"] == "instruction"]
q9 = sum(summ[c]["sp_adequate"] != "True" for c in bench)
add_adequate = {m: sum(fit[(c, m)]["adequate"] == "True" for c in fails) for m in ("SP+drive", "SP+ndt", "SP+retrieval")}

fig = plt.figure(figsize=(WIDTH, 9.3))
fig.text(0, 0.985, "Can skill and practice, mixed, produce each dataset's change?", fontsize=13, weight="bold", va="top")
fig.text(0, 0.957, f"{len(cids)} contrasts in 11 datasets, trials pooled over units; the extended Wong-Wang model calibrated to A, "
         "then changed to fit B.", fontsize=8.3, va="top", color="#444444")
rows = [("Question", "Measured", "Answer"),
        ("6. Skill + practice produce the observed\nchange adequately (p > .05)", f"{q6} of {len(cids)} contrasts", category(q6, len(cids))),
        ("7. Where they do, what is needed", "; ".join(f"{k}: {v}" for k, v in sorted(needed.items())) or "-", "per contrast\n(Figure 7)"),
        ("8. Where they do not, the single addition\nthat makes the fit adequate (lowest AIC)",
         "; ".join(f"{k.replace('SP+', '')}: {v}" for k, v in sorted(adds.items())) or "-",
         max(adds, key=adds.get).replace("SP+", "") if adds else "-"),
        ("   adequate with each addition (of the failures)", "; ".join(f"{k.replace('SP+', '')}: {v} of {len(fails)}" for k, v in add_adequate.items()), ""),
        ("   adequate with drive level and non-decision\n   time both added (of the failures)",
         f"{sum(summ[c]['two_additions_adequate'] == 'True' for c in fails)} of {len(fails)}", ""),
        ("9. Speed-accuracy benchmarks need more\nthan skill + practice", f"{q9} of {len(bench)}", {0: "neither", 1: "one", 2: "both"}[q9])]
y = 0.915
fig.add_artist(plt.Line2D([0, 1], [y + 0.012, y + 0.012], color="black", lw=0.8))
for i, row in enumerate(rows):
    for cx, v in zip([0, 0.47, 0.86], row):
        fig.text(cx, y, v, va="top", fontsize=8.2, weight="bold" if i == 0 else "normal")
    if i == 0:
        fig.add_artist(plt.Line2D([0, 1], [y - 0.022, y - 0.022], color="black", lw=0.8))
    y -= 0.035 if i == 0 else 0.048
fig.add_artist(plt.Line2D([0, 1], [y + 0.018, y + 0.018], color="black", lw=0.8))

# Per-contrast verdict table.
y -= 0.01
fig.text(0, y, "Per contrast", weight="bold", va="top")
y -= 0.028
heads = ["Id", "Task", "Change", "Instance share\nin A", "Skill + practice", "Lowest AIC"]
SHORT = {"S": "skill", "P": "practice", "SP": "skill + practice", "SP+drive": "SP + drive", "SP+ndt": "SP + ndt",
         "SP+retrieval": "SP + retrieval", "SP+drive+ndt": "SP + drive + ndt"}
xs = [0, 0.06, 0.37, 0.55, 0.68, 0.86]
for cx, h in zip(xs, heads):
    fig.text(cx, y, h, weight="bold", fontsize=7.3, va="top")
fig.add_artist(plt.Line2D([0, 1], [y - 0.026, y - 0.026], color="black", lw=0.6))
y -= 0.036
for c in cids:
    s = summ[c]
    verdict = s["needed"] if s["sp_adequate"] == "True" else f"no; {s['best_addition'].replace('SP+', '+ ')}"
    flag = "†" if summ[c]["few_units"] else ""
    vals = [c + flag, textwrap.shorten(info[c]["task"], 30, placeholder="…"), AXIS_LABEL[info[c]["axis"]].replace("Speed vs accuracy instruction", "Instruction"),
            f"{100 * float(s['instance_share_A']):.0f}%", verdict, SHORT[s["lowest_aic_model"]]]
    for cx, v in zip(xs, vals):
        fig.text(cx, y, v, fontsize=7.2, va="top")
    y -= 0.0185
fig.add_artist(plt.Line2D([0, 1], [y + 0.004, y + 0.004], color="black", lw=0.6))
notes = ["Skill = drive spread u0_sd changed with mean drive fixed; practice = stored instances logN changed; drive level = "
         "u0 and u0_sd scaled together; non-decision time = ter_mu shifted; retrieval speed = retrieval onset scale s_W.",
         "Adequate: chi-square over the 5th to 99th percentiles' percent change, with data (bootstrap over units) and simulation "
         "error, p > .05. AIC = chi-square + 2 x changed parameters.",
         "Instance share: % of A's simulated trials won by the retrieval route; practice can act only through these trials.",
         "† Two or three units (Ratcliff, three participants; Roitman, two monkeys): the bootstrap over units gives wide "
         "intervals, so the adequacy test is weak there.",
         "Sources: 4_output/mixture_summary.csv, mixture_fits.csv, pooled_profiles.csv. Model: WW-RTShapeAnatomy/model/wwsim.py."]
y -= 0.015
for k, n in enumerate(notes, 1):
    wrapped = textwrap.fill(f"{k}. {n}", 128)
    fig.text(0, y, wrapped, fontsize=7, va="top", color="#333333")
    y -= 0.0115 * (wrapped.count("\n") + 1) + 0.006
figures.append(("mix_summary", fig))

# ---- Figure 1: observed pooled profiles ----------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 7.0), sharex=True)
for ax, c in zip(axes.ravel(), cids):
    v, lo, hi = profile(c, "change_pct"), profile(c, "low_pct"), profile(c, "high_pct")
    col = AXIS_COLOUR[info[c]["axis"]]
    ax.fill_between(range(7), lo, hi, color=col, alpha=0.25, lw=0)
    ax.plot(range(7), v, "o-", ms=2.5, color=col)
    ax.axhline(0, color="black", lw=0.5)
    nA, nB = int(prof[[r["contrast"] for r in prof].index(c)]["n_A"]), int(prof[[r["contrast"] for r in prof].index(c)]["n_B"])
    ax.set_title(f"{title(c)}\nn = {nA:,} / {nB:,}", fontsize=6.8)
    ax.set_xticks(range(7)); ax.set_xticklabels(PCTS, fontsize=6.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator()); ax.tick_params(labelsize=6.5)
handles = [Rectangle((0, 0), 1, 1, color=v) for v in AXIS_COLOUR.values()]
fig.legend(handles, [AXIS_LABEL[k] for k in AXIS_COLOUR], loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.supxlabel("Percentile of the pooled RT distribution", fontsize=8, y=0.06)
fig.supylabel("Change, B relative to A (%), 95% interval", fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
figures.append(("mix_fig1_observed", fig))
captions["mix_fig1_observed"] = ("Figure 1. The data. Percent change from A to B at each percentile of the pooled RT distribution, "
    "every contrast, with 95% cluster-bootstrap intervals over units (bands); n = trials in A / B. Each panel has its own y scale.")

# ---- Figure 2: calibration of A ------------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 6.6))
for ax, c in zip(axes.ravel(), cids):
    r = res[c]
    ax.plot(PCTS, r["q_A_observed_ms"], "o", ms=3.5, color="#777777", label="observed A")
    ax.plot(PCTS, r["q_A_model_ms"], "-", lw=1.3, color="black", label="model A")
    ax.set_yscale("log")
    ax.set_title(f"{title(c)}\nRMS log error {r['calibration_rms_log']:.3f}", fontsize=6.8)
    ax.set_xticks([5, 25, 50, 75, 99]); ax.tick_params(labelsize=6.5)
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0, 0].legend(frameon=False, fontsize=6.5)
fig.supxlabel("Percentile", fontsize=8); fig.supylabel("RT (ms, log)", fontsize=8)
fig.tight_layout()
figures.append(("mix_fig2_calibration", fig))
captions["mix_fig2_calibration"] = ("Figure 2. Step 1, the starting point. The model's condition A after calibration (line) "
    "against the observed pooled A (points). RMS log error 0.05 is about 5% at a typical percentile. Calibration moved "
    "four parameters from the WW-CarefulFitting fit of the nearest condition (2_data_processing/baselines.csv).")

# ---- Figure 3: what each ingredient does ----------------------------------------------------------------
ING = ["skill (u0_sd x 0.5)", "practice (logN + 2)", "drive level (u0, u0_sd x 1.3)", "non-decision time (-50 ms)", "retrieval speed (s_W x 0.5)"]
ING_COLOUR = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
fig, axes = plt.subplots(1, 5, figsize=(WIDTH, 2.8), sharey=True)
for ax, ing, col in zip(axes, ING, ING_COLOUR):
    allv = []
    for c in cids:
        v = [float(s["change_pct"]) for s in steps if s["contrast"] == c and s["ingredient"] == ing]
        allv.append(v)
        ax.plot(range(7), v, "-", lw=0.6, alpha=0.45, color=col)
    ax.plot(range(7), np.median(allv, axis=0), "o-", ms=3, lw=1.6, color="black")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(textwrap.fill(ing, 18), fontsize=7.3)
    ax.set_xticks(range(7)); ax.set_xticklabels(PCTS, fontsize=6, rotation=90)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
axes[0].set_ylabel("Change in simulated percentile (%)")
fig.supxlabel("Percentile", fontsize=8)
fig.tight_layout()
figures.append(("mix_fig3_ingredients", fig))
captions["mix_fig3_ingredients"] = ("Figure 3. What each ingredient does. One step of each change applied to every contrast's "
    "calibrated A (thin lines; black = median over the 16 starting points). Ingredients that move the percentiles in the same "
    "pattern cannot be told apart by these data; the patterns differ most at the 5th-10th and 90th-99th percentiles.")

# ---- Figure 4: reachable region -------------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 7.0))
cmap = LinearSegmentedColormap.from_list("sp", ["#0072B2", "#BBBBBB", "#E69F00"])
for ax, c in zip(axes.ravel(), cids):
    g = res[c]["grid"]
    x = np.array([p["predicted_pct"][1] for p in g]); yv = np.array([p["predicted_pct"][6] for p in g])
    pv = np.array([p["p"] for p in g])
    pts = np.c_[x, yv]
    if len(np.unique(pts.round(2), axis=0)) >= 3:
        hull = ConvexHull(pts)
        ax.add_patch(Polygon(pts[hull.vertices], closed=True, color="#EEEEEE", zorder=0))
    ax.scatter(x, yv, c=pv, cmap=cmap, s=6, zorder=2)
    o = profile(c, "change_pct"); lo = profile(c, "low_pct"); hi = profile(c, "high_pct")
    ax.errorbar(o[1], o[6], xerr=[[o[1] - lo[1]], [hi[1] - o[1]]], yerr=[[o[6] - lo[6]], [hi[6] - o[6]]], fmt="*", ms=8,
                color="black", zorder=3, elinewidth=0.8)
    ax.axhline(0, color="black", lw=0.4); ax.axvline(0, color="black", lw=0.4)
    ax.set_title(title(c), fontsize=6.8); ax.tick_params(labelsize=6.5); minor(ax)
fig.supxlabel("Change at 10th percentile (%)", fontsize=8); fig.supylabel("Change at 99th percentile (%)", fontsize=8)
fig.subplots_adjust(left=0.1, right=0.86, bottom=0.07, top=0.96, hspace=0.5, wspace=0.42)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-3, 5))
cb = fig.colorbar(sm, cax=fig.add_axes((0.89, 0.35, 0.02, 0.3)))
cb.set_label("practice change (logN)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
figures.append(("mix_fig4_reachable", fig))
captions["mix_fig4_reachable"] = ("Figure 4. Step 3, what any mix can reach. Grey area: every change at the 10th and 99th "
    "percentiles that skill (u0_sd x 0.14 to x 2.7) and practice (logN -3 to +5) produce together from the calibrated A, "
    "on a 7 x 7 grid (points coloured by the practice change). Star: the observed change with its 95% interval. A star "
    "outside the grey area cannot be produced by these two mechanisms in this range. Points near +400% at the 99th "
    "percentile are mixes under which some simulated trials do not commit within 10 s.")

# ---- Figure 5: model fits per contrast --------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 7.2), sharex=True)
for ax, c in zip(axes.ravel(), cids):
    o, lo, hi = profile(c, "change_pct"), profile(c, "low_pct"), profile(c, "high_pct")
    ax.fill_between(range(7), lo, hi, color="#DDDDDD", lw=0)
    ax.plot(range(7), o, "o", ms=3, color="black")
    for m, ls in [("S", ":"), ("P", "--"), ("SP", "-")]:
        ax.plot(range(7), res[c]["fits"][m]["predicted_pct"], ls, lw=1.1, color=MODEL_COLOUR[m])
    best_add = min(("SP+drive", "SP+ndt", "SP+retrieval"), key=lambda m: float(fit[(c, m)]["aic"]))
    ax.plot(range(7), res[c]["fits"][best_add]["predicted_pct"], "-", lw=1.1, color=MODEL_COLOUR[best_add])
    sp_p = float(fit[(c, "SP")]["p"]); add_p = float(fit[(c, best_add)]["p"])
    ax.set_title(f"{title(c)}\nSP p = {sp_p:.2g}; {MODEL_LABEL[best_add]} p = {add_p:.2g}", fontsize=6.3)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(range(7)); ax.set_xticklabels(PCTS, fontsize=6.5); ax.tick_params(labelsize=6.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
h = [plt.Line2D([], [], marker="o", ls="", color="black")] + [plt.Line2D([], [], ls=ls, color=MODEL_COLOUR[m]) for m, ls in [("S", ":"), ("P", "--"), ("SP", "-")]] \
    + [plt.Line2D([], [], color=MODEL_COLOUR[m]) for m in ("SP+drive", "SP+ndt", "SP+retrieval")]
fig.legend(h, ["observed"] + [MODEL_LABEL[m] for m in MODELS], loc="lower center", ncol=4, frameon=False, fontsize=7)
fig.supxlabel("Percentile", fontsize=8, y=0.06); fig.supylabel("Change, B relative to A (%)", fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
figures.append(("mix_fig5_fits", fig))
captions["mix_fig5_fits"] = ("Figure 5. Step 2, the fits. Observed change (points, grey band = 95% interval) and the best fit of "
    "each model: skill only, practice only, both, and both plus the addition with the lowest AIC for that contrast (coloured "
    "as in the legend). p: chi-square adequacy; p > .05 = adequate.")

# ---- Figure 6: adequacy matrix ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
P = np.array([[float(fit[(c, m)]["p"]) for m in MODELS] for c in cids])
L = -np.log10(np.clip(P, 1e-12, 1))
im = ax.imshow(L, cmap="Greys", aspect="auto", vmin=0, vmax=12)
for i in range(len(cids)):
    for j in range(len(MODELS)):
        ok = P[i, j] > 0.05
        ax.text(j, i, "ok" if ok else f"{P[i, j]:.0e}" if P[i, j] > 1e-12 else "<1e-12", ha="center", va="center",
                fontsize=6.5 if not ok else 9, color="white" if L[i, j] > 6 else "black", weight="bold" if ok else "normal")
        if ok:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#009E73", lw=1.5))
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=25, ha="right")
ax.set_yticks(range(len(cids))); ax.set_yticklabels([title(c) for c in cids], fontsize=7)
cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.set_label("-log10 p (darker = worse fit)", fontsize=7)
fig.tight_layout()
figures.append(("mix_fig6_adequacy", fig))
captions["mix_fig6_adequacy"] = ("Figure 6. Which models are adequate. Each cell: chi-square p of a model for a contrast; "
    "ok in a green box = adequate (p > .05). Shading: -log10 p, capped at 12.")

# ---- Figure 7: amounts of skill and practice -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
for c in cids:
    ch = res[c]["fits"]["SP"]["changes"]
    ok = summ[c]["sp_adequate"] == "True"
    ax.plot(ch["s"], ch["p"], "o" if ok else "X", ms=8, color=AXIS_COLOUR[info[c]["axis"]], mec="black" if ok else "none", zorder=3)
    ax.annotate(c, (ch["s"], ch["p"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("Skill change s (log of the factor on drive spread u0_sd; < 0 = more precise)")
ax.set_ylabel("Practice change p (change in logN; > 0 = more instances)")
for lim, side in [(-3, "x"), (2, "x")]:
    ax.axvline(lim, color="#999999", ls=":", lw=0.8)
ax.axhline(-5, color="#999999", ls=":", lw=0.8); ax.axhline(6, color="#999999", ls=":", lw=0.8)
handles = [plt.Line2D([], [], marker="o", ls="", color=v) for v in AXIS_COLOUR.values()] + \
          [plt.Line2D([], [], marker="o", ls="", color="white", mec="black"), plt.Line2D([], [], marker="X", ls="", color="#777777")]
ax.legend(handles, [AXIS_LABEL[k] for k in AXIS_COLOUR] + ["adequate", "not adequate"], frameon=False, fontsize=7, loc="lower right", ncol=2)
minor(ax)
fig.tight_layout()
figures.append(("mix_fig7_amounts", fig))
captions["mix_fig7_amounts"] = ("Figure 7. How much skill and how much practice the skill + practice fit uses for each contrast. "
    "Dotted lines: search bounds; a point on a bound used all the change allowed. Circle = adequate fit; cross = not adequate.")

# ---- Figure 8: what the additions do ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 4.0), sharey=True)
for ax, (m, par, lab, tf) in zip(axes, [("SP+drive", "d", "drive level factor", np.exp), ("SP+ndt", "t", "non-decision shift (ms)", lambda v: v),
                                        ("SP+retrieval", "r", "retrieval onset factor", np.exp)]):
    for i, c in enumerate(cids):
        v = tf(float(fit[(c, m)][f"change_{par}"]))
        ok = fit[(c, m)]["adequate"] == "True"
        ax.plot(v, i, "o" if ok else "x", ms=5, color=MODEL_COLOUR[m] if ok else "#999999")
    ax.axvline(1 if par != "t" else 0, color="black", lw=0.5)
    if par != "t":
        ax.set_xscale("log")
    ax.set_title(MODEL_LABEL[m]); ax.set_xlabel(lab)
axes[0].set_yticks(range(len(cids))); axes[0].set_yticklabels([title(c) for c in cids], fontsize=7); axes[0].invert_yaxis()
fig.tight_layout()
figures.append(("mix_fig8_additions", fig))
captions["mix_fig8_additions"] = ("Figure 8. The added change in each three-change model: drive level (factor on u0 and u0_sd), "
    "non-decision time (ms), retrieval onset scale (factor; < 1 = faster). Filled circle = the model is adequate; grey cross = not.")

# ---- Figure 9: improvement from each addition ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
xs = np.arange(len(cids))
for k, m in enumerate(("SP+drive", "SP+ndt", "SP+retrieval")):
    gain = [float(fit[(c, "SP")]["aic"]) - float(fit[(c, m)]["aic"]) for c in cids]
    ax.bar(xs + (k - 1) * 0.27, np.clip(gain, 0.1, None), 0.27, color=MODEL_COLOUR[m], label=MODEL_LABEL[m])
ax.set_yscale("log"); ax.axhline(2, color="black", ls="--", lw=0.7)
ax.text(len(cids) - 0.5, 2.3, "AIC drop of 2", fontsize=6.5, ha="right")
ax.set_xticks(xs); ax.set_xticklabels(cids)
ax.set_ylabel("AIC of skill + practice minus AIC with the addition\n(log; values below 0.1 drawn at 0.1)")
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
figures.append(("mix_fig9_aic", fig))
captions["mix_fig9_aic"] = ("Figure 9. How much each addition improves on skill + practice, as the drop in AIC (larger = better "
    "fit for the extra parameter). Bars below the dashed line improve AIC by less than 2.")

# ---- Figure 10: instance route and practice ---------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
for c in cids:
    share = 100 * float(summ[c]["instance_share_A"])
    prac = [float(s["change_pct"]) for s in steps if s["contrast"] == c and s["ingredient"] == "practice (logN + 2)"]
    a1.plot(share, prac[1], "o", color=AXIS_COLOUR[info[c]["axis"]]); a1.annotate(c, (share, prac[1]), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    a2.plot(share, prac[6], "o", color=AXIS_COLOUR[info[c]["axis"]]); a2.annotate(c, (share, prac[6]), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
for ax, lab in [(a1, "10th"), (a2, "99th")]:
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Trials won by the instance route in A (%)")
    ax.set_ylabel(f"Change at {lab} percentile\nfrom logN + 2 (%)"); minor(ax)
fig.tight_layout()
figures.append(("mix_fig10_instance_share", fig))
captions["mix_fig10_instance_share"] = ("Figure 10. Practice works only through the instance route: the effect of one practice step "
    "(logN + 2) at the 10th and 99th percentiles against the share of A's simulated trials the instance route wins.")

with PdfPages(out / "SkillPractice-Mixtures_report.pdf") as pdf:
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
print("wrote 4_output/SkillPractice-Mixtures_report.pdf")
