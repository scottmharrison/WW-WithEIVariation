"""
17_make_mechanism_report.py

Purpose
    Write the report of the mechanism comparison: which of eight mechanisms, added to skill and practice,
    does the most in each contrast; the combination chosen for each contrast; whether all eight together
    can fit it; and whether any one combination fits all 16 contrasts.

Inputs
    2_data_processing/contrasts.csv; 4_output/pooled_accuracy_profiles_fine.csv;
    4_output/mech_selection/<contrast>.json (stage 15); 4_output/shared_combinations/<contrast>.json (stage 16)

Outputs
    4_output/mechanism_comparison.csv, 4_output/shared_combinations.csv (dictionaries beside them);
    4_output/SkillPractice-Mechanisms_report.pdf; 4_output/figures/mech_*.png (300 dpi)

Depends on
    16_fit_shared_combinations.py

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
MECHS = ["drive", "evidence", "control", "threshold", "urgency", "lapse", "ndt", "retrieval"]
MECH_LABEL = {"drive": "common drive", "evidence": "evidence", "control": "E/I control", "threshold": "threshold (WW + LCA)",
              "urgency": "urgency", "lapse": "lapse", "ndt": "non-decision time", "retrieval": "retrieval accuracy"}
# Okabe-Ito colour-blind-safe palette plus two greys; the same colour means the same mechanism everywhere.
MECH_COL = {"drive": "#009E73", "evidence": "#56B4E9", "control": "#D55E00", "threshold": "#0072B2", "urgency": "#E69F00",
            "lapse": "#CC79A7", "ndt": "#999999", "retrieval": "#F0E442"}
AXIS_LABEL = {"practice": "Time on task", "instruction": "Instruction", "item practice": "Word frequency", "skill": "Expertise",
              "drive level": "Stimulus strength"}
FINE = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

contrasts = list(csv.DictReader(open("2_data_processing/contrasts.csv")))
cids = [c["contrast"] for c in contrasts]
info = {c["contrast"]: c for c in contrasts}
sel = {c: json.loads((out / "mech_selection" / f"{c}.json").read_text()) for c in cids}
shared = {c: json.loads((out / "shared_combinations" / f"{c}.json").read_text())["fits"] for c in cids}
prof = list(csv.DictReader(open(out / "pooled_accuracy_profiles_fine.csv")))


def mechs_of(name):
    return [] if name == "SP" else name.split("+")[1:]


def label_of(name):
    m = mechs_of(name)
    return "skill + practice" if not m else "SP + " + " + ".join(MECH_LABEL[x] for x in m)


def title(c):
    return f"{c} {textwrap.shorten(info[c]['task'], 26, placeholder='…')}"


def obs(c, measure):
    rows = [r for r in prof if r["contrast"] == c and r["measure"] == measure]
    return np.array([float(r["change"]) for r in rows]), np.array([float(r["se"]) for r in rows])


def ok(f):
    return f["p"] == f["p"] and f["p"] > 0.05


def minor(ax, x=True, y=True):
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


# ---- Tables ----------------------------------------------------------------------------------------
rows = []
for c in cids:
    F = sel[c]["fits"]
    best_any = min((f for n, f in F.items()), key=lambda f: f["chi2"])   # the ceiling: lowest chi-square of any fit
    gains = {m: F["SP"]["aic"] - F[f"SP+{m}"]["aic"] for m in MECHS}
    s = F[sel[c]["selected"]]
    rows.append(dict(contrast=c, axis=info[c]["axis"], selected=sel[c]["selected"], selected_p=round(s["p"], 5), selected_adequate=ok(s),
                     selected_k=len(s["params"]), sp_p=round(F["SP"]["p"], 5), sp_adequate=ok(F["SP"]),
                     ceiling_chi2=best_any["chi2"], ceiling_model_p=round(F["SP+all"]["p"], 5) if F["SP+all"]["p"] == F["SP+all"]["p"] else "",
                     best_single=max(gains, key=gains.get), **{f"aic_gain_{m}": round(gains[m], 3) for m in MECHS}))
with open(out / "mechanism_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
combos = sorted({n for c in cids for n in shared[c]}, key=lambda n: (len(mechs_of(n)), n))
shared_rows = []
for n in combos:
    n_ok = sum(ok(shared[c][n]) for c in cids)
    shared_rows.append(dict(combination=n, n_mechanisms=len(mechs_of(n)), adequate_in=n_ok,
                            total_aic=round(sum(shared[c][n]["aic"] for c in cids), 2),
                            failing=";".join(c for c in cids if not ok(shared[c][n]))))
with open(out / "shared_combinations.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(shared_rows[0].keys())); w.writeheader(); w.writerows(shared_rows)
by = {r["contrast"]: r for r in rows}

figures, captions = [], {}

# ---- Summary page -----------------------------------------------------------------------------------
n_sel = sum(r["selected_adequate"] for r in rows)
best_shared = max(shared_rows, key=lambda r: (r["adequate_in"], -r["total_aic"]))
lowest_aic = min(shared_rows, key=lambda r: r["total_aic"])
fig = plt.figure(figsize=(WIDTH, 9.3))
fig.text(0, 0.985, "Which mechanisms account for each dataset?", fontsize=13, weight="bold", va="top")
fig.text(0, 0.957, "Eight mechanisms added to skill and practice, compared on correct RTs (12 percentiles), accuracy and error RTs\n"
         "in 16 contrasts; extended Wong-Wang model.", fontsize=8.2, va="top", color="#444444")
lines = [("Best combination for each contrast adequate", f"{n_sel} of 16"),
         ("Skill + practice alone adequate", f"{sum(r['sp_adequate'] for r in rows)} of 16"),
         ("One combination adequate in all 16", "yes: " + ", ".join(r["combination"] for r in shared_rows if r["adequate_in"] == 16)
          if any(r["adequate_in"] == 16 for r in shared_rows) else "no"),
         ("Shared combination adequate in the most contrasts", f"{label_of(best_shared['combination'])}: {best_shared['adequate_in']} of 16"),
         ("Shared combination with the lowest total AIC", f"{label_of(lowest_aic['combination'])} (fails: {lowest_aic['failing'] or 'none'})"),
         ("Mechanism most often the best single addition", ", ".join(f"{MECH_LABEL[m]} {k}" for m, k in Counter(r["best_single"] for r in rows).most_common(3)))]
y = 0.905
fig.add_artist(plt.Line2D([0, 1], [y + 0.012, y + 0.012], color="black", lw=0.8))
for head, val in lines:
    fig.text(0, y, head, fontsize=8.2, va="top", weight="bold")
    fig.text(0.47, y, textwrap.fill(val, 62), fontsize=8.2, va="top")
    y -= 0.042
fig.add_artist(plt.Line2D([0, 1], [y + 0.016, y + 0.016], color="black", lw=0.8))
y -= 0.012
fig.text(0, y, "Per contrast", weight="bold", va="top"); y -= 0.028
heads = ["Id", "Task", "Change", "Chosen combination", "p", "Best single"]
xs = [0, 0.055, 0.29, 0.43, 0.8, 0.88]
for cx, h in zip(xs, heads):
    fig.text(cx, y, h, weight="bold", fontsize=7.3, va="top")
fig.add_artist(plt.Line2D([0, 1], [y - 0.016, y - 0.016], color="black", lw=0.6)); y -= 0.026
for c in cids:
    r = by[c]
    comb = " + ".join(MECH_LABEL[m] for m in mechs_of(r["selected"])) or "(skill + practice only)"
    vals = [c, textwrap.shorten(info[c]["task"], 24, placeholder="…"), AXIS_LABEL[info[c]["axis"]], textwrap.shorten(comb, 44, placeholder="…"),
            f"{r['selected_p']:.2g}" + ("" if r["selected_adequate"] else " (no)"), MECH_LABEL[r["best_single"]].split(" (")[0]]
    for cx, v in zip(xs, vals):
        fig.text(cx, y, v, fontsize=6.9, va="top")
    y -= 0.0185
fig.add_artist(plt.Line2D([0, 1], [y + 0.004, y + 0.004], color="black", lw=0.6))
notes = ["Chosen combination: forward selection from skill + practice, adding the mechanism with the lowest AIC while AIC falls by "
         "more than 2. Adequate: chi-square p > .05; (no) = not adequate. Best single: the mechanism that lowers AIC most when added alone.",
         "Mechanisms: common drive (u0, u0_sd scaled); evidence (pool difference); E/I control (strength, ratio; Lo, Wang & Wang 2015); "
         "threshold of the circuit and of the instance route; urgency (input growing through the trial); lapse (fraction of trials with "
         "reduced drive and no evidence); non-decision time; retrieval accuracy (instance-route bias).",
         "Shared combinations: every combination chosen for any contrast, and their union, fitted to all 16 (Figure 5). P3 and S1 have "
         "no accuracy (12 terms), D1 has no error RTs in B (13 terms), so models with as many changes as terms have no p there.",
         "Caution: each fit's chi-square includes its own simulation error. Where the data's accuracy error is very small (D1 reaches "
         "100% correct), a model with noisier simulated accuracy is penalised less; this is why some single additions show a negative "
         "AIC drop in Figure 1 (D1, D2). Comparisons there are not reliable.",
         "Sources: 4_output/mechanism_comparison.csv, shared_combinations.csv, mech_selection/, shared_combinations/."]
y -= 0.014
for k, n in enumerate(notes, 1):
    wrapped = textwrap.fill(f"{k}. {n}", 128)
    fig.text(0, y, wrapped, fontsize=7, va="top", color="#333333"); y -= 0.0115 * (wrapped.count("\n") + 1) + 0.006
figures.append(("mech_summary", fig))

# ---- Figure 1: what each mechanism does on its own -------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 5.0))
G = np.array([[by[c][f"aic_gain_{m}"] for m in MECHS] for c in cids])
L = np.sign(G) * np.log10(1 + np.abs(G))
im = ax.imshow(L, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
for i, c in enumerate(cids):
    chosen = mechs_of(by[c]["selected"])
    for j, m in enumerate(MECHS):
        ax.text(j, i, f"{G[i, j]:.0f}", ha="center", va="center", fontsize=6.3, color="white" if abs(L[i, j]) > 2 else "black")
        if m in chosen:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6))
ax.set_xticks(range(len(MECHS))); ax.set_xticklabels([MECH_LABEL[m] for m in MECHS], rotation=30, ha="right")
ax.set_yticks(range(len(cids))); ax.set_yticklabels([title(c) for c in cids], fontsize=7)
cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.set_label("AIC drop when added alone (signed log10(1 + |drop|))", fontsize=7)
fig.tight_layout()
figures.append(("mech_fig1_single", fig))
captions["mech_fig1_single"] = ("Figure 1. What each mechanism does on its own: the drop in AIC when it alone is added to skill + "
    "practice (numbers; blue = better fit). Black boxes: mechanisms in the combination chosen for that contrast. A negative drop "
    "(red) cannot be a worse fit of a larger model; it comes from each fit's own simulation error in the chi-square (summary, note 5).")

# ---- Figure 2: how often each mechanism is used --------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
chosen_counts = Counter(m for r in rows for m in mechs_of(r["selected"]))
single_counts = Counter(r["best_single"] for r in rows)
ys = np.arange(len(MECHS))
a1.barh(ys, [single_counts[m] for m in MECHS], color=[MECH_COL[m] for m in MECHS], edgecolor="black", lw=0.3)
a2.barh(ys, [chosen_counts[m] for m in MECHS], color=[MECH_COL[m] for m in MECHS], edgecolor="black", lw=0.3)
a1.set_yticks(ys); a1.set_yticklabels([MECH_LABEL[m] for m in MECHS]); a1.invert_yaxis()
a1.set_xlabel("Contrasts where it is the\nbest single addition"); a2.set_xlabel("Contrasts whose chosen\ncombination includes it")
for ax in (a1, a2):
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2)); ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
fig.tight_layout()
figures.append(("mech_fig2_counts", fig))
captions["mech_fig2_counts"] = ("Figure 2. How often each mechanism is needed, out of 16 contrasts.")

# ---- Figure 3: selection paths ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
for c in cids:
    path = sel[c]["path"]; F = sel[c]["fits"]
    a = [F[n]["aic"] for n in path]
    ax.plot(range(len(a)), a, "o-", ms=3, lw=1, color="#555555" if by[c]["selected_adequate"] else "#D55E00", alpha=0.85)
    ax.annotate(c, (len(a) - 1, a[-1]), xytext=(3, 0), textcoords="offset points", fontsize=6.5, va="center")
ax.set_yscale("log")
ax.set_xlabel("Mechanisms added to skill + practice"); ax.set_ylabel("AIC (log)")
ax.set_xticks(range(6))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout()
figures.append(("mech_fig3_paths", fig))
captions["mech_fig3_paths"] = ("Figure 3. AIC along each contrast's selection path, one mechanism added per step until AIC no longer "
    "falls by more than 2. Grey: the chosen combination is adequate; orange: it is not.")

# ---- Figure 4: fits of the chosen combination ---------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 7.4), sharex=True)
for ax, c in zip(axes.ravel(), cids):
    q, se = obs(c, "correct_rt_pct")
    F = sel[c]["fits"]; s = F[by[c]["selected"]]
    ax.fill_between(range(12), q - 1.96 * se, q + 1.96 * se, color="#DDDDDD", lw=0)
    ax.plot(range(12), q, "o", ms=2.5, color="black")
    ax.plot(range(12), F["SP"]["predicted_correct_pct"], "-", lw=0.9, color="#888888")
    ax.plot(range(12), s["predicted_correct_pct"], "-", lw=1.3, color="#0072B2")
    acc = [r for r in prof if r["contrast"] == c and r["measure"] == "accuracy"]
    acc_txt = (f"\nacc {100 * float(acc[0]['change']):+.1f}; SP {100 * F['SP']['predicted_accuracy_change']:+.1f}; chosen "
               f"{100 * s['predicted_accuracy_change']:+.1f}") if acc and sel[c]["has_accuracy"] else ""
    ax.set_title(f"{title(c)}\nSP p = {F['SP']['p']:.2g}; chosen p = {s['p']:.2g}{acc_txt}", fontsize=5.8)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks([0, 2, 5, 9, 11]); ax.set_xticklabels([5, 20, 50, 90, 99], fontsize=6); ax.tick_params(labelsize=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
h = [plt.Line2D([], [], marker="o", ls="", color="black"), plt.Line2D([], [], color="#888888"), plt.Line2D([], [], color="#0072B2")]
fig.legend(h, ["observed", "skill + practice", "chosen combination"], loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.supxlabel("Percentile of correct RTs", fontsize=8, y=0.06); fig.supylabel("Change, B relative to A (%)", fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
figures.append(("mech_fig4_fits", fig))
captions["mech_fig4_fits"] = ("Figure 4. Observed change in 12 correct-RT percentiles (points; band = 95% interval), skill + "
    "practice (grey) and the chosen combination (blue). Titles: chi-square p and the observed and predicted change in accuracy (points).")

# ---- Figure 5: shared combinations ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(WIDTH, 0.6 + 0.3 * len(combos) + 1.2))
M = np.array([[1.0 if ok(shared[c][n]) else 0.0 for c in cids] for n in combos])
ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#FFFFFF", "#BFE3CC"]), vmin=0, vmax=1, aspect="auto")
for i, n in enumerate(combos):
    for j, c in enumerate(cids):
        p = shared[c][n]["p"]
        ax.text(j, i, "ok" if ok(shared[c][n]) else ("n/a" if p != p else f"{p:.0e}"), ha="center", va="center",
                fontsize=5.0 if not ok(shared[c][n]) else 6.0, color="black" if ok(shared[c][n]) else "#555555")
ax.set_yticks(range(len(combos)))
ax.set_yticklabels([f"{textwrap.shorten(label_of(n), 58, placeholder='…')}  [{r['adequate_in']}/16]" for n, r in zip(combos, shared_rows)], fontsize=6.5)
ax.set_xticks(range(len(cids))); ax.set_xticklabels(cids, fontsize=7)
fig.tight_layout()
figures.append(("mech_fig5_shared", fig))
captions["mech_fig5_shared"] = ("Figure 5. Each combination chosen for any contrast, and their union (last row), fitted to every "
    "contrast. Green 'ok' = adequate (p > .05); otherwise p, or n/a where the model has as many changes as observations. "
    "Brackets: contrasts fitted adequately.")

with PdfPages(out / "SkillPractice-Mechanisms_report.pdf") as pdf:
    for name, fig in figures:
        fig.savefig(out / "figures" / f"{name}.png", dpi=300 if name in captions else 150)
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
print("wrote 4_output/SkillPractice-Mechanisms_report.pdf, mechanism_comparison.csv, shared_combinations.csv")
