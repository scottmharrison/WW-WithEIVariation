"""
19_make_atlas_report.py

Purpose
    Write the mechanism atlas: what each of 14 processes in the extended Wong-Wang model does to the RT
    distribution (its location, spread, fast edge, slow tail and accuracy), from three starting points;
    how the processes compare in shape space; how much of each contrast's fitted change each mechanism
    carries; and why StarCraft 2 fits well in WW-CarefulFitting but not in the contrast fits here.

Inputs
    4_output/atlas/sweeps.json, 4_output/atlas/decomposition.json (18_mechanism_atlas.py);
    4_output/pooled_accuracy_profiles_fine.csv; 4_output/mech_selection/<contrast>.json;
    1_raw/ww_conditions.csv (WW-CarefulFitting fits); 2_data_processing/contrasts.csv

Outputs
    4_output/mechanism_fingerprints.csv (dictionary beside it); 4_output/SkillPractice-Atlas_report.pdf;
    4_output/figures/atlas_*.png (300 dpi)

Depends on
    18_mechanism_atlas.py

Last substantive change
    2026-11-07 — first version.

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
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter  # noqa: E402

# Arial has Helvetica's letter widths and is the Helvetica substitute installed on this machine.
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 6.5
out = Path("4_output")
(out / "figures").mkdir(exist_ok=True)
D = json.loads((out / "atlas" / "sweeps.json").read_text())
DEC = json.loads((out / "atlas" / "decomposition.json").read_text())
BASES = D["bases"]
SW = D["sweeps"]
EDGES = np.array(D["density_edges"])
CENTRES = np.sqrt(EDGES[:-1] * EDGES[1:])
PROCS = list(SW["A2"])
SHORT = {"skill": "skill", "practice": "practice", "drive": "common drive", "evidence": "evidence", "control_exc": "E/I excitatory",
         "control_inh": "E/I inhibitory", "threshold_ww": "circuit threshold", "threshold_lca": "instance threshold", "urgency": "urgency",
         "lapse": "lapse", "ndt": "non-decision time", "retrieval_acc": "retrieval accuracy", "circuit": "circuit strength",
         "retrieval_str": "retrieval strength"}
# One colour per process, used in every figure (Okabe-Ito colours plus darker and lighter variants).
PCOL = {"skill": "#0072B2", "practice": "#E69F00", "drive": "#009E73", "evidence": "#56B4E9", "control_exc": "#F4A582",
        "control_inh": "#B2182B", "threshold_ww": "#332288", "threshold_lca": "#88CCEE", "urgency": "#DDCC77", "lapse": "#CC79A7",
        "ndt": "#777777", "retrieval_acc": "#C7B000", "circuit": "#117733", "retrieval_str": "#AA4499"}
PCTS = np.arange(1, 100)
figures, captions = [], {}


def lv(base, p):
    return [l for l in SW[base][p]["levels"] if "q" in l]


def change(l0, l1):
    """Changes from level l0 to l1: percent at percentiles 10, 50, 99; tail-to-median ratio; accuracy
    (points); ex-Gaussian mu, sigma, tau (ms); Hill index."""
    q0, q1 = np.array(l0["q"]), np.array(l1["q"])
    return {"10th %": 100 * (q1[9] / q0[9] - 1), "median %": 100 * (q1[49] / q0[49] - 1), "99th %": 100 * (q1[98] / q0[98] - 1),
            "99th/median %": 100 * ((q1[98] / q1[49]) / (q0[98] / q0[49]) - 1), "accuracy (pts)": 100 * (l1["accuracy"] - l0["accuracy"]),
            "mu (ms)": l1["mu"] - l0["mu"], "sigma (ms)": l1["sigma"] - l0["sigma"], "tau (ms)": l1["tau"] - l0["tau"],
            "Hill index": l1["hill"] - l0["hill"]}


STATS = ["10th %", "median %", "99th %", "99th/median %", "accuracy (pts)", "mu (ms)", "sigma (ms)", "tau (ms)", "Hill index"]

# ---- Fingerprint table: lowest to highest level, from each starting point --------------------------------
fp_rows = []
for b in BASES:
    for p in PROCS:
        L = lv(b, p)
        ch = change(L[0], L[-1])
        fp_rows.append(dict(start=b, process=p, from_level=L[0]["level"], to_level=L[-1]["level"],
                            instance_share_from=round(L[0]["instance_share"], 3), **{k: round(v, 3) for k, v in ch.items()}))
with open(out / "mechanism_fingerprints.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys())); w.writeheader(); w.writerows(fp_rows)
FP = {(r["start"], r["process"]): r for r in fp_rows}

# ---- Summary page ---------------------------------------------------------------------------------------
fig = plt.figure(figsize=(WIDTH, 9.3))
fig.text(0, 0.985, "What each process does to the RT distribution", fontsize=13, weight="bold", va="top")
fig.text(0, 0.957, "14 processes of the extended Wong-Wang model, each moved alone from three fitted starting points; "
         "RT distributions of\n10,000 simulated trials per setting (correct responses).", fontsize=8.2, va="top", color="#444444")
heads = ["Process", "Range (lowest → highest)", "10th", "Median", "99th", "99th/med", "Acc", "tau"]
xs = [0, 0.2, 0.47, 0.56, 0.65, 0.74, 0.84, 0.92]
y = 0.905
fig.text(0, y + 0.02, "Change from the lowest to the highest setting, from the lexical-decision starting point (A2):", fontsize=8, va="top")
for cx, h in zip(xs, heads):
    fig.text(cx, y - 0.01, h, weight="bold", fontsize=7.4, va="top")
fig.add_artist(plt.Line2D([0, 1], [y - 0.03, y - 0.03], color="black", lw=0.6)); y -= 0.04
for p in PROCS:
    r = FP[("A2", p)]
    vals = [SHORT[p], f"{r['from_level']} → {r['to_level']}", f"{r['10th %']:+.0f}%", f"{r['median %']:+.0f}%", f"{r['99th %']:+.0f}%",
            f"{r['99th/median %']:+.0f}%", f"{r['accuracy (pts)']:+.1f}", f"{r['tau (ms)']:+.0f}"]
    for cx, v in zip(xs, vals):
        fig.text(cx, y, v, fontsize=7.2, va="top")
    y -= 0.0215
fig.add_artist(plt.Line2D([0, 1], [y + 0.005, y + 0.005], color="black", lw=0.6))
notes = ["Columns: percent change in the 10th percentile, median and 99th percentile of correct RTs; change in the ratio of the 99th "
         "percentile to the median (tail relative to the body); accuracy (points); ex-Gaussian tau (ms). A2 = lexical decision under speed "
         "instruction, calibrated to its data. Every process from every starting point: 4_output/mechanism_fingerprints.csv.",
         "Processes that act through the instance route (practice, instance threshold, retrieval accuracy and strength) do little where "
         "that route seldom wins (A2: 1%; flanker P9: 0%) and much where it often wins (StarCraft 2 Bronze S1: 59%) (Figure 2).",
         "StarCraft 2 has no correct/error label, so its starting point has no evidence term and chooses at chance; its accuracy values are "
         "not interpretable.",
         "Sources: 4_output/atlas/sweeps.json (3_analyses/18_mechanism_atlas.py; seed 20261107)."]
y -= 0.02
for k, n in enumerate(notes, 1):
    wrapped = textwrap.fill(f"{k}. {n}", 128)
    fig.text(0, y, wrapped, fontsize=7, va="top", color="#333333"); y -= 0.0115 * (wrapped.count("\n") + 1) + 0.006
figures.append(("atlas_summary", fig))

# ---- Figure 1: fingerprint heatmaps ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 5.0), sharey=True)
for ax, b in zip(axes, BASES):
    M = np.array([[FP[(b, p)][s] for s in STATS[:5]] for p in PROCS])
    L = np.sign(M) * np.log10(1 + np.abs(M))
    ax.imshow(L, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    for i in range(len(PROCS)):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=5.4, color="white" if abs(L[i, j]) > 1.6 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(["10th", "median", "99th", "99th/\nmedian", "acc"], fontsize=6.5)
    ax.set_title(f"{b}: {BASES[b]}\n(instance route {100 * lv(b, 'skill')[2]['instance_share']:.0f}%)", fontsize=7)
axes[0].set_yticks(range(len(PROCS))); axes[0].set_yticklabels([SHORT[p] for p in PROCS], fontsize=7)
fig.tight_layout()
figures.append(("atlas_fig1_fingerprints", fig))
captions["atlas_fig1_fingerprints"] = ("Figure 1. Fingerprint of each process: change from its lowest to its highest setting, from each "
    "starting point. Percent change at the 10th percentile, median and 99th percentile, change in the 99th/median ratio (tail relative "
    "to the body), and accuracy in points. Red = increase, blue = decrease (colour on a signed log scale). Instance-route share of the "
    "starting point in the titles.")

# ---- Figures 2-6: one row per process: density, survival, percentile change, statistics -------------------------
rows_per_page = 3
for page, start in enumerate(range(0, len(PROCS), rows_per_page)):
    procs = PROCS[start:start + rows_per_page]
    fig, axes = plt.subplots(len(procs), 4, figsize=(WIDTH, 2.35 * len(procs) + 0.3))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(procs):
        L = lv("A2", p)
        cmap = plt.get_cmap("viridis")
        cols = [cmap(i / max(len(L) - 1, 1)) for i in range(len(L))]
        a0, a1, a2, a3 = axes[r]
        for l, c in zip(L, cols):
            a0.plot(CENTRES, l["density"], color=c, lw=1, label=l["level"])
            q = np.array(l["q"])
            a1.plot(q, 1 - PCTS / 100, color=c, lw=1)
            a2.plot(PCTS, 100 * (q / np.array(L[0]["q"]) - 1), color=c, lw=1)
        a0.set_xscale("log"); a0.set_xlim(150, 6000)
        a0.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10))); a0.xaxis.set_minor_formatter(NullFormatter())
        a0.set_ylabel(textwrap.fill(SHORT[p], 16) + "\ndensity (per s)", fontsize=7)
        a0.legend(frameon=False, fontsize=5.5, loc="upper right", handlelength=1)
        a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(150, 10000); a1.set_ylim(0.01, 1)
        a1.xaxis.set_minor_formatter(NullFormatter())
        a2.axhline(0, color="black", lw=0.4); a2.yaxis.set_minor_locator(AutoMinorLocator()); a2.xaxis.set_minor_locator(AutoMinorLocator())
        x = np.arange(len(L))
        a3.plot(x, [l["mu"] for l in L], "o-", ms=2.5, color="#0072B2", label="mu")
        a3.plot(x, [l["sigma"] for l in L], "s-", ms=2.5, color="#009E73", label="sigma")
        a3.plot(x, [l["tau"] for l in L], "^-", ms=2.5, color="#D55E00", label="tau")
        a3b = a3.twinx()
        a3b.plot(x, [100 * l["accuracy"] for l in L], "D--", ms=2.5, color="black", lw=0.8, label="accuracy")
        a3b.set_ylabel("accuracy (%)", fontsize=6.5); a3b.tick_params(labelsize=6)
        a3.set_xticks(x); a3.set_xticklabels([l["level"] for l in L], rotation=45, fontsize=5.5, ha="right")
        a3.set_ylabel("ms", fontsize=6.5); a3.yaxis.set_minor_locator(AutoMinorLocator())
        if r == 0:
            a0.set_title("Density of correct RTs"); a1.set_title("Slow tail: P(RT > t)"); a2.set_title("Change at each percentile (%)")
            a3.set_title("Ex-Gaussian and accuracy")
        if r == len(procs) - 1:
            a0.set_xlabel("RT (ms, log)"); a1.set_xlabel("RT (ms, log)"); a2.set_xlabel("Percentile")
    fig.tight_layout()
    name = f"atlas_fig{2 + page}_processes_{page + 1}"
    figures.append((name, fig))
    captions[name] = (f"Figure {2 + page}. " + "; ".join(SW["A2"][p]["label"] for p in procs) + ". From the lexical-decision starting "
        "point (A2), each process alone at each setting (colours, light = highest). Columns: density of correct RTs (log time); survival "
        "function on log-log axes (the slow tail, to the 99th percentile); percent change at each percentile from the lowest setting; "
        "ex-Gaussian mu (blue circles), sigma (green squares) and tau (orange triangles) in ms, left axis, and accuracy (black diamonds, "
        "dashed), right axis, across settings.")
n_proc_figs = (len(PROCS) + rows_per_page - 1) // rows_per_page
fnum = 2 + n_proc_figs

# ---- Shape space ------------------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 6.8))
for c, b in enumerate(BASES):
    for p in PROCS:
        L = lv(b, p)
        q0 = np.array(L[0]["q"])
        med = [100 * (l["q"][49] / q0[49] - 1) for l in L]
        tail = [100 * ((l["q"][98] / l["q"][49]) / (q0[98] / q0[49]) - 1) for l in L]
        acc = [100 * (l["accuracy"] - L[0]["accuracy"]) for l in L]
        dmu = [l["mu"] - L[0]["mu"] for l in L]
        dtau = [l["tau"] - L[0]["tau"] for l in L]
        for ax, xv, yv in [(axes[0, c], med, tail), (axes[1, c], med, acc), (axes[2, c], dmu, dtau)]:
            ax.plot(xv, yv, "-", lw=1, color=PCOL[p]); ax.plot(xv[-1], yv[-1], "o", ms=3, color=PCOL[p])
    axes[0, c].set_title(f"{b}: {BASES[b]}", fontsize=7)
    for r in range(3):
        axes[r, c].axhline(0, color="black", lw=0.4); axes[r, c].axvline(0, color="black", lw=0.4)
        axes[r, c].tick_params(labelsize=6); axes[r, c].xaxis.set_minor_locator(AutoMinorLocator()); axes[r, c].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0, c].set_xlabel("median change (%)", fontsize=6.5); axes[1, c].set_xlabel("median change (%)", fontsize=6.5)
    axes[2, c].set_xlabel("mu change (ms)", fontsize=6.5)
axes[0, 0].set_ylabel("tail relative to median,\n99th/median change (%)"); axes[1, 0].set_ylabel("accuracy change (pts)")
axes[2, 0].set_ylabel("tau change (ms)")
for ax in axes[0]:
    ax.set_ylim(-80, 250)
h = [plt.Line2D([], [], color=PCOL[p], marker="o", ms=3) for p in PROCS]
fig.legend(h, [SHORT[p] for p in PROCS], loc="lower center", ncol=5, frameon=False, fontsize=6.3)
fig.tight_layout(rect=(0, 0.08, 1, 1))
name = f"atlas_fig{fnum}_shape_space"
figures.append((name, fig))
captions[name] = (f"Figure {fnum}. Shape space. Each line is one process traced from its lowest setting (origin) to its highest (dot), "
    "from each starting point. Top: change in the median against change in the tail relative to the median (99th/median), so a process "
    "that rescales the whole distribution runs along the horizontal axis and one that reshapes the tail runs vertically. Middle: speed "
    "against accuracy. Bottom: ex-Gaussian mu against tau. Top row clipped at -80% and +250%.")
fnum += 1

# ---- Percentile-change profile at the highest setting, every process, three starting points --------------
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 6.6), sharex=True)
for ax, p in zip(axes.ravel(), PROCS):
    for b, ls in zip(BASES, ["-", "--", ":"]):
        L = lv(b, p)
        ax.plot(PCTS, 100 * (np.array(L[-1]["q"]) / np.array(L[0]["q"]) - 1), ls, lw=1.1, color=PCOL[p], label=b)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title(textwrap.shorten(SHORT[p], 22), fontsize=7); ax.tick_params(labelsize=6)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
for ax in axes.ravel()[len(PROCS):]:
    ax.axis("off")
h = [plt.Line2D([], [], color="black", ls=ls) for ls in ["-", "--", ":"]]
axes.ravel()[len(PROCS)].legend(h, [f"{b}: {BASES[b]}" for b in BASES], frameon=False, fontsize=6.5, loc="center")
fig.supxlabel("Percentile of correct RTs", fontsize=7.5); fig.supylabel("Change, highest vs lowest setting (%)", fontsize=7.5)
fig.tight_layout()
name = f"atlas_fig{fnum}_profiles"
figures.append((name, fig))
captions[name] = (f"Figure {fnum}. Where in the distribution each process acts: percent change at each percentile from its lowest to its "
    "highest setting, from the three starting points (line style). A flat line rescales or shifts the distribution; a rising line acts "
    "on the slow tail; a falling line acts on the fast edge. Each panel has its own y scale.")
fnum += 1

# ---- Decomposition of the fitted combinations -------------------------------------------------------------------
prof = list(csv.DictReader(open(out / "pooled_accuracy_profiles_fine.csv")))
contrasts = list(csv.DictReader(open("2_data_processing/contrasts.csv")))
info = {c["contrast"]: c for c in contrasts}
fine = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
MCOL = {"skill + practice": "#888888", "drive": "#009E73", "evidence": "#56B4E9", "control": "#D55E00", "threshold": "#332288",
        "urgency": "#DDCC77", "lapse": "#CC79A7", "ndt": "#8C510A", "retrieval": "#C7B000"}
fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 7.6), sharex=True)
for ax, c in zip(axes.ravel(), [c["contrast"] for c in contrasts]):
    parts = DEC[c]
    qa = np.array(parts["A"]["q"])
    obs = np.array([float(r["change"]) for r in prof if r["contrast"] == c and r["measure"] == "correct_rt_pct"])
    ax.plot(fine, obs, "o", ms=2.5, color="black", zorder=5)
    ax.plot(PCTS, 100 * (np.array(parts["combined"]["q"]) / qa - 1), "-", lw=1.6, color="black")
    for part, s in parts.items():
        if part in ("A", "combined") or "q" not in s:
            continue
        ax.plot(PCTS, 100 * (np.array(s["q"]) / qa - 1), "-", lw=1, color=MCOL.get(part, "#999999"))
    sel = json.loads((out / "mech_selection" / f"{c}.json").read_text())["selected"]
    ax.set_title(f"{c} {textwrap.shorten(info[c]['task'], 20, placeholder='…')}\n{textwrap.shorten(sel.replace('SP', 'SP').replace('+', ' + '), 40, placeholder='…')}", fontsize=5.8)
    ax.axhline(0, color="black", lw=0.4); ax.tick_params(labelsize=6); ax.yaxis.set_minor_locator(AutoMinorLocator())
h = [plt.Line2D([], [], marker="o", ls="", color="black"), plt.Line2D([], [], color="black", lw=1.6)] + \
    [plt.Line2D([], [], color=v) for v in MCOL.values()]
fig.legend(h, ["observed", "fitted combination"] + [f"{k} alone" if k != "skill + practice" else "skill + practice alone" for k in MCOL],
           loc="lower center", ncol=4, frameon=False, fontsize=6.3)
fig.supxlabel("Percentile of correct RTs", fontsize=7.5, y=0.075); fig.supylabel("Change, B relative to A (%)", fontsize=7.5)
fig.tight_layout(rect=(0, 0.09, 1, 1))
name = f"atlas_fig{fnum}_decomposition"
figures.append((name, fig))
captions[name] = (f"Figure {fnum}. The fitted changes taken apart. For each contrast's chosen combination (stage 15; title), the "
    "observed change (points), the fitted combination (black), and each of its parts applied alone at its fitted value (coloured). Where "
    "the parts do not add up to the combination, the mechanisms interact.")
fnum += 1

# ---- StarCraft 2: why the fits differ between projects -------------------------------------------------------------
conds = {r["cond_id"]: r for r in csv.DictReader(open("1_raw/ww_conditions.csv"))}
keys = ["W_self", "W_inhib", "u_star", "u0", "u0_sd", "logN", "N_logsd", "s_W", "rho_max", "sigL", "ter_sd"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.4), gridspec_kw=dict(width_ratios=[1, 1.2]))
ratios = [abs(float(conds["sc2__27"][k]) / float(conds["sc2__21"][k])) for k in keys]
kcol = ["#117733" if k in ("W_self", "W_inhib", "u_star") else "#AA4499" if k in ("s_W", "rho_max", "logN", "N_logsd", "sigL") else "#999999" for k in keys]
# Lines drawn from a ratio of 1 (no difference), so the length shows the size of the difference on the log scale.
a1.hlines(range(len(keys)), 1, ratios, color=kcol, lw=2.5)
a1.scatter(ratios, range(len(keys)), color=kcol, s=14, zorder=3)
a1.axvline(1, color="black", lw=0.6); a1.set_xscale("log"); a1.set_xlim(0.1, 10)
a1.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10]); a1.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
a1.xaxis.set_minor_formatter(NullFormatter())
a1.set_yticks(range(len(keys))); a1.set_yticklabels(keys, fontsize=7); a1.invert_yaxis()
a1.set_xlabel("Grandmaster / Bronze ratio, WW-CarefulFitting\npooled-league fits (log scale)")
qa = [float(r["value_A"]) for r in prof if r["contrast"] == "S1" and r["measure"] == "correct_rt_pct"]
qb = [float(r["value_B"]) for r in prof if r["contrast"] == "S1" and r["measure"] == "correct_rt_pct"]
a2.plot(fine, qa, "o-", ms=3, lw=0.6, color="#999999", label="Bronze (observed)")
a2.plot(fine, qb, "s-", ms=3, lw=0.6, color="black", label="Grandmaster (observed)")
for v in np.arange(1, 134) * 45.2:
    a2.axhline(v, color="#EEEEEE", lw=0.4, zorder=0)
a2.set_yscale("log"); a2.set_ylim(100, 6000); a2.set_xlabel("Percentile"); a2.set_ylabel("RT (ms, log); grey lines = 45.2 ms steps")
a2.legend(frameon=False, fontsize=6.5); a2.yaxis.set_minor_formatter(NullFormatter())
fig.tight_layout()
name = f"atlas_fig{fnum}_starcraft"
figures.append((name, fig))
captions[name] = (f"Figure {fnum}. Why StarCraft 2 fits well in WW-CarefulFitting but not here. Left: ratio of each fitted parameter, "
    "Grandmaster to Bronze, in WW-CarefulFitting's pooled-league fits: the circuit's weights (green; tipping point u* 24.0 to 9.1 Hz) and "
    "the retrieval route (purple; rho_max x4.4, s_W x0.17) differ most, and neither is among the mechanisms fitted here. Right: the "
    "observed percentiles sit on 45.2 ms recording steps (Grandmaster's 5th and 20th percentiles are both 135.6 ms), a staircase that a "
    "smooth model cannot follow when judged percentile by percentile; WW-CarefulFitting's binned likelihood is built around the steps. "
    "WW-CarefulFitting also judges fits against other models (BIC), here against an absolute chi-square test.")

with PdfPages(out / "SkillPractice-Atlas_report.pdf") as pdf:
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
print("wrote 4_output/SkillPractice-Atlas_report.pdf and 4_output/mechanism_fingerprints.csv")
