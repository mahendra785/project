"""Build the journal manuscript, supplementary material and figures from the committed result files.

Every number in manuscript.tex / supplementary.tex is computed here from
  docs/research/results_v2/summary_all.json   (revision: frozen deep backbones, Kaggle sessions 1-3)
  docs/research/results/mgcrc_summary.json    (preliminary: random forest, cross-fitting)
  docs/research/results/simulation_E7.json    (controlled simulation, Amendment 2)

    python paper/journal/build.py
"""

from __future__ import annotations

import json
import math
import os
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"
S = json.loads((REPO / "docs/research/results_v2/summary_all.json").read_text())
PRE = {e["dataset"]: e for e in json.loads((REPO / "docs/research/results/mgcrc_summary.json").read_text())}
E7 = json.loads((REPO / "docs/research/results/simulation_E7.json").read_text())
E7D = json.loads((REPO / "docs/research/results/simulation_E7_posthoc_diagnostic.json").read_text())["rows"]
E8 = {ds: json.loads((REPO / f"docs/research/results/posthoc_E8_{ds}_rf.json").read_text())
      for ds in ("kvasir-seg", "fracatlas")}
TORCH = json.loads((REPO / "docs/research/results_v2/manifests/kvasir-seg__unet-r34_seed0.json").read_text())["env"]["torch"].split("+")[0]

DS = [("kvasir-seg", "Kvasir-SEG"), ("fracatlas", "FracAtlas"), ("msd-hepatic-vessel", "MSD Task08")]
AR = [("unet-r34", "U-Net/ResNet-34"), ("unet-mitb2", "U-Net/MiT-B2")]
ALPHAS = ("0.05", "0.10", "0.15")
NAMES = {
    "pix": "Pixel CRC", "joint": "Joint CRC, thresholding (reference)", "max": "Max-form CRC, thresholding",
    "dil": "Joint CRC, dilation family", "strat": "Joint CRC, CCRA-S strata", "ltt": "LTT ($\\delta=0.1$)",
    "le_union": "Joint CRC, lesion-equalized (ours)", "le_max": "Max-form, lesion-equalized (ours)",
    "joint_multi": "Joint CRC + $R^{\\mathrm{mul}}$", "pix_tc": "Pixel CRC (C2)", "joint_tc": "Joint CRC, thresholding (C2)",
    "max_tc": "Max-form CRC (C2)", "dil_tc": "Joint CRC, dilation (C2)", "strat_tc": "Joint CRC, CCRA-S (C2)",
    "ltt_tc": "LTT (C2)", "joint_multi_tc": "Joint CRC + $R^{\\mathrm{mul}}$ (C2)",
    "le_fixed_tc": "Lesion-equalized, fixed $\\rho=0.5,t_0=0.2$ (C3)",
}
NAVY, TEAL, ORANGE, GREY, INK = "#21295C", "#028090", "#C8553D", "#6B7280", "#1F2937"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ accessors and formatting
def cfg(ds, ar):
    return S[f"{ds}|{ar}"]


def m(ds, ar, a, meth, k):
    return cfg(ds, ar)["methods"][f"{a}|{meth}"][k]


def rel(ds, ar, a, meth):
    return m(ds, ar, a, meth, "rel_vs_joint")


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def sg(v, d=1):  # signed number with a typeset minus
    return f"{v:+.{d}f}".replace("-", "$-$")


def pc(x, d=1):
    return sg(100 * x, d) + "\\%"


def pcn(x, d=1):  # unsigned magnitude
    return f"{abs(100 * x):.{d}f}\\%"


def ci2(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def cipc(c, d=1):
    return f"[{sg(100 * c[0], d)}, {sg(100 * c[1], d)}]"


def lam(ds, ar, a, meth):
    return m(ds, ar, a, meth, "lam")["mean"] / 1000.0  # stored as grid index on 1001 points


def pm(ds, ar, k):
    return cfg(ds, ar)["pool_metrics"][k]


def gap(ds, ar, a, k):
    return cfg(ds, ar)["gap"][a][k]


# ------------------------------------------------------------------ supplementary table numbers (order of supplementary())
SUP = dict(seeds="S1", counts="S2", validity="S3", gap="S4", abl="S5--S6", verdicts="S7", e8="S8", methods="S9--S26")


# ------------------------------------------------------------------ eligible subgroups and the CRC fallback
NEED = {"0.05": 19, "0.10": 9, "0.15": 6}  # smallest n_r with (0 + 1)/(n_r + 1) <= alpha (B = 1)


def _pmf(N, K, n, x):
    return math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)


def eligible(ds):
    """Expected eligible units per split and the exact per-partition probability of the CRC fallback.

    Partitions are uniform random permutations (mgcrc_v2.partitions), so the number of eligible units in a split of
    size n is Hypergeometric(N, K, n); averaged over the three seeds.
    """
    cnt = cfg(ds, "unet-r34")["counts"]
    out = {}
    for seed in range(3):
        N = cnt["n_units"][seed]
        n_tune, n_cal = int(0.2 * N), int(0.4 * N)
        sizes = dict(tune=n_tune, cal=n_cal, test=N - n_tune - n_cal, tc=n_tune + n_cal)
        for risk, K in (("les", N), ("sml", cnt["n_small"][seed]), ("mul", cnt["n_multi"][seed])):
            for split, n in sizes.items():
                d = out.setdefault((risk, split), dict(mean_n=[], fallback={a: [] for a in NEED}, empty=[]))
                d["mean_n"].append(n * K / N)
                d["empty"].append(_pmf(N, K, n, 0) if K < N else 0.0)
                for a, need in NEED.items():
                    d["fallback"][a].append(sum(_pmf(N, K, n, x) for x in range(0, min(need, K + 1))) if K < N
                                            else float(n < need))
    return {k: dict(mean_n=float(np.mean(v["mean_n"])), empty=float(np.mean(v["empty"])),
                    fallback={a: float(np.mean(x)) for a, x in v["fallback"].items()}) for k, v in out.items()}


def empty_prob():
    """Largest per-partition probability of an empty eligible subgroup in any unstratified rule (C1, C2, tuning)."""
    E = [eligible(ds) for ds, _ in DS]
    return max([e[(r, s)]["empty"] for e in E for r in ("les", "sml", "mul") for s in ("cal", "tc")]
               + [e[("sml", "tune")]["empty"] for e in E])


def fallback_paragraph():
    E = {ds: eligible(ds) for ds, _ in DS}
    kv, t8 = E["kvasir-seg"], E["msd-hepatic-vessel"]
    worst_main = max(e[(r, s)]["fallback"][a] for ds, e in E.items() for r in ("les", "sml") for s in ("cal", "tc")
                     for a in NEED if not (ds == "msd-hepatic-vessel" and r == "sml" and a == "0.05"))
    empty = empty_prob()
    assert all(e[("les", "cal")]["fallback"][a] == 0 for e in E.values() for a in NEED)
    n_cal = ", ".join(str(int(0.4 * cfg(ds, "unet-r34")["counts"]["n_units"][0])) for ds, _ in DS[:2])
    return (
        "Because partitions are uniformly random, the number of eligible units in a split is hypergeometric, and the "
        "probability of the fallback can be computed exactly from the pool counts. For pixel and lesion risk every "
        f"calibration unit is eligible ({n_cal} and {int(0.4 * cfg(DS[2][0], 'unet-r34')['counts']['n_units'][0])} units), "
        "so the fallback never occurs. For the small-lesion risk it occurs in practice only on Task08 at $\\alpha=0.05$, "
        f"where about {t8[('sml', 'cal')]['mean_n']:.0f} calibration units are eligible: in "
        f"{pc1(t8[('sml', 'cal')]['fallback']['0.05'])} of partitions under C1 (with probability below "
        f"{sci(t8[('sml', 'tc')]['fallback']['0.05'])} under C2, which calibrates on 50\\% more units), and the joint set "
        "is then the whole volume. In every "
        f"other configuration and level its probability is below {sci(worst_main)} per partition. "
        "The multi-lesion risk, used only by the exploratory rule of Section~\\ref{sec:multi}, has about "
        f"{kv[('mul', 'cal')]['mean_n']:.0f} eligible calibration units on Kvasir-SEG and falls back in "
        f"{pc1(kv[('mul', 'cal')]['fallback']['0.05'])}, {pc1(kv[('mul', 'cal')]['fallback']['0.10'])} and "
        f"{pc1(kv[('mul', 'cal')]['fallback']['0.15'])} of partitions at $\\alpha=0.05$, 0.10 and 0.15. The stratified "
        "baseline calibrates each stratum on about a third of the calibration units, so its fallback is more frequent; "
        "its per-stratum counts depend on the predicted mass and were not recorded. An empty eligible subgroup has "
        f"probability below {sci(empty)} per partition for every unstratified rule.")


def pc1(x):
    return f"{100 * x:.1f}\\%"


def sci(x):
    """Smallest power of ten above x, as LaTeX (an upper bound for a probability)."""
    assert 0 < x < 1, x
    return f"$10^{{{int(np.floor(np.log10(x))) + 1}}}$"


def span(vals, fmt="{:.0f}", unit="\\%"):
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return f"{lo}{unit}" if lo == hi else f"{lo}--{hi}{unit}"


# ------------------------------------------------------------------ figures
def fig_gap():
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    al = [0.05, 0.10, 0.15]
    for j, (ds, dn) in enumerate(DS):
        for i, (k, lab) in enumerate((("G_multi", "$G_{K\\geq2}$"), ("G_small", "$G_{\\mathrm{sml}}$"))):
            ax = axs[i, j]
            for o, ((ar, an), col) in enumerate(zip(AR, (NAVY, TEAL))):
                y = [gap(ds, ar, a, k)["mean"] for a in ALPHAS]
                lo = [gap(ds, ar, a, k)["ci"][0] for a in ALPHAS]
                hi = [gap(ds, ar, a, k)["ci"][1] for a in ALPHAS]
                x = np.array(al) + (o - 0.5) * 0.006
                ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o-", color=col, ms=3.5,
                            capsize=2.5, lw=1.3, label=an)
            ax.axhline(1, color=ORANGE, ls="--", lw=1)
            ax.set_xticks(al)
            if i == 0:
                ax.set_title(dn, fontsize=9.5, color=NAVY, fontweight="bold")
            if j == 0:
                ax.set_ylabel(lab + " = risk / $\\alpha$")
            if i == 1:
                ax.set_xlabel("$\\alpha$")
    axs[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "gap.pdf", bbox_inches="tight")
    plt.close(fig)


COST_ROWS = [("pix", "Pixel CRC", "C1"), ("max", "Max-form CRC", "C1"), ("dil", "Dilation family", "C1"),
             ("strat", "CCRA-S strata", "C1"), ("le_union", "Lesion-equalized", "C1"),
             ("ltt", "LTT (high-prob. guarantee)", "LTT"),
             ("joint_tc", "Joint thresholding, C2 data", "C2/C3"), ("le_fixed_tc", "Lesion-eq. fixed, C3 data", "C2/C3")]


def fig_cost():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    y = np.arange(len(COST_ROWS))[::-1].astype(float)
    groups = [g for _, _, g in COST_ROWS]
    seps = [(y[i] + y[i + 1]) / 2 for i in range(len(groups) - 1) if groups[i] != groups[i + 1]]
    for ax, (ar, an) in zip(axs, AR):
        v = np.array([rel("kvasir-seg", ar, "0.10", mm)["mean"] for mm, _, _ in COST_ROWS]) * 100
        c = np.array([rel("kvasir-seg", ar, "0.10", mm)["ci"] for mm, _, _ in COST_ROWS]) * 100
        cols = [ORANGE if mm == "ltt" else TEAL if mm.startswith("le") else GREY for mm, _, _ in COST_ROWS]
        ax.barh(y, v, color=cols, height=0.62)
        ax.errorbar(v, y, xerr=[v - c[:, 0], c[:, 1] - v], fmt="none", ecolor=INK, capsize=2, lw=0.9)
        ax.axvline(0, color=INK, lw=0.8)
        for s_ in seps:
            ax.axhline(s_, color=GREY, ls=":", lw=0.8)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab, _ in COST_ROWS])
        ax.set_title(f"Kvasir-SEG, {an}, $\\alpha=0.10$", fontsize=9, color=NAVY, fontweight="bold")
        ax.set_xlabel("set size vs joint thresholding, %\n$\\leftarrow$ smaller      larger $\\rightarrow$")
    fig.tight_layout()
    fig.savefig(FIG / "cost.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_regime():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    mk = {"kvasir-seg": "o", "fracatlas": "s", "msd-hepatic-vessel": "^"}
    for ds, dn in DS:
        for ar, an in AR:
            rec = pm(ds, ar, "lesion_recall")["mean"]
            col = NAVY if ar == "unet-r34" else TEAL
            axs[0].scatter(rec, m(ds, ar, "0.10", "joint", "set_to_truth")["mean"], marker=mk[ds], color=col, s=36, zorder=3)
            axs[0].scatter(rec, m(ds, ar, "0.10", "dil", "set_to_truth")["mean"], marker=mk[ds], facecolor="none",
                           edgecolor=col, s=36, zorder=3)
            mp = cfg(ds, ar)["missed_profile"]["missed"]
            axs[1].scatter(rec, mp["frac_peak_below_bg"], marker=mk[ds], color=col, s=36, zorder=3)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("lesion recall of the backbone (pool)")
    axs[0].set_ylabel("set-to-truth ratio at $\\alpha=0.10$")
    axs[0].set_title("Size of the jointly guaranteed set", fontsize=9, color=NAVY, fontweight="bold")
    axs[1].set_xlabel("lesion recall of the backbone (pool)")
    axs[1].set_ylabel("missed lesions with peak $\\leq$ bg $q_{99}$")
    axs[1].set_ylim(0, 1.02)
    axs[1].set_title("Are the missed lesions detectable?", fontsize=9, color=NAVY, fontweight="bold")
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker=mk[d], ls="", color=GREY, label=n) for d, n in DS]
    h += [Line2D([], [], marker="o", ls="", color=NAVY, label="ResNet-34"), Line2D([], [], marker="o", ls="", color=TEAL,
                                                                                  label="MiT-B2"),
          Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=GREY, label="dilation family")]
    axs[0].legend(handles=h, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "regime.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ tables
# The narrow variants (narrow=True) put CIs and standard deviations on a second line, so that the tables fit the
# 119 mm text block of the Springer Nature template upright.
def two(top, bottom, align="c"):
    """A table cell on two lines (the second line is braced so that a leading '[' is not read as \\\\[length])."""
    return f"\\makecell[{align}]{{{top}\\\\{{{bottom}}}}}"


def bb(an, narrow):
    """Backbone name, split after 'U-Net/' in narrow tables."""
    return two("U-Net/", an.split("/", 1)[1], "l") if narrow else an


def table_backbones(narrow=False):
    rows = []
    for ds, dn in DS:
        for ar, an in AR:
            c = cfg(ds, ar)
            cnt = c["counts"]

            def ms(k, d=3):
                mean, sd = f"{pm(ds, ar, k)['mean']:.{d}f}", f"$\\pm${pm(ds, ar, k)['sd']:.{d}f}"
                return two(mean, sd) if narrow else mean + sd
            rows.append(f"{dn} & {bb(an, narrow)} & {cnt['n_units'][0]} & {min(cnt['n_multi'])}--{max(cnt['n_multi'])} & "
                        f"{min(cnt['n_small'])}--{max(cnt['n_small'])} & {ms('dice')} & {ms('iou')} & {ms('lesion_recall')} & "
                        f"{ms('lesion_precision')} & {ms('fp_components_per_unit', 2)} \\\\")
    return r"""\begin{table}[t]
\caption{Performance of the frozen backbones on the calibration pool, before conformal calibration.}
\label{tab:backbones}
\scriptsize
\setlength{\tabcolsep}{""" + ("2pt" if narrow else "2.5pt") + r"""}
\begin{tabular}{@{}llccccccccc@{}}
\toprule
Dataset & Backbone & $n$ & $K\geq2$ & small & Dice & IoU & """ + (
        r"\makecell{Les.\\recall} & \makecell{Les.\\precision} & \makecell{FP /\\unit}" if narrow
        else r"Les.\ recall & Les.\ precision & FP / unit") + r""" \\
\midrule
""" + ("\n\\addlinespace[2pt]\n" if narrow else "\n").join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def verdict_g(ds, ar):
    return cfg(ds, ar)["H1"]


def table_gap(narrow=False):
    rows = []
    for ds, dn in DS:
        e = eligible(ds)
        for ar, an in AR:
            gm, gs, ga = gap(ds, ar, "0.10", "G_multi"), gap(ds, ar, "0.10", "G_small"), gap(ds, ar, "0.10", "G_all")
            cm = two(f2(gm["mean"]), ci2(gm["ci"])) if narrow else f"{f2(gm['mean'])} {ci2(gm['ci'])}"
            cs = two(f2(gs["mean"]), ci2(gs["ci"])) if narrow else f"{f2(gs['mean'])} {ci2(gs['ci'])}"
            rows.append(f"{dn} & {bb(an, narrow)} & {f2(ga['mean'])} & {cm} & {e[('mul', 'test')]['mean_n']:.0f} & "
                        f"{cs} & {e[('sml', 'test')]['mean_n']:.0f} & "
                        f"{lam(ds, ar, '0.10', 'pix'):.3f} & {verdict_g(ds, ar)} \\\\")
    return r"""\begin{table}[t]
\caption{RQ1: lesion-miss rates at the pixel-CRC threshold divided by $\alpha$ ($\alpha=0.10$).}
\label{tab:gap}
\scriptsize
\setlength{\tabcolsep}{""" + ("1.5pt" if narrow else "3pt") + r"""}
\begin{tabular}{@{}llccccccc@{}}
\toprule
Dataset & Backbone & $G_{\mathrm{all}}$ & """ + (
        r"\makecell{$G_{K\geq2}$\\{[95\% CI]}} & $\bar n_{K\geq2}$ & \makecell{$G_{\mathrm{sml}}$\\{[95\% CI]}}" if narrow
        else r"$G_{K\geq2}$ [95\% CI] & $\bar n_{K\geq2}$ & $G_{\mathrm{sml}}$ [95\% CI]") + r""" & $\bar n_{\mathrm{sml}}$ & $\hat\lambda^{\mathrm{pix}}$ & H1 \\
\midrule
""" + ("\n\\addlinespace[2pt]\n" if narrow else "\n").join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def method_rows(ds, ar, a, meths):
    out = []
    for mm in meths:
        r = cfg(ds, ar)["methods"][f"{a}|{mm}"]
        rv = r["rel_vs_joint"]
        dd = 2 if mm.startswith("le") else 1
        d = "ref." if mm == "joint" else f"{pc(rv['mean'], dd)} {cipc(rv['ci'], dd)}"
        cells = [f3(r[k]["mean"]) + ("$^{*}$" if k in r["claims"] else "")
                 for k in ("risk_pix", "risk_les", "risk_small", "risk_multi")]
        out.append(f"{NAMES[mm]} & " + " & ".join(cells) + f" & {r['set_to_truth']['mean']:.2f} & {d} \\\\")
    return out


def table_main():
    body = []
    for ar, an in AR:
        body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{Kvasir-SEG, {an}}}}} \\\\")
        body += method_rows("kvasir-seg", ar, "0.10", ["pix", "joint", "max", "dil", "strat", "le_union", "le_max"])
        body.append("\\addlinespace[1pt]")
        body += method_rows("kvasir-seg", ar, "0.10", ["ltt"])
        body.append("\\midrule" if ar == "unet-r34" else "")
    return r"""\begin{table}[t]
\caption{RQ2 and RQ3 on Kvasir-SEG ($\alpha=0.10$, comparison C1): mean test risks and set sizes; $^{*}$ marks a guaranteed risk.}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lcccccl@{}}
\toprule
Method & $R^{\mathrm{pix}}$ & $R^{\mathrm{les}}$ & $R^{\mathrm{sml}}$ & $R^{\mathrm{mul}}$ & set/truth & $\Delta$ vs joint [95\% CI] \\
\midrule
""" + "\n".join(b for b in body if b) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table_weak(narrow=False):
    rows = []
    for ds, dn in DS[1:]:
        for ar, an in AR:
            v = [m(ds, ar, "0.10", mm, "set_to_truth")["mean"] for mm in ("pix", "joint", "max", "ltt", "dil")]
            d = rel(ds, ar, "0.10", "dil")
            dv = two(pc(d["mean"]), cipc(d["ci"])) if narrow else f"{pc(d['mean'])} {cipc(d['ci'])}"
            rows.append(f"{dn} & {bb(an, narrow)} & {f3(pm(ds, ar, 'lesion_recall')['mean'])} & "
                        + " & ".join(f"{x:.1f}" for x in v) + f" & {dv} \\\\")
    return r"""\begin{table}[t]
\caption{Set-to-truth ratios in the weak-detector regime (FracAtlas and Task08, $\alpha=0.10$, C1).}
\label{tab:weak}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccccccl@{}}
\toprule
 & & & \multicolumn{5}{c}{set/truth} & \\
\cmidrule(lr){4-8}
Dataset & Backbone & """ + (
        r"\makecell{Les.\\recall} & Pixel & Joint & \makecell{Max-\\form} & LTT & Dilation & \makecell{Dilation vs\\joint [95\% CI]}"
        if narrow else r"Les.\ recall & Pixel & Joint & Max-form & LTT & Dilation & Dilation vs joint [95\% CI]") + r""" \\
\midrule
""" + ("\n\\addlinespace[2pt]\n" if narrow else "\n").join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""


ABL = [("main", "Registered setting ($\\tau=0.5$, $q=33$, 8-conn., all cal.)"), ("abltau25", "$\\tau=0.25$"),
       ("abltau75", "$\\tau=0.75$"), ("ablq25", "small = below $q_{25}$"), ("ablq50", "small = below $q_{50}$"),
       ("ablconn4", "4-connected candidates"), ("ablncal50", "$n_{\\mathrm{cal}}=50$"), ("ablncal100", "$n_{\\mathrm{cal}}=100$"),
       ("ablncal200", "$n_{\\mathrm{cal}}=200$")]


def abl_entry(tag, ds):
    if tag == "main":
        c = cfg(ds, "unet-r34")
        return dict(gap=c["gap"]["0.10"], jp=c["methods"]["0.10|pix"]["rel_vs_joint"], le=c["methods"]["0.10|le_union"]["rel_vs_joint"])
    a = S["_ablations"].get(f"{tag}|{ds}|unet-r34")
    if a is None:
        return None
    return dict(gap=a["gap"]["0.10"], jp=a["joint_vs_pix"]["0.1"], le=a["le_rel"]["0.1"])


def table_ablation(ds="kvasir-seg", label="tab:ablation", caption_ds="Kvasir-SEG"):
    rows = []
    for tag, name in ABL:
        e = abl_entry(tag, ds)
        if e is None:
            continue
        g = e["gap"]
        rows.append(f"{name} & {f2(g['G_multi']['mean'])} {ci2(g['G_multi']['ci'])} & {f2(g['G_small']['mean'])} & "
                    f"{pc(e['jp']['mean'])} & {pc(e['le']['mean'], 2)} & {e['le']['p_smaller']:.2f} \\\\")
    return f"""\\begin{{table}}[t]
\\caption{{Sensitivity analyses ({caption_ds}, U-Net/ResNet-34, $\\alpha=0.10$, mean over three seeds).}}
\\label{{{label}}}
\\footnotesize
\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\toprule
Setting & $G_{{K\\geq2}}$ [95\\% CI] & $G_{{\\mathrm{{sml}}}}$ & Pixel vs joint & LE vs joint & $P$ \\\\
\\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table_prelim():
    rows = []
    for ds, dn in DS[:2]:
        e = PRE[ds]
        g = next(x for x in e["gap"] if abs(x["alpha"] - 0.1) < 1e-9)
        R = {r["method"]: r for r in e["rows"] if abs(r["alpha"] - 0.1) < 1e-9}
        rows.append(f"{dn} & {e['mean_dice']:.3f} & {g['G_multi']:.2f} [{g['G_multi_lo']:.2f}, {g['G_multi_hi']:.2f}] & "
                    f"{pc(R['pixel_crc']['rel'])} & {pc(R['ltt_threshold']['rel'])} & {pc(R['union_dilation']['rel'])} & "
                    f"{pc(R['ours_union']['rel'])} & {e['verdicts']['H1']} \\\\")
    return r"""\begin{table}[h]
\caption{Preliminary study with cross-fitted random-forest posteriors ($\alpha=0.10$).}
\label{tab:prelim}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
Dataset & Dice & $G_{K\geq2}$ [split range] & $\Delta$ pixel & $\Delta$ LTT & $\Delta$ dilation & $\Delta$ lesion-eq. & H1 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""


# ------------------------------------------------------------------ prose
def numbers():
    K = "kvasir-seg"
    n = {}
    for ar in ("unet-r34", "unet-mitb2"):
        t = "r" if ar == "unet-r34" else "m"
        for a in ALPHAS:
            n[f"G{t}{a}"] = gap(K, ar, a, "G_multi")
            n[f"Gs{t}{a}"] = gap(K, ar, a, "G_small")
            n[f"Ga{t}{a}"] = gap(K, ar, a, "G_all")
            for mm in ("pix", "max", "dil", "strat", "ltt", "le_union", "joint_tc", "le_fixed_tc", "joint_multi",
                       "dil_tc", "strat_tc", "ltt_tc"):
                n[f"{mm}{t}{a}"] = rel(K, ar, a, mm)
            for mm in ("pix", "joint", "max", "ltt", "joint_multi", "dil"):
                n[f"S{mm}{t}{a}"] = m(K, ar, a, mm, "set_to_truth")["mean"]
    return n


def abstract():
    n = numbers()
    return (
        "Conformal risk control (CRC) can bound the expected fraction of lesion pixels a segmentation model misses, "
        "but large lesions dominate this average. We define lesion-miss and small-lesion-miss risks on ground-truth "
        "connected components, averaged over the images or volumes containing them, and show that the maximum "
        "of the per-risk CRC thresholds controls each of these risks, together with pixel risk, in expectation under "
        "exchangeability. We characterise protocols satisfying this assumption and propose a lesion-equalized "
        "prediction-set family. In a pre-registered study with frozen U-Net backbones on three public datasets, every "
        "guaranteed mean risk stayed below its level on held-out units. On Kvasir-SEG, where the detectors found most "
        f"polyps, pixel-level CRC let images with several polyps miss lesions at {n['Gr0.10']['mean']:.1f}--"
        f"{n['Gm0.10']['mean']:.1f} times the target level (95\\% CIs excluding 1). Joint control guaranteed the lesion "
        "and small-lesion risks, with mean joint sets about "
        f"{100 * (n['Sjointr0.15'] / n['Spixr0.15'] - 1):.0f}\\% larger than pixel-only sets at $\\alpha=0.15$ but "
        f"{n['Sjointr0.05'] / n['Spixr0.05']:.1f} times as large at $\\alpha=0.05$, and did not remove the multi-polyp gap, which it "
        "does not control. On FracAtlas and MSD liver tumours, detectors found about half of the lesions and guaranteed "
        "sets were nearly whole images. The tested lesion-equalized sets gave no material efficiency gain on real data."
    )


def results():
    n = numbers()
    K = "kvasir-seg"
    maxex = max((r[c]["mean"] - float(k.split("|")[0]))
                for key, v in S.items() if not key.startswith("_")
                for k, r in v["methods"].items() for c in r["claims"] if r[c]["mean"] == r[c]["mean"])
    pre = next(x for x in PRE[K]["gap"] if abs(x["alpha"] - 0.1) < 1e-9)
    t08m, t08r = "msd-hepatic-vessel|unet-mitb2", "msd-hepatic-vessel|unet-r34"
    gs08m, gs08r = S[t08m]["gap"]["0.10"]["G_small"], S[t08r]["gap"]["0.10"]["G_small"]
    lam_weak = min(lam(ds, ar, "0.10", "pix") for ds, _ in DS[1:] for ar, _ in AR)
    dil_rng = [rel(ds, ar, "0.10", "dil")["mean"] for ds, _ in DS[1:] for ar, _ in AR]
    pixw = [rel(ds, ar, "0.10", "pix")["mean"] for ds, _ in DS[1:] for ar, _ in AR]
    h2 = {f"{ar}{a}": cfg(K, ar)["H2"][a] for ar, _ in AR for a in ALPHAS}
    mp = {f"{ds}|{ar}": cfg(ds, ar)["missed_profile"]["missed"] for ds, _ in DS for ar, _ in AR}
    det = {f"{ds}|{ar}": cfg(ds, ar)["detectability"] for ds, _ in DS for ar, _ in AR}
    chosen_k = [cfg(K, ar)["chosen_family"].get("threshold", 0) for ar, _ in AR]
    # upper bounds on the lesion-equalized vs thresholding differences at alpha = 0.10, rounded up to 0.1 points
    le = math.ceil(1000 * max(abs(rel(K, ar, "0.10", "le_union")["mean"]) for ar, _ in AR)) / 10
    le_mx = math.ceil(1000 * max(abs(rel(K, ar, "0.10", "le_max")["mean"] - rel(K, ar, "0.10", "max")["mean"])
                                 for ar, _ in AR)) / 10
    nmul = cfg(K, "unet-r34")["H3"]["n_multi_pool"]
    h3k = cfg(K, "unet-r34")["H3"]["per_alpha"]["0.10"]
    h3k_m = cfg(K, "unet-mitb2")["H3"]["per_alpha"]["0.10"]
    bgk = m(K, "unet-r34", "0.10", "joint_multi", "fp_to_truth")["mean"], m(K, "unet-r34", "0.10", "joint", "fp_to_truth")["mean"]
    best7 = min(E7["rows"], key=lambda r: r["rel"])
    abl = {tag: abl_entry(tag, K) for tag, _ in ABL}
    gm_abl = [abl[t]["gap"]["G_multi"]["mean"] for t in abl if abl[t]]
    AB = S["_ablations"]
    fa_max = max(v["gap"]["0.10"][g]["mean"] for k, v in AB.items() if "|fracatlas|" in k for g in ("G_multi", "G_small"))
    t8 = {k.split("|")[0]: v["gap"]["0.10"]["G_small"]["mean"] for k, v in AB.items() if "|msd-hepatic-vessel|" in k}
    sim_bg = [r["frac_small_lesions_not_above_background_p99"] for r in E7D if r["gamma"] >= 1.5]
    h3w = {ds: max(cfg(ds, ar)["H3"]["per_alpha"][a]["set_change"] for ar, _ in AR for a in ALPHAS) for ds, _ in DS[1:]}
    EK, ET = eligible(K), eligible("msd-hepatic-vessel")
    rec_k = [pm(K, ar, "lesion_recall")["mean"] for ar, _ in AR]
    gs05_excl = [an.split("/")[1] for ar, an in AR if gap(K, ar, "0.05", "G_small")["ci"][0] > 1]
    gs05_incl = [an.split("/")[1] for ar, an in AR if gap(K, ar, "0.05", "G_small")["ci"][0] <= 1]
    # claims the prose makes about the data; the build stops if a rerun changes any of them
    for t in "rm":
        c1 = sorted(["dil", "ltt", "strat"], key=lambda mm: n[f"{mm}{t}0.10"]["mean"])
        c2 = sorted(["dil_tc", "ltt_tc", "strat_tc"], key=lambda mm: n[f"{mm}{t}0.10"]["mean"])
        assert c1 == [x[:-3] for x in c2], "C2 ordering of the baselines differs from C1"
        assert n[f"joint_tc{t}0.10"]["mean"] < n[f"le_union{t}0.10"]["mean"]
        assert all(n[f"{mm}{t}0.10"]["mean"] > 0 for mm in ("dil", "strat", "ltt"))
    assert all(gap(K, ar, a, "G_multi")["ci"][0] > 1 for ar, _ in AR for a in ALPHAS)
    assert all(gap(K, ar, "0.10", "G_all")["mean"] < 1 for ar, _ in AR)
    assert not any(v["supported"] for ds, _ in DS for ar, _ in AR for v in cfg(ds, ar)["H2"].values())
    assert all(cfg(ds, ar)["chosen_family"].get("threshold", 0) == 1.0 for ds, _ in DS[1:] for ar, _ in AR)
    # sensitivity analyses (Section 5.3 states these without repeating the table)
    jp = {t: abs(abl[t]["jp"]["mean"]) for t in abl if abl[t]}
    assert min(gm_abl) > 1 and abl["abltau75"]["gap"]["G_multi"]["mean"] > abl["main"]["gap"]["G_multi"]["mean"]
    assert jp["abltau25"] < jp["main"] < jp["abltau75"] and jp["ablq25"] > jp["main"] and jp["ablq50"] < 0.25 * jp["main"]
    assert jp["ablncal50"] > jp["ablncal100"] > jp["ablncal200"] >= jp["main"]
    assert abs(jp["ablconn4"] - jp["main"]) < 1e-9
    assert fa_max < 1 and t8["ablq25"] > 1 and t8["abltau75"] > 1
    assert all(abs(v["le_rel"]["0.1"]["mean"]) < 1e-12 for k, v in AB.items() if "|kvasir-seg|" not in k)
    # orderings stated in Section 5.2 without numbers
    for t in "rm":
        assert n[f"dil{t}0.10"]["mean"] > 0 and n[f"strat{t}0.10"]["mean"] > 0 and n[f"ltt{t}0.10"]["mean"] > 0
        assert n[f"max{t}0.10"]["mean"] < 0 and n[f"max{t}0.05"]["mean"] < 0 < n[f"max{t}0.15"]["mean"] and n[f"joint_tc{t}0.10"]["mean"] < 0
    assert all(abs(x) > 0.05 for x in pixw) and all(x < 0 for x in dil_rng)
    assert all(abs(h2[f"{ar}{a}"]["rel"]) < 0.01 for ar, _ in AR for a in ALPHAS)

    s = []
    s.append(r"\subsection{Pixel-to-lesion risk gap}\label{sec:gap}")
    s.append(table_gap())
    s.append(r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{gap.pdf}
\caption{RQ1 across levels: the gap ratios $G_{K\geq2}$ (top) and $G_{\mathrm{sml}}$ (bottom).}
\label{fig:gap}
\end{figure}""")
    s.append(
        "Table~\\ref{tab:gap} reports, at $\\alpha=0.10$, lesion-miss rates at the pixel-CRC threshold divided by "
        "$\\alpha$, as means over partitions and seeds with 95\\% clustered-bootstrap CIs: $G_{\\mathrm{all}}$ over all "
        "lesion-bearing units, $G_{K\\geq2}$ over units with at least two lesions, and $G_{\\mathrm{sml}}$, the small-lesion "
        "miss rate over units with a small lesion. A ratio $G>1$ means that the lesion-level risk exceeds the level to "
        "which the pixel guarantee was calibrated. In the same table, $\\bar n$ is the expected number of eligible test "
        "units per partition, $\\hat\\lambda^{\\mathrm{pix}}$ the pixel-CRC threshold, and H1 the pre-registered verdict "
        "defined in Section~\\ref{sec:stats}. Fig.~\\ref{fig:gap} shows the same estimates at all three levels, with a "
        f"dashed line at $G=1$; Supplementary Table~{SUP['gap']} lists them.")
    s.append(
        "Kvasir-SEG provides the strongest evidence for the pixel-to-lesion risk gap, because it is the only dataset "
        f"whose backbones find most lesions (lesion recall {min(rec_k):.2f}--{max(rec_k):.2f}). There, the pixel guarantee hides missed lesions in images with several "
        "polyps: at the pixel-CRC threshold their lesion-miss rate was about three times $\\alpha$ at $\\alpha=0.10$, and "
        "the CI excluded 1 at every level for both backbones, whereas the lesion-miss rate over all lesion-bearing images "
        "stayed below $\\alpha$ (Table~\\ref{tab:gap}, Fig.~\\ref{fig:gap}). Both H1 verdicts on Kvasir-SEG are PROCEED. "
        f"The preliminary random-forest estimate of {pre['G_multi']:.2f}, whose split range included 1 "
        "(Appendix~\\ref{app:prelim}), is thus replaced by a clear effect, although it rests on a small subgroup of about "
        f"{EK[('mul', 'test')]['mean_n']:.0f} multi-polyp test images per partition and is less precise than the estimate over "
        "all images. The small-lesion ratio was close to 1 "
        "at $\\alpha=0.10$ and above 1 at $\\alpha=0.05$, where its CI excluded 1 for "
        + (" and ".join(gs05_excl) if gs05_excl else "neither backbone")
        + (f" but not for {' and '.join(gs05_incl)}." if gs05_incl else "."))
    s.append(
        "The ratios on FracAtlas and Task08 are lower, but they are not evidence that pixel-level control protects "
        "lesions. With these weak detectors the pixel-CRC threshold is already "
        f"$\\hat\\lambda^{{\\mathrm{{pix}}}}\\ge{lam_weak:.3f}$, so the pixel set covers most of each image or volume, "
        "and a lesion-level risk can hardly exceed $\\alpha$ when the set approaches the whole unit. The registered H1 "
        "verdicts in Table~\\ref{tab:gap} must be read in this light: FracAtlas returns KILL for both backbones, and "
        "Task08 returns PROCEED with MiT-B2 through the small-lesion arm and INCONCLUSIVE with ResNet-34, but none of "
        "these verdicts measures lesion protection by an informative set.")

    s.append(r"\subsection{Joint risk control and baseline comparison}\label{sec:joint}")
    s.append(
        "\\textbf{Empirical risk control.} On the held-out test units, the mean of every risk a method guarantees "
        "(Table~\\ref{tab:methods}), each evaluated on its eligible units, stayed below its level for every method, level "
        f"and configuration; the largest seed-averaged excess over the level was {sg(maxex, 4)} (Supplementary "
        f"Table~{SUP['validity']}). This is empirical behaviour consistent with Theorem~\\ref{{thm:joint}} and "
        "Propositions~\\ref{prop:max}--\\ref{prop:frozen}, not a proof; the guarantees rest on the theorems and their "
        "assumptions. Guaranteed and empirical risks are kept apart throughout: a starred entry in Table~\\ref{tab:main} "
        "is a risk the method guarantees, and an unstarred entry is an empirical risk reported only for comparison, so a "
        "low unstarred value is not evidence of control. No method in Table~\\ref{tab:main} guarantees "
        "$R^{\\mathrm{mul}}$, which is exploratory, and the max-form rules do not guarantee $R^{\\mathrm{sml}}$. All "
        f"methods in Table~\\ref{{tab:main}} calibrate on the same {N_CAL_K} units (comparison C1), and each risk is averaged "
        f"over its eligible units: a calibration split, and likewise a test split, holds about {N_CAL_K} eligible units for "
        f"$R^{{\\mathrm{{pix}}}}$ and $R^{{\\mathrm{{les}}}}$, {EK[('sml', 'cal')]['mean_n']:.0f} for $R^{{\\mathrm{{sml}}}}$ "
        f"and {EK[('mul', 'cal')]['mean_n']:.0f} for $R^{{\\mathrm{{mul}}}}$. The column set/truth is "
        "$\\sum|S|/\\sum|Y|$, and $\\Delta$ is the paired relative change in set size versus joint thresholding "
        "(negative: smaller) with its 95\\% clustered-bootstrap CI.")
    s.append(table_main())
    s.append(r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{cost.pdf}
\caption{Change in set size relative to joint thresholding on Kvasir-SEG ($\alpha=0.10$).}
\label{fig:cost}
\end{figure}""")
    s.append(
        "\\textbf{Cost of the lesion-level guarantee.} On Kvasir-SEG the lesion-level constraints, not the pixel "
        "constraint, determine the joint threshold, so the joint set is larger than the pixel-only set by an amount that "
        "depends strongly on $\\alpha$ (Table~\\ref{tab:main} and Fig.~\\ref{fig:cost} give $\\alpha=0.10$). The paired "
        "change in size of the pixel-only set relative to the joint set (ResNet-34, MiT-B2) was "
        f"{pc(n['pixr0.15']['mean'])} and {pc(n['pixm0.15']['mean'])} at $\\alpha=0.15$ but "
        f"{pc(n['pixr0.05']['mean'], 0)} and {pc(n['pixm0.05']['mean'], 0)} at $\\alpha=0.05$, where the set-to-truth "
        f"ratio rises from {n['Spixr0.05']:.2f} to {n['Sjointr0.05']:.2f} (ResNet-34). This rise at small $\\alpha$ is a "
        "practical limitation of joint control. The preliminary impression that lesion-level control is nearly free "
        "(Appendix~\\ref{app:prelim}) came from a weak backbone for which the pixel constraint was binding.")
    s.append(
        "\\textbf{Comparison with other procedures (C1).} All efficiency conclusions rest on the primary comparison C1, "
        "in which every method calibrates on the same units. Under the same expectation guarantee, joint thresholding "
        "produced smaller sets on Kvasir-SEG at $\\alpha=0.10$ than the dilation family and the CCRA-S-style strata "
        "(Table~\\ref{tab:main}, Fig.~\\ref{fig:cost}). Fig.~\\ref{fig:cost} shows each method's change in set size "
        "relative to joint thresholding on a symmetric-log axis with 95\\% clustered-bootstrap CIs: negative values (left "
        "of zero) mean smaller sets, and pixel CRC is smaller because it does not guarantee the lesion-level risks. The "
        "rows below its dotted lines differ in guarantee or data budget: LTT, and the last two rows, which calibrate on "
        "50\\% more units (C2, C3). The strata calibrate each risk on about a third of the calibration "
        "units, so the finite-sample correction is paid on three smaller subgroups. LTT sets were larger, but LTT "
        "certifies a stronger statement: each population risk is at most $\\alpha$ with probability at least $1-\\delta$ "
        "over the calibration draw, whereas the other rules bound expected risks. Its larger sets are the price of that "
        "guarantee and do not show that LTT is less efficient for what it guarantees. The max-form rule gave smaller sets "
        "than the union rule at $\\alpha=0.10$ and 0.05 (slightly larger ones at 0.15), consistent with "
        "Proposition~\\ref{prop:max}: it does not guarantee "
        f"$R^{{\\mathrm{{sml}}}}$, and its small-lesion risk sat at the level "
        f"({m(K, 'unet-r34', '0.05', 'max', 'risk_small')['mean']:.3f} at $\\alpha=0.05$), within the "
        "$\\alpha/\\pi_{\\mathrm{sml}}$ bound. In the secondary comparison C2, where the baselines receive 50\\% more "
        "calibration units, joint thresholding became smaller, the ordering of the baselines was unchanged, and joint "
        "thresholding with C2 data was smaller than our lesion-equalized family with C1 data (Fig.~\\ref{fig:cost}).")
    s.append(table_weak())
    s.append(r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{regime.pdf}
\caption{Two regimes: set size (left) and detectability of missed lesions (right) against backbone lesion recall.}
\label{fig:regime}
\end{figure}""")
    s.append(
        "\\textbf{Weak-detector limitation.} On FracAtlas and Task08 the backbones found about half of the lesions "
        "(Table~\\ref{tab:backbones}), and the missed lesions received posteriors near zero. Any threshold-ranked set that "
        "meets the guarantee must then include almost every pixel (Table~\\ref{tab:weak}, Fig.~\\ref{fig:regime}). "
        "Set/truth in Table~\\ref{tab:weak} is a ratio of pixel counts, not a percentage or a coverage probability "
        "(Section~\\ref{sec:baselines}): about 255 is the whole letterboxed FracAtlas image and about 311 the whole Task08 "
        "volume. The left panel of Fig.~\\ref{fig:regime} plots the set-to-truth ratio of the jointly guaranteed set at "
        "$\\alpha=0.10$, on a log scale, against the lesion recall of the frozen backbone, with open markers for the "
        "dilation family; its right panel, used in Section~\\ref{sec:error}, gives the fraction of the lesions missed at "
        "the joint threshold whose peak posterior does not exceed the 99th percentile of the unit's background. The "
        "guarantees hold under their assumptions but are nearly vacuous: pixel-only and joint sets both covered most of "
        "the unit. On Task08 at $\\alpha=0.05$ the joint set is moreover the whole volume whenever the "
        f"small-lesion subgroup of the calibration split is below 19 units ({pc1(ET[('sml', 'cal')]['fallback']['0.05'])} of "
        "partitions; Section~\\ref{sec:stats}). In this regime, ranking pixels by distance to the predicted mask (the "
        "dilation family) gave smaller sets than thresholding under the same joint guarantee (last column of "
        "Table~\\ref{tab:weak}, with 95\\% clustered-bootstrap CIs); this comparison is exploratory and was not "
        "pre-registered. These datasets support the validity of the guarantees but not their "
        "practical usefulness, and they do not show that the risk-control method fails: they show what any guarantee "
        "can deliver with detectors of this quality.")

    s.append(r"\subsection{Ablation and sensitivity analysis}\label{sec:ablation}")
    s.append(table_ablation())
    s.append(
        "The sensitivity analyses in Table~\\ref{tab:ablation} are robustness checks of the main analysis, not independent "
        "confirmation. Except in its first row, their intervals come from a smaller bootstrap ($B=50$ resamples of 50 "
        "partitions) and are less precise than those of the main analysis. In Table~\\ref{tab:ablation}, ``Pixel vs "
        "joint'' is the paired change in set size of pixel-only CRC relative to joint thresholding, so its magnitude is "
        "the cost of the lesion-level guarantee; ``LE vs joint'' compares the lesion-equalized family with joint "
        "thresholding, and $P$ is the fraction of partitions in which the lesion-equalized set is smaller. Three "
        "findings matter for interpretation. "
        "First, the multi-lesion gap on Kvasir-SEG persisted under every definition and grew with a stricter hit "
        "criterion. Second, the cost of the "
        "lesion-level guarantee depends strongly on the definitions: it increased with $\\tau$ and with a narrower "
        "definition of small lesions, and nearly vanished when small lesions were defined by the median. Third, the cost "
        "increased as the calibration split shrank, because the finite-sample term $B/(n_r+1)$ is paid on the eligible "
        "subgroups, which shrink with it. Four-connected candidate regions changed nothing. The dataset-specific "
        f"analyses for FracAtlas and Task08 are in Supplementary Tables~{SUP['abl']}: every FracAtlas ratio stayed below 1, "
        "the Task08 small-lesion ratio exceeded 1 under the $q_{25}$ definition and at $\\tau=0.75$, and the "
        "lesion-equalized and threshold families tied exactly in every weak-detector ablation.")

    s.append(r"\subsection{Simulation and multi-lesion analysis}\label{sec:multi}")
    s.append(r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{sim.pdf}
\caption{Controlled simulation: change in set size of lesion-equalized sets relative to joint thresholding.}
\label{fig:sim}
\end{figure}""")
    s.append(
        "\\textbf{Simulation (synthetic data).} The controlled simulation, registered in Amendment~2, tests the mechanism "
        "of Proposition~\\ref{prop:le} on synthetic scans: lesion confidence scales as $a\\propto r^{\\gamma}$ with lesion "
        "radius $r$, with 600 scans and 500 partitions per setting. Fig.~\\ref{fig:sim} shows the change in set size of "
        "lesion-equalized sets relative to joint thresholding as small lesions become fainter, for two background "
        "levels $b$. Its shaded settings are those in which a post-hoc diagnostic finds that most small lesions do not "
        "exceed the background; the shading describes the simulation only and is not a clinically established "
        "threshold. Lesion-equalized sets were smaller than joint "
        f"thresholding when small lesions were moderately under-confident, by up to {abs(best7['rel']) * 100:.0f}\\% "
        f"($\\gamma={best7['gamma']:g}$, $b={best7['background']:g}$, $\\alpha={best7['alpha']:g}$; Fig.~\\ref{{fig:sim}}). "
        "The gain vanished in the harder settings, $\\gamma\\geq1.5$, where a post-hoc diagnostic shows that "
        f"{min(sim_bg) * 100:.0f}--{max(sim_bg) * 100:.0f}\\% of small lesions do not rise above the background, so the "
        f"pre-registered requirement of a gain at every $\\gamma\\geq1$ {E7['verdict'].lower()}. This simulated improvement "
        "does not establish a real-data benefit: it shows a mechanism under which a gain is possible, and on the real "
        "datasets no material gain was observed (Section~\\ref{sec:error}).")
    s.append(
        "\\textbf{Multi-lesion risk (exploratory).} This analysis is exploratory, and its results must be read with its "
        f"main limitation, subgroup size: the Kvasir-SEG pools hold only {min(nmul)}--{max(nmul)} multi-lesion images, below "
        "the pre-registered minimum of 30 in at least one seed, so H3 is not evaluable there, and a calibration split "
        f"contains about {EK[('mul', 'cal')]['mean_n']:.0f} of them. Adding $R^{{\\mathrm{{mul}}}}$, the lesion risk on units "
        "with $K\\geq2$, to the joint rule is valid by Lemma~\\ref{lem:sub}, and its guarantee applies to the multi-lesion "
        "subgroup. The joint rule of Theorem~\\ref{thm:joint} does not control this risk: under it the multi-lesion "
        f"miss rate on Kvasir-SEG stayed at {h3k['multi_joint'] / 0.10:.2f}$\\,\\alpha$ and "
        f"{h3k_m['multi_joint'] / 0.10:.2f}$\\,\\alpha$ at $\\alpha=0.10$, close to its level under pixel CRC. "
        "With so few eligible calibration units, the rule that adds $R^{\\mathrm{mul}}$ fell back to the "
        f"whole image in {pc1(EK[('mul', 'cal')]['fallback']['0.10'])} of partitions at $\\alpha=0.10$ and in "
        f"{pc1(EK[('mul', 'cal')]['fallback']['0.05'])} at $\\alpha=0.05$, and otherwise needed nearly complete sets. "
        f"At $\\alpha=0.10$ the multi-lesion miss rate fell from {h3k['multi_joint']:.3f} to {h3k['multi_with']:.3f} "
        f"(ResNet-34) and from {h3k_m['multi_joint']:.3f} to {h3k_m['multi_with']:.3f} (MiT-B2), but the sets grew by "
        f"{pcn(h3k['set_change'], 0)} and {pcn(h3k_m['set_change'], 0)} (background ratio "
        f"{bgk[1]:.2f}$\\to${bgk[0]:.2f}). This is not a practical improvement. On Task08 "
        f"({min(cfg('msd-hepatic-vessel', 'unet-r34')['counts']['n_multi'])}--"
        f"{max(cfg('msd-hepatic-vessel', 'unet-r34')['counts']['n_multi'])} multi-lesion units) and FracAtlas "
        f"({min(cfg('fracatlas', 'unet-r34')['counts']['n_multi'])}--{max(cfg('fracatlas', 'unet-r34')['counts']['n_multi'])}), "
        "H3 is formally supported at every level, but the joint sets already cover nearly the whole unit, so the result "
        "has limited practical meaning; the additional risk changed set size by at most "
        f"{h3w['msd-hepatic-vessel'] * 100:.1f}\\% on Task08 and by up to {h3w['fracatlas'] * 100:.0f}\\% on FracAtlas. "
        "The earlier post-hoc multi-lesion analysis (E8) of the preliminary study is reported in Supplementary "
        f"Table~{SUP['e8']}.")

    s.append(r"\subsection{Error analysis and practical implications}\label{sec:error}")
    s.append(
        "\\textbf{Lesion-equalized sets on real data (H2).} H2 was not supported in any configuration. On Kvasir-SEG the "
        f"tuning split selected plain thresholding in {span([100 * c for c in chosen_k])} of partitions, and the paired "
        "change in set size was below 1\\% in magnitude at every level. At $\\alpha=0.10$ (Table~\\ref{tab:main}), joint "
        f"lesion-equalized sets differed from joint thresholding by less than {le:.1f}\\%, and max-form lesion-equalized "
        f"sets from max-form thresholding by less than {le_mx:.1f} percentage points. The largest "
        f"effect, {pc(h2['unet-r340.15']['rel'], 2)} {cipc(h2['unet-r340.15']['ci'], 2)} at $\\alpha=0.15$ (ResNet-34), "
        f"has a CI below zero, but the set was smaller in only {h2['unet-r340.15']['p_smaller'] * 100:.0f}\\% of partitions "
        "and the change is negligible in size. On FracAtlas and Task08 the tuning split chose thresholding in every "
        "partition; on Task08 at $\\alpha=0.05$ the tuning split holds about "
        f"{ET[('sml', 'tune')]['mean_n']:.0f} small-lesion units, fewer than the 19 the rule needs, so every member falls "
        "back to the whole volume and the tie is resolved in favour of thresholding. The fixed family of comparison C3 "
        "tied joint thresholding calibrated on the same data (Fig.~\\ref{fig:cost}). For the construction tested "
        "here---candidate regions defined as connected components of the posterior above $t_0$, each rescaled by a power "
        "of its own peak, with $(\\rho,t_0)$ chosen from a 16-member grid---the data therefore do not show a material "
        "efficiency improvement. This is a limitation of the tested candidate-generation and score-transformation "
        "approach, not evidence that lesion-aware score transformations in general cannot reduce set size; we present "
        "the family as a valid candidate whose efficiency advantage remains unestablished.")
    ks = [f"{ds}|{ar}" for ds, _ in DS for ar, _ in AR]
    found_med = np.mean([cfg(K, ar)["missed_profile"]["found"]["median_size"] for ar, _ in AR])

    def rng(key, keys):
        v = sorted(mp[k][key] * 100 for k in keys)
        return f"{v[0]:.0f}--{v[-1]:.0f}\\%"

    s.append(
        "\\textbf{Which lesions are missed.} At the joint threshold ($\\alpha=0.10$), the lesions missed on Kvasir-SEG "
        f"were mostly small (median {mp['kvasir-seg|unet-r34']['median_size']:.0f} and "
        f"{mp['kvasir-seg|unet-mitb2']['median_size']:.0f} pixels, against {found_med:,.0f} for found lesions; "
        f"{rng('frac_small', ks[:2])} small). About half ({rng('frac_no_candidate_t0_0p2', ks[:2])}) had a peak posterior "
        "below 0.2, too low to form a candidate at $t_0=0.2$, and many had a peak no higher than the unit's "
        "99th-percentile background (Fig.~\\ref{fig:regime}, right). On Task08 the missed tumours were almost all small "
        f"({rng('frac_small', ks[4:])}), with median peak posterior below {max(mp[k]['median_peak'] for k in ks[4:]):.3f}. "
        f"Over all small lesions, the peak exceeded the background in {det['kvasir-seg|unet-r34']['small_peak_above_bg'] * 100:.0f}\\% "
        f"of cases on Kvasir-SEG but only {det['msd-hepatic-vessel|unet-r34']['small_peak_above_bg'] * 100:.0f}\\% on Task08. "
        "This is a diagnostic of the tested family: the lesions that drive the joint threshold either form no "
        "candidate under its rule or do not stand out from the background in the posterior, which is the failing "
        "precondition of Proposition~\\ref{prop:le}(d) and the same mechanism that ends the simulated gain at "
        "$\\gamma\\geq1.5$. It explains why this candidate-generation rule and peak normalisation failed here; it does "
        "not show that every score transformation must fail, and transformations that generate candidates differently "
        "or use image content beyond the posterior are not covered by this analysis.")
    s.append(
        "\\textbf{Practical implications.} (i) For multi-lesion tasks, pixel-level CRC should not be reported alone: on "
        "Kvasir-SEG its lesion-miss rate in multi-polyp images was several times the nominal level. (ii) In our "
        "experiments (C1, Kvasir-SEG), joint thresholding needed smaller sets than the dilation family and the stratified "
        "rule under the same expectation guarantee, but its cost rises steeply at small $\\alpha$ and with small eligible "
        "subgroups. (iii) With detectors that miss about half of the lesions, guaranteed sets cover nearly the whole unit "
        "and should be reported as uninformative. These are statistical findings on public datasets; the study does not "
        "evaluate clinical outcomes or deployment safety.")
    return "\n\n".join(s)


def conclusion():
    n = numbers()
    nk = cfg("kvasir-seg", "unet-r34")["counts"]["n_multi"]
    return (
        "\\textbf{Limitations.} The theorems hold under their stated assumptions; the following limitations restrict how "
        "far the empirical findings generalise, not the validity of the theorems. (1) Only Kvasir-SEG has detectors "
        "competent enough for the guarantees to be informative, and its pools contain few multi-lesion images "
        f"({min(nk)}--{max(nk)}), so "
        "the multi-lesion results rest on small subgroups and H3 was not evaluable. (2) The FracAtlas and Task08 backbones "
        "find about half of the lesions; the Task08 2-D slice networks are far from the state of the art for liver tumours. "
        "(3) Kvasir-SEG and FracAtlas have no patient identifiers, so their splits are image-wise; only Task08 is "
        "patient-disjoint. (4) The guarantees are marginal over eligible units---lesion-bearing units, or units with a "
        "small lesion---and are neither per-patient guarantees nor valid under distribution shift. (5) FracAtlas "
        "annotations are coarse fracture regions. (6) No clinical validation was performed; this is methodological "
        "research on public datasets, not a medical device.\n\n"
        "\\textbf{Conclusion.} We formulated lesion-miss and small-lesion-miss risks on ground-truth components, showed "
        "that the maximum of the per-risk CRC thresholds controls each of them, together with pixel FNR, in expectation "
        "over the units on which each risk is defined, and stated which protocols satisfy the required assumptions. The "
        "strongest real-data finding comes from Kvasir-SEG: pixel-level CRC left images with several polyps with "
        f"lesion-miss rates of about {n['Gr0.10']['mean']:.0f} times the nominal level. Joint control guaranteed the lesion "
        "and small-lesion risks at a set-size cost that grows as $\\alpha$ decreases, but it does not control the "
        "multi-lesion subgroup, whose gap closes only when that risk is added explicitly, at the price of near-total "
        "sets in our data. With weak detectors the guarantees remained valid but "
        "nearly vacuous. The tested lesion-equalized construction is valid, but it did not improve efficiency on the real "
        "datasets in this study; this is a limitation of its candidate-generation and peak-normalisation rule, and other "
        "score transformations remain to be evaluated. Future work should evaluate stronger 3-D segmentation models, larger multi-lesion cohorts "
        "with patient identifiers, score families that use information beyond posterior confidence, and extensions to "
        "distribution shift~\\cite{barber2023beyond}.")


def appendix():
    k = PRE["kvasir-seg"]
    g = next(x for x in k["gap"] if abs(x["alpha"] - 0.1) < 1e-9)
    return (
        "Before the revision, the method was evaluated with a CPU random-forest pixel classifier over a multiscale filter "
        "bank. Posteriors came from 5-fold cross-fitting at $256^2$, with 3 seeds $\\times$ 1000 partitions "
        "(tune/calibrate/test 20/40/40), and baselines were calibrated on tune $\\cup$ calibrate. By "
        "Proposition~\\ref{prop:crossfit}, these results certify validity only for units of the cross-fitted pool and say "
        "nothing about new patients scored by a deployed model. The backbone was weak (out-of-fold Dice "
        f"{PRE['kvasir-seg']['mean_dice']:.3f} on Kvasir-SEG and {PRE['fracatlas']['mean_dice']:.3f} on FracAtlas), and "
        "the pixel constraint was binding, so lesion-level control appeared nearly free. All methods met their guaranteed "
        "risks on held-out pool units. The multi-lesion gap on Kvasir-SEG was "
        f"{g['G_multi']:.2f} with a split range [{g['G_multi_lo']:.2f}, {g['G_multi_hi']:.2f}] that included 1; H2 failed at "
        "every level. A post-hoc analysis (E8, not pre-registered) of a multi-lesion risk is reported in Supplementary "
        f"Table~{SUP['e8']} and remains post hoc. Table~\\ref{{tab:prelim}} summarises these results (seeds 0--2 "
        "$\\times$ 1000 partitions, image-wise); its intervals are 2.5--97.5\\% split ranges, not confidence intervals, "
        "and $\\Delta$ is the change in set size versus joint thresholding.\n\n" + table_prelim())


def datasets_paragraph():
    k, f, t = cfg("kvasir-seg", "unet-r34"), cfg("fracatlas", "unet-r34"), cfg("msd-hepatic-vessel", "unet-r34")
    return (
        "We use three public datasets with different lesion structure. Kvasir-SEG~\\cite{jha2020kvasir} contains 1000 "
        "colonoscopy images with polyp masks; most images contain a single polyp. FracAtlas~\\cite{abedeen2023fracatlas} "
        "contains 719 radiographs with fracture polygons, which we rasterise at native resolution. Overlapping polygons merge "
        "into one component, and many images contain several fracture regions. MSD Task08~\\cite{antonelli2022msd} "
        "contains 303 portal-venous CT volumes of patients with liver tumours; we use the tumour label and ignore "
        "vessels. Many patients have several tumours. The pools contain "
        f"{k['counts']['n_units'][0]}, {f['counts']['n_units'][0]} and {t['counts']['n_units'][0]} units, of which "
        f"{min(k['counts']['n_multi'])}--{max(k['counts']['n_multi'])}, {min(f['counts']['n_multi'])}--{max(f['counts']['n_multi'])} "
        f"and {min(t['counts']['n_multi'])}--{max(t['counts']['n_multi'])} have at least two lesions (ranges over seeds; "
        "Table~\\ref{tab:backbones}).")


def deviations():
    return (
        "Three registered analyses were narrowed: the 17-member fixed-family ablation listed in Amendment~3 was not run, "
        "only the registered fixed member of comparison C3 was evaluated; the ablations used a smaller bootstrap ($B=50$ "
        "resamples of 50 partitions) than the main analysis; and calibration-size ablations larger than the calibration "
        "split (Task08: 100 and 200 units; FracAtlas: 200) use the whole split. The dilation-versus-threshold comparison in "
        "the weak-detector regime was not a registered hypothesis and is reported as exploratory. In the code used for the "
        "reported runs, a risk with no eligible calibration unit was dropped, in the CRC rules and in LTT, instead of "
        "setting $\\hat\\lambda^r=1$; for the "
        f"unstratified rules this event has probability below {sci(empty_prob())} per partition, whereas for the "
        "per-stratum rules of the stratified baseline it was not recorded. The released code follows Equation~\\eqref{eq:crc}. Operational events that did not affect any result, and the full registration history, "
        "are listed in the supplementary material.")


KEYWORDS = ("Conformal risk control, Distribution-free uncertainty, Lesion detection, "
            "Medical image segmentation, Prediction sets, Uncertainty quantification")
assert KEYWORDS.split(", ") == sorted(KEYWORDS.split(", "), key=str.lower), "keywords must be alphabetical"
N_CAL_K = int(0.4 * S["kvasir-seg|unet-r34"]["counts"]["n_units"][0])  # calibration units per Kvasir-SEG partition

# Code availability. Set PUMS_CODE_URL to the public repository (or anonymised review link) before submission.
CODE_URL = os.environ.get("PUMS_CODE_URL", "https://github.com/mahendra785/project").strip()


def code_availability():
    what = ("The complete implementation---losses, threshold calibration, score-family selection, training, "
            "statistical analysis and aggregation---together with the configuration, the pre-registration, the notebooks "
            "that download the data and train the backbones, and a script that regenerates every number, table and figure "
            "of this article")
    if CODE_URL:
        return (f"{what} are publicly available at \\url{{{CODE_URL}}}. Readers can reproduce all reported results by "
                "following the step-by-step instructions in the file REPRODUCING.md of that repository, either from the "
                "released results or by rerunning every experiment from scratch.")
    return (f"{what} are provided with this submission as supplementary code, with instructions for reproducing the "
            "principal results.")
CONTRIB = ("Mahendra Sajjan Choudhary: Conceptualization, Methodology, Software, Formal analysis, Investigation, "
           "Visualization, Writing -- original draft. Vansh Bhatiya: Software, Validation, Data curation, Investigation, "
           "Writing -- review \\& editing. Shashank Mouli Satapathy: Supervision, Methodology, Writing -- review \\& editing.")


# ------------------------------------------------------------------ supplementary
SUPP_METHODS = ["pix", "joint", "max", "dil", "strat", "le_union", "le_max", "joint_multi", "ltt", "pix_tc", "joint_tc",
                "max_tc", "dil_tc", "strat_tc", "joint_multi_tc", "ltt_tc", "le_fixed_tc"]


def supp_method_table(ds, dn, ar, an, a):
    rows = []
    for mm in SUPP_METHODS:
        r = cfg(ds, ar)["methods"][f"{a}|{mm}"]
        rv = r["rel_vs_joint"]
        dd = 2 if mm.startswith("le") else 1
        d = "ref." if mm == "joint" else f"{pc(rv['mean'], dd)} {cipc(rv['ci'], dd)}"
        cl = r["claims"]
        risk = " & ".join(f"{r[k]['mean']:.3f}{'*' if k in cl else ''}" for k in ("risk_pix", "risk_les", "risk_small", "risk_multi"))
        rows.append(f"{NAMES[mm]} & {risk} & {r['set_to_truth']['mean']:.2f} & {r['fp_to_truth']['mean']:.2f} & "
                    f"{r['lam']['mean'] / 1000:.3f} & {d} & {rv['p_smaller']:.2f} & {'yes' if r['valid'] else 'NO'} \\\\")
    return (f"\\begin{{table}}[p]\\caption{{{dn}, {an}, $\\alpha={a}$. Risks marked * are guaranteed by the method; "
            "the others are reported only. Set/truth and bg (background ratio) are ratios of pixel counts, not "
            "percentages. $\\hat\\lambda$: mean threshold; $\\Delta$: paired change in set size versus joint "
            "thresholding under C1 (negative: smaller), with 95\\% clustered-bootstrap CI; $P$: fraction of partitions "
            "in which the set is smaller than joint thresholding; valid: every guaranteed mean risk at most "
            "$\\alpha+0.005$. Rows marked C2 and C3 calibrate on tuning $\\cup$ calibration.}\n"
            "\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\begin{tabular}{@{}lcccccccccc@{}}\\toprule\nMethod & $R^{pix}$ & $R^{les}$ & $R^{sml}$ & $R^{mul}$ & set/truth & bg & "
            "$\\hat\\lambda$ & $\\Delta$ [95\\% CI] & $P$ & valid\\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}")


def supp_eligible_table():
    rows = []
    risks = (("les", "$R^{pix}$, $R^{les}$ (lesion-bearing)", "n_units"), ("sml", "$R^{sml}$ (small lesion)", "n_small"),
             ("mul", "$R^{mul}$ ($K\\geq2$)", "n_multi"))
    for ds, dn in DS:
        e = eligible(ds)
        cnt = cfg(ds, "unet-r34")["counts"]
        for r, name, key in risks:
            fb = " & ".join(f"{100 * e[(r, 'cal')]['fallback'][a]:.1f}" for a in NEED)
            fb2 = f"{100 * e[(r, 'tc')]['fallback']['0.05']:.1f}"
            rows.append(f"{dn} & {name} & {span(cnt[key], unit='')} & {e[(r, 'tune')]['mean_n']:.1f} & "
                        f"{e[(r, 'cal')]['mean_n']:.1f} & {e[(r, 'test')]['mean_n']:.1f} & {fb} & {fb2} \\\\")
    return ("\\begin{table}[h]\\caption{Eligible units per split and frequency of the CRC fallback. Pool: eligible pool "
            "units (range over seeds). Tune, cal, test: expected eligible units per partition (20/40/40 split). Fallback: "
            "probability (\\%) that a split holds fewer eligible units than the CRC rule needs (19, 9 and 6 at "
            "$\\alpha=0.05,0.10,0.15$), so that $\\hat\\lambda^r=1$; computed exactly from the hypergeometric distribution of "
            "uniformly random partitions and averaged over seeds, i.e.\\ the expected fraction of the 1000 partitions. "
            "The multi-lesion risk is used only by the exploratory rule Joint CRC $+\\,R^{mul}$, and the tuning split only "
            "for family selection.}\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}llrrrrrrrr@{}}\\toprule & & & \\multicolumn{3}{c}{expected eligible} & "
            "\\multicolumn{3}{c}{fallback, C1 (\\%)} & C2 (\\%)\\\\\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-10}\n"
            "Dataset & Risk & Pool & tune & cal & test & 0.05 & 0.10 & 0.15 & 0.05\\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}")


def supp_e8_table():
    rows = []
    for ds, dn in DS[:2]:
        for r in E8[ds]["summary"]:
            p = r["price_of_multi"]
            rows.append(f"{dn} & {r['alpha']:.2f} & {r['thr_joint3_str']['mean']:.2f} & {r['thr_joint3_multi']['mean']:.3f} & "
                        f"{r['thr_joint4_str']['mean']:.2f} & {r['thr_joint4_multi']['mean']:.3f} & "
                        f"{sg(100 * p['mean'])}\\% [{sg(100 * p['lo'])}, {sg(100 * p['hi'])}] \\\\")
    return ("\\begin{table}[h]\\caption{POST-HOC analysis E8 of the preliminary study (random forest, cross-fitted "
            "posteriors; not pre-registered). Joint thresholding with three risks versus four risks (adding $R^{mul}$). "
            "Set/truth and $R^{mul}$ are means over partitions; brackets are 2.5--97.5\\% split ranges, not confidence "
            "intervals. Validity holds only for units of the cross-fitted pool.}\\small\n"
            "\\begin{tabular}{@{}lcccccc@{}}\\toprule & & \\multicolumn{2}{c}{three risks} & \\multicolumn{2}{c}{four risks} & "
            "\\\\\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            "Dataset & $\\alpha$ & set/truth & $R^{mul}$ & set/truth & $R^{mul}$ & set-size cost of $R^{mul}$ [split range]"
            "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}")


def supplementary():
    parts = [r"""\documentclass[10pt]{article}
\usepackage[margin=2cm,a4paper]{geometry}
\usepackage{booktabs,graphicx,amsmath,amssymb}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\title{Supplementary material\\[4pt]\large Lesion-Level Joint Conformal Risk Control for Medical Image Segmentation}
\date{}
\begin{document}
\maketitle
All numbers are generated by \texttt{paper/journal/build.py} from the committed result files. Model seeds 0--2; 1000 tune/calibrate/test partitions (20/40/40) per seed; 95\% CIs from a clustered bootstrap of pool units ($B=200$ resamples $\times$ 100 partitions; ablations $B=50\times50$). Each risk is averaged over its eligible units: lesion-bearing units for $R^{pix}$ and $R^{les}$, units with a small lesion for $R^{sml}$, and units with at least two lesions for $R^{mul}$.

\section*{A. Backbones and eligible subgroups}"""]
    rows = []
    for ds, dn in DS:
        for ar, an in AR:
            pmx = cfg(ds, ar)["pool_metrics"]
            for i in range(3):
                rows.append(f"{dn} & {an} & {i} & " + " & ".join(f"{pmx[k]['per_seed'][i]:.3f}" for k in
                            ("dice", "iou", "lesion_recall", "lesion_precision")) + f" & {pmx['fp_components_per_unit']['per_seed'][i]:.2f} & "
                            f"{cfg(ds, ar)['small_cut'][i]:.0f} \\\\")
    parts.append("\\begin{table}[h]\\caption{Pool metrics per model seed (before calibration) and small-lesion cut "
                 "$a_{small}$ (33rd percentile of lesion areas in the $D_{train}$ labels, pixels or voxels at working "
                 "resolution).}\\small\n"
                 "\\begin{tabular}{@{}lllcccccc@{}}\\toprule Dataset & Backbone & Seed & Dice & IoU & Les.\\ recall & "
                 "Les.\\ prec. & FP/unit & $a_{small}$\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}")
    parts.append(supp_eligible_table())
    maxex = []
    for ds, dn in DS:
        for ar, an in AR:
            for a in ALPHAS:
                worst = max(cfg(ds, ar)["methods"][f"{a}|{mm}"][c]["mean"] - float(a)
                            for mm in SUPP_METHODS for c in cfg(ds, ar)["methods"][f"{a}|{mm}"]["claims"])
                maxex.append(f"{dn} & {an} & {a} & {sg(worst, 4)} \\\\")
    parts.append("\\section*{B. Empirical validity and the gap at every level}\\begin{table}[h]\\caption{Largest mean "
                 "test risk minus $\\alpha$ over all methods of comparisons C1--C3 and over all risks each method "
                 "guarantees. Negative: every guaranteed mean risk is below its level. These are empirical observations on "
                 "held-out units, not a proof of the guarantees.}\\small\n"
                 "\\begin{tabular}{@{}llcc@{}}\\toprule Dataset & Backbone & $\\alpha$ & max excess\\\\\\midrule\n" +
                 "\n".join(maxex) + "\n\\bottomrule\\end{tabular}\\end{table}")
    grows = []
    for ds, dn in DS:
        for ar, an in AR:
            for a in ALPHAS:
                g = cfg(ds, ar)["gap"][a]
                grows.append(f"{dn} & {an} & {a} & {f2(g['G_all']['mean'])} {ci2(g['G_all']['ci'])} & "
                             f"{f2(g['G_multi']['mean'])} {ci2(g['G_multi']['ci'])} & {f2(g['G_small']['mean'])} {ci2(g['G_small']['ci'])} \\\\")
    parts.append("\\begin{table}[h]\\caption{Gap ratios at the pixel-CRC threshold with 95\\% clustered-bootstrap "
                 "CIs.}\\small\n\\begin{tabular}{@{}lllccc@{}}\\toprule Dataset & Backbone & "
                 "$\\alpha$ & $G_{all}$ & $G_{K\\ge2}$ & $G_{sml}$\\\\\\midrule\n" + "\n".join(grows) +
                 "\n\\bottomrule\\end{tabular}\\end{table}")
    parts.append("\\section*{C. Sensitivity analyses on FracAtlas and Task08}Tables S5 and S6 repeat the sensitivity "
                 "analyses of Section~5.3 for FracAtlas and Task08. Except in the first row, intervals come from a smaller "
                 "bootstrap ($B=50$ resamples of 50 partitions). Pixel vs joint is the paired change in set size of "
                 "pixel-only CRC relative to joint thresholding; LE vs joint compares the lesion-equalized family with "
                 "joint thresholding, and $P$ is the fraction of partitions in which the lesion-equalized set is smaller.")
    parts.append(table_ablation("fracatlas", "tab:ablfa", "FracAtlas").replace("[t]", "[h]"))
    parts.append(table_ablation("msd-hepatic-vessel", "tab:ablt8", "MSD Task08").replace("[t]", "[h]"))
    parts.append("\\section*{D. Hypothesis verdicts and post-hoc analysis}")
    vr = []
    for ds, dn in DS:
        for ar, an in AR:
            c = cfg(ds, ar)
            h2 = ", ".join(f"{a}: {'supported' if v['supported'] else 'failed' if v['failed'] else 'partial'}" for a, v in c["H2"].items())
            h3 = ("not evaluable ($<$30 multi-lesion units)" if not c["H3"]["evaluable"] else
                  ", ".join(f"{a}: {'supported' if v['supported'] else 'not supported'}" for a, v in c["H3"]["per_alpha"].items()))
            vr.append(f"{dn} & {an} & {c['H1']} & {h2} & {h3} \\\\")
    parts.append("\\begin{table}[h]\\caption{Pre-registered verdicts (Amendment~3). H1: PROCEED if $G_{K\\ge2}$ or "
                 "$G_{sml}$ is at least 1.5 at $\\alpha=0.10$; KILL if both are at most 1.1 at every level; otherwise "
                 "INCONCLUSIVE. H2: supported if the mean paired change is negative, the set is smaller in at least 95\\% "
                 "of partitions, the CI lies below zero and validity holds; failed if the mean change is not negative; "
                 "partial otherwise. H3: evaluated only with at least 30 multi-lesion pool units in every seed.}\\scriptsize\n"
                 "\\begin{tabular}{@{}llp{1.8cm}p{4.6cm}p{4.6cm}@{}}\\toprule Dataset & Backbone & H1 & H2 & H3\\\\\\midrule\n"
                 + "\n".join(vr) + "\n\\bottomrule\\end{tabular}\\end{table}")
    parts.append(supp_e8_table())
    parts.append("\\clearpage\\section*{E. Full method tables}")
    for ds, dn in DS:
        for ar, an in AR:
            for a in ALPHAS:
                parts.append(supp_method_table(ds, dn, ar, an, a))
    parts.append(r"""\clearpage\section*{F. Registration history and protocol deviations}
\textbf{Registration history.} The pre-registration (\texttt{docs/research/preregistration-mgcrc.md}) was committed before any posterior existed. Amendment~1 (14 October 2026) fixed the CPU random-forest backbone of the preliminary study. Amendment~2 (14 October 2026) registered the controlled simulation before it was run. Amendment~3 (20 October 2026) registered the revision protocol of this paper---frozen deep backbones, the equal-data comparisons C1--C3, H1--H3 with their decision rules, the clustered bootstrap and the ablations---before any revised result existed.

\textbf{Deviations that affect the scope of the analyses.} (i) The 17-member fixed-family ablation of Amendment~3 was not run; only the registered fixed member of comparison C3 was evaluated, and the posteriors were not retained after the compute sessions. (ii) Ablations used a smaller bootstrap ($B=50$ resamples of 50 partitions) than the main analysis ($B=200\times100$). (iii) Calibration-size ablations larger than the calibration split (Task08: 100 and 200 $>$ 72 units; FracAtlas: 200 $>$ 172) use the whole split. (iv) The comparison of the dilation and threshold families in the weak-detector regime was not a registered hypothesis and is exploratory. (v) In the code used for the reported runs, a risk with no eligible calibration unit was dropped, in the CRC rules and in LTT, instead of setting $\hat\lambda^r=1$. For the unstratified rules this event has probability below """ + sci(empty_prob()) + r""" per partition (Table~S2); for the per-stratum rules of the stratified baseline it was not recorded. The released code implements the stated rule for CRC and LTT.

\textbf{Operational events without effect on any result.} Three Kaggle sessions stopped before any training or result---two because of notebook packaging and one because no accelerator was attached---and were restarted unchanged. Fifty-nine lesion-free FracAtlas radiographs are truncated in the official archive and were skipped when reading the training negatives.

\section*{G. Reproduction}
\textbf{Code.} Training: \texttt{backend/scripts/experiments/kaggle\_train.py}; calibration and bootstrap: \texttt{mgcrc\_v2.py}; aggregation and hypothesis verdicts: \texttt{aggregate\_v2.py}; losses, score families and the threshold rule: \texttt{backend/app/calibration/}. The three Kaggle notebooks (\texttt{scripts/kaggle/pums\_session*.ipynb}, with \texttt{HOW\_TO\_RUN.md}) download every dataset from its official source, check SHA-256 sums or file sizes, train the backbones and write every result file. They embed the code version used for the reported runs, so they reproduce the reported numbers exactly; the repository version of \texttt{mgcrc\_v2.py} additionally contains the corrected empty-subgroup rule (Section~F, item v). \texttt{paper/journal/build.py} regenerates every number, table and figure of the manuscript and this supplement.

\textbf{Configuration.} Seeds $m\in\{0,1,2\}$. The $D_{train}/D_{pool}$ split (40/60) uses \texttt{numpy.random.default\_rng(5000+m)}, with 10\% of $D_{train}$ held out for checkpoint selection; the 1000 tune/calibration/test partitions and the bootstrap use \texttt{default\_rng(7919m+17)}; network initialisation uses seed $m$. U-Net with ResNet-34 or MiT-B2 encoder (\texttt{segmentation\_models\_pytorch} 0.5.0, ImageNet weights); AdamW, learning rate $3\times10^{-4}$, one-cycle schedule, batch size 8, 60 epochs (25 for Task08), binary cross-entropy plus soft Dice, flip and gamma augmentation, horizontal-flip test-time averaging. Inputs: grey level, zero-padded and resized to $352^2$ (Kvasir-SEG, FracAtlas); CT windowed to $[-100,250]$ HU and resized in-plane to $256^2$ (Task08). Lesions: full-connectivity components of at least 10 pixels or voxels at native resolution. Threshold grid: 1001 equally spaced values of $\lambda\in[0,1]$; loss bound $B=1$; LTT with Hoeffding--Bentkus $p$-values, fixed-sequence testing and $\delta=0.1$; three strata at the terciles of the calibration units' total predicted mass. Hardware: Kaggle notebooks with an NVIDIA Tesla T4 GPU (16\,GB), 4 CPU cores and 31\,GB of RAM; Python 3.13, PyTorch """ + TORCH + r""". Run times: training one backbone took 7--13 minutes for the 2-D datasets and 23--30 minutes for Task08; after a one-off per-unit sort, each calibration partition takes about a millisecond.
\end{document}""")
    return "\n\n".join(parts)


def summary_md():
    lines = ["# Revision results (frozen deep backbones), generated by paper/journal/build.py: do not edit", ""]
    for ds, dn in DS:
        for ar, an in AR:
            c = cfg(ds, ar)
            lines.append(f"## {dn} · {an}")
            lines.append(f"Pool {c['counts']['n_units'][0]} · multi-lesion {c['counts']['n_multi']} · Dice "
                         f"{pm(ds, ar, 'dice')['mean']:.3f} · lesion recall {pm(ds, ar, 'lesion_recall')['mean']:.3f} · "
                         f"H1 **{c['H1']}** · H2 " + ", ".join(f"{a}: {'S' if v['supported'] else 'F' if v['failed'] else 'P'}"
                                                            for a, v in c["H2"].items()))
            lines.append("")
            lines.append("| α | G_K≥2 [CI] | G_small [CI] | pixel vs joint | LTT vs joint | dilation vs joint | lesion-eq vs joint |")
            lines.append("|---|---|---|---|---|---|---|")
            for a in ALPHAS:
                g = c["gap"][a]
                r = {mm: c["methods"][f"{a}|{mm}"]["rel_vs_joint"]["mean"] for mm in ("pix", "ltt", "dil", "le_union")}
                lines.append(f"| {a} | {g['G_multi']['mean']:.2f} [{g['G_multi']['ci'][0]:.2f}, {g['G_multi']['ci'][1]:.2f}] | "
                             f"{g['G_small']['mean']:.2f} [{g['G_small']['ci'][0]:.2f}, {g['G_small']['ci'][1]:.2f}] | "
                             + " | ".join(f"{100 * r[x]:+.1f}%" for x in ("pix", "ltt", "dil", "le_union")) + " |")
            lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ overleaf
# Used only when sn-jnl.cls is absent (e.g. a blank Overleaf project): article layout plus the sn-jnl front-matter macros.
FALLBACK_CLS = r"""
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{snfallback}[2026/10/26 article-based fallback used when sn-jnl.cls is absent]
\DeclareOption*{}\ProcessOptions\relax
\LoadClass[11pt,a4paper]{article}
\RequirePackage[margin=2.5cm]{geometry}
\RequirePackage{amsmath,amsthm}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{url}
\bibliographystyle{unsrtnat}
\newtheoremstyle{thmstyleone}{\topsep}{\topsep}{\itshape}{}{\bfseries}{.}{.5em}{}
\newtheoremstyle{thmstyletwo}{\topsep}{\topsep}{\normalfont}{}{\itshape}{.}{.5em}{}
\newtheoremstyle{thmstylethree}{\topsep}{\topsep}{\normalfont}{}{\bfseries}{.}{.5em}{}
\newcommand{\fnm}[1]{#1}\newcommand{\sur}[1]{#1}
\newcommand{\orgdiv}[1]{#1}\newcommand{\orgname}[1]{#1}\newcommand{\orgaddress}[1]{#1}
\newcommand{\street}[1]{#1}\newcommand{\city}[1]{#1}\newcommand{\postcode}[1]{#1}
\newcommand{\state}[1]{#1}\newcommand{\country}[1]{#1}
\newcommand{\backmatter}{}
\newcommand{\bmhead}[1]{\paragraph*{#1}}
\newenvironment{appendices}{\appendix}{}
\makeatletter
\def\sn@authors{}\def\sn@affils{}\def\sn@sep{}
\newif\ifsn@corr
\renewcommand{\author}{\@ifstar{\sn@corrtrue\sn@author}{\sn@corrfalse\sn@author}}
\newcommand{\sn@author}[2][]{%
  \protected@xdef\sn@authors{\sn@authors\sn@sep#2\noexpand\textsuperscript{#1\ifsn@corr,*\fi}}%
  \gdef\sn@sep{, }}
\newcommand{\email}[1]{\ifsn@corr\gdef\sn@corremail{#1}\fi}
\renewcommand{\title}[2][]{\gdef\@title{#2}}
\newcommand{\affil}[2][]{\g@addto@macro\sn@affils{\textsuperscript{#1}#2\par}}
\renewcommand{\abstract}[1]{\gdef\sn@abstract{#1}}
\newcommand{\keywords}[1]{\gdef\sn@keywords{#1}}
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries\@title\par}\vskip 1.2em
    {\large\sn@authors\par}\vskip .6em
    {\small\sn@affils}
    \ifdefined\sn@corremail{\small\textsuperscript{*}Corresponding author: \texttt{\sn@corremail}\par}\fi
  \end{center}
  \begin{quote}\small\noindent\textbf{Abstract.} \sn@abstract\par\medskip
    \noindent\textbf{Keywords:} \sn@keywords\end{quote}\bigskip}
\makeatother
"""


def overleaf_main(src):
    """manuscript.tex for the Overleaf project: references stay in refs.bib, loaded by \\bibliography{refs} at the end."""
    head = ("% main.tex for Overleaf (generated by paper/journal/build.py).\n"
            "% Project files: main.tex, refs.bib (all references), snfallback.cls, and the figures\n"
            "% gap.pdf, cost.pdf, regime.pdf, sim.pdf, mech.pdf. Uses sn-jnl.cls when the Springer Nature\n"
            "% template is present, otherwise the plain fallback layout in snfallback.cls.\n\n")
    safe = (r"\graphicspath{{figures/}{./}}" "\n"
            r"\newcommand{\safegraphic}[1]{\IfFileExists{figures/#1}{\includegraphics[width=\linewidth]{#1}}"
            r"{\IfFileExists{#1}{\includegraphics[width=\linewidth]{#1}}"
            r"{\fbox{\parbox[c][4cm][c]{0.95\linewidth}{\centering Figure file \texttt{#1} not uploaded yet}}}}}")
    assert "\\bibliography{refs}" in src and "filecontents" not in src
    out = src.replace(r"\documentclass[sn-mathphys-num]{sn-jnl}",
                      r"\IfFileExists{sn-jnl.cls}{\documentclass[sn-mathphys-num]{sn-jnl}}{\documentclass{snfallback}}", 1)
    assert out != src, "documentclass line not found"
    out = out.replace(r"\graphicspath{{figures/}}", safe, 1)
    out = re.sub(r"\\includegraphics\[width=\\linewidth\]\{([a-z]+\.pdf)\}", r"\\safegraphic{\1}", out)
    return head + out


# ------------------------------------------------------------------ Springer Nature template (journal's own PUMS.tex)
# The template from the journal (springer_template/: PUMS.tex, sn-jnl.cls, sn-mathphys.bst, optidef.sty) is kept
# unchanged; only its placeholders are replaced by this paper's content, and its class files are copied as they are.
SPRINGER = HERE / "springer_template"
SPRINGER_EXTRA = r"""%%%% Additional packages and macros used by this article
\usepackage{booktabs}
\usepackage{multirow}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Prob}{\mathbb{P}}
\newcommand{\1}{\mathbb{1}}
\newcommand{\lhat}{\hat\lambda}
\newcommand{\pix}{\mathrm{pix}}
\newcommand{\les}{\mathrm{les}}
\newcommand{\sml}{\mathrm{sml}}
\newcommand{\mul}{\mathrm{mul}}
%%%%
"""
SPRINGER_THEOREMS = r"""
%% Additional environments used by this article (Propositions are numbered on their own, as cited in the review)
\theoremstyle{thmstyleone}%
\newtheorem{lemma}{Lemma}%
\newtheorem{corollary}{Corollary}%
\theoremstyle{thmstylethree}%
\newtheorem{assumption}{Assumption}%
"""


def _between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def _decl(src, name):
    """Text of one '\\subsection*{name} ...' declaration line of the manuscript."""
    m = re.search(r"\\subsection\*\{" + re.escape(name) + r"\} (.+)", src)
    assert m, name
    return m.group(1).strip()


def springer_export(src):
    """Fill the journal's Springer Nature template (springer_template/PUMS.tex) with the generated manuscript."""
    t = (SPRINGER / "PUMS.tex").read_text(encoding="utf-8")

    def put(old, new):
        nonlocal t
        assert t.count(old) == 1, old[:60]
        t = t.replace(old, new)

    title = re.search(r"\\title\[[^\]]*\]\{[^\n]*\}\n", src).group(0)
    affil = re.search(r"\\affil\[1\]\{[^\n]*\}\n", src).group(0)
    body = _between(src, "\\maketitle\n", "\\backmatter").split("\\maketitle\n", 1)[1].strip()
    supp = _between(src, "\\bmhead{Supplementary information}", "\\section*{Declarations}").strip()
    appx = _between(src, "\\begin{appendices}", "\\end{appendices}") + "\\end{appendices}"
    abstract = re.search(r"\\abstract\{(.*)\}\n\n\\keywords", src, re.S).group(1)

    put("\\newcommand{\\xmark}{\\ding{55}}%\n%%%%\n", "\\newcommand{\\xmark}{\\ding{55}}%\n%%%%\n" + SPRINGER_EXTRA)
    put("\\newtheorem{proposition}[theorem]{Proposition}% \n", "\\newtheorem{proposition}{Proposition}% \n")
    put("\\newtheorem{definition}{Definition}%\n", "\\newtheorem{definition}{Definition}%\n" + SPRINGER_THEOREMS)
    put(re.search(r"%\\title\[ENHANCING[^\n]*\n\n", t).group(0), "")
    put(re.search(r"\\title\[GAN[^\n]*\n", t).group(0), title)
    # author lines (with the authors' e-mail addresses) come from the manuscript, followed by its affiliation
    authors = re.findall(r"\\author\*?\[1\]\{[^\n]*\}\n", src)
    tpl_authors = re.findall(r"\\author\*?\[1\]\{[^\n]*\}\n", t)
    assert len(authors) == len(tpl_authors) == 3, (authors, tpl_authors)
    put("".join(tpl_authors), "".join(authors) + "\n" + affil)
    # sn-jnl's author font switches on \boldmath and does not switch it off, so the abstract's math would print bold
    put(re.search(r"\\abstract\{Cataracts.*?\}\n", t, re.S).group(0), "\\abstract{\\unboldmath " + abstract + "}\n")
    put(re.search(r"\\keywords\{Cataract[^\n]*\n", t).group(0), "\\keywords{" + KEYWORDS + "}\n")
    put(_between(t, "\\section{Introduction}\\label{sec1}", "\\section{Declarations}"),
        body + "\n\n\\bmhead{Supplementary information}\n" + supp.split("}", 1)[1].strip() + "\n\n")
    decl = (
        "\\section{Declarations}\n"
        f"\\subsection{{Ethical Approval}} {_decl(src, 'Ethics approval and consent to participate')}\n\n"
        f"\\subsection{{Availability of supporting data}} {_decl(src, 'Data availability')}\n\n"
        f"\\subsection{{Code availability}} {_decl(src, 'Code availability')}\n\n"
        f"\\subsection{{Competing interests}} {_decl(src, 'Competing interests')}\n\n"
        f"\\subsection{{Funding}} {_decl(src, 'Funding')}\n\n"
        f"\\subsection*{{Authors' contributions}} {_decl(src, 'Authors' + chr(39) + ' contributions')}\n\n"
        + appx + "\n\n")
    put(_between(t, "\\section{Declarations}", "%\\bibliographystyle{spbasic}"), decl)

    # The journal's text block is 119 mm wide: the three widest tables are replaced by their upright narrow variants
    # (CIs and standard deviations on a second line), and the sensitivity table gets tighter column padding.
    for label, table in (("tab:backbones", table_backbones), ("tab:gap", table_gap), ("tab:weak", table_weak)):
        i = t.index("\\label{" + label + "}")
        a = t.rindex("\\begin{table}[t]", 0, i)
        b = t.index("\\end{table}", i) + len("\\end{table}")
        t = t[:a] + table(narrow=True) + t[b:]
    put("\\label{tab:ablation}\n\\footnotesize\n", "\\label{tab:ablation}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n")
    # On the smaller journal page, top-only floats left half-empty pages (4, 10, 12); let floats go here or on a page.
    for env in ("table", "figure", "algorithm"):
        t = t.replace("\\begin{" + env + "}[t]", "\\begin{" + env + "}[!htbp]")

    # Under sn-jnl a bare '|' before some tokens (e.g. '|\alpha|') stops XeLaTeX with "Missing $"; every '|' of the
    # body is in math (checked), so write it as the equivalent \vert, which compiles with pdfLaTeX and XeLaTeX alike.
    head, body = t.split("\\begin{document}", 1)
    assert "\\|" not in body
    t = head + "\\begin{document}" + body.replace("|", "\\vert ")

    out = HERE / "springer"
    out.mkdir(exist_ok=True)
    (out / "PUMS.tex").write_text(t, encoding="utf-8")
    (out / "PUMS.bib").write_text((HERE / "refs.bib").read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("sn-jnl.cls", "sn-mathphys.bst", "optidef.sty"):  # copied unchanged
        (out / name).write_bytes((SPRINGER / name).read_bytes())
    for p in FIG.glob("*.pdf"):
        (out / p.name).write_bytes(p.read_bytes())
    with zipfile.ZipFile(HERE / "springer_project.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, p.name)
    return t


# ------------------------------------------------------------------ main
def main():
    FIG.mkdir(exist_ok=True)
    fig_gap()
    fig_cost()
    fig_regime()
    env = json.loads(next((REPO / "docs/research/results_v2/manifests").glob("kvasir-seg__unet-r34_seed0.json")).read_text())["env"]
    tokens = dict(
        ABSTRACT=abstract(), KEYWORDS=KEYWORDS, RESULTS=results(), CONCLUSION=conclusion(), APPENDIX_PRELIM=appendix(),
        DATASETS_PARAGRAPH=datasets_paragraph(), TABLE_BACKBONES=table_backbones(), DEVIATIONS=deviations(), FALLBACK=fallback_paragraph(),
        TORCH_VERSION=env["torch"].split("+")[0], SMP_VERSION=env["smp"], CONTRIBUTIONS=CONTRIB,
        ETHICS_TASK08=", MSD Task08", DATA_TASK08=" and MSD Task08~\\cite{antonelli2022msd}",
        CODE_AVAILABILITY=code_availability(),
    )
    src = (HERE / "manuscript.src.tex").read_text(encoding="utf-8")
    src = re.sub(r"\A(%[^\n]*\n)+", "", src)  # drop the source-file banner
    for k, v in tokens.items():
        src = src.replace(f"@@{k}@@", v)
    left = re.findall(r"@@[A-Z0-9_]+@@", src)
    assert not left, left
    (HERE / "manuscript.tex").write_text(src, encoding="utf-8")
    ol = HERE / "overleaf"
    ol.mkdir(exist_ok=True)
    (ol / "main.tex").write_text(overleaf_main(src), encoding="utf-8")
    (ol / "refs.bib").write_text((HERE / "refs.bib").read_text(encoding="utf-8"), encoding="utf-8")
    (ol / "snfallback.cls").write_text(FALLBACK_CLS.lstrip(), encoding="utf-8")
    for p in FIG.glob("*.pdf"):
        (ol / p.name).write_bytes(p.read_bytes())
    with zipfile.ZipFile(HERE / "overleaf_project.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ol.iterdir()):
            z.write(p, p.name)
    if SPRINGER.is_dir():
        springer_export(src)
    (HERE / "supplementary.tex").write_text(supplementary(), encoding="utf-8")
    (REPO / "docs/research/results_v2/summary_v2.md").write_text(summary_md(), encoding="utf-8")
    words = len(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", tokens["ABSTRACT"]).split())
    print(f"manuscript.tex written; abstract ~{words} words; figures: {sorted(p.name for p in FIG.glob('*.pdf'))}")


if __name__ == "__main__":
    main()
