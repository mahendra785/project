"""Write paper/generated/*.tex from the committed result files (no hand-typed numbers).

    python scripts/experiments/paper_macros.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import REPO_ROOT  # noqa: E402

RES = REPO_ROOT / "docs" / "research" / "results"
OUT = REPO_ROOT / "paper" / "generated"
SHORT = {
    "pixel_crc": "Pixel CRC~\\cite{angelopoulos2024crc}",
    "union_threshold": "Joint, threshold",
    "max_threshold": "Joint max-form, threshold",
    "ltt_threshold": "LTT~\\cite{angelopoulos2021ltt}, threshold",
    "union_dilation": "Joint, morphological~\\cite{mossina2025morph}",
    "union_stratified": "Joint, CCRA-S strata~\\cite{luo2025ccra}",
    "ours_union": "\\textbf{Ours}, joint",
    "ours_max": "\\textbf{Ours}, max-form",
}
ORDER = list(SHORT)
DS_NAME = {"fracatlas": "FracAtlas", "kvasir-seg": "Kvasir-SEG"}


def f(v, d=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def macro_name(s: str) -> str:
    return "".join(ch for ch in s.title() if ch.isalpha())


def main_table(block, alpha) -> str:
    rows = {r["method"]: r for r in block["rows"] if r["alpha"] == alpha}
    out = ["\\begin{tabular}{lccccr}", "\\toprule",
           "Method & pixel FNR & lesion miss & small miss & set/truth & $\\Delta$ vs joint-thr. \\\\", "\\midrule"]
    for m in ORDER:
        r = rows.get(m)
        if r is None:
            continue
        def cell(k, claimed):
            v = r[k]
            s = f(v)
            return f"\\textcolor{{red}}{{{s}}}" if claimed and v is not None and np.isfinite(v) and v > alpha + 0.005 else s
        joint = m != "pixel_crc"
        delta = "ref." if m == "union_threshold" else f"{100*r['rel']:+.1f}\\%"
        out.append(f"{SHORT[m]} & {cell('risk_pix', True)} & {cell('risk_les', joint)} & {cell('risk_small', joint)} & "
                   f"{f(r['set_to_truth'], 2)} & {delta} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def gap_table(blocks) -> str:
    out = ["\\begin{tabular}{llccc}", "\\toprule",
           "Dataset & $\\alpha$ & $G$ (all) & $G\\mid K{\\ge}2$ & $G_{\\mathrm{small}}$ \\\\", "\\midrule"]
    for b in blocks:
        for g in b["gap"]:
            out.append(f"{DS_NAME.get(b['dataset'], b['dataset'])} & {g['alpha']:.2f} & {f(g['G'],2)} & "
                       f"{f(g['G_multi'],2)} [{f(g['G_multi_lo'],2)}, {f(g['G_multi_hi'],2)}] & "
                       f"{f(g['G_small'],2)} [{f(g['G_small_lo'],2)}, {f(g['G_small_hi'],2)}] \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out.append("\\end{tabular}")
    return "\n".join(out)


def sim_table(sim) -> str:
    rows = sim["rows"]
    gammas = sorted({r["gamma"] for r in rows})
    out = ["\\begin{tabular}{l" + "c" * len(gammas) + "}", "\\toprule",
           "background, $\\alpha$ & " + " & ".join(f"$\\gamma={g:g}$" for g in gammas) + " \\\\", "\\midrule"]
    for b in sorted({r["background"] for r in rows}):
        for a in sorted({r["alpha"] for r in rows}):
            cells = []
            for g in gammas:
                r = next(x for x in rows if x["gamma"] == g and x["background"] == b and x["alpha"] == a)
                s = f"{100*r['rel']:+.1f}"
                cells.append(f"\\textbf{{{s}}}" if r["p_smaller"] >= 0.95 else s)
            out.append(f"$b={b:g}$, $\\alpha={a:.2f}$ & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = json.loads((RES / "mgcrc_summary.json").read_text())
    sim = json.loads((RES / "simulation_E7.json").read_text())
    macros = []
    for b in blocks:
        tag = macro_name(DS_NAME.get(b["dataset"], b["dataset"]))
        (OUT / f"table_main_{b['dataset']}.tex").write_text(main_table(b, 0.1))
        m0 = b["meta"][0]
        macros += [f"\\newcommand{{\\{tag}Units}}{{{m0['n_units']}}}",
                   f"\\newcommand{{\\{tag}Multi}}{{{m0['n_multi']}}}",
                   f"\\newcommand{{\\{tag}WithSmall}}{{{m0['n_with_small']}}}",
                   f"\\newcommand{{\\{tag}SmallCut}}{{{m0['small_cut_px']:.0f}}}",
                   f"\\newcommand{{\\{tag}Seeds}}{{{len(b['meta'])}}}",
                   f"\\newcommand{{\\{tag}Reps}}{{{m0['reps']}}}",
                   f"\\newcommand{{\\{tag}Dice}}{{{f(b.get('mean_dice'))}}}",
                   f"\\newcommand{{\\{tag}HOne}}{{{b['verdicts']['H1']}}}"]
        for a, v in b["verdicts"]["H2"].items():
            macros.append(f"\\newcommand{{\\{tag}HTwo{macro_name(str(int(float(a)*100)).replace('5','five').replace('1','one').replace('0','zero'))}}}{{{v}}}")
        for r in b["rows"]:
            if r["alpha"] == 0.1 and r["method"] in ("ours_union", "ltt_threshold", "union_dilation", "pixel_crc"):
                macros.append(f"\\newcommand{{\\{tag}{macro_name(r['method'].replace('_', ' '))}Rel}}{{{100*r['rel']:+.1f}\\%}}")
    (OUT / "table_gap.tex").write_text(gap_table(blocks))
    (OUT / "table_sim.tex").write_text(sim_table(sim))
    e8 = [r"\begin{tabular}{llcccc}", r"\toprule",
          r"Dataset & $\alpha$ & miss$\mid K{\ge}2$, joint & miss$\mid K{\ge}2$, +cond. & set/truth, joint & "
          r"set/truth, +cond. \\",
          r"\midrule"]
    for ds in ("fracatlas", "kvasir-seg"):
        path = RES / f"posthoc_E8_{ds}_rf.json"
        if not path.exists():
            continue
        for s in json.loads(path.read_text())["summary"]:
            price = 100 * s["price_of_multi"]["mean"]
            e8.append(f"{DS_NAME[ds]} & {s['alpha']:.2f} & {f(s['thr_joint3_multi']['mean'])} & "
                      f"{f(s['thr_joint4_multi']['mean'])} & {f(s['thr_joint3_str']['mean'], 2)} & "
                      f"{f(s['thr_joint4_str']['mean'], 2)} ({price:+.0f}" + r"\%) \\")
    e8 += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_e8.tex").write_text("\n".join(e8))
    macros.append(f"\\newcommand{{\\SimVerdict}}{{{sim['verdict']}}}")
    best = min(sim["rows"], key=lambda r: r["rel"])
    macros.append(f"\\newcommand{{\\SimBest}}{{{100*best['rel']:.1f}\\%}}")
    macros.append(f"\\newcommand{{\\SimBestGamma}}{{{best['gamma']:g}}}")
    worst = max(sim["rows"], key=lambda r: r["rel"])
    macros.append(f"\\newcommand{{\\SimWorst}}{{{100*worst['rel']:+.1f}\\%}}")
    # abstract sentence, assembled from verdicts only
    def h2_word(b):
        vs = set(b["verdicts"]["H2"].values())
        return "supported" if vs == {"SUPPORTED"} else "not supported" if vs == {"FAILED"} else "partially supported"
    real = "; ".join(f"{h2_word(b)} on {DS_NAME.get(b['dataset'], b['dataset'])}" for b in blocks)
    violated = sorted({f"{k.split('|')[1]} (alpha={k.split('|')[0]})" for b in blocks
                       for k, v in b["verdicts"]["validity"].items() if v != "ok"})
    validity = ("Every method that claims a guarantee met it on held-out data" if not violated else
                "Guarantee violations were observed for " + ", ".join(violated).replace("_", r"\_"))

    def rel_range(method):
        v = [100 * r["rel"] for b in blocks for r in b["rows"] if r["method"] == method]
        return min(v), max(v)

    pix_cost = max(abs(100 * r["rel"]) for b in blocks for r in b["rows"] if r["method"] == "pixel_crc")
    ltt, dil, strat = rel_range("ltt_threshold"), rel_range("union_dilation"), rel_range("union_stratified")
    pct = r"\%"
    abstract = (
        f"{validity}. Joint lesion-level control cost at most {pix_cost:.1f}{pct} in set size over "
        f"pixel-only CRC. Learn-Then-Test (a high-probability guarantee) needed {ltt[0]:.0f}--{ltt[1]:.0f}{pct} "
        f"larger sets and morphological sets {dil[0]:.0f}--{dil[1]:.0f}{pct} larger; CCRA-S strata ranged "
        f"{strat[0]:+.0f} to {strat[1]:+.0f}{pct}. The pre-registered claim that lesion-equalized sets are "
        f"smaller than joint thresholding was {real}. In a controlled simulation they reduced set size by up to "
        f"{-best['rel']*100:.0f}{pct} when small lesions are moderately less confident, and fell back to "
        "thresholding when lesions are indistinguishable from background."
    )
    macros.append(r"\newcommand{\ResultValidity}{" + validity + "}")
    macros.append(r"\newcommand{\ResultAbstract}{" + abstract + "}")
    (OUT / "results.tex").write_text("\n".join(macros) + "\n")
    print((OUT / "results.tex").read_text())


if __name__ == "__main__":
    main()
