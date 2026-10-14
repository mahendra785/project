"""Aggregate MG-CRC results over seeds and apply the pre-registered decision rules.

Reads   docs/research/results/<tag>_<dataset>_<backbone>_seed<k>_tau<t>.json
Writes  docs/research/results/summary.md          human-readable tables + verdicts
        docs/research/results/mgcrc_summary.json  served by GET /api/research/mgcrc
        paper/data/*.csv                          pgfplots inputs for the paper

Nothing in the outputs is typed by hand: every number and every verdict string is
computed here from the result files, using the thresholds written in
docs/research/preregistration-mgcrc.md.

    python scripts/experiments/aggregate.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "docs" / "research" / "results"
PAPER_DATA = REPO_ROOT / "paper" / "data"
PATTERN = re.compile(r"^(?P<tag>[a-z0-9]+)_(?P<ds>.+)_(?P<bb>[a-z0-9-]+)_seed(?P<seed>\d+)_tau(?P<tau>[\d.]+)\.json$")

METHOD_ORDER = ["pixel_crc", "union_threshold", "max_threshold", "ltt_threshold", "union_dilation",
                "union_stratified", "ours_union", "ours_max"]
METHOD_LABEL = {
    "pixel_crc": "Pixel CRC (pixel FNR only)",
    "union_threshold": "Joint CRC, thresholding",
    "max_threshold": "Joint CRC max-form, thresholding",
    "ltt_threshold": "Learn-Then-Test (δ=0.1), thresholding",
    "union_dilation": "Joint CRC, morphological sets",
    "union_stratified": "Joint CRC, CCRA-S strata",
    "ours_union": "**Ours**: joint CRC, lesion-equalized",
    "ours_max": "**Ours**: max-form, lesion-equalized",
}
#: which risks each method claims to control (for the validity check)
CLAIMS = {
    "pixel_crc": ("risk_pix",),
    "union_threshold": ("risk_pix", "risk_les", "risk_small"),
    "max_threshold": ("risk_pix", "risk_les", "risk_small"),
    "ltt_threshold": ("risk_pix", "risk_les", "risk_small"),
    "union_dilation": ("risk_pix", "risk_les", "risk_small"),
    "union_stratified": ("risk_pix", "risk_les", "risk_small"),
    "ours_union": ("risk_pix", "risk_les", "risk_small"),
    "ours_max": ("risk_pix", "risk_les", "risk_small"),
}
MC_SLACK = 0.005


def load(tag: str = "main"):
    runs = collections.defaultdict(list)
    for f in sorted(RESULTS.glob("*.json")):
        m = PATTERN.match(f.name)
        if not m or m["tag"] != tag:
            continue
        runs[(m["ds"], m["bb"], float(m["tau"]))].append(json.loads(f.read_text()))
    return runs


def combine(runs):
    """Per (dataset, backbone, tau): mean over seeds of per-seed means; seed sd."""
    out = {}
    for key, seeds in runs.items():
        table = collections.defaultdict(lambda: collections.defaultdict(list))
        for s in seeds:
            for row in s["summary"]:
                for k in ("risk_pix", "risk_les", "risk_small", "risk_les_multi", "set_to_truth", "fp_to_truth", "lam"):
                    table[(row["alpha"], row["method"])][k].append(row[k]["mean"])
                    table[(row["alpha"], row["method"])][k + "_lo"].append(row[k]["lo"])
                    table[(row["alpha"], row["method"])][k + "_hi"].append(row[k]["hi"])
            for p in s["paired_vs_union_threshold"]:
                t = table[(p["alpha"], p["method"])]
                t["rel"].append(p["rel_change_mean"])
                t["rel_lo"].append(p["lo"])
                t["rel_hi"].append(p["hi"])
                t["p_smaller"].append(p["p_smaller"])
        rows = []
        for (alpha, method), v in sorted(table.items()):
            r = dict(alpha=alpha, method=method, n_seeds=len(v["set_to_truth"]))
            for k, vals in v.items():
                a = np.asarray(vals, float)
                r[k] = float(np.nanmean(a))
                if not k.endswith(("_lo", "_hi")):
                    r[k + "_seed_sd"] = float(np.nanstd(a)) if len(a) > 1 else None
            rows.append(r)
        gaps = collections.defaultdict(lambda: collections.defaultdict(list))
        for s in seeds:
            for g in s["gap"]:
                for k in ("G", "G_multi", "G_small"):
                    gaps[g["alpha"]][k].append(g[k]["mean"])
                    gaps[g["alpha"]][k + "_lo"].append(g[k]["lo"])
                    gaps[g["alpha"]][k + "_hi"].append(g[k]["hi"])
                gaps[g["alpha"]]["n_cal_multi"].append(g["n_cal_multi_mean"])
                gaps[g["alpha"]]["n_cal_small"].append(g["n_cal_small_mean"])
        gap_rows = [dict(alpha=a, **{k: float(np.nanmean(v)) for k, v in d.items()}) for a, d in sorted(gaps.items())]
        fam = collections.Counter()
        for s in seeds:
            for k, c in s["chosen_family"].items():
                fam[k] += c
        out[key] = dict(meta=[s["meta"] for s in seeds], rows=rows, gap=gap_rows, chosen_family=dict(fam))
    return out


def verdicts(block):
    rows = {(r["alpha"], r["method"]): r for r in block["rows"]}
    gap = {g["alpha"]: g for g in block["gap"]}
    # H1
    kill = all(g["G_multi"] <= 1.1 and g["G_small"] <= 1.1 for g in gap.values())
    g10 = gap.get(0.1)
    proceed = g10 is not None and (g10["G_multi"] >= 1.5 or g10["G_small"] >= 1.5)
    h1 = "KILL" if kill else "PROCEED" if proceed else "INCONCLUSIVE"
    # H2 per alpha
    h2 = {}
    for alpha in sorted({a for a, _ in rows}):
        o = rows.get((alpha, "ours_union"))
        if o is None:
            continue
        valid = all(o[k] <= alpha + MC_SLACK for k in CLAIMS["ours_union"] if np.isfinite(o[k]))
        if o["rel"] < 0 and o["p_smaller"] >= 0.95 and valid:
            h2[alpha] = "SUPPORTED"
        elif o["rel"] >= 0:
            h2[alpha] = "FAILED"
        else:
            h2[alpha] = "PARTIAL"
    validity = {}
    for (alpha, method), r in rows.items():
        bad = [k for k in CLAIMS.get(method, ()) if np.isfinite(r[k]) and r[k] > alpha + MC_SLACK]
        validity[f"{alpha}|{method}"] = "ok" if not bad else "VIOLATED: " + ",".join(bad)
    return dict(H1=h1, H2={str(k): v for k, v in h2.items()}, validity=validity)


def fmt(v, d=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def write_markdown(blocks) -> str:
    lines = ["# MG-CRC results (generated by `scripts/experiments/aggregate.py` — do not edit)", ""]
    for (ds, bb, tau), b in sorted(blocks.items()):
        v = b["verdicts"]
        m0 = b["meta"][0]
        dice = b.get("mean_dice")
        lines += [f"## {ds} · backbone `{bb}` · τ = {tau:g}", "",
                  f"Units with lesions: {m0['n_units']} · multi-lesion: {m0['n_multi']} · with a small lesion: "
                  f"{m0['n_with_small']} (small := < {m0['small_cut_px']:.0f} px) · seeds: {len(b['meta'])} · "
                  f"repetitions/seed: {m0['reps']}" + (f" · backbone Dice (out-of-fold): {dice:.3f}" if dice else ""), "",
                  f"**H1 (gap): {v['H1']}** · **H2 (efficiency): "
                  + ", ".join(f"α={a}: {s}" for a, s in v["H2"].items()) + "**", "",
                  "### Gap at the pixel-CRC threshold (test split)", "",
                  "| α | G (all) | G \\| K≥2 [95% split interval] | G_small [95%] | n_cal multi | n_cal small |",
                  "|---|---|---|---|---|---|"]
        for g in b["gap"]:
            lines.append(f"| {g['alpha']:.2f} | {fmt(g['G'],2)} | {fmt(g['G_multi'],2)} [{fmt(g['G_multi_lo'],2)}, "
                         f"{fmt(g['G_multi_hi'],2)}] | {fmt(g['G_small'],2)} [{fmt(g['G_small_lo'],2)}, "
                         f"{fmt(g['G_small_hi'],2)}] | {g['n_cal_multi']:.0f} | {g['n_cal_small']:.0f} |")
        lines += ["", "### Main comparison (test split; mean over seeds of the mean over repetitions)", "",
                  "| α | method | pixel FNR | lesion miss | small miss | lesion miss \\| K≥2 | set / truth [95%] | Δ vs joint-threshold [95%] | P(smaller) | validity |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        rows = {(r["alpha"], r["method"]): r for r in b["rows"]}
        for alpha in sorted({a for a, _ in rows}):
            for meth in METHOD_ORDER:
                r = rows.get((alpha, meth))
                if r is None:
                    continue
                rel = "ref." if meth == "union_threshold" else f"{100*r['rel']:+.1f}% [{100*r['rel_lo']:+.1f}, {100*r['rel_hi']:+.1f}]"
                lines.append(
                    f"| {alpha:.2f} | {METHOD_LABEL[meth]} | {fmt(r['risk_pix'])} | {fmt(r['risk_les'])} | "
                    f"{fmt(r['risk_small'])} | {fmt(r['risk_les_multi'])} | {fmt(r['set_to_truth'],2)} "
                    f"[{fmt(r['set_to_truth_lo'],2)}, {fmt(r['set_to_truth_hi'],2)}] | {rel} | "
                    f"{'—' if meth == 'union_threshold' else fmt(r['p_smaller'],3)} | {v['validity'][f'{alpha}|{meth}']} |")
        lines += ["", "Families chosen on the tuning split (count over seeds × repetitions):", ""]
        for k, c in sorted(b["chosen_family"].items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"- α={k.split('|')[0]} `{k.split('|')[1]}`: {c}")
        lines.append("")
    return "\n".join(lines)


def write_pgf(blocks):
    PAPER_DATA.mkdir(parents=True, exist_ok=True)
    for (ds, bb, tau), b in blocks.items():
        rows = {(r["alpha"], r["method"]): r for r in b["rows"]}
        alphas = sorted({a for a, _ in rows})
        name = f"{ds}_{bb}_tau{tau:g}"
        with open(PAPER_DATA / f"str_{name}.csv", "w") as f:
            f.write("alpha," + ",".join(METHOD_ORDER) + "\n")
            for a in alphas:
                f.write(f"{a}," + ",".join(fmt(rows.get((a, m), {}).get("set_to_truth", np.nan), 4) for m in METHOD_ORDER) + "\n")
        with open(PAPER_DATA / f"risk_{name}.csv", "w") as f:
            f.write("alpha,method,risk_pix,risk_les,risk_small\n")
            for (a, m), r in sorted(rows.items()):
                f.write(f"{a},{m},{fmt(r['risk_pix'],4)},{fmt(r['risk_les'],4)},{fmt(r['risk_small'],4)}\n")
        with open(PAPER_DATA / f"gap_{name}.csv", "w") as f:
            f.write("alpha,G,G_multi,G_multi_lo,G_multi_hi,G_small,G_small_lo,G_small_hi\n")
            for g in b["gap"]:
                f.write(",".join(fmt(g[k], 4) for k in ("alpha", "G", "G_multi", "G_multi_lo", "G_multi_hi",
                                                          "G_small", "G_small_lo", "G_small_hi")) + "\n")


def main():
    runs = load("main")
    if not runs:
        sys.exit("no result files found")
    blocks = combine(runs)
    post = REPO_ROOT / "backend" / "var" / "posteriors"
    for (ds, bb, tau), b in blocks.items():
        dices = [json.loads(p.read_text())["mean_dice"] for p in sorted((post / ds).glob(f"{bb}_seed*.json"))]
        b["mean_dice"] = float(np.mean(dices)) if dices else None
        b["verdicts"] = verdicts(b)
    md = write_markdown(blocks)
    (RESULTS / "summary.md").write_text(md, encoding="utf-8")
    (RESULTS / "mgcrc_summary.json").write_text(json.dumps(
        [dict(dataset=ds, backbone=bb, tau=tau, **b) for (ds, bb, tau), b in sorted(blocks.items())], indent=2))
    write_pgf(blocks)
    sys.stdout.buffer.write(md.encode("utf-8"))


if __name__ == "__main__":
    main()
