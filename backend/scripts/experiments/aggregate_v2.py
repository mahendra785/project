"""Aggregate the revision runs (Amendment 3) into one summary JSON + tables for the paper.

Seed handling: model seeds are independent replicates. Point estimates are the
mean over seeds of each seed's partition mean. CIs come from the clustered
bootstrap: replicate b of the seed-averaged estimand is the mean over seeds of
each seed's b-th bootstrap replicate (independent resampling per seed's pool).

    python scripts/experiments/aggregate_v2.py
"""

from __future__ import annotations

import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mgcrc_v2 as V  # noqa: E402

from app.core.config import REPO_ROOT  # noqa: E402

RES = REPO_ROOT / "docs" / "research" / "results_v2"
SLACK = 0.005
M, K = list(V.METHODS), list(V.METRICS)


def load(tag):
    runs = collections.defaultdict(dict)
    for f in sorted(RES.glob(f"{tag}_*.pkl")):
        d = pickle.loads(f.read_bytes())
        meta = d["meta"]
        runs[(meta["dataset"], meta["arch"])][meta["seed"]] = d
    return runs


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x.size else [np.nan, np.nan]


def seed_avg_boot(seeds, getter):
    """Replicate-wise mean over seeds of getter(run)[b] -> array over b."""
    arrs = [getter(r) for r in seeds.values() if r.get("boot") is not None]
    if not arrs:
        return None
    b = min(len(a) for a in arrs)
    return np.nanmean(np.stack([a[:b] for a in arrs]), axis=0)


def summarise_config(seeds: dict):
    any_run = next(iter(seeds.values()))
    alphas = any_run["meta"]["alphas"]
    out = dict(seeds=sorted(seeds), alphas=alphas)
    # pool and backbone facts
    pm = [r["meta"]["pool_metrics"] for r in seeds.values()]
    out["pool_metrics"] = {k: dict(mean=float(np.mean([p[k] for p in pm])), sd=float(np.std([p[k] for p in pm], ddof=1))
                                   if len(pm) > 1 else 0.0, per_seed=[p[k] for p in pm])
                           for k in ("dice", "iou", "lesion_recall", "lesion_precision", "fp_components_per_unit")}
    out["counts"] = {k: [r["meta"][k] for r in seeds.values()] for k in ("n_units", "n_multi", "n_small", "n_cal", "n_tune")}
    out["small_cut"] = [r["meta"]["small_cut"] for r in seeds.values()]
    per = {}
    for ai, a in enumerate(alphas):
        for mi, m in enumerate(M):
            row = {}
            for ki, k in enumerate(K):
                point = float(np.mean([np.nanmean(r["main"][:, ai, mi, ki]) for r in seeds.values()]))
                split = ci(np.concatenate([r["main"][:, ai, mi, ki] for r in seeds.values()]))
                bt = seed_avg_boot(seeds, lambda r: r["boot"][:, ai, mi, ki])
                row[k] = dict(mean=point, split=split, ci=ci(bt) if bt is not None else None,
                              per_seed=[float(np.nanmean(r["main"][:, ai, mi, ki])) for r in seeds.values()])
            # paired relative change in set size vs joint thresholding (C1 reference)
            j, s = M.index("joint"), K.index("set_to_truth")
            rel = np.concatenate([r["main"][:, ai, mi, s] / r["main"][:, ai, j, s] - 1 for r in seeds.values()])
            bt = seed_avg_boot(seeds, lambda r: r["boot_rel"][:, ai, mi])
            row["rel_vs_joint"] = dict(mean=float(np.nanmean(rel)), split=ci(rel), p_smaller=float(np.mean(rel < 0)),
                                       ci=ci(bt) if bt is not None else None,
                                       per_seed=[float(np.nanmean(r["main"][:, ai, mi, s] / r["main"][:, ai, j, s] - 1))
                                                 for r in seeds.values()])
            claims = {"pix": ["risk_pix"], "pix_tc": ["risk_pix"]}.get(m, ["risk_pix", "risk_les", "risk_small"])
            if m.startswith("joint_multi"):
                claims = ["risk_pix", "risk_les", "risk_small", "risk_multi"]
            if m.startswith("max"):
                claims = ["risk_pix", "risk_les"]  # Proposition 1: small-lesion risk only at alpha/pi
            if m == "le_max":
                claims = ["risk_pix", "risk_les"]
            row["claims"] = claims
            row["valid"] = bool(all(row[c]["mean"] <= a + SLACK for c in claims if np.isfinite(row[c]["mean"])))
            per[f"{a:.2f}|{m}"] = row
    out["methods"] = per
    # gap (RQ1): at the pixel-CRC threshold calibrated on cal (C1)
    gap = {}
    for ai, a in enumerate(alphas):
        mi = M.index("pix")
        g = {}
        for name, k in (("G_all", "risk_les"), ("G_multi", "risk_multi"), ("G_small", "risk_small")):
            ki = K.index(k)
            point = float(np.mean([np.nanmean(r["main"][:, ai, mi, ki]) for r in seeds.values()])) / a
            bt = seed_avg_boot(seeds, lambda r: r["boot"][:, ai, mi, ki])
            g[name] = dict(mean=point, ci=[c / a for c in ci(bt)] if bt is not None else None,
                           split=[c / a for c in ci(np.concatenate([r["main"][:, ai, mi, ki] for r in seeds.values()]))])
        gap[f"{a:.2f}"] = g
    out["gap"] = gap
    g10 = gap["0.10"]
    out["H1"] = ("PROCEED" if max(g10["G_multi"]["mean"], g10["G_small"]["mean"]) >= 1.5 else
                 "KILL" if all(max(gap[f"{a:.2f}"]["G_multi"]["mean"], gap[f"{a:.2f}"]["G_small"]["mean"]) <= 1.1
                               for a in alphas) else "INCONCLUSIVE")
    # H2 per alpha (C1)
    h2 = {}
    for a in alphas:
        r = per[f"{a:.2f}|le_union"]
        rv = r["rel_vs_joint"]
        ok_ci = rv["ci"] is not None and rv["ci"][1] < 0
        h2[f"{a:.2f}"] = dict(rel=rv["mean"], ci=rv["ci"], p_smaller=rv["p_smaller"], valid=r["valid"],
                             supported=bool(rv["mean"] < 0 and rv["p_smaller"] >= 0.95 and ok_ci and r["valid"]),
                             failed=bool(rv["mean"] >= 0))
    out["H2"] = h2
    # H3 (joint + multi-lesion risk), only if >= 30 multi-lesion pool units
    n_multi = min(out["counts"]["n_multi"])
    h3 = {}
    for a in alphas:
        jm, j = per[f"{a:.2f}|joint_multi"], per[f"{a:.2f}|joint"]
        h3[f"{a:.2f}"] = dict(multi_joint=j["risk_multi"]["mean"], multi_joint_ci=j["risk_multi"]["ci"],
                              multi_with=jm["risk_multi"]["mean"], multi_with_ci=jm["risk_multi"]["ci"],
                              set_change=jm["rel_vs_joint"]["mean"], set_change_ci=jm["rel_vs_joint"]["ci"],
                              bg_joint=j["fp_to_truth"]["mean"], bg_with=jm["fp_to_truth"]["mean"],
                              supported=bool(jm["risk_multi"]["mean"] <= a + SLACK) if n_multi >= 30 else None)
    out["H3"] = dict(evaluable=n_multi >= 30, n_multi_pool=out["counts"]["n_multi"], per_alpha=h3)
    chosen = collections.Counter()
    for r in seeds.values():
        chosen.update(r["chosen"])
    total = sum(chosen.values())
    out["chosen_family"] = {k: v / total for k, v in chosen.most_common()}
    return out


def detectability(seeds: dict):
    """Per small / all lesion: peak posterior vs the unit's 99th-percentile background."""
    rows = []
    for r in seeds.values():
        cut = r["meta"]["small_cut"]
        for les in r["lesions"]:
            for pk, sz in zip(les["peak"], les["size"]):
                rows.append((pk, sz, les["bg_q99"], sz < cut))
    a = np.asarray(rows, float) if rows else np.zeros((0, 4))
    small = a[a[:, 3] == 1] if len(a) else a
    def frac(x, t0):
        return float(np.mean(x[:, 0] > np.maximum(x[:, 2], t0))) if len(x) else float("nan")
    return dict(n_lesions=int(len(a)), n_small=int(len(small)),
                small_peak_above_bg=frac(small, 0.0), small_peak_above_bg_and_t0_0p2=frac(small, 0.2),
                all_peak_above_bg=frac(a, 0.0), small_peak_median=float(np.median(small[:, 0])) if len(small) else None,
                small_bg_q99_median=float(np.median(small[:, 2])) if len(small) else None)


def missed_lesion_profile(seeds: dict, alpha=0.10, reps=200):
    """Lesions missed by joint thresholding (C1) on test units: size and peak vs bg, over random partitions."""
    lambdas = V.lambda_grid()
    missed, found = [], []
    for seed, r in seeds.items():
        meta = r["meta"]
        cache = V.build_cache(meta["dataset"], meta["arch"], seed, meta["tau"], False, lambdas)
        keep = [i for i, u in enumerate(cache["stats"]["threshold"]) if u.n_gt > 0 and u.k > 0]
        units = [cache["stats"]["threshold"][i] for i in keep]
        base = V.Mats(units, meta["small_cut"], lambdas)
        les = [cache["lesions"][i] for i in keep]
        rng = np.random.default_rng(99 + seed)
        n = len(keep)
        w = np.ones(n)
        for tune, cal, test in V.partitions(n, rng, reps):
            idx = V.union_idx(V.curves(base, cal, w), alpha)
            cutoff = 1 - lambdas[idx]
            for t in test:
                u, l = units[t], les[t]
                for k in range(u.k):
                    rec = (float(u.lesion_size[k]), float(l["peak"][k]), float(l["bg_q99"]),
                           bool(u.lesion_size[k] < meta["small_cut"]))
                    (missed if u.lesion_stat[k] < cutoff else found).append(rec)
    m, f = np.asarray(missed, float), np.asarray(found, float)
    def desc(x):
        if not len(x):
            return None
        return dict(n=int(len(x)), median_size=float(np.median(x[:, 0])), frac_small=float(np.mean(x[:, 3])),
                    median_peak=float(np.median(x[:, 1])), frac_peak_below_bg=float(np.mean(x[:, 1] <= x[:, 2])),
                    frac_no_candidate_t0_0p2=float(np.mean(x[:, 1] < 0.2)))
    return dict(alpha=alpha, missed=desc(m), found=desc(f))


def main():
    summary = {}
    main_runs = load("main")
    for (ds, arch), seeds in sorted(main_runs.items()):
        key = f"{ds}|{arch}"
        s = summarise_config(seeds)
        s["detectability"] = detectability(seeds)
        try:
            s["missed_profile"] = missed_lesion_profile(seeds)
        except FileNotFoundError:
            s["missed_profile"] = None
        summary[key] = s
        print(f"{key}: seeds {s['seeds']}  H1 {s['H1']}  H2 " +
              " ".join(f"{a}:{'SUPPORTED' if v['supported'] else 'FAILED' if v['failed'] else 'partial'}"
                       f"({100 * v['rel']:+.1f}%)" for a, v in s["H2"].items()))
    # ablations: any tag other than main
    abl = {}
    for f in sorted(RES.glob("*.pkl")):
        tag = f.name.split("_")[0]
        if tag == "main":
            continue
        d = pickle.loads(f.read_bytes())
        meta = d["meta"]
        key = f"{tag}|{meta['dataset']}|{meta['arch']}"
        abl.setdefault(key, {})[meta["seed"]] = d
    abl_out = {}
    for key, seeds in abl.items():
        s = summarise_config(seeds)
        abl_out[key] = dict(meta={k: next(iter(seeds.values()))["meta"][k] for k in ("tau", "small_q", "conn4", "ncal")},
                            gap=s["gap"], H2=s["H2"],
                            le_rel={a: s["methods"][f"{a:.2f}|le_union"]["rel_vs_joint"] for a in s["alphas"]},
                            joint_vs_pix={a: s["methods"][f"{a:.2f}|pix"]["rel_vs_joint"] for a in s["alphas"]},
                            validity={m: s["methods"][f"0.10|{m}"]["valid"] for m in M})
    summary["_ablations"] = abl_out
    (RES / "summary_v2.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {RES / 'summary_v2.json'}")


if __name__ == "__main__":
    main()
