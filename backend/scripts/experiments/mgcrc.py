"""Multi-granularity CRC experiments (E1 gap, E2/E3 main comparison, E4 validity).

Protocol: docs/research/preregistration-mgcrc.md (committed before this script
was run on any posterior). Summary:

  pool        all annotated units with out-of-fold posteriors (train_backbone.py)
  repetition  random partition of the pool: tune 20% / cal 40% / test 40%
  baselines   calibrate on tune U cal (60%) — they need no tuning, so they get
              the extra data; this biases the comparison AGAINST our method
  ours        family chosen on tune (smallest set under the joint rule), then
              calibrated on cal only
  metric      set-to-truth ratio  sum|S| / sum|Y| on test units, at a threshold
              whose guarantee covers pixel FNR, lesion miss and small-lesion miss

    python scripts/experiments/mgcrc.py --dataset fracatlas --seed 0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.calibration.families import ScoreFamily, default_family_grid  # noqa: E402
from app.calibration.joint import (  # noqa: E402
    crc_index,
    evaluate,
    ltt_index,
    max_index,
    select_family,
    stratified_indices,
    union_index,
)
from app.calibration.losses import DEFAULT_TAU, lambda_grid, stack, unit_stats  # noqa: E402
from app.core.config import BACKEND_ROOT, REPO_ROOT  # noqa: E402

POST_ROOT = BACKEND_ROOT / "var" / "posteriors"
CACHE_ROOT = BACKEND_ROOT / "var" / "stats"
OUT_ROOT = REPO_ROOT / "docs" / "research" / "results"
ALPHAS = (0.05, 0.10, 0.15)
SPLIT = (0.2, 0.4, 0.4)
LTT_DELTA = 0.1
N_STRATA = 3
SMALL_QUANTILE = 33.0  # small lesion := below the 33rd pct of GT component sizes


def families() -> list[ScoreFamily]:
    return default_family_grid() + [ScoreFamily("dilation")]


def load_posteriors(dataset: str, seed: int, backbone: str):
    d = np.load(POST_ROOT / dataset / f"{backbone}_seed{seed}.npz")
    return d["prob"].astype(np.float32), d["labels"], d["valid"]


def unit_cache(dataset, seed, backbone, tau, lambdas):
    """Per family: list of UnitStats (cached — the expensive pass over pixels)."""
    path = CACHE_ROOT / f"{dataset}_{backbone}_seed{seed}_tau{tau:g}_g{lambdas.size}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    prob, labels, valid = load_posteriors(dataset, seed, backbone)
    out = {"_mass": prob.reshape(len(prob), -1).sum(1)}
    for fam in families():
        t = time.time()
        units = []
        for p, lab, v in zip(prob, labels, valid):
            s = fam.score(p)
            s[~v] = 0.0  # letterbox padding enters the set only at lambda = 1
            units.append(unit_stats(s, lab, tau=tau, lambdas=lambdas))
        out[fam.label] = units
        print(f"    stats {fam.label:<40} {time.time()-t:5.1f}s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def small_cut(units) -> float:
    sizes = np.concatenate([u.lesion_size for u in units if u.k])
    return float(np.percentile(sizes, SMALL_QUANTILE))


def run(dataset, seed, backbone, reps, tau, alphas, lambdas):
    cache = unit_cache(dataset, seed, backbone, tau, lambdas)
    cut = small_cut(cache["threshold"])
    stacks = {k: stack(v, small_max_size=cut, lambdas=lambdas) for k, v in cache.items() if k != "_mass"}
    keep = [i for i, u in enumerate(cache["threshold"]) if u.n_gt > 0 and u.k > 0]
    mass = cache["_mass"][keep]
    base = stacks["threshold"]
    n = len(base)
    le_labels = [f.label for f in default_family_grid()]
    rng = np.random.default_rng(seed * 7919 + 1)
    rows, gap_rows, chosen = [], [], []

    n_tune, n_cal = int(SPLIT[0] * n), int(SPLIT[1] * n)
    for r in range(reps):
        perm = rng.permutation(n)
        tune, cal, test = perm[:n_tune], perm[n_tune:n_tune + n_cal], perm[n_tune + n_cal:]
        tc = np.concatenate([tune, cal])
        for alpha in alphas:
            res = {}
            b_tc, b_test = base.subset(tc), base.subset(test)
            i_pix, _ = crc_index(b_tc.pix_rate, alpha)
            res["pixel_crc"] = evaluate(b_test, i_pix, lambdas)
            res["union_threshold"] = evaluate(b_test, union_index(b_tc, alpha)[0], lambdas)
            res["max_threshold"] = evaluate(b_test, max_index(b_tc, alpha)[0], lambdas)
            res["ltt_threshold"] = evaluate(b_test, ltt_index(b_tc, alpha, LTT_DELTA)[0], lambdas)
            d = stacks["dilation"]
            res["union_dilation"] = evaluate(d.subset(test), union_index(d.subset(tc), alpha)[0], lambdas)
            # CCRA-S-style strata on total predicted mass; cutpoints from tune U cal (label-free)
            q = np.quantile(mass[tc], np.linspace(0, 1, N_STRATA + 1)[1:-1])
            strat_tc, strat_test = np.digitize(mass[tc], q), np.digitize(mass[test], q)
            per = stratified_indices(b_tc, strat_tc, alpha)
            res["union_stratified"] = evaluate(
                b_test, np.asarray([per.get(int(g), (lambdas.size - 1, False))[0] for g in strat_test]), lambdas)
            # ours: family on tune, calibrate on cal
            tune_stacks = {k: stacks[k].subset(tune) for k in le_labels}
            for rule, fn in (("union", union_index), ("max", max_index)):
                fam = select_family(tune_stacks, alpha, rule)
                st = stacks[fam]
                res[f"ours_{rule}"] = evaluate(st.subset(test), fn(st.subset(cal), alpha)[0], lambdas)
                if rule == "union":
                    chosen.append(dict(rep=r, alpha=alpha, family=fam))
            for m, ev in res.items():
                rows.append(dict(rep=r, alpha=alpha, method=m, **{k: getattr(ev, k) for k in ev.__slots__}))
            # E1 gap at the pixel-calibrated threshold, on test, full and K>=2
            ev_pix = res["pixel_crc"]
            gap_rows.append(dict(rep=r, alpha=alpha, G=ev_pix.risk_les / alpha,
                                 G_multi=ev_pix.risk_les_multi / alpha, G_small=ev_pix.risk_small / alpha,
                                 n_cal=len(tc), n_cal_multi=int(base.multi[tc].sum()),
                                 n_cal_small=int(base.has_small[tc].sum())))
    meta = dict(dataset=dataset, seed=seed, backbone=backbone, tau=tau, reps=reps, n_units=n,
                n_multi=int(base.multi.sum()), n_with_small=int(base.has_small.sum()),
                small_cut_px=cut, split=SPLIT, ltt_delta=LTT_DELTA, grid=int(lambdas.size))
    return rows, gap_rows, chosen, meta


def summarise(rows, gap_rows, chosen, meta):
    import collections

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["alpha"], r["method"])].append(r)
    summary = []
    for (alpha, method), rs in sorted(by.items()):
        def stat(k):
            v = np.asarray([x[k] for x in rs], dtype=float)
            v = v[np.isfinite(v)]
            return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
        summary.append(dict(alpha=alpha, method=method, **{k: stat(k) for k in
                       ("risk_pix", "risk_les", "risk_small", "risk_les_multi", "set_to_truth", "fp_to_truth", "lam")}))
    paired = []
    for alpha in sorted({r["alpha"] for r in rows}):
        ref = {r["rep"]: r["set_to_truth"] for r in rows if r["alpha"] == alpha and r["method"] == "union_threshold"}
        for method in sorted({r["method"] for r in rows}):
            cur = {r["rep"]: r["set_to_truth"] for r in rows if r["alpha"] == alpha and r["method"] == method}
            rel = np.asarray([cur[k] / ref[k] - 1 for k in ref])
            paired.append(dict(alpha=alpha, method=method, rel_change_mean=float(rel.mean()),
                               lo=float(np.percentile(rel, 2.5)), hi=float(np.percentile(rel, 97.5)),
                               p_smaller=float((rel < 0).mean())))
    gaps = []
    for alpha in sorted({g["alpha"] for g in gap_rows}):
        gs = [g for g in gap_rows if g["alpha"] == alpha]
        def gstat(k):
            v = np.asarray([g[k] for g in gs], float)
            v = v[np.isfinite(v)]
            return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
        gaps.append(dict(alpha=alpha, G=gstat("G"), G_multi=gstat("G_multi"), G_small=gstat("G_small"),
                         n_cal=gs[0]["n_cal"], n_cal_multi_mean=float(np.mean([g["n_cal_multi"] for g in gs])),
                         n_cal_small_mean=float(np.mean([g["n_cal_small"] for g in gs]))))
    fam_counts = collections.Counter((c["alpha"], c["family"]) for c in chosen)
    return dict(meta=meta, summary=summary, paired_vs_union_threshold=paired, gap=gaps,
                chosen_family={f"{a}|{f}": c for (a, f), c in sorted(fam_counts.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="unet-r34")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    lambdas = lambda_grid()
    t = time.time()
    rows, gap_rows, chosen, meta = run(args.dataset, args.seed, args.backbone, args.reps, args.tau, ALPHAS, lambdas)
    out = summarise(rows, gap_rows, chosen, meta)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_{args.dataset}_{args.backbone}_seed{args.seed}_tau{args.tau:g}"
    (OUT_ROOT / f"{stem}.json").write_text(json.dumps(out, indent=2))
    import csv
    with open(OUT_ROOT / f"{stem}_reps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{stem}: {time.time()-t:.0f}s")
    for s in out["summary"]:
        print(f"  a={s['alpha']:.2f} {s['method']:<18} pix {s['risk_pix']['mean']:.3f} les {s['risk_les']['mean']:.3f} "
              f"small {s['risk_small']['mean']:.3f}  STR {s['set_to_truth']['mean']:.2f} "
              f"[{s['set_to_truth']['lo']:.2f},{s['set_to_truth']['hi']:.2f}]")
    for p in out["paired_vs_union_threshold"]:
        if p["method"].startswith("ours") or p["method"] in ("ltt_threshold", "union_dilation", "union_stratified", "max_threshold"):
            print(f"  a={p['alpha']:.2f} {p['method']:<18} vs union_threshold: {100*p['rel_change_mean']:+.1f}% "
                  f"[{100*p['lo']:+.1f},{100*p['hi']:+.1f}]  P(smaller)={p['p_smaller']:.3f}")
    for g in out["gap"]:
        print(f"  GAP a={g['alpha']:.2f}  G={g['G']['mean']:.2f}  G|K>=2={g['G_multi']['mean']:.2f} "
              f"[{g['G_multi']['lo']:.2f},{g['G_multi']['hi']:.2f}]  G_small={g['G_small']['mean']:.2f} "
              f"[{g['G_small']['lo']:.2f},{g['G_small']['hi']:.2f}]  n_cal_multi~{g['n_cal_multi_mean']:.0f}")


if __name__ == "__main__":
    main()
