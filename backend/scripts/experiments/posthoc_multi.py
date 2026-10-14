"""E8 — POST-HOC (not pre-registered): close the multi-lesion gap with a conditional risk.

The pre-registered joint guarantee bounds the lesion-miss rate averaged over ALL
units; on multi-lesion units it can still exceed alpha. Adding the risk
L^les restricted to units with K >= 2 is valid under CRC for the same reason as
L^small: conditionally on the label-defined event K >= 2, calibration and test
units are exchangeable, and the shared threshold is applied to every unit.

Same caches, splits and seeds as mgcrc.py. Methods: joint-threshold and ours,
each with and without the conditional risk.

    python scripts/experiments/posthoc_multi.py --dataset kvasir-seg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mgcrc  # noqa: E402

from app.calibration.families import default_family_grid  # noqa: E402
from app.calibration.joint import evaluate, select_family, union_index  # noqa: E402
from app.calibration.losses import lambda_grid, stack  # noqa: E402

R3 = ("pix", "les", "small")
R4 = ("pix", "les", "small", "les_multi")


def pick(tune, alpha, risks):
    best, size = None, np.inf
    for k, st in tune.items():
        i, _ = union_index(st, alpha, risks)
        s = st.set_size[:, i].sum() / st.n_gt.sum()
        if s < size - 1e-12:
            best, size = k, s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="rf")
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    lambdas = lambda_grid()
    le = [f.label for f in default_family_grid()]
    out = []
    for seed in (0, 1, 2):
        cache = mgcrc.unit_cache(args.dataset, seed, args.backbone, 0.5, lambdas)
        cut = mgcrc.small_cut(cache["threshold"])
        stacks = {k: stack(cache[k], small_max_size=cut, lambdas=lambdas) for k in le}
        base = stacks["threshold"]
        n = len(base)
        rng = np.random.default_rng(seed * 7919 + 1)  # identical splits to mgcrc.py
        n_tune, n_cal = int(mgcrc.SPLIT[0] * n), int(mgcrc.SPLIT[1] * n)
        for r in range(args.reps):
            perm = rng.permutation(n)
            tune, cal, test = perm[:n_tune], perm[n_tune:n_tune + n_cal], perm[n_tune + n_cal:]
            tc = np.concatenate([tune, cal])
            b_tc, b_test = base.subset(tc), base.subset(test)
            tune_st = {k: stacks[k].subset(tune) for k in le}
            for alpha in mgcrc.ALPHAS:
                row = dict(seed=seed, rep=r, alpha=alpha)
                for name, risks in (("joint3", R3), ("joint4", R4)):
                    ev = evaluate(b_test, union_index(b_tc, alpha, risks)[0], lambdas)
                    row[f"thr_{name}_str"], row[f"thr_{name}_multi"] = ev.set_to_truth, ev.risk_les_multi
                    row[f"thr_{name}_les"], row[f"thr_{name}_pix"] = ev.risk_les, ev.risk_pix
                    fam = pick(tune_st, alpha, risks)
                    st = stacks[fam]
                    ev = evaluate(st.subset(test), union_index(st.subset(cal), alpha, risks)[0], lambdas)
                    row[f"ours_{name}_str"], row[f"ours_{name}_multi"] = ev.set_to_truth, ev.risk_les_multi
                out.append(row)
    summ = []
    for alpha in mgcrc.ALPHAS:
        rs = [x for x in out if x["alpha"] == alpha]
        def m(k):
            v = np.asarray([x[k] for x in rs], float)
            v = v[np.isfinite(v)]
            return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
        price = np.asarray([x["thr_joint4_str"] / x["thr_joint3_str"] - 1 for x in rs])
        ours = np.asarray([x["ours_joint4_str"] / x["thr_joint4_str"] - 1 for x in rs])
        summ.append(dict(alpha=alpha, **{k: m(k) for k in rs[0] if k not in ("seed", "rep", "alpha")},
                         price_of_multi=dict(mean=float(price.mean()), lo=float(np.percentile(price, 2.5)),
                                             hi=float(np.percentile(price, 97.5))),
                         ours_vs_thr_joint4=dict(mean=float(ours.mean()), lo=float(np.percentile(ours, 2.5)),
                                                 hi=float(np.percentile(ours, 97.5)), p_smaller=float((ours < 0).mean()))))
        s = summ[-1]
        print(f"{args.dataset} a={alpha:.2f}  miss|K>=2: joint3 {s['thr_joint3_multi']['mean']:.3f} -> joint4 "
              f"{s['thr_joint4_multi']['mean']:.3f}  | set/truth {s['thr_joint3_str']['mean']:.2f} -> "
              f"{s['thr_joint4_str']['mean']:.2f} (price {100*s['price_of_multi']['mean']:+.1f}%)  | ours vs thr (joint4) "
              f"{100*s['ours_vs_thr_joint4']['mean']:+.1f}% P(smaller)={s['ours_vs_thr_joint4']['p_smaller']:.2f}")
    (mgcrc.OUT_ROOT / f"posthoc_E8_{args.dataset}_{args.backbone}.json").write_text(
        json.dumps(dict(label="POST-HOC, not pre-registered", dataset=args.dataset, summary=summ), indent=2))


if __name__ == "__main__":
    main()
