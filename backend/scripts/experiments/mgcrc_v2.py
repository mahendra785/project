"""Revision harness for lesion-level joint CRC (pre-registration Amendment 3).

Differences from mgcrc.py (the preliminary study, kept unchanged):

* Posteriors come from ONE frozen model per seed (train_frozen.py), so the
  calibration and test units are i.i.d. given the model (exact CRC setting).
* The small-lesion cut is read from the model's D_train labels (manifest), never
  from the pool.
* C1 (primary): every method calibrates on the SAME cal split; the tune split is
  used only by the lesion-equalized family choice. C2: baselines on tune U cal.
  C3: lesion-equalized with a fixed family (rho=0.5, t0=0.2) on tune U cal.
* Every quantity is computed with unit weights, so the same code serves the
  clustered bootstrap (each resampled unit keeps all its copies in one split) that
  gives CIs accounting for the finite pool and the overlapping partitions.

    python scripts/experiments/mgcrc_v2.py --dataset kvasir-seg --arch unet-r34 --seed 0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.calibration.families import ScoreFamily, connectivity, default_family_grid  # noqa: E402
from app.calibration.losses import lambda_grid, unit_stats  # noqa: E402
from app.core.config import BACKEND_ROOT, REPO_ROOT  # noqa: E402

POST_ROOT = BACKEND_ROOT / "var" / "posteriors_frozen"
CACHE_ROOT = BACKEND_ROOT / "var" / "stats_v2"
OUT_ROOT = REPO_ROOT / "docs" / "research" / "results_v2"
ALPHAS = (0.05, 0.10, 0.15)
SPLIT = (0.2, 0.4, 0.4)
LTT_DELTA = 0.1
N_STRATA = 3
FIXED_FAMILY = "lesion_equalized(rho=0.5,t0=0.2)"
LOSS_BOUND = 1.0

METHODS = (
    # C1: calibrate on cal only (primary)
    "pix", "joint", "max", "dil", "strat", "ltt", "le_union", "le_max", "joint_multi",
    # C2: baselines on tune U cal
    "pix_tc", "joint_tc", "max_tc", "dil_tc", "strat_tc", "ltt_tc", "joint_multi_tc",
    # C3: fixed lesion-equalized family on tune U cal
    "le_fixed_tc",
)
METRICS = ("risk_pix", "risk_les", "risk_small", "risk_multi", "set_to_truth", "fp_to_truth", "lam")


# --------------------------------------------------------------------------- data
def lesion_equalized_conn(p, rho, t0, conn):
    """Lesion-equalized score with an explicit candidate connectivity (ablation)."""
    s = p.copy()
    lab, count = ndimage.label(p >= t0, structure=ndimage.generate_binary_structure(p.ndim, conn))
    if count == 0 or rho == 0:
        return s
    peaks = ndimage.maximum(p, lab, index=np.arange(1, count + 1))
    div = np.ones(count + 1)
    div[1:] = np.power(np.maximum(peaks, t0), rho)
    inside = lab > 0
    s[inside] = np.minimum(p[inside] / div[lab[inside]], 1.0)
    return s


def families(conn4: bool):
    fams = {f.label: f.score for f in default_family_grid()}
    fams["dilation"] = ScoreFamily("dilation").score
    if conn4:  # 2-D only: 4-connected candidates for every lesion-equalized member
        for f in default_family_grid():
            if f.kind == "lesion_equalized":
                fams["c4:" + f.label] = (lambda p, f=f: lesion_equalized_conn(p, f.rho, f.t0, 1))
    return fams


def iter_units(dataset, arch, seed):
    path = POST_ROOT / dataset / f"{arch}_seed{seed}"
    if path.is_dir():  # 3-D: one file per volume
        for f in sorted(path.glob("*.npz")):
            d = np.load(f)
            yield d["prob"].astype(np.float32), d["labels"], d["valid"]
        return
    d = np.load(path.with_suffix(".npz"))
    for p, lab, v in zip(d["prob"], d["labels"], d["valid"]):
        yield p.astype(np.float32), lab, v


def build_cache(dataset, arch, seed, tau, conn4, lambdas):
    path = CACHE_ROOT / f"{dataset}_{arch}_seed{seed}_tau{tau:g}{'_c4' if conn4 else ''}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    fams = families(conn4)
    stats = {k: [] for k in fams}
    mass, lesions = [], []  # lesions: per-unit arrays for the error analysis
    t = time.time()
    for p, lab, v in iter_units(dataset, arch, seed):
        p = np.where(v, p, 0.0).astype(np.float32)
        mass.append(float(p.sum()))
        for name, score in fams.items():
            s = score(p)
            s[~v] = 0.0
            stats[name].append(unit_stats(s, lab, tau=tau, lambdas=lambdas))
        bg = p[(lab == 0) & v]
        k = int(lab.max())
        peaks = ndimage.maximum(p, lab, index=np.arange(1, k + 1)) if k else np.zeros(0)
        sizes = np.bincount(lab.ravel(), minlength=k + 1)[1:] if k else np.zeros(0, int)
        lesions.append(dict(peak=np.asarray(peaks, float), size=np.asarray(sizes),
                            bg_q99=float(np.percentile(bg, 99)) if bg.size else 0.0))
    out = dict(stats=stats, mass=np.asarray(mass), lesions=lesions)
    print(f"    stats: {len(mass)} units x {len(fams)} families ({time.time() - t:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


class Mats:
    """Per-unit loss curves of one family on the grid (rows = units with GT)."""

    def __init__(self, units, small_cut, lambdas):
        cut = 1.0 - lambdas
        n, g = len(units), lambdas.size
        self.pix = np.empty((n, g), np.float32)
        self.les = np.empty((n, g), np.float32)
        self.small = np.zeros((n, g), np.float32)
        self.has_small = np.zeros(n, bool)
        self.size = np.empty((n, g), np.float64)
        self.n_gt = np.empty(n, np.float64)
        self.multi = np.zeros(n, bool)
        for i, u in enumerate(units):
            self.pix[i] = u.pix_miss / u.n_gt
            st = np.sort(u.lesion_stat)
            self.les[i] = np.searchsorted(st, cut, side="left") / st.size
            sm = np.sort(u.lesion_stat[u.lesion_size < small_cut])
            if sm.size:
                self.has_small[i] = True
                self.small[i] = np.searchsorted(sm, cut, side="left") / sm.size
            self.size[i] = u.set_size
            self.n_gt[i] = u.n_gt
            self.multi[i] = u.k >= 2
        self.mx = np.maximum(np.maximum(self.pix, self.les), np.where(self.has_small[:, None], self.small, 0.0))


# --------------------------------------------------------------------------- rules (weighted)
def curves(m: Mats, idx, w):
    """Weighted mean loss curves and the effective n of each risk on subset idx."""
    wi = w[idx]
    out = {}
    for name, rows in (("pix", None), ("les", None), ("small", m.has_small[idx]), ("multi", m.multi[idx])):
        ww = wi if rows is None else wi * rows
        n = ww.sum()
        mat = {"pix": m.pix, "les": m.les, "small": m.small, "multi": m.les}[name]
        out[name] = ((ww @ mat[idx]) / n if n > 0 else np.zeros(mat.shape[1]), n)
    out["max"] = None
    return out


def max_curve(m: Mats, idx, w):
    wi = w[idx]
    return (wi @ m.mx[idx]) / wi.sum(), wi.sum()


def crc_idx(curve, n, alpha):
    if n <= 0:
        return 0
    feas = np.flatnonzero((n * curve + LOSS_BOUND) / (n + 1) <= alpha)
    return int(feas[0]) if feas.size else curve.size - 1


def union_idx(cv, alpha, risks=("pix", "les", "small")):
    return max(crc_idx(*cv[r], alpha) for r in risks)


def hb_p(r_hat, n, alpha):
    r = np.minimum(r_hat, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = r * np.log(r / alpha) + (1 - r) * np.log((1 - r) / (1 - alpha))
    h1 = np.where(r <= 0, -np.log(1 - alpha), h1)
    return np.minimum(np.minimum(np.exp(-n * h1), np.e * sstats.binom.cdf(np.ceil(n * r_hat), int(round(n)), alpha)), 1)


def ltt_idx(cv, alpha, delta=LTT_DELTA):
    p = np.maximum.reduce([hb_p(cv[r][0], cv[r][1], alpha) for r in ("pix", "les", "small") if cv[r][1] > 0])
    i = p.size - 1
    while i - 1 >= 0 and p[i - 1] <= delta:
        i -= 1
    return i


def evaluate(m: Mats, idx, w, index):
    wi = w[idx]
    ii = np.broadcast_to(np.asarray(index), (idx.size,))
    rows = idx
    pix = m.pix[rows, ii]
    les = m.les[rows, ii]
    size = m.size[rows, ii]
    hs, mu = m.has_small[rows], m.multi[rows]
    gt = (wi * m.n_gt[rows]).sum()

    def wmean(v, mask=None):
        ww = wi if mask is None else wi * mask
        return float((ww * v).sum() / ww.sum()) if ww.sum() > 0 else np.nan

    covered = m.n_gt[rows] * (1 - pix)
    return (wmean(pix), wmean(les), wmean(m.small[rows, ii], hs), wmean(les, mu),
            float((wi * size).sum() / gt), float((wi * (size - covered)).sum() / gt), float(np.mean(ii)))


# --------------------------------------------------------------------------- one partition
def run_partition(fam_mats, base, dil, le_labels, tune, cal, test, w, mass, alphas, lambdas, fixed):
    tc = np.concatenate([tune, cal])
    res = np.full((len(alphas), len(METHODS), len(METRICS)), np.nan)
    chosen = []
    cv = {"cal": curves(base, cal, w), "tc": curves(base, tc, w)}
    cvd = {"cal": curves(dil, cal, w), "tc": curves(dil, tc, w)}
    mx = {"cal": max_curve(base, cal, w), "tc": max_curve(base, tc, w)}
    tune_cv = {lab: curves(fam_mats[lab], tune, w) for lab in le_labels}
    tune_mx = {lab: max_curve(fam_mats[lab], tune, w) for lab in le_labels}
    gt_tune = {lab: (w[tune] * fam_mats[lab].n_gt[tune]).sum() for lab in le_labels}
    fixed_cv = curves(fam_mats[fixed], tc, w)
    strata = {}
    for sub in ("cal", "tc"):
        cidx = cal if sub == "cal" else tc
        q = np.quantile(mass[cidx], np.linspace(0, 1, N_STRATA + 1)[1:-1])
        sc, st = np.digitize(mass[cidx], q), np.digitize(mass[test], q)
        strata[sub] = (st, {g: curves(base, cidx[sc == g], w) for g in range(N_STRATA)})
    for ai, a in enumerate(alphas):
        out = {}
        for sfx, sub in (("", "cal"), ("_tc", "tc")):
            c = cv[sub]
            out["pix" + sfx] = (base, crc_idx(*c["pix"], a))
            out["joint" + sfx] = (base, union_idx(c, a))
            out["max" + sfx] = (base, crc_idx(*mx[sub], a))
            out["dil" + sfx] = (dil, union_idx(cvd[sub], a))
            out["ltt" + sfx] = (base, ltt_idx(c, a))
            out["joint_multi" + sfx] = (base, union_idx(c, a, ("pix", "les", "small", "multi")) if c["multi"][1] > 0
                                        else union_idx(c, a))
            # CCRA-S-style strata of total predicted mass; cutpoints from the calibration units
            st, scv = strata[sub]
            per = {g: union_idx(scv[g], a) for g in range(N_STRATA)}
            out["strat" + sfx] = (base, np.asarray([per[g] for g in st]))
        # ours: family chosen on tune (smallest tuned set), calibrated on cal
        for rule in ("union", "max"):
            best, best_size, best_i = None, np.inf, 0
            for lab in le_labels:
                m = fam_mats[lab]
                i = union_idx(tune_cv[lab], a) if rule == "union" else crc_idx(*tune_mx[lab], a)
                size = (w[tune] * m.size[tune, i]).sum() / gt_tune[lab]
                if size < best_size - 1e-12:
                    best, best_size = lab, size
            m = fam_mats[best]
            i = union_idx(curves(m, cal, w), a) if rule == "union" else crc_idx(*max_curve(m, cal, w), a)
            out[f"le_{rule}"] = (m, i)
            if rule == "union":
                chosen.append(best)
        out["le_fixed_tc"] = (fam_mats[fixed], union_idx(fixed_cv, a))
        for mi, name in enumerate(METHODS):
            m, i = out[name]
            res[ai, mi] = evaluate(m, test, w, i)
    return res, chosen


def partitions(n, rng, reps, units=None):
    """Random tune/cal/test partitions over distinct units (clusters)."""
    n_tune, n_cal = int(SPLIT[0] * n), int(SPLIT[1] * n)
    for _ in range(reps):
        perm = rng.permutation(n)
        yield perm[:n_tune], perm[n_tune:n_tune + n_cal], perm[n_tune + n_cal:]


def run(dataset, arch, seed, reps, boot, boot_reps, tau, small_q, conn4, ncal, alphas, workers_seed=0):
    lambdas = lambda_grid()
    manifest = json.loads((POST_ROOT / dataset / f"{arch}_seed{seed}.json").read_text())
    small_cut = manifest["small_cuts_from_train"][f"q{small_q:g}"]
    cache = build_cache(dataset, arch, seed, tau, conn4, lambdas)
    keep = [i for i, u in enumerate(cache["stats"]["threshold"]) if u.n_gt > 0 and u.k > 0]
    mass = cache["mass"][keep]
    fam_mats = {k: Mats([v[i] for i in keep], small_cut, lambdas) for k, v in cache["stats"].items()}
    base, dil = fam_mats["threshold"], fam_mats["dilation"]
    le_labels = [f.label for f in default_family_grid()]
    if conn4:
        le_labels = ["threshold"] + [k for k in fam_mats if k.startswith("c4:")]
    n = len(keep)
    rng = np.random.default_rng(seed * 7919 + 17)
    w1 = np.ones(n)

    def one(tune, cal, test, w):
        if ncal:
            cal = cal[:ncal]
        return run_partition(fam_mats, base, dil, le_labels, tune, cal, test, w, mass, alphas, lambdas, FIXED_FAMILY)

    t = time.time()
    main_res, chosen = [], []
    for tune, cal, test in partitions(n, rng, reps):
        r, c = one(tune, cal, test, w1)
        main_res.append(r)
        chosen.extend(c)
    main_res = np.stack(main_res)  # [R, A, M, K]
    print(f"    {reps} partitions: {time.time() - t:.0f}s", flush=True)

    boot_res = boot_rel = None
    if boot:
        t = time.time()
        boot_res, boot_rel = [], []
        j, k_str = METHODS.index("joint"), METRICS.index("set_to_truth")
        for b in range(boot):
            draw = rng.integers(0, n, n)
            ids, counts = np.unique(draw, return_counts=True)
            w = np.zeros(n)
            w[ids] = counts
            acc = []
            for tune, cal, test in partitions(ids.size, rng, boot_reps):
                acc.append(one(ids[tune], ids[cal], ids[test], w)[0])
            acc = np.stack(acc)
            boot_res.append(np.nanmean(acc, axis=0))  # mean over partitions
            boot_rel.append(np.nanmean(acc[:, :, :, k_str] / acc[:, :, j:j + 1, k_str] - 1, axis=0))
        boot_res, boot_rel = np.stack(boot_res), np.stack(boot_rel)  # [B, A, M, K], [B, A, M]
        print(f"    bootstrap {boot} x {boot_reps}: {time.time() - t:.0f}s", flush=True)

    lesions = [cache["lesions"][i] for i in keep]
    meta = dict(dataset=dataset, arch=arch, seed=seed, tau=tau, small_q=small_q, small_cut=small_cut, conn4=conn4,
                ncal=ncal, reps=reps, boot=boot, boot_reps=boot_reps, n_units=n, n_multi=int(base.multi.sum()),
                n_small=int(base.has_small.sum()), split=SPLIT, ltt_delta=LTT_DELTA, grid=int(lambdas.size),
                alphas=list(alphas), methods=list(METHODS), metrics=list(METRICS),
                n_cal=int(SPLIT[1] * n) if not ncal else min(ncal, int(SPLIT[1] * n)), n_tune=int(SPLIT[0] * n),
                pool_metrics=manifest["pool_metrics"], fixed_family=FIXED_FAMILY)
    return dict(meta=meta, main=main_res, boot=boot_res, boot_rel=boot_rel, chosen=chosen, lesions=lesions,
                base_stat_les=[u.lesion_stat for u in [cache["stats"]["threshold"][i] for i in keep]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arch", default="unet-r34")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot-reps", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--small-q", type=float, default=33.0)
    ap.add_argument("--conn4", action="store_true")
    ap.add_argument("--ncal", type=int, default=0)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t = time.time()
    out = run(args.dataset, args.arch, args.seed, args.reps, args.boot, args.boot_reps, args.tau,
              args.small_q, args.conn4, args.ncal, ALPHAS)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_{args.dataset}_{args.arch}_seed{args.seed}"
    (OUT_ROOT / f"{stem}.pkl").write_bytes(pickle.dumps(out))
    m = out["main"].mean(0)
    print(f"{stem}: {time.time() - t:.0f}s  n={out['meta']['n_units']} multi={out['meta']['n_multi']}")
    for ai, a in enumerate(ALPHAS):
        for mi, name in enumerate(METHODS):
            v = m[ai, mi]
            print(f"  a={a:.2f} {name:<15} pix {v[0]:.3f} les {v[1]:.3f} small {v[2]:.3f} multi {v[3]:.3f} "
                  f"STR {v[4]:.3f}")


if __name__ == "__main__":
    main()
