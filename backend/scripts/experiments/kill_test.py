"""Gate A kill test: does pixel-level conformal risk control leave lesions unprotected?

Two phases, deliberately separated so the minimum-lesion-size m is chosen from
the size histogram BEFORE any gap number is computed:

    # phase 1 — K3 data audit only; prints the size histogram and stops
    python scripts/experiments/kill_test.py --dataset kvasir-seg --audit-only

    # phase 2 — gap test with m fixed from the audit
    python scripts/experiments/kill_test.py --dataset kvasir-seg --m 10 --seeds 3

FIXED PROTOCOL (see docs/research/w1/theory-memo.md, "Assumptions"):
  hit criterion   GT component hit at lambda iff >= tau of its pixels are in S_lambda
                  (tau = 0.5). Ground-truth components only — never predicted ones.
  connectivity    8-connectivity, 2-D.
  exclusion       |Y_i| = 0 after size filtering -> excluded from both losses.
  alphas          {0.05, 0.10, 0.15}.
  prediction set  S_lambda = {p >= 1 - lambda} (thresholding), lambda grid of 200.
  selection       lambda_hat = first lambda with (n R_n(lambda) + 1)/(n+1) <= alpha,
                  identical to app.calibration.conformal.ConformalRiskControl.

This script REUSES: load_slices / split_by_case (patient-wise splits), resize_pair /
as_preprocessed (the same 256^2 working geometry the app uses), CalibrationSample /
CalibrationSet + _loss_curve (the app's own pixel-FNR curve), LAMBDA_GRID_SIZE, and
the RandomForestPixelSegmenter adapter. It does NOT add modules to app/: if Gate A
fails, nothing here needs undoing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy import ndimage  # noqa: E402

from app.calibration.conformal import LAMBDA_GRID_SIZE, _loss_curve  # noqa: E402
from app.calibration.datasets import CalibrationSample, CalibrationSet  # noqa: E402
from app.core.config import BACKEND_ROOT, REPO_ROOT, get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.datasets.loaders import Slice2D, load_slices, split_by_case  # noqa: E402
from app.datasets.medical import as_preprocessed, resize_pair  # noqa: E402
from app.datasets.registry import get_spec  # noqa: E402
from app.models.adapters.rf_pixel_segmenter import RandomForestPixelSegmenter  # noqa: E402
from app.schemas.common import DataProvenance  # noqa: E402

# 8-connectivity in 2-D: a 3x3 all-ones structuring element.
CONNECTIVITY = np.ones((3, 3), dtype=bool)
TAU = 0.5
ALPHAS = (0.05, 0.10, 0.15)
OUT_DIR = REPO_ROOT / "docs" / "research" / "w1"


# ----------------------------------------------------------------- components --
def components(mask: np.ndarray, m: int) -> tuple[np.ndarray, int]:
    """Label GT components (8-conn) and merge those below m pixels into background.

    Returns the filtered label image and K (number of surviving components).
    """
    labels, count = ndimage.label(mask, structure=CONNECTIVITY)
    if count == 0:
        return labels, 0
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = np.flatnonzero(sizes >= m) + 1
    if keep.size == count:
        return labels, count
    filtered = np.where(np.isin(labels, keep), labels, 0)
    # Relabel densely so K matches the number of surviving components.
    relabelled, k = ndimage.label(filtered > 0, structure=CONNECTIVITY)
    return relabelled, k


# ----------------------------------------------------------------------- audit --
@dataclass
class Audit:
    dataset: str
    resolution: str
    n_units: int
    n_units_excluded_empty: int
    multi_lesion_fraction: float
    mean_k: float
    n_components: int
    frac_components_below_1pct_area: float
    size_percentiles_px: dict
    histogram: dict
    n_train: int
    n_cal: int
    n_test: int


def audit(slices: list[Slice2D], dataset: str, size: int | None, m: int) -> Audit:
    """K3 statistics at native resolution (size=None) or the working resolution."""
    ks, sizes, excluded = [], [], 0
    for item in slices:
        mask = item.mask
        if size is not None:
            _, mask = resize_pair(item.image, item.mask, size)
        labels, k = components(mask, m)
        if k == 0:
            excluded += 1
            continue
        ks.append(k)
        area = mask.size
        sizes.extend((ndimage.sum(labels > 0, labels, index=np.arange(1, k + 1)) / area).tolist())

    sizes_arr = np.asarray(sizes)
    px = sizes_arr * (size**2 if size else np.mean([s.mask.size for s in slices]))
    edges = [0, 10, 50, 200, 1000, 5000, 20000, 100000, np.inf]
    hist, _ = np.histogram(px, bins=edges)
    train, cal, test = split_by_case(slices)
    return Audit(
        dataset=dataset,
        resolution=f"{size}x{size}" if size else "native",
        n_units=len(slices),
        n_units_excluded_empty=excluded,
        multi_lesion_fraction=float(np.mean(np.asarray(ks) >= 2)) if ks else 0.0,
        mean_k=float(np.mean(ks)) if ks else 0.0,
        n_components=int(len(sizes)),
        frac_components_below_1pct_area=float(np.mean(sizes_arr < 0.01)) if sizes else 0.0,
        size_percentiles_px={
            f"p{p}": round(float(np.percentile(px, p)), 1) for p in (1, 5, 25, 50, 75, 95)
        } if sizes else {},
        histogram={f"[{edges[i]},{edges[i+1]})": int(hist[i]) for i in range(len(hist))},
        n_train=len(train),
        n_cal=len(cal),
        n_test=len(test),
    )


def print_audit(a: Audit) -> None:
    print(f"\n  K3 audit — {a.dataset} @ {a.resolution}")
    print(f"    units {a.n_units}  (excluded as empty after m-filter: {a.n_units_excluded_empty})")
    print(f"    split train/cal/test = {a.n_train}/{a.n_cal}/{a.n_test}")
    print(f"    multi-lesion fraction (K>=2): {a.multi_lesion_fraction:.3f}   mean K = {a.mean_k:.2f}")
    print(f"    components: {a.n_components}   below 1% of image area: {a.frac_components_below_1pct_area:.3f}")
    print(f"    size percentiles (px): {a.size_percentiles_px}")
    print("    size histogram (px):")
    for k, v in a.histogram.items():
        print(f"      {k:>18}  {v}")


# -------------------------------------------------------------------- model --
def get_model(dataset: str, train: list[Slice2D], size: int, seed: int) -> RandomForestPixelSegmenter:
    path = BACKEND_ROOT / "var" / "models" / f"rf_pixel_seg_{dataset}_seed{seed}.joblib"
    model = RandomForestPixelSegmenter(model_path=path, seed=seed)
    if not model.is_trained:
        images, masks = zip(*(resize_pair(s.image, s.mask, size) for s in train))
        model.train(list(images), list(masks), dataset_id=dataset,
                    n_cases=len({s.case_id for s in train}))
    model.load()
    return model


def build_samples(model, slices: list[Slice2D], size: int, m: int):
    """(CalibrationSet with m-filtered truth, per-image component order stats, |M_hat| per image)."""
    samples, order_stats, mhat_sizes, dropped = [], [], [], 0
    for item in slices:
        image, mask = resize_pair(item.image, item.mask, size)
        labels, k = components(mask, m)
        if k == 0:
            dropped += 1
            continue
        p = np.clip(model.predict_segmentation(as_preprocessed(image)).foreground_probability, 0, 1)
        truth = labels > 0
        samples.append(CalibrationSample(foreground_probability=p, ground_truth=truth))
        # Lemma 2: component k is hit at cutoff c iff its ceil(tau*|Y^k|)-th largest score >= c.
        stats = []
        for comp in range(1, k + 1):
            scores = np.sort(p[labels == comp])[::-1]
            stats.append(scores[int(np.ceil(TAU * scores.size)) - 1])
        order_stats.append(np.sort(np.asarray(stats)))
        mhat_sizes.append(int((p >= 0.5).sum()))
    cs = CalibrationSet(samples=samples, name="kill", provenance=DataProvenance.COMPUTED, description="")
    return cs, order_stats, np.asarray(mhat_sizes), dropped


# ------------------------------------------------------------------- losses --
def lesion_curve(order_stats: list[np.ndarray], lambdas: np.ndarray) -> np.ndarray:
    """Mean lesion-miss rate at each lambda (one searchsorted per image, Lemma 2)."""
    cutoffs = 1.0 - lambdas
    losses = np.zeros((len(order_stats), lambdas.size))
    for i, s in enumerate(order_stats):
        losses[i] = np.searchsorted(s, cutoffs, side="left") / s.size
    return losses.mean(axis=0)


def set_size_curve(cs: CalibrationSet, lambdas: np.ndarray) -> np.ndarray:
    """Total |S_lambda| over the set at each lambda."""
    cutoffs = 1.0 - lambdas
    total = np.zeros(lambdas.size)
    for s in cs.samples:
        p = np.sort(s.foreground_probability.ravel())
        total += p.size - np.searchsorted(p, cutoffs, side="left")
    return total


def select(curve: np.ndarray, n: int, alpha: float, lambdas: np.ndarray) -> tuple[float, int, bool]:
    corrected = (n * curve + 1.0) / (n + 1)
    feasible = np.flatnonzero(corrected <= alpha)
    if feasible.size:
        return float(lambdas[feasible[0]]), int(feasible[0]), True
    return float(lambdas[-1]), int(lambdas.size - 1), False


# ---------------------------------------------------------------------- main --
def run_gap(dataset: str, m: int, size: int, seeds: list[int]) -> list[dict]:
    lambdas = np.linspace(0.0, 1.0, LAMBDA_GRID_SIZE)
    rows = []
    for seed in seeds:
        slices = load_slices(dataset, max_cases=100_000, slices_per_case=3)
        train, cal, test = split_by_case(slices, seed=seed)
        model = get_model(dataset, train, size, seed)

        cal_cs, cal_os, cal_mhat, cal_dropped = build_samples(model, cal, size, m)
        test_cs, test_os, test_mhat, _ = build_samples(model, test, size, m)
        n = len(cal_cs)

        R_pix = _loss_curve(cal_cs, lambdas)          # the app's own pixel-FNR curve
        R_les = lesion_curve(cal_os, lambdas)
        S = set_size_curve(cal_cs, lambdas)
        R_pix_test = _loss_curve(test_cs, lambdas)
        R_les_test = lesion_curve(test_os, lambdas)
        mhat_total = max(int(cal_mhat.sum()), 1)
        ks = np.asarray([len(s) for s in cal_os])

        assert np.all(np.diff(R_pix) <= 1e-12) and np.all(np.diff(R_les) <= 1e-12), "loss not monotone"

        for alpha in ALPHAS:
            l_pix, i_pix, f_pix = select(R_pix, n, alpha, lambdas)
            l_les, i_les, f_les = select(R_les, n, alpha, lambdas)
            l_un = max(l_pix, l_les); i_un = max(i_pix, i_les)
            rows.append(dict(
                dataset=dataset, seed=seed, alpha=alpha, m=m, tau=TAU, n_cal=n,
                n_cal_dropped_empty=cal_dropped, n_test=len(test_cs),
                multi_lesion_fraction=round(float(np.mean(ks >= 2)), 3), mean_k=round(float(ks.mean()), 2),
                empty_mhat_frac=round(float((cal_mhat == 0).mean()), 3),
                lambda_pix=round(l_pix, 3), feasible_pix=f_pix,
                R_pix_at_lpix=round(float(R_pix[i_pix]), 4), R_les_at_lpix=round(float(R_les[i_pix]), 4),
                G=round(float(R_les[i_pix] / alpha), 3),
                R_les_test_at_lpix=round(float(R_les_test[i_pix]), 4),
                R_pix_test_at_lpix=round(float(R_pix_test[i_pix]), 4),
                lambda_les=round(l_les, 3), feasible_les=f_les,
                R_pix_at_lles=round(float(R_pix[i_les]), 4), R_les_at_lles=round(float(R_les[i_les]), 4),
                lambda_union=round(l_un, 3),
                infl_pix=round(float(S[i_pix] / mhat_total), 2),
                infl_les=round(float(S[i_les] / mhat_total), 2),
                infl_union=round(float(S[i_un] / mhat_total), 2),
            ))
            r = rows[-1]
            print(f"  seed {seed} a={alpha:.2f}  l_pix={l_pix:.3f}{'' if f_pix else '!'} "
                  f"R_pix={r['R_pix_at_lpix']:.3f} R_les={r['R_les_at_lpix']:.3f}  G={r['G']:.2f}  "
                  f"| l_les={l_les:.3f}{'' if f_les else '!'} l_union={l_un:.3f}  "
                  f"infl pix/les/union={r['infl_pix']}/{r['infl_les']}/{r['infl_union']}  "
                  f"| test: R_les@l_pix={r['R_les_test_at_lpix']:.3f}")
    return rows


def write_report(dataset: str, rows: list[dict], audits: list[Audit], m_note: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"kill_{dataset}.json").write_text(
        json.dumps({"rows": rows, "audits": [asdict(a) for a in audits], "m_justification": m_note}, indent=2),
        encoding="utf-8")
    import csv
    with open(OUT_DIR / f"kill_{dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Aggregate over seeds per alpha.
    lines = [f"# Gate A kill test — {dataset}", "",
             f"Protocol: tau={TAU}, 8-connectivity, m={rows[0]['m']} px @ working res, "
             f"empty-GT excluded, thresholding sets, lambda grid {LAMBDA_GRID_SIZE}, "
             f"seeds={sorted({r['seed'] for r in rows})}.", "", f"m justification: {m_note}", "",
             "| alpha | n_cal | multi-lesion frac | lambda_pix | R_pix | R_les(lambda_pix) | **G** | R_les test | lambda_les | lambda_union | infl pix | infl les | infl union |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for alpha in ALPHAS:
        rs = [r for r in rows if r["alpha"] == alpha]
        def agg(k, fmt="{:.3f}"):
            v = np.asarray([r[k] for r in rs]);
            return (fmt.format(v.mean()) + (f" ± {v.std():.3f}" if len(v) > 1 else ""))
        flag = "" if all(r["feasible_pix"] for r in rs) else " (infeasible!)"
        lines.append(f"| {alpha:.2f} | {rs[0]['n_cal']} | {agg('multi_lesion_fraction')} | {agg('lambda_pix')}{flag} | "
                     f"{agg('R_pix_at_lpix')} | {agg('R_les_at_lpix')} | **{agg('G', '{:.2f}')}** | {agg('R_les_test_at_lpix')} | "
                     f"{agg('lambda_les')} | {agg('lambda_union')} | {agg('infl_pix','{:.2f}')} | {agg('infl_les','{:.2f}')} | {agg('infl_union','{:.2f}')} |")
    lines += ["", "## K3 audit"]
    for a in audits:
        lines += [f"", f"**{a.resolution}** — units {a.n_units}, excluded-empty {a.n_units_excluded_empty}, "
                  f"train/cal/test {a.n_train}/{a.n_cal}/{a.n_test}, multi-lesion {a.multi_lesion_fraction:.3f}, "
                  f"mean K {a.mean_k:.2f}, components {a.n_components}, <1% area {a.frac_components_below_1pct_area:.3f}",
                  f"", f"size percentiles px: `{a.size_percentiles_px}`", "", "| bin (px) | count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in a.histogram.items()]
    path = OUT_DIR / f"kill_{dataset}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--audit-only", action="store_true", help="Phase 1: histogram only, no gap test")
    ap.add_argument("--m", type=int, default=None, help="Min lesion size in px at working res (REQUIRED for the gap test)")
    ap.add_argument("--m-note", default="", help="One-line justification for m, recorded in the report")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337])
    ap.add_argument("--size", type=int, default=None, help="Working resolution (default: settings.inference_size)")
    args = ap.parse_args()
    configure_logging("WARNING")
    size = args.size or get_settings().inference_size
    spec = get_spec(args.dataset)
    print(f"\n{spec.name}  |  working res {size}  |  tau={TAU}  8-conn")

    t0 = time.perf_counter()
    slices = load_slices(args.dataset, max_cases=100_000, slices_per_case=3)
    audits = [audit(slices, args.dataset, None, m=0), audit(slices, args.dataset, size, m=0)]
    for a in audits:
        print_audit(a)
    if args.audit_only or args.m is None:
        if args.m is None and not args.audit_only:
            print("\n  --m not given: choose it from the histogram above, then re-run with --m N --m-note '...'.")
        return

    print(f"\n  m = {args.m} px  —  {args.m_note or '(no justification given)'}")
    print_audit(audit(slices, args.dataset, size, m=args.m))
    print("\n  Gap test:")
    rows = run_gap(args.dataset, args.m, size, args.seeds)
    path = write_report(args.dataset, rows, audits + [audit(slices, args.dataset, size, m=args.m)], args.m_note)
    print(f"\n  report -> {path.relative_to(REPO_ROOT)}   ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
