"""Live lesion-level CRC demo on real frozen-model posteriors.

One frozen-model bundle (scripts/build_demo_bundle.py, from the output of
experiments/kaggle_train.py) is loaded once. Per-unit loss curves are computed for
three score families and cached; after that every request draws a fresh random
tune/calibrate/test partition and re-runs the calibration rules of
app.calibration.joint, the same functions the experiments used. Nothing here is a
recorded number: every risk and set size is measured on the held-out test units of
the partition that was just drawn.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from app.calibration.families import ScoreFamily
from app.calibration.joint import crc_index, evaluate, ltt_index, risk_matrix, union_index
from app.calibration.losses import DEFAULT_TAU, StackedStats, UnitStats, lambda_grid, stack, unit_stats
from app.core.config import BACKEND_ROOT
from app.core.logging import get_logger

logger = get_logger(__name__)

DEMO_ROOT = BACKEND_ROOT / "var" / "demo"
DEFAULT_BUNDLE = DEMO_ROOT / "kvasir-seg_unet-r34_seed0.npz"
#: tune / calibrate / test, as in the experiments (pre-registration Amendment 3, P1)
SPLIT = (0.2, 0.4, 0.4)
LTT_DELTA = 0.1
N_STRATA = 3
JOINT = ("pix", "les", "small")

FAMILIES: dict[str, ScoreFamily] = {
    "threshold": ScoreFamily("threshold"),
    "dilation": ScoreFamily("dilation"),
    "lesion_equalized": ScoreFamily("lesion_equalized", rho=0.5, t0=0.2),
}


@dataclass(frozen=True, slots=True)
class Method:
    id: str
    label: str
    family: str
    rule: str  # crc_pix | joint | joint_multi | ltt | strata
    guaranteed: tuple[str, ...]
    guarantee: str
    ours: bool = False


METHODS: tuple[Method, ...] = (
    Method("pixel", "Pixel CRC (standard)", "threshold", "crc_pix", ("pix",), "expectation"),
    Method("joint", "Joint CRC (this work)", "threshold", "joint", JOINT, "expectation, each risk", ours=True),
    Method("joint_multi", "Joint CRC + multi-lesion risk (this work, exploratory)", "threshold", "joint_multi",
           JOINT + ("multi",), "expectation, each risk", ours=True),
    Method("ltt", "Learn-Then-Test", "threshold", "ltt", JOINT, "high probability (δ = 0.1)"),
    Method("dilation", "Morphological (dilation) family", "dilation", "joint", JOINT, "expectation, each risk"),
    Method("strata", "CCRA-S-style strata", "threshold", "strata", JOINT, "per stratum (approximate)"),
    Method("lesion_eq", "Lesion-equalized sets (this work, ρ = 0.5, t₀ = 0.2)", "lesion_equalized", "joint", JOINT,
           "expectation, each risk", ours=True),
)
METHOD_BY_ID = {m.id: m for m in METHODS}


@dataclass
class DemoData:
    prob: np.ndarray  # [N, H, W] float16
    labels: np.ndarray  # [N, H, W] uint8, GT components
    valid: np.ndarray  # [N, H, W] bool, False on letterbox padding
    ids: list[str]
    jpeg: np.ndarray
    jpeg_offsets: np.ndarray
    manifest: dict
    small_cut: float
    lambdas: np.ndarray
    units: dict[str, list[UnitStats]] = field(default_factory=dict)
    stats: dict[str, StackedStats] = field(default_factory=dict)
    mass: np.ndarray | None = None
    peak: list[np.ndarray] = field(default_factory=list)
    bg_q99: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.ids)

    def image_jpeg(self, uid: int) -> bytes:
        return self.jpeg[self.jpeg_offsets[uid]:self.jpeg_offsets[uid + 1]].tobytes()

    def score(self, uid: int, family: str) -> np.ndarray:
        p = np.where(self.valid[uid], self.prob[uid].astype(np.float32), 0.0)
        s = FAMILIES[family].score(p)
        s[~self.valid[uid]] = 0.0
        return s


class LesionDemo:
    """Thread-safe lazy holder of the demo data."""

    def __init__(self, bundle: Path = DEFAULT_BUNDLE):
        self.bundle = bundle
        self._data: DemoData | None = None
        self._error: str | None = None
        self._lock = threading.Lock()
        self._validity_cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------ loading
    @property
    def available(self) -> bool:
        return self.bundle.exists()

    def data(self) -> DemoData:
        with self._lock:
            if self._data is None:
                self._data = self._load()
            return self._data

    def _load(self) -> DemoData:
        d = np.load(self.bundle)
        manifest = json.loads(str(d["manifest"]))
        lambdas = lambda_grid()
        data = DemoData(
            prob=d["prob"], labels=d["labels"], valid=d["valid"].astype(bool), ids=[str(i) for i in d["ids"]],
            jpeg=d["jpeg"], jpeg_offsets=d["jpeg_offsets"], manifest=manifest,
            small_cut=float(manifest["small_cuts_from_train"]["q33"]), lambdas=lambdas,
        )
        digest = hashlib.sha256(self.bundle.read_bytes()[:1 << 20] + str(self.bundle.stat().st_size).encode()).hexdigest()[:16]
        cache = self.bundle.with_name(f"{self.bundle.stem}.stats_{digest}.pkl")
        if cache.exists():
            data.units, data.mass, data.peak, data.bg_q99 = pickle.loads(cache.read_bytes())
        else:
            logger.info("lesion demo: computing loss curves for %d units x %d families", data.n, len(FAMILIES))
            units = {f: [] for f in FAMILIES}
            mass, peak, bg = [], [], []
            for i in range(data.n):
                lab = data.labels[i]
                for f in FAMILIES:
                    units[f].append(unit_stats(data.score(i, f), lab, tau=DEFAULT_TAU, lambdas=lambdas))
                p = np.where(data.valid[i], data.prob[i].astype(np.float32), 0.0)
                mass.append(float(p.sum()))
                k = int(lab.max())
                peak.append(np.asarray(ndimage.maximum(p, lab, index=np.arange(1, k + 1)) if k else [], float))
                b = p[(lab == 0) & data.valid[i]]
                bg.append(float(np.percentile(b, 99)) if b.size else 0.0)
            data.units, data.mass, data.peak, data.bg_q99 = units, np.asarray(mass), peak, np.asarray(bg)
            cache.write_bytes(pickle.dumps((data.units, data.mass, data.peak, data.bg_q99)))
        if any(u.k == 0 or u.n_gt == 0 for u in data.units["threshold"]):
            raise ValueError("every demo unit must contain at least one lesion")
        data.stats = {f: stack(us, small_max_size=data.small_cut, lambdas=lambdas) for f, us in data.units.items()}
        return data

    # ------------------------------------------------------------------ calibration
    @staticmethod
    def partition(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perm = np.random.default_rng(seed).permutation(n)
        n_tune, n_cal = int(SPLIT[0] * n), int(SPLIT[1] * n)
        return np.sort(perm[:n_tune]), np.sort(perm[n_tune:n_tune + n_cal]), np.sort(perm[n_tune + n_cal:])

    def _calibrate(self, data: DemoData, method: Method, cal: np.ndarray, test: np.ndarray, alpha: float):
        """Grid index for every test unit (per-unit only for the stratified rule) and feasibility."""
        st = data.stats[method.family].subset(cal)
        if method.rule == "crc_pix":
            i, ok = crc_index(risk_matrix(st, "pix")[0], alpha)
        elif method.rule == "joint":
            i, ok = union_index(st, alpha, JOINT)
        elif method.rule == "joint_multi":
            i, ok = union_index(st, alpha, JOINT + ("les_multi",))
        elif method.rule == "ltt":
            i, ok = ltt_index(st, alpha, delta=LTT_DELTA, risks=JOINT)
        elif method.rule == "strata":
            q = np.quantile(data.mass[cal], np.linspace(0, 1, N_STRATA + 1)[1:-1])
            sc, stt = np.digitize(data.mass[cal], q), np.digitize(data.mass[test], q)
            per = {g: union_index(st.subset(np.flatnonzero(sc == g)), alpha, JOINT) for g in range(N_STRATA)}
            return np.asarray([per[g][0] for g in stt]), all(v[1] for v in per.values()), [per[g][0] for g in range(N_STRATA)]
        else:
            raise ValueError(method.rule)
        return np.full(test.size, i), ok, [i]

    def run(self, alpha: float, seed: int) -> dict:
        data = self.data()
        tune, cal, test = self.partition(data.n, seed)
        thr = data.stats["threshold"]
        st_cal = thr.subset(cal)
        methods, idx_by_method = [], {}
        for m in METHODS:
            idx, ok, grid_idx = self._calibrate(data, m, cal, test, alpha)
            idx_by_method[m.id] = idx
            ev = evaluate(data.stats[m.family].subset(test), idx, data.lambdas)
            methods.append(dict(
                id=m.id, label=m.label, family=m.family, ours=m.ours, guaranteed=list(m.guaranteed),
                guarantee=m.guarantee, feasible=bool(ok),
                lam=[float(data.lambdas[g]) for g in grid_idx], cutoff=[float(1 - data.lambdas[g]) for g in grid_idx],
                test=dict(pix=_f(ev.risk_pix), les=_f(ev.risk_les), small=_f(ev.risk_small),
                          multi=_f(ev.risk_les_multi), set_to_truth=_f(ev.set_to_truth), bg_to_truth=_f(ev.fp_to_truth)),
            ))
        curves = {}
        for r, key in (("pix", "pix"), ("les", "les"), ("small", "small"), ("multi", "les_multi")):
            mat = risk_matrix(st_cal, key)[0]
            n = mat.shape[0]
            mean = mat.mean(axis=0, dtype=np.float64) if n else np.zeros(data.lambdas.size)
            curves[r] = dict(n=int(n), corrected=[round(float(v), 5) for v in (n * mean + 1) / (n + 1)],
                             lam_hat=float(data.lambdas[crc_index(mat, alpha)[0]]) if n else 1.0)
        units = [self._unit_summary(data, int(u), j, idx_by_method) for j, u in enumerate(test)]
        units.sort(key=lambda u: -u["significance"])
        return dict(
            alpha=alpha, seed=seed, n_tune=int(tune.size), n_cal=int(cal.size), n_test=int(test.size),
            eligible_cal=dict(pix=int(st_cal.n_gt.size), les=int(st_cal.n_gt.size), small=int(st_cal.has_small.sum()),
                              multi=int(st_cal.multi.sum())),
            grid_size=int(data.lambdas.size), methods=methods, curves=curves, units=units,
        )

    def _unit_summary(self, data: DemoData, uid: int, j: int, idx_by_method: dict) -> dict:
        lesions = []
        for k in range(data.units["threshold"][uid].k):
            found = {}
            for m in METHODS:
                stat = data.units[m.family][uid].lesion_stat[k]
                found[m.id] = bool(stat >= 1 - data.lambdas[idx_by_method[m.id][j]])
            size = int(data.units["threshold"][uid].lesion_size[k])
            lesions.append(dict(index=k + 1, size=size, small=size < data.small_cut,
                                stat=float(data.units["threshold"][uid].lesion_stat[k]),
                                peak=float(data.peak[uid][k]), found=found))
        rescued = sum(1 for le in lesions if le["found"]["joint"] and not le["found"]["pixel"])
        missed_pix = sum(1 for le in lesions if not le["found"]["pixel"])
        missed_joint = sum(1 for le in lesions if not le["found"]["joint"])
        st = data.stats["threshold"]
        set_px = {m: int(st.set_size[uid, idx_by_method[m][j]]) for m in ("pixel", "joint")}
        significance = (100 * rescued + 30 * (len(lesions) >= 2) + 10 * missed_pix
                        + 5 * sum(le["small"] for le in lesions if le["found"]["joint"] and not le["found"]["pixel"]))
        cutoffs = {m.id: float(1 - data.lambdas[idx_by_method[m.id][j]]) for m in METHODS}
        return dict(uid=uid, id=data.ids[uid], k=len(lesions), multi=len(lesions) >= 2, lesions=lesions, cutoffs=cutoffs,
                    rescued=rescued, missed_pixel=missed_pix, missed_joint=missed_joint,
                    gt_pixels=int(st.n_gt[uid]), set_pixels=set_px, bg_q99=float(data.bg_q99[uid]),
                    significance=int(significance))

    def rescues(self, alpha: float, splits: int) -> dict:
        """Images where pixel CRC misses a ground-truth polyp and the joint rule finds it.

        Scans run seeds 0..splits-1 (the same partitions as `run`), so each hit carries a
        seed the page can open to show that exact calibration.
        """
        key = ("rescues", round(alpha, 4), splits)
        if key in self._validity_cache:
            return self._validity_cache[key]
        data = self.data()
        by_id = {m.id: m for m in METHODS}
        thr = data.units["threshold"]
        hits: dict[int, dict] = {}
        n_test = np.zeros(data.n, int)
        for s in range(splits):
            _, cal, test = self.partition(data.n, s)
            cut_p = 1 - data.lambdas[self._calibrate(data, by_id["pixel"], cal, test, alpha)[0][0]]
            cut_j = 1 - data.lambdas[self._calibrate(data, by_id["joint"], cal, test, alpha)[0][0]]
            n_test[test] += 1
            for u in test:
                stat = thr[u].lesion_stat
                rescued = [k + 1 for k in range(thr[u].k) if cut_j <= stat[k] < cut_p]
                if not rescued:
                    continue
                h = hits.setdefault(int(u), dict(n_rescued=0, seed=s, lesions=set()))
                h["n_rescued"] += 1
                h["lesions"].update(rescued)
        units = []
        for u, h in hits.items():
            sizes = thr[u].lesion_size
            lesions = sorted(h["lesions"])
            units.append(dict(
                uid=u, id=data.ids[u], k=int(thr[u].k), multi=bool(thr[u].k >= 2), lesions=lesions,
                small=bool(any(sizes[k - 1] < data.small_cut for k in lesions)),
                stat=float(min(thr[u].lesion_stat[k - 1] for k in lesions)),
                n_test=int(n_test[u]), n_rescued=h["n_rescued"], seed=h["seed"],
            ))
        units.sort(key=lambda d: (-d["n_rescued"] / max(1, d["n_test"]), -d["n_rescued"]))
        res = dict(alpha=alpha, splits=splits, units=units)
        self._validity_cache[key] = res
        return res

    def validity(self, alpha: float, reps: int, seed: int = 0) -> dict:
        key = (round(alpha, 4), reps, seed)
        if key in self._validity_cache:
            return self._validity_cache[key]
        data = self.data()
        acc = {m.id: [] for m in METHODS}
        for r in range(reps):
            _, cal, test = self.partition(data.n, 10_000 + seed * 7919 + r)
            joint_size = None
            rows = {}
            for m in METHODS:
                idx, _, _ = self._calibrate(data, m, cal, test, alpha)
                ev = evaluate(data.stats[m.family].subset(test), idx, data.lambdas)
                rows[m.id] = ev
                if m.id == "joint":
                    joint_size = ev.set_to_truth
            for m in METHODS:
                ev = rows[m.id]
                acc[m.id].append((ev.risk_pix, ev.risk_les, ev.risk_small, ev.risk_les_multi, ev.set_to_truth,
                                  ev.set_to_truth / joint_size - 1))
        out = []
        names = ("pix", "les", "small", "multi", "set_to_truth", "rel_vs_joint")
        for m in METHODS:
            a = np.asarray(acc[m.id], float)
            out.append(dict(id=m.id, label=m.label, ours=m.ours, guaranteed=list(m.guaranteed), guarantee=m.guarantee,
                            stats={n: dict(mean=_f(np.nanmean(a[:, c])), p05=_f(np.nanpercentile(a[:, c], 5)),
                                           p95=_f(np.nanpercentile(a[:, c], 95)),
                                           frac_above=_f(np.nanmean(a[:, c] > alpha)) if c < 4 else None)
                                   for c, n in enumerate(names)}))
        res = dict(alpha=alpha, reps=reps, methods=out)
        self._validity_cache[key] = res
        return res

    def status(self) -> dict:
        if not self.available:
            return dict(available=False, reason=f"Demo bundle not found at {self.bundle.name}. Run "
                        "experiments/kaggle_train.py, then scripts/build_demo_bundle.py.")
        data = self.data()
        st = data.stats["threshold"]
        k = np.asarray([u.k for u in data.units["threshold"]])
        man = data.manifest
        return dict(
            available=True, dataset=man["dataset"], arch=man["arch"], seed=man["seed"], size=int(data.prob.shape[-1]),
            n_units=data.n, n_multi=int((k >= 2).sum()), n_small=int(st.has_small.sum()), n_lesions=int(k.sum()),
            small_cut=data.small_cut, tau=DEFAULT_TAU, split=list(SPLIT), pool_metrics=man["pool_metrics"],
            best_val_dice=man["best_val_dice"], epochs=man["epochs"], env=man.get("env", {}),
            methods=[dict(id=m.id, label=m.label, family=m.family, guaranteed=list(m.guaranteed),
                          guarantee=m.guarantee, ours=m.ours) for m in METHODS],
        )


def _f(v) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


_demo: LesionDemo | None = None


def get_demo() -> LesionDemo:
    global _demo
    if _demo is None:
        _demo = LesionDemo()
    return _demo
