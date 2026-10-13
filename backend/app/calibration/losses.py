r"""Structured, set-monotone segmentation losses on a fixed lambda grid.

For one calibration unit (an image or a volume) with ground truth :math:`Y` and
ground-truth connected components :math:`Y^1..Y^K`, and a score map :math:`s`
defining :math:`S_\lambda=\{s\ge 1-\lambda\}`:

.. math::
   L^{pix}(\lambda) &= 1 - |S_\lambda\cap Y| / |Y| \\
   L^{les}(\lambda) &= \tfrac1K\,\#\{k : |S_\lambda\cap Y^k| < \tau |Y^k|\} \\
   L^{small}(\lambda) &= \tfrac1{K_s}\,\#\{k \in \text{small} : |S_\lambda\cap Y^k| < \tau |Y^k|\}

All three are non-increasing in :math:`\lambda` for any nested family (theory memo,
Lemma 1). Units with :math:`Y=\emptyset` are excluded from all three; units with no
small lesion are excluded from :math:`L^{small}` only — each guarantee is therefore
*conditional on the structure being present*, and the reported :math:`n` is per loss.

Everything is reduced to integer counts on the grid once per unit (Lemma 2: a
component is hit at cutoff :math:`c` iff its :math:`\lceil\tau|Y^k|\rceil`-th largest
score is :math:`\ge c`), so re-calibrating on thousands of random splits costs a
few array sums rather than a pass over pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from app.calibration.families import connectivity

#: Lambda grid. 1001 points => 0.001 resolution; finer than the CRC finite-sample
#: slack 1/(n+1) for every calibration size used in this project.
GRID_SIZE = 1001
#: Default lesion hit fraction (theory memo §3, fixed before any result).
DEFAULT_TAU = 0.5


def lambda_grid(size: int = GRID_SIZE) -> np.ndarray:
    return np.linspace(0.0, 1.0, size)


@dataclass(slots=True)
class UnitStats:
    """Grid-reduced loss ingredients for one unit under one score family."""

    n_pixels: int
    n_gt: int
    #: [G] number of ground-truth pixels outside S_lambda.
    pix_miss: np.ndarray
    #: [G] |S_lambda|.
    set_size: np.ndarray
    #: [K] tau-order statistic of each GT component (Lemma 2).
    lesion_stat: np.ndarray
    #: [K] component sizes in pixels/voxels.
    lesion_size: np.ndarray

    @property
    def k(self) -> int:
        return int(self.lesion_stat.size)


def gt_components(mask: np.ndarray, min_size: int = 0) -> np.ndarray:
    """Label GT components with full connectivity, dropping those below min_size."""
    labels, count = ndimage.label(mask, structure=connectivity(mask.ndim))
    if count == 0 or min_size <= 0:
        return labels
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    keep = np.flatnonzero(sizes >= min_size) + 1
    filtered = np.where(np.isin(labels, keep), labels, 0)
    return ndimage.label(filtered > 0, structure=connectivity(mask.ndim))[0]


def unit_stats(
    score: np.ndarray,
    labels: np.ndarray,
    *,
    tau: float = DEFAULT_TAU,
    lambdas: np.ndarray | None = None,
) -> UnitStats:
    """Reduce one unit's score map + labelled GT to grid counts."""
    lambdas = lambda_grid() if lambdas is None else lambdas
    cutoffs = 1.0 - lambdas
    flat = score.ravel()
    lab = labels.ravel()
    truth = lab > 0

    all_sorted = np.sort(flat)
    set_size = flat.size - np.searchsorted(all_sorted, cutoffs, side="left")
    gt_sorted = np.sort(flat[truth])
    pix_miss = np.searchsorted(gt_sorted, cutoffs, side="left")

    k = int(lab.max()) if truth.any() else 0
    stats = np.empty(k)
    sizes = np.empty(k, dtype=np.int64)
    if k:
        order = np.argsort(lab[truth], kind="stable")
        comp_ids = lab[truth][order]
        comp_scores = flat[truth][order]
        bounds = np.searchsorted(comp_ids, np.arange(1, k + 2))
        for j in range(k):
            sc = np.sort(comp_scores[bounds[j] : bounds[j + 1]])[::-1]
            sizes[j] = sc.size
            stats[j] = sc[int(np.ceil(tau * sc.size)) - 1]
    return UnitStats(
        n_pixels=int(flat.size),
        n_gt=int(truth.sum()),
        pix_miss=pix_miss.astype(np.int64),
        set_size=set_size.astype(np.int64),
        lesion_stat=stats,
        lesion_size=sizes,
    )


@dataclass(slots=True)
class StackedStats:
    """UnitStats for many units, stacked for vectorised curve computation.

    Only units with non-empty GT are stacked; empty-GT units contribute set size
    only and are kept in ``empty_set_size``.
    """

    pix_rate: np.ndarray  # [N, G] per-unit pixel FNR
    les_rate: np.ndarray  # [N, G] per-unit lesion-miss rate
    small_rate: np.ndarray  # [N, G] per-unit small-lesion miss rate (nan if none)
    has_small: np.ndarray  # [N] bool
    set_size: np.ndarray  # [N, G]
    n_gt: np.ndarray  # [N]
    n_pixels: np.ndarray  # [N]
    multi: np.ndarray  # [N] bool, K >= 2

    def __len__(self) -> int:
        return int(self.n_gt.size)

    def subset(self, idx: np.ndarray) -> "StackedStats":
        return StackedStats(*(getattr(self, f)[idx] for f in self.__slots__))


def stack(units: list[UnitStats], *, small_max_size: float, lambdas: np.ndarray | None = None) -> StackedStats:
    """Stack non-empty units; ``small_max_size`` is the small-lesion size cut (exclusive)."""
    lambdas = lambda_grid() if lambdas is None else lambdas
    cutoffs = 1.0 - lambdas
    kept = [u for u in units if u.n_gt > 0 and u.k > 0]
    n, g = len(kept), lambdas.size
    pix = np.empty((n, g), dtype=np.float32)
    les = np.empty((n, g), dtype=np.float32)
    small = np.full((n, g), np.nan, dtype=np.float32)
    has_small = np.zeros(n, dtype=bool)
    for i, u in enumerate(kept):
        pix[i] = u.pix_miss / u.n_gt
        st = np.sort(u.lesion_stat)
        les[i] = np.searchsorted(st, cutoffs, side="left") / st.size
        sm = np.sort(u.lesion_stat[u.lesion_size < small_max_size])
        if sm.size:
            has_small[i] = True
            small[i] = np.searchsorted(sm, cutoffs, side="left") / sm.size
    return StackedStats(
        pix_rate=pix,
        les_rate=les,
        small_rate=small,
        has_small=has_small,
        set_size=np.stack([u.set_size for u in kept]) if kept else np.empty((0, g)),
        n_gt=np.asarray([u.n_gt for u in kept]),
        n_pixels=np.asarray([u.n_pixels for u in kept]),
        multi=np.asarray([u.k >= 2 for u in kept]),
    )
