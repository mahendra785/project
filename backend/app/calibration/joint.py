r"""Multi-granularity risk control: selection rules over a lambda grid.

All rules consume :class:`~app.calibration.losses.StackedStats` (per-unit loss
rates on a shared grid) and return a grid index. Guarantees, with :math:`n_r` the
number of calibration units on which risk :math:`r` is defined:

``crc``     single risk, :math:`\mathbb E[L^r_{n+1}(\hat\lambda)]\le\alpha` (CRC, Thm 1).
``union``   :math:`\hat\lambda^\cup=\max_r\hat\lambda^r`; every risk satisfies its own
            marginal bound simultaneously (theory memo, Corollary 1). No Bonferroni
            split is needed because these are expectation bounds, each controlled
            on its own.
``max``     CRC on :math:`\max_r L^r_i`; bounds the expected *per-unit worst* of the
            risks — strictly stronger than ``union``, at a larger :math:`\lambda`.
``ltt``     Learn-Then-Test with Hoeffding–Bentkus p-values and fixed-sequence
            testing from the largest :math:`\lambda` down; intersection-union over
            risks. High-probability: with prob. :math:`\ge 1-\delta` over the
            calibration draw, every population risk is :math:`\le\alpha`.
            (Angelopoulos, Bates, Candès, Jordan, Lei 2021.)
``stratified`` per-stratum ``union`` (the CCRA-S construction of Luo & Zhou 2025,
            extended to the joint risks): a separate :math:`\hat\lambda_g` per stratum
            of a label-free image feature, guarantee within each stratum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sstats

from app.calibration.losses import StackedStats

RISKS = ("pix", "les", "small")
LOSS_BOUND = 1.0


def risk_matrix(st: StackedStats, risk: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit loss matrix [n_r, G] for one risk and the row mask it used."""
    if risk == "pix":
        return st.pix_rate, np.ones(len(st), dtype=bool)
    if risk == "les":
        return st.les_rate, np.ones(len(st), dtype=bool)
    if risk == "small":
        return st.small_rate[st.has_small], st.has_small
    raise ValueError(risk)


def max_form_matrix(st: StackedStats, risks: tuple[str, ...]) -> np.ndarray:
    out = np.zeros_like(st.pix_rate)
    for r in risks:
        if r == "small":
            m = np.where(st.has_small[:, None], st.small_rate, 0.0)
        else:
            m = risk_matrix(st, r)[0]
        out = np.maximum(out, m)
    return out


def crc_index(losses: np.ndarray, alpha: float) -> tuple[int, bool]:
    """First grid index with (n R_n + B)/(n+1) <= alpha. Losses [n, G], non-increasing."""
    n = losses.shape[0]
    if n == 0:
        return 0, True
    corrected = (n * losses.mean(axis=0, dtype=np.float64) + LOSS_BOUND) / (n + 1)
    feasible = np.flatnonzero(corrected <= alpha)
    if feasible.size:
        return int(feasible[0]), True
    return int(losses.shape[1] - 1), False


def union_index(st: StackedStats, alpha: float, risks: tuple[str, ...] = RISKS) -> tuple[int, bool]:
    idx, ok = 0, True
    for r in risks:
        i, f = crc_index(risk_matrix(st, r)[0], alpha)
        idx, ok = max(idx, i), ok and f
    return idx, ok


def max_index(st: StackedStats, alpha: float, risks: tuple[str, ...] = RISKS) -> tuple[int, bool]:
    return crc_index(max_form_matrix(st, risks), alpha)


def hb_pvalue(r_hat: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Hoeffding–Bentkus p-value for H0: R > alpha (Bates et al. 2021, RCPS)."""
    r = np.minimum(r_hat, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = r * np.log(r / alpha) + (1 - r) * np.log((1 - r) / (1 - alpha))
    h1 = np.where(r <= 0, -np.log(1 - alpha), h1)
    hoeffding = np.exp(-n * h1)
    bentkus = np.e * sstats.binom.cdf(np.ceil(n * r_hat), n, alpha)
    return np.minimum(np.minimum(hoeffding, bentkus), 1.0)


def ltt_index(
    st: StackedStats, alpha: float, delta: float = 0.1, risks: tuple[str, ...] = RISKS
) -> tuple[int, bool]:
    """Fixed-sequence LTT from lambda = 1 downward; FWER <= delta."""
    p = None
    for r in risks:
        m = risk_matrix(st, r)[0]
        pr = hb_pvalue(m.mean(axis=0, dtype=np.float64), m.shape[0], alpha)
        p = pr if p is None else np.maximum(p, pr)  # intersection-union
    g = p.size
    idx = g - 1
    if p[idx] > delta:
        return idx, False
    while idx - 1 >= 0 and p[idx - 1] <= delta:
        idx -= 1
    return idx, True


def stratified_indices(
    st: StackedStats,
    strata: np.ndarray,
    alpha: float,
    risks: tuple[str, ...] = RISKS,
) -> dict[int, tuple[int, bool]]:
    """Per-stratum union index (CCRA-S style). ``strata`` gives each unit's stratum."""
    out = {}
    for g in np.unique(strata):
        out[int(g)] = union_index(st.subset(np.flatnonzero(strata == g)), alpha, risks)
    return out


@dataclass(slots=True)
class Evaluation:
    """Test-time behaviour of a chosen threshold (or per-unit thresholds)."""

    risk_pix: float
    risk_les: float
    risk_small: float
    risk_les_multi: float
    set_to_truth: float
    fp_to_truth: float
    lam: float


def evaluate(st: StackedStats, index: np.ndarray | int, lambdas: np.ndarray) -> Evaluation:
    """Risks and set size on ``st`` at a grid index (scalar or one per unit)."""
    n = len(st)
    idx = np.broadcast_to(np.asarray(index), (n,))
    rows = np.arange(n)
    pix = st.pix_rate[rows, idx]
    les = st.les_rate[rows, idx]
    small = st.small_rate[rows, idx][st.has_small]
    size = st.set_size[rows, idx].astype(np.float64)
    covered = st.n_gt * (1.0 - pix)
    total_gt = float(st.n_gt.sum())
    return Evaluation(
        risk_pix=float(pix.mean()),
        risk_les=float(les.mean()),
        risk_small=float(small.mean()) if small.size else float("nan"),
        risk_les_multi=float(les[st.multi].mean()) if st.multi.any() else float("nan"),
        set_to_truth=float(size.sum() / total_gt),
        fp_to_truth=float((size - covered).sum() / total_gt),
        lam=float(np.mean(lambdas[idx])),
    )


def select_family(
    tuning: dict[str, StackedStats], alpha: float, rule: str = "union"
) -> str:
    """Choose the family whose own calibrated set is smallest on the TUNING split.

    The tuning split must be disjoint from calibration and test. This is what
    keeps family selection from voiding the CRC guarantee: conditional on the
    chosen family, the calibration units are still exchangeable with the test
    unit and the score transform is fixed.
    """
    pick = {"union": union_index, "max": max_index}[rule]
    best, best_size = None, np.inf
    for label, st in tuning.items():
        idx, _ = pick(st, alpha)
        size = st.set_size[:, idx].sum() / st.n_gt.sum()
        if size < best_size - 1e-12:
            best, best_size = label, size
    assert best is not None
    return best
