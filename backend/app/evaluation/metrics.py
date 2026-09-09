"""Reusable evaluation metrics.

These are the standard definitions, implemented plainly so they can be checked
against hand-computed values in tests (see `tests/test_evaluation.py` and the
`/api/evaluation/self-test` endpoint).
"""

from __future__ import annotations

import numpy as np


def dice_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    r"""Sørensen-Dice: :math:`2|A \cap B| / (|A| + |B|)`.

    Two empty masks are defined as a perfect match (1.0), the convention used
    by MONAI and the medical segmentation literature — a model correctly
    predicting "nothing here" should not be scored 0.
    """
    a = np.asarray(prediction).astype(bool)
    b = np.asarray(target).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    r"""Jaccard index: :math:`|A \cap B| / |A \cup B|`. Empty/empty = 1.0."""
    a = np.asarray(prediction).astype(bool)
    b = np.asarray(target).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def multiclass_dice(
    prediction: np.ndarray, target: np.ndarray, n_classes: int, ignore_background: bool = True
) -> dict[int, float]:
    """Per-class Dice for an integer label map."""
    start = 1 if ignore_background else 0
    return {
        class_id: dice_coefficient(prediction == class_id, target == class_id)
        for class_id in range(start, n_classes)
    }


def uncertainty_error_correlation(
    uncertainty: np.ndarray, error: np.ndarray
) -> float:
    r"""Point-biserial correlation between uncertainty and per-pixel error.

    This is the single most important sanity check on an uncertainty method: if
    uncertainty does not correlate with where the model is actually wrong, the
    map is decorative. Returns NaN-free 0.0 when either input is constant.
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(error, dtype=np.float64).ravel()
    if u.size != e.size or u.size < 2:
        return 0.0
    if u.std() < 1e-12 or e.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(u, e)[0, 1])


def selective_risk(
    error: np.ndarray, uncertainty: np.ndarray, coverage: float
) -> float:
    """Error rate over the `coverage` fraction of most-confident predictions."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(error, dtype=np.float64).ravel()
    if u.size == 0:
        return 0.0
    n_keep = max(1, int(round(coverage * u.size)))
    keep = np.argsort(u)[:n_keep]
    return float(e[keep].mean())


def risk_coverage_from_uncertainty(
    error: np.ndarray, uncertainty: np.ndarray, n_points: int = 20
) -> list[tuple[float, float]]:
    """(coverage, selective risk) pairs, sweeping coverage from low to full."""
    coverages = np.linspace(0.05, 1.0, n_points)
    return [
        (float(c), selective_risk(error, uncertainty, float(c))) for c in coverages
    ]


def area_under_risk_coverage(
    error: np.ndarray, uncertainty: np.ndarray, n_points: int = 20
) -> float:
    """AURC — lower is better. Summarises the whole selective-prediction curve."""
    points = risk_coverage_from_uncertainty(error, uncertainty, n_points)
    if not points:
        return 0.0
    coverages = np.array([p[0] for p in points])
    risks = np.array([p[1] for p in points])
    return float(np.trapezoid(risks, coverages) / (coverages[-1] - coverages[0]))
