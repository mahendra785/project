"""Calibration metrics: ECE, Brier, reliability bins, risk-coverage curves.

All operate on flat arrays of (confidence, correctness) so they are reusable
for pixels, objects, or whole images.
"""

from __future__ import annotations

import numpy as np

from app.schemas.calibration import ReliabilityBin, RiskCoveragePoint


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> list[ReliabilityBin]:
    """Equal-width binning of confidences with per-bin empirical accuracy."""
    conf = np.clip(np.asarray(confidence, dtype=np.float64).ravel(), 0.0, 1.0)
    corr = np.asarray(correct, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        # Half-open bins, closed at the top for the final bin.
        in_bin = (conf > lower) & (conf <= upper) if i > 0 else (conf >= lower) & (conf <= upper)
        count = int(in_bin.sum())
        bins.append(
            ReliabilityBin(
                bin_lower=float(lower),
                bin_upper=float(upper),
                count=count,
                mean_confidence=float(conf[in_bin].mean()) if count else 0.0,
                empirical_accuracy=float(corr[in_bin].mean()) if count else 0.0,
            )
        )
    return bins


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    r"""Standard equal-width ECE (Naeini et al. 2015; Guo et al. 2017).

    .. math:: \mathrm{ECE} = \sum_b \frac{|B_b|}{n}\,
              \bigl|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\bigr|
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    total = conf.size
    if total == 0:
        return 0.0
    error = 0.0
    for bin_ in reliability_bins(confidence, correct, n_bins):
        if bin_.count == 0:
            continue
        error += (bin_.count / total) * abs(
            bin_.empirical_accuracy - bin_.mean_confidence
        )
    return float(error)


def brier_score(probability: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error between a predicted probability and a binary outcome."""
    p = np.asarray(probability, dtype=np.float64).ravel()
    y = np.asarray(outcome, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def risk_coverage_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_points: int = 21,
) -> list[RiskCoveragePoint]:
    """Selective risk as a function of coverage (El-Yaniv & Wiener, 2010).

    At each confidence threshold we *accept* the predictions above it; coverage
    is the accepted fraction and risk is the error rate among accepted
    predictions. A useful uncertainty estimator makes risk fall monotonically
    as coverage shrinks — this curve is how that claim gets tested rather than
    asserted.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()
    if conf.size == 0:
        return []

    thresholds = np.quantile(conf, np.linspace(0.0, 0.98, n_points))
    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        accepted = conf >= threshold
        coverage = float(accepted.mean())
        if coverage <= 0:
            continue
        points.append(
            RiskCoveragePoint(
                coverage=coverage,
                risk=float(1.0 - corr[accepted].mean()),
                threshold=float(threshold),
            )
        )
    # De-duplicate identical coverages produced by tied quantiles.
    seen: set[float] = set()
    unique: list[RiskCoveragePoint] = []
    for point in points:
        key = round(point.coverage, 4)
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return sorted(unique, key=lambda p: p.coverage)
