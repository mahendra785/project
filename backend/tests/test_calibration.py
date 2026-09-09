"""Calibration tests, including the finite-sample conformal guarantee itself."""

from __future__ import annotations

import numpy as np
import pytest

from app.calibration.conformal import (
    LAMBDA_GRID_SIZE,
    BaselineThresholdCalibration,
    ConformalRiskControl,
    _loss_curve,
)
from app.calibration.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
    risk_coverage_curve,
)
from app.calibration.registry import get_method, implemented_ids, list_methods
from app.calibration.temperature import TemperatureScaling, fit_temperature
from app.schemas.common import ImplementationStatus


# ------------------------------------------------------------------ metrics --
def test_ece_is_zero_for_perfect_calibration():
    """A model that is right exactly `confidence` of the time has ECE 0."""
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.5, 1.0, size=200_000)
    correct = (rng.uniform(size=confidence.size) < confidence).astype(float)
    assert expected_calibration_error(confidence, correct, n_bins=15) < 0.01


def test_ece_detects_total_overconfidence():
    """Always claims 1.0, always wrong -> ECE must be 1.0."""
    confidence = np.ones(1000)
    correct = np.zeros(1000)
    assert expected_calibration_error(confidence, correct, n_bins=15) == pytest.approx(1.0)


def test_brier_score_bounds():
    assert brier_score(np.ones(10), np.ones(10)) == 0.0
    assert brier_score(np.ones(10), np.zeros(10)) == 1.0


def test_reliability_bins_partition_all_samples():
    rng = np.random.default_rng(1)
    confidence = rng.uniform(0, 1, size=5000)
    correct = rng.integers(0, 2, size=5000).astype(float)
    bins = reliability_bins(confidence, correct, n_bins=15)
    assert sum(b.count for b in bins) == confidence.size


def test_risk_coverage_is_monotone_for_informative_confidence():
    """With confidence that perfectly ranks errors, risk must fall as coverage does."""
    confidence = np.linspace(0.0, 1.0, 1000)
    correct = (confidence > 0.3).astype(float)
    curve = risk_coverage_curve(confidence, correct, n_points=15)
    risks = [point.risk for point in curve]
    assert risks == sorted(risks), "risk should not decrease as coverage grows"


# --------------------------------------------------------------- conformal --
def test_loss_curve_is_monotone_non_increasing(calibration_set):
    """The CRC theorem requires a monotone loss; verify ours actually is."""
    lambdas = np.linspace(0.0, 1.0, LAMBDA_GRID_SIZE)
    losses = _loss_curve(calibration_set, lambdas)
    assert np.all(np.diff(losses) <= 1e-12)


def test_conformal_selects_smallest_feasible_lambda(calibration_set):
    method = ConformalRiskControl()
    result = method.calibrate(calibration_set, target_risk=0.2, n_bins=10)

    lambdas = np.linspace(0.0, 1.0, LAMBDA_GRID_SIZE)
    losses = _loss_curve(calibration_set, lambdas)
    n = len(calibration_set)
    corrected = (n * losses + 1.0) / (n + 1)

    index = int(np.argmin(np.abs(lambdas - result.threshold)))
    if not result.validity_warnings:
        assert corrected[index] <= 0.2 + 1e-12
        if index > 0:
            assert corrected[index - 1] > 0.2, "a smaller lambda was also feasible"


def test_conformal_threshold_is_monotone_in_alpha(calibration_set):
    """Looser risk targets must never require a *larger* prediction set."""
    method = ConformalRiskControl()
    thresholds = [
        method.calibrate(calibration_set, target_risk=alpha, n_bins=10).threshold
        for alpha in (0.05, 0.15, 0.30, 0.45)
    ]
    assert thresholds == sorted(thresholds, reverse=True)


def test_finite_sample_correction_is_conservative(calibration_set):
    """CRC must never pick a smaller lambda than the uncorrected baseline.

    This is the whole point of the `+B/(n+1)` term, and it is the difference
    between the two methods PUMS ships.
    """
    conformal = ConformalRiskControl().calibrate(
        calibration_set, target_risk=0.25, n_bins=10
    )
    baseline = BaselineThresholdCalibration().calibrate(
        calibration_set, target_risk=0.25, n_bins=10
    )
    assert conformal.threshold >= baseline.threshold


def test_conformal_reports_when_target_is_unreachable(calibration_set):
    """An impossible alpha must produce a warning, not a silent bad threshold."""
    result = ConformalRiskControl().calibrate(
        calibration_set, target_risk=0.001, n_bins=10
    )
    if result.empirical_risk > 0.001:
        assert result.validity_warnings, "unreachable target must be flagged"


def test_calibration_result_carries_assumptions(calibration_set):
    result = ConformalRiskControl().calibrate(calibration_set, target_risk=0.1, n_bins=10)
    joined = " ".join(result.assumptions).lower()
    assert "exchangeab" in joined
    assert result.provenance.value == "synthetic"


def test_apply_produces_a_mask_of_the_right_shape(calibration_set):
    method = ConformalRiskControl()
    result = method.calibrate(calibration_set, target_risk=0.2, n_bins=10)
    probability = calibration_set.samples[0].foreground_probability
    applied = method.apply(result, probability)

    assert applied.abstain_mask.shape == probability.shape
    assert 0.0 <= applied.abstention_rate <= 1.0


# ------------------------------------------------------------- temperature --
def test_fit_temperature_recovers_a_known_scaling():
    """Generate perfectly calibrated data, over-sharpen it, recover T."""
    rng = np.random.default_rng(3)
    p_true = rng.uniform(0.02, 0.98, size=60_000)
    y = (rng.uniform(size=p_true.size) < p_true).astype(float)

    true_t = 0.5  # over-confident: logits divided by 0.5 == doubled
    logits = np.log(p_true / (1 - p_true))
    p_sharp = 1.0 / (1.0 + np.exp(-logits / true_t))

    recovered = fit_temperature(p_sharp, y)
    # Recovering T means *undoing* the sharpening, i.e. T ~= 1/true_t.
    assert recovered == pytest.approx(1.0 / true_t, rel=0.2)


def test_temperature_scaling_reduces_ece(calibration_set):
    result = TemperatureScaling().calibrate(
        calibration_set, target_risk=0.1, n_bins=10
    )
    baseline = ConformalRiskControl().calibrate(
        calibration_set, target_risk=0.1, n_bins=10
    )
    assert result.expected_calibration_error <= baseline.expected_calibration_error + 1e-9


def test_temperature_scaling_cannot_change_decisions(calibration_set):
    """Monotone rescaling must leave the thresholded mask untouched."""
    method = TemperatureScaling()
    result = method.calibrate(calibration_set, target_risk=0.1, n_bins=10)
    probability = calibration_set.samples[0].foreground_probability

    from app.calibration.temperature import _logit, _sigmoid

    scaled = _sigmoid(_logit(probability) / result.threshold)
    np.testing.assert_array_equal(scaled >= 0.5, probability >= 0.5)


# ---------------------------------------------------------------- registry --
def test_planned_methods_refuse_to_run(calibration_set):
    method = get_method("spatially_aware_conformal")
    assert method.status is ImplementationStatus.PLANNED
    with pytest.raises(NotImplementedError):
        method.calibrate(calibration_set, target_risk=0.1, n_bins=10)


def test_registry_separates_implemented_from_planned():
    implemented = set(implemented_ids())
    assert "conformal_risk_control" in implemented
    assert "temperature_scaling" in implemented
    assert "adaptive_conformal_risk_control" not in implemented
    assert len(list_methods()) > len(implemented)
