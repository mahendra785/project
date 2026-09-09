"""Evaluation metric tests with analytically known answers."""

from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import (
    area_under_risk_coverage,
    dice_coefficient,
    iou_score,
    multiclass_dice,
    risk_coverage_from_uncertainty,
    selective_risk,
    uncertainty_error_correlation,
)


def test_dice_identical_masks():
    mask = np.array([[1, 1], [0, 0]], dtype=bool)
    assert dice_coefficient(mask, mask) == 1.0


def test_dice_disjoint_masks():
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[0, 1], [0, 0]], dtype=bool)
    assert dice_coefficient(a, b) == 0.0


def test_dice_partial_overlap_hand_computed():
    a = np.zeros((4, 4), dtype=bool)
    a[:2, :] = True  # 8 px
    b = np.zeros((4, 4), dtype=bool)
    b[1:3, :] = True  # 8 px, 4 overlapping
    assert dice_coefficient(a, b) == pytest.approx(2 * 4 / 16)
    assert iou_score(a, b) == pytest.approx(4 / 12)


def test_empty_masks_are_a_perfect_match():
    empty = np.zeros((4, 4), dtype=bool)
    assert dice_coefficient(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0


def test_dice_and_iou_agree_on_the_algebraic_identity():
    """iou = dice / (2 - dice) must hold for any pair of masks."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.random((16, 16)) > 0.5
        b = rng.random((16, 16)) > 0.5
        dice = dice_coefficient(a, b)
        assert iou_score(a, b) == pytest.approx(dice / (2 - dice), abs=1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice_coefficient(np.zeros((4, 4)), np.zeros((5, 5)))


def test_multiclass_dice_skips_background_by_default():
    prediction = np.array([[0, 1], [2, 2]])
    scores = multiclass_dice(prediction, prediction, n_classes=3)
    assert set(scores) == {1, 2}
    assert all(value == 1.0 for value in scores.values())


def test_selective_risk_rewards_informative_uncertainty():
    uncertainty = np.linspace(0.0, 1.0, 100)
    error = (uncertainty > 0.5).astype(float)
    assert selective_risk(error, uncertainty, coverage=0.5) == 0.0
    assert selective_risk(error, uncertainty, coverage=1.0) == pytest.approx(0.5, abs=0.01)


def test_selective_risk_rejects_bad_coverage():
    with pytest.raises(ValueError):
        selective_risk(np.zeros(10), np.zeros(10), coverage=0.0)


def test_uninformative_uncertainty_gives_flat_risk_curve():
    rng = np.random.default_rng(0)
    error = (rng.random(2000) < 0.3).astype(float)
    uncertainty = rng.random(2000)  # independent of error
    curve = risk_coverage_from_uncertainty(error, uncertainty, n_points=10)
    risks = np.array([r for _, r in curve])
    assert risks.std() < 0.06, "random uncertainty should not sort errors"


def test_aurc_is_lower_for_better_uncertainty():
    rng = np.random.default_rng(1)
    error = (rng.random(2000) < 0.3).astype(float)
    informative = error + rng.normal(0, 0.05, error.size)
    uninformative = rng.random(error.size)
    assert area_under_risk_coverage(error, informative) < area_under_risk_coverage(
        error, uninformative
    )


def test_correlation_extremes():
    values = np.linspace(0, 1, 100)
    assert uncertainty_error_correlation(values, values) == pytest.approx(1.0)
    assert uncertainty_error_correlation(values, -values) == pytest.approx(-1.0)
    assert uncertainty_error_correlation(np.ones(100), values) == 0.0


def test_service_self_tests_all_pass():
    from app.evaluation.service import EvaluationService

    results = EvaluationService.self_test()
    assert results
    for result in results:
        assert result.passed, f"{result.metric}: {result.scenario}"
