"""Uncertainty estimator and hierarchy tests.

The important properties are analytic: a uniform distribution must give exactly
1.0, a one-hot distribution exactly 0.0, and the normalisation must make maps
comparable across class counts. Those are checked here rather than asserted in
prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.uncertainty.estimators import (
    LeastConfidenceEstimator,
    MarginConfidenceEstimator,
    PredictiveEntropyEstimator,
    get_estimator,
)
from app.uncertainty.hierarchical import HierarchicalUncertaintyEngine


def _uniform(n_classes: int, size: int = 8) -> np.ndarray:
    return np.full((n_classes, size, size), 1.0 / n_classes)


def _one_hot(n_classes: int, size: int = 8) -> np.ndarray:
    probs = np.zeros((n_classes, size, size))
    probs[0] = 1.0
    return probs


@pytest.mark.parametrize("n_classes", [2, 3, 5, 21])
def test_uniform_distribution_is_maximally_uncertain(n_classes):
    result = PredictiveEntropyEstimator().estimate(_uniform(n_classes))
    np.testing.assert_allclose(result.values, 1.0, atol=1e-9)


@pytest.mark.parametrize("n_classes", [2, 3, 5, 21])
def test_one_hot_distribution_is_certain(n_classes):
    result = PredictiveEntropyEstimator().estimate(_one_hot(n_classes))
    np.testing.assert_allclose(result.values, 0.0, atol=1e-6)


def test_entropy_matches_hand_computed_value():
    # p = [0.5, 0.25, 0.25] -> H = 1.5 nats/bits(base e: 1.0397), /log(3)=0.9464
    probs = np.array([0.5, 0.25, 0.25]).reshape(3, 1, 1)
    expected = -(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25)) / np.log(3)
    result = PredictiveEntropyEstimator().estimate(probs)
    assert result.values[0, 0] == pytest.approx(expected, abs=1e-9)


def test_normalisation_makes_class_counts_comparable():
    """A uniform 3-class and a uniform 21-class map must both read 1.0.

    This is the property that lets one colour scale serve every adapter.
    """
    three = PredictiveEntropyEstimator().estimate(_uniform(3)).values.mean()
    twentyone = PredictiveEntropyEstimator().estimate(_uniform(21)).values.mean()
    assert three == pytest.approx(twentyone, abs=1e-9)


def test_margin_and_least_confidence_bounds():
    for estimator in (MarginConfidenceEstimator(), LeastConfidenceEstimator()):
        certain = estimator.estimate(_one_hot(4)).values
        uncertain = estimator.estimate(_uniform(4)).values
        np.testing.assert_allclose(certain, 0.0, atol=1e-6)
        np.testing.assert_allclose(uncertain, 1.0, atol=1e-6)


def test_estimator_rejects_bad_shapes():
    with pytest.raises(ValueError):
        PredictiveEntropyEstimator().estimate(np.ones((8, 8)))
    with pytest.raises(ValueError):
        PredictiveEntropyEstimator().estimate(np.ones((1, 8, 8)))


def test_unnormalised_input_is_renormalised():
    """Adapters may hand us near-normalised floats; the estimator must cope."""
    probs = _uniform(4) * 3.0
    result = PredictiveEntropyEstimator().estimate(probs)
    np.testing.assert_allclose(result.values, 1.0, atol=1e-9)


def test_uncertainty_derives_from_real_probabilities(segmenter, preprocessed):
    """Guards the core integrity rule: the map is a function of the posteriors.

    Recomputing entropy independently from the returned probabilities must
    reproduce the map exactly — impossible if anything synthetic were injected.
    """
    output = segmenter.predict_segmentation(preprocessed)
    probs = np.clip(output.probabilities, 1e-12, 1.0)
    probs /= probs.sum(axis=0, keepdims=True)
    expected = -(probs * np.log(probs)).sum(axis=0) / np.log(probs.shape[0])

    actual = get_estimator().estimate(output.probabilities).values
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_hierarchy_levels_are_consistent(settings, segmenter, preprocessed):
    output = segmenter.predict_segmentation(preprocessed)
    uncertainty = get_estimator().estimate(output.probabilities)
    result = HierarchicalUncertaintyEngine(settings).build(
        uncertainty,
        output.probabilities,
        output.labels,
        output.class_labels,
        threshold=0.5,
    )
    payload = result.payload

    # Level 1 must be the mean of Level 3.
    assert payload.global_.score == pytest.approx(result.uncertainty_map.mean(), abs=1e-9)
    # The reported high-uncertainty fraction must match the mask.
    assert payload.global_.high_uncertainty_fraction == pytest.approx(
        (result.uncertainty_map >= 0.5).mean(), abs=1e-9
    )
    # Every Level 2 region must lie inside the image and above the threshold.
    for obj in payload.objects:
        assert 0 <= obj.bbox.x and obj.bbox.x + obj.bbox.width <= payload.pixel.width
        assert 0 <= obj.bbox.y and obj.bbox.y + obj.bbox.height <= payload.pixel.height
        assert obj.peak_uncertainty >= obj.uncertainty
        if obj.source == "high_uncertainty_region":
            assert obj.uncertainty >= 0.5
    assert payload.pixel.height == preprocessed.gray.shape[0]
    assert len(payload.pixel.preview_grid) == 32


def test_boundary_uncertainty_exceeds_interior(settings, segmenter, preprocessed):
    """A sanity property of a working uncertainty map on real output.

    Class boundaries are where a dense model is genuinely ambiguous, so mean
    uncertainty on boundary pixels should exceed the interior. If this ever
    fails, the map is not tracking model behaviour.
    """
    output = segmenter.predict_segmentation(preprocessed)
    uncertainty = get_estimator().estimate(output.probabilities)
    payload = HierarchicalUncertaintyEngine(settings).build(
        uncertainty, output.probabilities, output.labels, output.class_labels
    ).payload

    assert payload.global_.boundary_uncertainty > payload.global_.interior_uncertainty
