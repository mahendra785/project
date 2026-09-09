"""Distribution-shift and risk-engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.preprocessing.pipeline import PreprocessedImage
from app.risk.engine import RiskAssessmentService, RiskInputs
from app.schemas.common import RiskLevel, ShiftStatus
from app.shift.detector import CompositeIntensityShiftDetector
from app.shift.reference import build_profile_from_images, get_reference_profile


def _wrap(gray: np.ndarray) -> PreprocessedImage:
    from app.preprocessing.pipeline import PreprocessingPipeline
    from app.core.config import get_settings

    pipeline = PreprocessingPipeline(get_settings())
    stats, histogram = pipeline._summarise(gray)
    return PreprocessedImage(
        gray=gray,
        rgb=np.dstack([gray] * 3),
        display=gray,
        original_height=gray.shape[0],
        original_width=gray.shape[1],
        original_channels=1,
        working_size=gray.shape[0],
        image_format="TEST",
        is_dicom=False,
        checksum="test",
        size_bytes=0,
        statistics=stats,
        histogram=histogram,
    )


# ------------------------------------------------------------------- shift --
def test_reference_image_scores_as_in_distribution(settings):
    from app.calibration.datasets import _make_phantom

    reference = get_reference_profile()
    image, _ = _make_phantom(np.random.default_rng(99), 128)
    result = CompositeIntensityShiftDetector().detect(
        _wrap(image), reference, thresholds=settings.shift_thresholds
    )
    assert result.status is ShiftStatus.ID
    assert result.shift_score < settings.shift_threshold_mild


def test_severe_shift_is_detected(settings):
    """An inverted, heavily-noised image must not be called in-distribution."""
    from app.calibration.datasets import _make_phantom

    rng = np.random.default_rng(5)
    base, _ = _make_phantom(rng, 128)
    corrupted = np.clip(1.0 - base + rng.normal(0, 0.2, base.shape), 0, 1)

    result = CompositeIntensityShiftDetector().detect(
        _wrap(corrupted), get_reference_profile(), thresholds=settings.shift_thresholds
    )
    assert result.status in (ShiftStatus.MODERATE_SHIFT, ShiftStatus.SEVERE_SHIFT)
    assert result.shift_score > settings.shift_threshold_moderate


def test_shift_score_increases_with_corruption_strength(settings):
    from app.calibration.datasets import _make_phantom

    rng = np.random.default_rng(11)
    base, _ = _make_phantom(rng, 128)
    reference = get_reference_profile()
    detector = CompositeIntensityShiftDetector()

    scores = [
        detector.detect(
            _wrap(np.clip(base * (1 - strength) + strength * (1 - base), 0, 1)),
            reference,
            thresholds=settings.shift_thresholds,
        ).shift_score
        for strength in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert scores[-1] > scores[0], "full inversion must score above no corruption"


def test_indicator_weights_sum_to_one(settings):
    from app.calibration.datasets import _make_phantom

    image, _ = _make_phantom(np.random.default_rng(2), 128)
    result = CompositeIntensityShiftDetector().detect(
        _wrap(image), get_reference_profile(), thresholds=settings.shift_thresholds
    )
    assert sum(i.weight for i in result.indicators) == pytest.approx(1.0)
    for indicator in result.indicators:
        assert 0.0 <= indicator.normalised <= 1.0
    # The composite must equal the weighted sum it claims to be.
    assert result.shift_score == pytest.approx(
        min(sum(i.normalised * i.weight for i in result.indicators), 1.0), abs=1e-6
    )


def test_custom_reference_profile_can_be_built():
    images = [np.full((32, 32), 0.5) for _ in range(4)]
    profile = build_profile_from_images(images, name="flat", description="test")
    assert profile.n_samples == 4
    assert profile.mean == pytest.approx(0.5)


# -------------------------------------------------------------------- risk --
class _Global:
    def __init__(self, score, fraction):
        self.score = score
        self.high_uncertainty_fraction = fraction


class _Uncertainty:
    def __init__(self, score, fraction=0.05, objects=()):
        self.global_ = _Global(score, fraction)
        self.objects = list(objects)


def test_low_uncertainty_confident_input_is_low_risk(settings):
    service = RiskAssessmentService(settings)
    result = service.assess(
        RiskInputs(
            uncertainty=_Uncertainty(0.05, 0.01),
            confidence=0.98,
            calibration=None,
            shift=None,
        )
    )
    assert result.risk_level is RiskLevel.LOW
    assert not result.review_recommended


def test_high_uncertainty_area_forces_review(settings):
    service = RiskAssessmentService(settings)
    result = service.assess(
        RiskInputs(
            uncertainty=_Uncertainty(0.10, fraction=0.9),
            confidence=0.99,
            calibration=None,
            shift=None,
        )
    )
    assert result.risk_level is RiskLevel.REVIEW_REQUIRED
    assert any("high_uncertainty_area" in o for o in result.triggered_overrides)


def test_risk_score_equals_sum_of_declared_contributions(settings):
    """No hidden terms: the score must be exactly what the factors say it is."""
    service = RiskAssessmentService(settings)
    result = service.assess(
        RiskInputs(
            uncertainty=_Uncertainty(0.4, 0.1),
            confidence=0.7,
            calibration=None,
            shift=None,
        )
    )
    assert result.risk_score == pytest.approx(
        sum(f.contribution for f in result.factors), abs=1e-3
    )
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_every_factor_explains_itself(settings):
    service = RiskAssessmentService(settings)
    result = service.assess(
        RiskInputs(
            uncertainty=_Uncertainty(0.3, 0.1), confidence=0.8, calibration=None, shift=None
        )
    )
    assert len(result.factors) == 4
    for factor in result.factors:
        assert factor.rationale
        assert 0.0 <= factor.normalised <= 1.0
        assert factor.contribution == pytest.approx(factor.normalised * factor.weight, abs=1e-4)


def test_disclaimer_is_always_present(settings):
    result = RiskAssessmentService(settings).assess(
        RiskInputs(
            uncertainty=_Uncertainty(0.1, 0.0), confidence=0.9, calibration=None, shift=None
        )
    )
    assert "not a diagnostic result" in result.disclaimer.lower()
    assert any("not a diagnosis" in r.lower() for r in result.reasons)
