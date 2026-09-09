"""Risk assessment — the decision-support layer.

DESIGN RULE: every number that moves the risk score is a named `RiskFactor`
carrying its raw value, its normalisation, its weight and its contribution.
There are no unexplained constants inside the scoring function, and the full
rule set is echoed in the response so any result can be recomputed by hand.

THIS IS NOT A DIAGNOSIS. The output is a research decision-support signal about
*the model's own reliability* on this input — never about the patient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.schemas.calibration import CalibrationResult
from app.schemas.common import ReferralPriority, RiskLevel, ShiftStatus
from app.schemas.risk import RiskAssessment, RiskFactor
from app.schemas.shift import DistributionShiftResult
from app.schemas.uncertainty import HierarchicalUncertainty


@dataclass(slots=True)
class RiskInputs:
    uncertainty: HierarchicalUncertainty
    confidence: float
    calibration: CalibrationResult | None
    shift: DistributionShiftResult | None


class RiskAssessmentService:
    """Combines confidence, uncertainty, calibration and shift into a band."""

    #: Fraction of the image above the uncertainty threshold that, on its own,
    #: forces review regardless of the weighted score.
    OVERRIDE_HIGH_UNCERTAINTY_AREA = 0.35
    #: A severe input-domain shift invalidates the conformal guarantee, so the
    #: calibrated numbers can no longer be trusted — that alone forces review.
    OVERRIDE_ON_SEVERE_SHIFT = True
    #: Any single region this uncertain forces review even if the image-level
    #: average looks benign (small lesions must not be averaged away).
    OVERRIDE_REGION_UNCERTAINTY = 0.85

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def assess(self, inputs: RiskInputs) -> RiskAssessment:
        weights = self.settings.risk_weights
        thresholds = self.settings.risk_thresholds
        factors = self._factors(inputs, weights)

        score = float(np.clip(sum(f.contribution for f in factors), 0.0, 1.0))
        level = self._band(score, thresholds)

        overrides, level = self._apply_overrides(inputs, level)
        reasons = self._reasons(inputs, factors, overrides)

        return RiskAssessment(
            risk_score=round(score, 4),
            risk_level=level,
            referral_priority=self._priority(level),
            review_recommended=level
            in (RiskLevel.HIGH, RiskLevel.REVIEW_REQUIRED),
            reasons=reasons,
            factors=factors,
            weights=weights,
            thresholds=thresholds,
            triggered_overrides=overrides,
        )

    # ------------------------------------------------------------ factors --
    def _factors(self, inputs: RiskInputs, weights: dict[str, float]) -> list[RiskFactor]:
        global_uncertainty = inputs.uncertainty.global_
        factors: list[RiskFactor] = []

        # 1. Predictive uncertainty. Already in [0, 1], used directly.
        factors.append(
            self._factor(
                "uncertainty",
                "Mean predictive uncertainty",
                raw=global_uncertainty.score,
                normalised=float(np.clip(global_uncertainty.score, 0.0, 1.0)),
                weight=weights["uncertainty"],
                rationale=(
                    "Mean normalised predictive entropy over all pixels. Higher "
                    "means the model's own distribution is closer to uniform."
                ),
            )
        )

        # 2. Low confidence. A 2-class model cannot go below 0.5, so we rescale
        #    from the achievable range [0.5, 1] rather than [0, 1] — otherwise
        #    a perfectly confident binary model would still score 0.5 here.
        confidence = float(np.clip(inputs.confidence, 0.0, 1.0))
        normalised_confidence = float(np.clip((1.0 - confidence) * 2.0, 0.0, 1.0))
        factors.append(
            self._factor(
                "confidence",
                "Low model confidence",
                raw=confidence,
                normalised=normalised_confidence,
                weight=weights["confidence"],
                rationale=(
                    "Mean max-class probability, rescaled from the achievable "
                    "range [0.5, 1.0] so a confident binary decision scores 0."
                ),
            )
        )

        # 3. Distribution shift.
        shift_score = inputs.shift.shift_score if inputs.shift else 0.0
        factors.append(
            self._factor(
                "distribution_shift",
                "Input-domain shift",
                raw=shift_score,
                normalised=float(np.clip(shift_score, 0.0, 1.0)),
                weight=weights["distribution_shift"],
                rationale=(
                    "Composite intensity-domain shift score. Shift breaks the "
                    "exchangeability the calibration guarantee depends on."
                    if inputs.shift
                    else "Shift detection was disabled for this analysis; treated as 0."
                ),
            )
        )

        # 4. Calibration quality. ECE of 0.2+ is severe miscalibration, so we
        #    saturate the factor there rather than at the theoretical max of 1.
        ece = (
            inputs.calibration.expected_calibration_error if inputs.calibration else 0.0
        )
        factors.append(
            self._factor(
                "calibration",
                "Calibration error",
                raw=ece,
                normalised=float(np.clip(ece / 0.20, 0.0, 1.0)),
                weight=weights["calibration"],
                rationale=(
                    "Expected Calibration Error on the calibration set, saturating "
                    "at ECE=0.20. Measured on synthetic phantoms unless a real "
                    "calibration split is registered."
                    if inputs.calibration
                    else "Calibration was not applied for this analysis; treated as 0."
                ),
            )
        )
        return factors

    @staticmethod
    def _factor(
        factor_id: str,
        name: str,
        *,
        raw: float,
        normalised: float,
        weight: float,
        rationale: str,
    ) -> RiskFactor:
        return RiskFactor(
            id=factor_id,
            name=name,
            raw_value=round(float(raw), 4),
            normalised=round(float(normalised), 4),
            weight=weight,
            contribution=round(float(normalised * weight), 4),
            rationale=rationale,
        )

    # ---------------------------------------------------------- overrides --
    def _apply_overrides(
        self, inputs: RiskInputs, level: RiskLevel
    ) -> tuple[list[str], RiskLevel]:
        overrides: list[str] = []

        area = inputs.uncertainty.global_.high_uncertainty_fraction
        if area >= self.OVERRIDE_HIGH_UNCERTAINTY_AREA:
            overrides.append(
                f"high_uncertainty_area>={self.OVERRIDE_HIGH_UNCERTAINTY_AREA:.2f} "
                f"(observed {area:.2f})"
            )

        if (
            self.OVERRIDE_ON_SEVERE_SHIFT
            and inputs.shift
            and inputs.shift.status is ShiftStatus.SEVERE_SHIFT
        ):
            overrides.append("severe_distribution_shift")

        peak = max(
            (obj.peak_uncertainty for obj in inputs.uncertainty.objects), default=0.0
        )
        if peak >= self.OVERRIDE_REGION_UNCERTAINTY:
            overrides.append(
                f"region_peak_uncertainty>={self.OVERRIDE_REGION_UNCERTAINTY:.2f} "
                f"(observed {peak:.2f})"
            )

        if overrides:
            return overrides, RiskLevel.REVIEW_REQUIRED
        return overrides, level

    # ------------------------------------------------------------ banding --
    @staticmethod
    def _band(score: float, thresholds: dict[str, float]) -> RiskLevel:
        if score >= thresholds["high"]:
            return RiskLevel.REVIEW_REQUIRED
        if score >= thresholds["moderate"]:
            return RiskLevel.HIGH
        if score >= thresholds["low"]:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    @staticmethod
    def _priority(level: RiskLevel) -> ReferralPriority:
        if level is RiskLevel.REVIEW_REQUIRED:
            return ReferralPriority.URGENT_REVIEW
        if level is RiskLevel.HIGH:
            return ReferralPriority.ELEVATED
        return ReferralPriority.ROUTINE

    # ------------------------------------------------------------ reasons --
    def _reasons(
        self,
        inputs: RiskInputs,
        factors: list[RiskFactor],
        overrides: list[str],
    ) -> list[str]:
        reasons: list[str] = []

        dominant = max(factors, key=lambda f: f.contribution)
        if dominant.contribution > 0:
            reasons.append(
                f"Largest contribution: {dominant.name.lower()} "
                f"({dominant.contribution:.3f} of the total score)."
            )

        n_regions = len(inputs.uncertainty.objects)
        if n_regions:
            reasons.append(
                f"{n_regions} localised high-uncertainty region(s) identified for "
                "clinician review."
            )
        else:
            reasons.append(
                "No connected region exceeded the high-uncertainty threshold."
            )

        if inputs.shift and inputs.shift.status is not ShiftStatus.ID:
            reasons.append(
                f"Input flagged as {inputs.shift.status.value.replace('_', ' ').lower()} "
                f"(score {inputs.shift.shift_score:.2f}) relative to reference domain "
                f"'{inputs.shift.reference_domain}'."
            )

        if inputs.calibration and inputs.calibration.validity_warnings:
            reasons.append(
                "Calibration validity warning: "
                + inputs.calibration.validity_warnings[0]
            )

        for override in overrides:
            reasons.append(f"Override triggered: {override}.")

        reasons.append(
            "Research decision-support signal about model reliability — not a "
            "diagnosis and not clinically validated."
        )
        return reasons
