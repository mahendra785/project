"""App-facing joint (pixel + lesion + small-lesion) conformal risk control.

Wraps the research modules (`families`, `losses`, `joint`) behind the same
`BaseCalibrationMethod` interface as the existing CRC, so the API, the method
catalogue and the UI all get the new guarantee without special cases.

Two registrations share this class:

* ``lesion_aware_crc`` — lesion-equalized family (this project's method);
* ``morphological_conformal`` — dilation family (Mossina & Friedrich, MICCAI
  2025), the main structure-aware baseline, with the same joint guarantee.
"""

from __future__ import annotations

import numpy as np

from app.calibration.base import (
    EXCHANGEABILITY,
    MARGINAL_ONLY,
    MONOTONE_LOSS,
    AppliedCalibration,
    BaseCalibrationMethod,
)
from app.calibration.conformal import _pixel_confidence_and_correctness
from app.calibration.datasets import CalibrationSet
from app.calibration.families import ScoreFamily
from app.calibration.joint import evaluate, union_index
from app.calibration.losses import DEFAULT_TAU, gt_components, lambda_grid, stack, unit_stats
from app.calibration.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
    risk_coverage_curve,
)
from app.schemas.calibration import CalibrationResult
from app.schemas.common import ImplementationStatus

LESION_LEVEL = (
    "Lesion-level guarantee: a ground-truth lesion (connected component) counts "
    "as found only if at least tau of its pixels are inside the prediction set; "
    "the expected fraction of missed lesions, and of missed small lesions, is "
    "controlled at alpha simultaneously with the pixel false-negative rate."
)
FAMILY_FROM_TUNING = (
    "Family chosen on a tuning split disjoint from calibration; choosing it on "
    "the calibration split would void the guarantee."
)


class StructuredConformalRiskControl(BaseCalibrationMethod):
    """Union-threshold CRC over pixel FNR, lesion miss and small-lesion miss."""

    status = ImplementationStatus.IMPLEMENTED

    def __init__(
        self,
        method_id: str,
        name: str,
        description: str,
        family: ScoreFamily,
        *,
        reference: str | None,
        tau: float = DEFAULT_TAU,
        small_max_px: float = 64.0,
        status: ImplementationStatus = ImplementationStatus.IMPLEMENTED,
    ) -> None:
        self.id = method_id
        self.name = name
        self.description = description
        self.family = family
        self.reference = reference
        self.tau = tau
        self.small_max_px = small_max_px
        self.status = status
        self.assumptions = [EXCHANGEABILITY, MONOTONE_LOSS, MARGINAL_ONLY, LESION_LEVEL]
        if family.kind == "lesion_equalized":
            self.assumptions.append(FAMILY_FROM_TUNING)

    def calibrate(
        self, calibration_set: CalibrationSet, *, target_risk: float, n_bins: int
    ) -> CalibrationResult:
        if len(calibration_set) == 0:
            raise ValueError("Calibration set is empty")
        lambdas = lambda_grid()
        units = [
            unit_stats(
                self.family.score(s.foreground_probability),
                gt_components(s.ground_truth),
                tau=self.tau,
                lambdas=lambdas,
            )
            for s in calibration_set.samples
        ]
        st = stack(units, small_max_size=self.small_max_px, lambdas=lambdas)
        if len(st) == 0:
            raise ValueError("No calibration unit contains a lesion")
        risks = ("pix", "les", "small") if st.has_small.sum() >= 1 else ("pix", "les")
        index, feasible = union_index(st, target_risk, risks)
        ev = evaluate(st, index, lambdas)
        warnings: list[str] = []
        if not feasible:
            warnings.append(
                f"No threshold meets alpha={target_risk:.3f} for every risk on this "
                "calibration set; the reported threshold does NOT control all risks."
            )
        if "small" not in risks:
            warnings.append("No small lesion in the calibration set: small-lesion risk not controlled.")
        warnings.append(
            f"Calibration-set risks at lambda_hat: pixel FNR {ev.risk_pix:.3f}, "
            f"lesion miss {ev.risk_les:.3f}, small-lesion miss "
            f"{'n/a' if np.isnan(ev.risk_small) else f'{ev.risk_small:.3f}'} "
            f"(n={len(st)} units with lesions, {int(st.has_small.sum())} with small lesions)."
        )

        confidence, correct, p_fg, truth = _pixel_confidence_and_correctness(calibration_set)
        return CalibrationResult(
            method=self.id,
            method_name=self.name,
            status=self.status,
            target_risk=target_risk,
            threshold=float(lambdas[index]),
            empirical_risk=float(max(ev.risk_pix, ev.risk_les, np.nan_to_num(ev.risk_small))),
            coverage=float(1.0 - ev.risk_les),
            expected_calibration_error=expected_calibration_error(confidence, correct, n_bins),
            brier_score=brier_score(p_fg, truth),
            reliability_bins=reliability_bins(confidence, correct, n_bins),
            risk_coverage_curve=risk_coverage_curve(confidence, correct),
            calibration_set_size=len(st),
            provenance=calibration_set.provenance,
            assumptions=self.assumptions,
            validity_warnings=warnings,
        )

    def apply(self, result: CalibrationResult, foreground_probability: np.ndarray) -> AppliedCalibration:
        score = self.family.score(foreground_probability)
        in_set = score >= 1.0 - result.threshold
        band = in_set & ~(foreground_probability >= 0.5)
        return AppliedCalibration(
            abstention_rate=float(band.mean()),
            calibrated_confidence=float(
                np.maximum(foreground_probability, 1.0 - foreground_probability).mean()
            ),
            abstain_mask=band,
        )
