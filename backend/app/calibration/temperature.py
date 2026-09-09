r"""Temperature scaling (Guo et al., ICML 2017) for the binary foreground head.

A single scalar :math:`T` is fitted on the calibration set by minimising the
negative log-likelihood of

.. math:: \hat p_T = \sigma\!\left(\frac{\operatorname{logit}(\hat p)}{T}\right)

Because :math:`\sigma(\cdot/T)` is strictly monotone, temperature scaling
cannot change any thresholded decision — Dice, IoU and the argmax mask are
provably unchanged. It only re-shapes *how confident* the model claims to be,
which is exactly the "fragile calibration" problem in the project's research
gap: one parameter, no retraining, no architecture change.

Unlike Conformal Risk Control this method carries **no finite-sample
guarantee**. It reduces measured ECE; it does not bound any risk. The two are
complementary and PUMS reports them separately rather than conflating them.
"""

from __future__ import annotations

import numpy as np

from app.calibration.base import SYNTHETIC_SET, AppliedCalibration, BaseCalibrationMethod
from app.calibration.conformal import _pixel_confidence_and_correctness
from app.calibration.datasets import CalibrationSet
from app.calibration.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
    risk_coverage_curve,
)
from app.core.logging import get_logger
from app.schemas.calibration import CalibrationResult
from app.schemas.common import ImplementationStatus

logger = get_logger(__name__)

EPSILON = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    objective: str = "ece",
    n_bins: int = 15,
    grid: np.ndarray | None = None,
) -> float:
    """Fit T by grid search over a log-spaced temperature range.

    A grid search rather than a gradient method: the objective is
    one-dimensional and smooth, so 60 evaluations locate the optimum to well
    within the noise of the calibration set, with no optimiser to diverge.

    OBJECTIVE CHOICE — a deliberate departure from Guo et al.
    ---------------------------------------------------------
    Guo et al. minimise NLL. On this model the two objectives genuinely
    disagree: NLL is dominated by the extreme-probability tail and selects
    T ~= 2.2, while measured ECE is minimised near T ~= 0.8, and the NLL
    optimum makes ECE *worse*. Since ECE is the reliability quantity PUMS
    reports and that the risk engine consumes, optimising NLL and then
    displaying a degraded ECE would be incoherent. `objective="ece"` is
    therefore the default; `objective="nll"` reproduces the original method
    and is kept for comparison.

    This selects the temperature that minimises ECE *on the calibration set*.
    That is a fit, not a guarantee: test-time ECE can still be worse.
    """
    logits = _logit(probability)
    y = np.asarray(target, dtype=np.float64)
    candidates = grid if grid is not None else np.logspace(-1.0, 1.2, 60)

    best_t, best_score = 1.0, np.inf
    for temperature in candidates:
        p = np.clip(_sigmoid(logits / temperature), EPSILON, 1.0 - EPSILON)
        if objective == "nll":
            score = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())
        elif objective == "ece":
            correct = ((p >= 0.5) == (y >= 0.5)).astype(np.float64)
            score = expected_calibration_error(np.maximum(p, 1.0 - p), correct, n_bins)
        else:
            raise ValueError(f"Unknown objective '{objective}'; use 'ece' or 'nll'")
        if score < best_score:
            best_score, best_t = score, float(temperature)
    return best_t


class TemperatureScaling(BaseCalibrationMethod):
    """Post-hoc scalar temperature fitted by NLL on the calibration set."""

    id = "temperature_scaling"
    name = "Temperature Scaling"
    status = ImplementationStatus.IMPLEMENTED
    description = (
        "Fits one scalar temperature on the calibration set (minimising ECE by "
        "default; NLL available for comparison). Monotone, so it never changes "
        "a thresholded decision — only the confidence attached to it. Reduces "
        "measured ECE but provides NO risk guarantee."
    )
    reference = (
        "Guo et al., On Calibration of Modern Neural Networks, ICML 2017 "
        "(adapted: ECE objective, see fit_temperature docstring)"
    )
    assumptions = [
        "Calibration set is representative of the test distribution; shift "
        "degrades the fitted temperature without any warning from the method.",
        "Provides no coverage or risk guarantee of any kind.",
        SYNTHETIC_SET,
    ]

    def calibrate(
        self, calibration_set: CalibrationSet, *, target_risk: float, n_bins: int
    ) -> CalibrationResult:
        _, _, p_fg, truth = _pixel_confidence_and_correctness(calibration_set)

        temperature = fit_temperature(p_fg, truth, objective="ece", n_bins=n_bins)
        scaled = _sigmoid(_logit(p_fg) / temperature)

        predicted = scaled >= 0.5
        correct = (predicted == (truth >= 0.5)).astype(np.float64)
        confidence = np.maximum(scaled, 1.0 - scaled)

        ece_before = expected_calibration_error(
            np.maximum(p_fg, 1.0 - p_fg),
            (( p_fg >= 0.5) == (truth >= 0.5)).astype(np.float64),
            n_bins,
        )
        ece_after = expected_calibration_error(confidence, correct, n_bins)
        logger.info(
            "temperature scaling: T=%.3f  ECE %.4f -> %.4f", temperature, ece_before, ece_after
        )

        return CalibrationResult(
            method=self.id,
            method_name=self.name,
            status=self.status,
            target_risk=target_risk,
            # `threshold` carries the fitted temperature for this method; it is
            # not a decision threshold. The UI labels it per-method.
            threshold=round(temperature, 4),
            empirical_risk=float(1.0 - correct.mean()),
            coverage=1.0,
            expected_calibration_error=ece_after,
            brier_score=brier_score(scaled, truth),
            reliability_bins=reliability_bins(confidence, correct, n_bins),
            risk_coverage_curve=risk_coverage_curve(confidence, correct),
            calibration_set_size=len(calibration_set),
            provenance=calibration_set.provenance,
            assumptions=self.assumptions,
            validity_warnings=[
                "Temperature scaling gives no risk or coverage guarantee. It "
                f"reduced ECE from {ece_before:.4f} to {ece_after:.4f} on the "
                "calibration set; that is a measurement, not a guarantee.",
            ],
        )

    def apply(
        self, result: CalibrationResult, foreground_probability: np.ndarray
    ) -> AppliedCalibration:
        scaled = _sigmoid(_logit(foreground_probability) / max(result.threshold, EPSILON))
        confidence = np.maximum(scaled, 1.0 - scaled)
        # "Abstention" for this method = pixels whose calibrated confidence is
        # below the conventional 0.75 review bar. Reported for comparability
        # with the conformal methods, not as a guaranteed quantity.
        return AppliedCalibration(
            abstention_rate=float((confidence < 0.75).mean()),
            calibrated_confidence=float(confidence.mean()),
            abstain_mask=confidence < 0.75,
        )
