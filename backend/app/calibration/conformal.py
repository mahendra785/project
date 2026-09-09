r"""Split Conformal Risk Control for binary segmentation.

Implements the finite-sample procedure of Angelopoulos, Bates, Fisch, Lei &
Schuster, *Conformal Risk Control*, ICLR 2024.

Setup
-----
For a threshold :math:`\lambda \in [0, 1]` the prediction set on image *i* is

.. math:: S_\lambda(X_i) = \{\, u : \hat p(u) \ge 1 - \lambda \,\}

so the set **grows** as :math:`\lambda` grows. With the false-negative-rate loss

.. math:: L_i(\lambda) = 1 - \frac{|S_\lambda(X_i) \cap Y_i|}{|Y_i|}

each :math:`L_i` is non-increasing in :math:`\lambda` and bounded by
:math:`B = 1`, satisfying the theorem's requirements.

Guarantee
---------
Choosing

.. math:: \hat\lambda = \inf\Bigl\{ \lambda :
          \frac{n\,\hat R_n(\lambda) + B}{n + 1} \le \alpha \Bigr\}

yields :math:`\mathbb{E}[L_{n+1}(\hat\lambda)] \le \alpha` for an exchangeable
test point. The :math:`+B` numerator term is what makes the bound valid at
finite *n* rather than only asymptotically.

Scope of validity in PUMS
-------------------------
The guarantee is **marginal** and requires **exchangeability**. Under the
distribution shift this project targets, exchangeability fails and the
guarantee lapses — PUMS therefore emits an explicit `validity_warnings` entry
whenever the shift detector flags the input, rather than quietly continuing to
display a number that no longer means what it says. Restoring validity under
shift (weighted/adaptive conformal) is future work, not a present claim.
"""

from __future__ import annotations

import numpy as np

from app.calibration.base import (
    EXCHANGEABILITY,
    MARGINAL_ONLY,
    MONOTONE_LOSS,
    SYNTHETIC_SET,
    AppliedCalibration,
    BaseCalibrationMethod,
)
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

#: Resolution of the lambda grid searched. 200 steps => 0.005 granularity,
#: finer than the statistical noise of an 80-sample calibration set.
LAMBDA_GRID_SIZE = 200


def _loss_curve(calibration_set: CalibrationSet, lambdas: np.ndarray) -> np.ndarray:
    """Mean false-negative-rate loss at each lambda, over the calibration set.

    Vectorised via the per-image distribution of foreground-pixel scores: for a
    given image, FNR(lambda) is simply the fraction of true-foreground pixels
    whose predicted probability falls below `1 - lambda`.
    """
    losses = np.zeros((len(calibration_set), lambdas.size), dtype=np.float64)
    for i, sample in enumerate(calibration_set.samples):
        truth = sample.ground_truth
        if not truth.any():
            continue
        scores = np.sort(sample.foreground_probability[truth])
        # searchsorted gives, for each cutoff, how many true pixels fall below it.
        cutoffs = 1.0 - lambdas
        below = np.searchsorted(scores, cutoffs, side="left")
        losses[i] = below / scores.size
    return losses.mean(axis=0)


def _pixel_confidence_and_correctness(
    calibration_set: CalibrationSet,
    max_pixels_per_image: int = 4000,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample (confidence, correct, p_foreground, truth) across the set.

    Subsampling keeps ECE/reliability computation fast while remaining an
    unbiased estimate; the sample is drawn with a fixed seed so results are
    reproducible run to run.
    """
    rng = np.random.default_rng(seed)
    confidences: list[np.ndarray] = []
    correctness: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []

    for sample in calibration_set.samples:
        p_fg = sample.foreground_probability.ravel()
        truth = sample.ground_truth.ravel()
        if p_fg.size > max_pixels_per_image:
            idx = rng.choice(p_fg.size, size=max_pixels_per_image, replace=False)
            p_fg, truth = p_fg[idx], truth[idx]

        predicted = p_fg >= 0.5
        confidences.append(np.maximum(p_fg, 1.0 - p_fg))
        correctness.append((predicted == truth).astype(np.float64))
        probabilities.append(p_fg)
        truths.append(truth.astype(np.float64))

    return (
        np.concatenate(confidences),
        np.concatenate(correctness),
        np.concatenate(probabilities),
        np.concatenate(truths),
    )


class ConformalRiskControl(BaseCalibrationMethod):
    """Split CRC with a false-negative-rate loss (Angelopoulos et al., 2024)."""

    id = "conformal_risk_control"
    name = "Conformal Risk Control (split, FNR loss)"
    status = ImplementationStatus.IMPLEMENTED
    description = (
        "Selects the smallest prediction-set threshold whose finite-sample "
        "corrected empirical risk meets the target alpha, using a monotone "
        "false-negative-rate loss."
    )
    reference = "Angelopoulos et al., Conformal Risk Control, ICLR 2024"
    assumptions = [EXCHANGEABILITY, MONOTONE_LOSS, MARGINAL_ONLY, SYNTHETIC_SET]

    #: Bound B on the loss. FNR is a rate, so B = 1.
    LOSS_BOUND = 1.0

    def calibrate(
        self, calibration_set: CalibrationSet, *, target_risk: float, n_bins: int
    ) -> CalibrationResult:
        n = len(calibration_set)
        if n == 0:
            raise ValueError("Calibration set is empty")

        lambdas = np.linspace(0.0, 1.0, LAMBDA_GRID_SIZE)
        empirical = _loss_curve(calibration_set, lambdas)

        # Finite-sample corrected risk from Theorem 1.
        corrected = (n * empirical + self.LOSS_BOUND) / (n + 1)
        feasible = np.flatnonzero(corrected <= target_risk)

        if feasible.size:
            index = int(feasible[0])
            lambda_hat = float(lambdas[index])
            warnings: list[str] = []
        else:
            # No lambda meets alpha even at the most inclusive setting: report
            # the most conservative option and say so, rather than silently
            # returning a threshold that does not control the risk.
            index = int(np.argmin(corrected))
            lambda_hat = float(lambdas[index])
            warnings = [
                f"No threshold achieves the corrected risk target alpha={target_risk:.3f} "
                f"on this calibration set (best achievable: {float(corrected[index]):.3f}). "
                "The reported threshold does NOT control the risk at the requested level."
            ]

        confidence, correct, p_fg, truth = _pixel_confidence_and_correctness(
            calibration_set
        )
        ece = expected_calibration_error(confidence, correct, n_bins)

        # Coverage here = fraction of true-foreground pixels retained by the
        # calibrated set, i.e. 1 - empirical FNR.
        coverage = float(1.0 - empirical[index])

        return CalibrationResult(
            method=self.id,
            method_name=self.name,
            status=self.status,
            target_risk=target_risk,
            threshold=lambda_hat,
            empirical_risk=float(empirical[index]),
            coverage=coverage,
            expected_calibration_error=ece,
            brier_score=brier_score(p_fg, truth),
            reliability_bins=reliability_bins(confidence, correct, n_bins),
            risk_coverage_curve=risk_coverage_curve(confidence, correct),
            calibration_set_size=n,
            provenance=calibration_set.provenance,
            assumptions=self.assumptions,
            validity_warnings=warnings,
        )

    def apply(
        self, result: CalibrationResult, foreground_probability: np.ndarray
    ) -> AppliedCalibration:
        """Apply lambda_hat to a new image.

        The calibrated prediction set is ``{p_fg >= 1 - lambda_hat}``, which is
        deliberately more inclusive than the argmax decision ``{p_fg >= 0.5}``.
        Pixels inside the calibrated set but outside the argmax decision form
        the *ambiguity band*: locations the risk-controlled rule refuses to call
        background even though the point prediction does. Its size is the most
        directly actionable number calibration produces, so it is reported as
        the abstention rate.
        """
        cutoff = 1.0 - result.threshold
        in_set = foreground_probability >= cutoff
        point_foreground = foreground_probability >= 0.5
        ambiguity_band = in_set & ~point_foreground

        return AppliedCalibration(
            abstention_rate=float(ambiguity_band.mean()),
            calibrated_confidence=float(
                np.maximum(foreground_probability, 1.0 - foreground_probability).mean()
            ),
            abstain_mask=ambiguity_band,
        )


class BaselineThresholdCalibration(BaseCalibrationMethod):
    """Reference point: pick the threshold that hits the target risk empirically.

    This is the *uncorrected* version of CRC — no `+B/(n+1)` finite-sample term
    and therefore no guarantee. It exists so the effect of the conformal
    correction is visible side by side rather than taken on faith.
    """

    id = "baseline_threshold"
    name = "Empirical Threshold (no guarantee)"
    status = ImplementationStatus.BASELINE
    description = (
        "Chooses the smallest threshold whose raw empirical risk meets alpha. "
        "No finite-sample correction, therefore no risk-control guarantee — "
        "included as the control condition for Conformal Risk Control."
    )
    reference = None
    assumptions = [
        "No statistical guarantee of any kind. Comparison baseline only.",
        SYNTHETIC_SET,
    ]

    def calibrate(
        self, calibration_set: CalibrationSet, *, target_risk: float, n_bins: int
    ) -> CalibrationResult:
        n = len(calibration_set)
        lambdas = np.linspace(0.0, 1.0, LAMBDA_GRID_SIZE)
        empirical = _loss_curve(calibration_set, lambdas)

        feasible = np.flatnonzero(empirical <= target_risk)
        index = int(feasible[0]) if feasible.size else int(np.argmin(empirical))

        confidence, correct, p_fg, truth = _pixel_confidence_and_correctness(
            calibration_set
        )

        return CalibrationResult(
            method=self.id,
            method_name=self.name,
            status=self.status,
            target_risk=target_risk,
            threshold=float(lambdas[index]),
            empirical_risk=float(empirical[index]),
            coverage=float(1.0 - empirical[index]),
            expected_calibration_error=expected_calibration_error(
                confidence, correct, n_bins
            ),
            brier_score=brier_score(p_fg, truth),
            reliability_bins=reliability_bins(confidence, correct, n_bins),
            risk_coverage_curve=risk_coverage_curve(confidence, correct),
            calibration_set_size=n,
            provenance=calibration_set.provenance,
            assumptions=self.assumptions,
            validity_warnings=[
                "This method provides NO coverage or risk guarantee; it is a "
                "comparison baseline for Conformal Risk Control."
            ],
        )

    def apply(
        self, result: CalibrationResult, foreground_probability: np.ndarray
    ) -> AppliedCalibration:
        return ConformalRiskControl().apply(result, foreground_probability)
