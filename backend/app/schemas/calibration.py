"""Calibration schemas.

Assumption disclosure is part of the contract: every calibration result carries
the assumptions its validity depends on, and whether they are believed to hold
for the current input. Conformal guarantees are *marginal* and require
exchangeability between the calibration set and the test point; under
distribution shift they are NOT valid and we say so explicitly.
"""

from __future__ import annotations

from app.schemas.common import ApiModel, DataProvenance, ImplementationStatus


class CalibrationMethodInfo(ApiModel):
    id: str
    name: str
    status: ImplementationStatus
    description: str
    reference: str | None = None
    #: Assumptions that must hold for this method's guarantee to be meaningful.
    assumptions: list[str] = []


class ReliabilityBin(ApiModel):
    bin_lower: float
    bin_upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float


class RiskCoveragePoint(ApiModel):
    coverage: float
    risk: float
    threshold: float


class CalibrationResult(ApiModel):
    method: str
    method_name: str
    status: ImplementationStatus

    #: Target expected loss alpha requested by the caller.
    target_risk: float
    #: lambda_hat: the calibrated threshold selected on the calibration set.
    threshold: float
    #: Empirical risk of `threshold` measured ON THE CALIBRATION SET.
    empirical_risk: float
    #: Fraction of calibration points retained/covered at `threshold`.
    coverage: float
    #: Expected Calibration Error of the model on the calibration set.
    expected_calibration_error: float
    brier_score: float | None = None

    #: Calibrated per-pixel abstention rate on the *current* image at
    #: `threshold` — i.e. what fraction of this image the calibrated rule
    #: flags as not-confidently-predicted.
    applied_abstention_rate: float | None = None
    calibrated_confidence: float | None = None
    applied: bool = False

    reliability_bins: list[ReliabilityBin] = []
    risk_coverage_curve: list[RiskCoveragePoint] = []

    calibration_set_size: int
    provenance: DataProvenance
    assumptions: list[str] = []
    #: Set when we detect a reason the guarantee should not be trusted.
    validity_warnings: list[str] = []
