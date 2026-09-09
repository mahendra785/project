"""Distribution-shift / OOD schemas."""

from __future__ import annotations

from app.schemas.common import (
    ApiModel,
    DataProvenance,
    ImplementationStatus,
    ShiftStatus,
)


class ShiftIndicator(ApiModel):
    """One interpretable component of the composite shift score."""

    id: str
    name: str
    #: Raw statistic value (units depend on the indicator).
    value: float
    #: Same statistic mapped monotonically into [0, 1].
    normalised: float
    weight: float
    description: str


class DistributionSummary(ApiModel):
    """First/second-order summary plus an intensity histogram."""

    name: str
    mean: float
    std: float
    skewness: float
    kurtosis: float
    entropy: float
    histogram: list[float]
    bin_edges: list[float]


class DistributionShiftResult(ApiModel):
    method: str
    status_label: ImplementationStatus

    #: Composite score in [0, 1]; 0 = indistinguishable from reference.
    shift_score: float
    status: ShiftStatus
    reference_domain: str
    current_domain: str
    thresholds: dict[str, float]

    indicators: list[ShiftIndicator]
    reference_distribution: DistributionSummary
    current_distribution: DistributionSummary

    #: Predicted degradation signals. These are *heuristic couplings*, not
    #: measured performance drops — never present them as accuracy numbers.
    confidence_degradation: float
    uncertainty_increase: float

    provenance: DataProvenance
    notes: list[str] = []
