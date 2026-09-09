"""Evaluation / benchmarking schemas.

IMPORTANT: PUMS ships no medical benchmark results. The evaluation endpoint
reports (a) metrics computed over the analyses this instance has actually run
and (b) metric *definitions* with their implementation status. Any figure that
would constitute a research claim is absent by design until a real dataset run
produces it.
"""

from __future__ import annotations

from app.schemas.common import ApiModel, DataProvenance, ImplementationStatus


class MetricDefinition(ApiModel):
    id: str
    name: str
    category: str
    status: ImplementationStatus
    description: str
    #: True when the metric needs ground truth we do not have at inference time.
    requires_ground_truth: bool
    higher_is_better: bool


class MetricValue(ApiModel):
    id: str
    name: str
    category: str
    value: float | None
    unit: str | None = None
    sample_size: int
    provenance: DataProvenance
    #: Filled when `value is None` — states exactly why it cannot be computed.
    unavailable_reason: str | None = None


class RiskCoverageSeriesPoint(ApiModel):
    coverage: float
    selective_risk: float
    threshold: float


class EvaluationReport(ApiModel):
    generated_at: str
    #: Number of completed analyses on this instance that fed the metrics.
    session_sample_size: int
    metrics: list[MetricValue]
    definitions: list[MetricDefinition]
    risk_coverage_curve: list[RiskCoverageSeriesPoint]
    uncertainty_vs_confidence: list[dict]
    notes: list[str]


class SelfTestResult(ApiModel):
    """Unit-style verification of the metric implementations on constructed
    inputs with analytically known answers. This is the honest way to show the
    evaluation code works without inventing dataset results."""

    metric: str
    scenario: str
    expected: float
    actual: float
    passed: bool
