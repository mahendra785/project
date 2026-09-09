"""Top-level analysis request/response schemas — the unified PUMS payload."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.calibration import CalibrationResult
from app.schemas.common import AnalysisStatus, ApiModel, TaskType
from app.schemas.prediction import PredictionResult
from app.schemas.risk import RiskAssessment
from app.schemas.shift import DistributionShiftResult
from app.schemas.uncertainty import HierarchicalUncertainty


class AnalysisOptions(ApiModel):
    """Per-request configuration. All fields have server-side defaults."""

    model_id: str | None = None
    task: TaskType = TaskType.SEGMENTATION
    calibration_method: str = "conformal_risk_control"
    target_risk: float = Field(default=0.10, ge=0.001, le=0.5)
    uncertainty_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    reference_domain: str = "synthetic_reference_v1"
    apply_calibration: bool = True
    detect_shift: bool = True
    notes: str | None = None


class ImageMetadata(ApiModel):
    filename: str
    content_type: str
    size_bytes: int
    original_width: int
    original_height: int
    original_channels: int
    working_width: int
    working_height: int
    format: str
    is_dicom: bool = False
    #: SHA-256 of the uploaded bytes; makes runs traceable and de-duplicable.
    checksum: str
    #: Extra tags harvested from DICOM headers, when present.
    tags: dict = {}
    preview_url: str | None = None


class AnalysisMetadata(ApiModel):
    created_at: datetime
    duration_ms: float
    model_id: str
    model_name: str
    task: TaskType
    options: AnalysisOptions
    image: ImageMetadata
    pipeline_version: str
    seed: int


class AnalysisResponse(ApiModel):
    analysis_id: str
    status: AnalysisStatus
    metadata: AnalysisMetadata
    prediction: PredictionResult
    uncertainty: HierarchicalUncertainty
    calibration: CalibrationResult | None = None
    distribution_shift: DistributionShiftResult | None = None
    risk_assessment: RiskAssessment
    warnings: list[str] = []


class AnalysisSummary(ApiModel):
    """Compact row used by dashboard lists."""

    analysis_id: str
    created_at: datetime
    filename: str
    model_id: str
    task: TaskType
    status: AnalysisStatus
    mean_uncertainty: float
    confidence: float
    shift_score: float | None
    shift_status: str | None
    risk_level: str
    risk_score: float


class AnalysisListResponse(ApiModel):
    items: list[AnalysisSummary]
    total: int


class DashboardStats(ApiModel):
    """Aggregates over *this instance's* completed analyses only."""

    total_analyses: int
    completed_analyses: int
    failed_analyses: int
    active_model_id: str
    active_model_name: str
    mean_uncertainty: float | None
    mean_confidence: float | None
    high_risk_count: int
    review_required_count: int
    shift_flagged_count: int
    mean_shift_score: float | None
    mean_ece: float | None
    #: Histogram of per-analysis mean uncertainty over 10 bins in [0, 1].
    uncertainty_histogram: list[int]
    risk_breakdown: dict[str, int]
    shift_breakdown: dict[str, int]
    recent: list[AnalysisSummary]
    backend_available: bool = True
