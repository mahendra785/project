"""Shared enums and small value objects used across the PUMS API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    """Base model: forbids unknown fields so contract drift fails loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TaskType(str, Enum):
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    CLASSIFICATION = "classification"


class ImplementationStatus(str, Enum):
    """Honest maturity label attached to every research component we expose.

    IMPLEMENTED   — code exists in this repository and runs in the pipeline.
    BASELINE      — implemented, but deliberately a simple reference method.
    EXPERIMENTAL  — implemented, not validated; results are not research claims.
    PLANNED       — not implemented; present only as an extension point.
    """

    IMPLEMENTED = "implemented"
    BASELINE = "baseline"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"


class DataProvenance(str, Enum):
    """Where a displayed number came from. Never leave this ambiguous."""

    #: Computed from the uploaded image by code in this repository.
    COMPUTED = "computed"
    #: Derived from a deterministic synthetic calibration/reference set that
    #: ships with the repo. Useful for wiring, NOT a research measurement.
    SYNTHETIC = "synthetic"
    #: Static placeholder for a UI surface whose backend is not implemented.
    MOCK = "mock"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ShiftStatus(str, Enum):
    ID = "ID"
    MILD_SHIFT = "MILD_SHIFT"
    MODERATE_SHIFT = "MODERATE_SHIFT"
    SEVERE_SHIFT = "SEVERE_SHIFT"


class ReferralPriority(str, Enum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT_REVIEW = "urgent_review"


class BoundingBox(ApiModel):
    """Axis-aligned box in *inference-resolution* pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


class ErrorResponse(ApiModel):
    code: str
    message: str
    detail: dict = {}
