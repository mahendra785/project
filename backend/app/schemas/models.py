"""Model-registry schemas."""

from __future__ import annotations

from app.schemas.common import ApiModel, ImplementationStatus, TaskType


class ModelInfo(ApiModel):
    id: str
    name: str
    #: e.g. "classical-probabilistic", "cnn", "transformer", "foundation".
    model_type: str
    tasks: list[TaskType]
    status: ImplementationStatus
    #: True when the adapter's dependencies are installed and it can run now.
    available: bool
    unavailable_reason: str | None = None

    #: Which research components this adapter is wired to.
    uncertainty_method: str
    calibration_methods: list[str]
    ood_detector: str

    #: Honest provenance of the weights/algorithm.
    trained_on: str
    is_medical: bool
    description: str
    parameters: dict = {}
    input_channels: int = 1
    class_labels: list[str] = []


class ModelListResponse(ApiModel):
    models: list[ModelInfo]
    default_model_id: str
