"""Prediction schemas.

Raw PyTorch / NumPy objects never cross the API boundary. Dense outputs are
transported either as a run-length-encoded label map (compact, lossless) or as
a PNG rendered by a dedicated media endpoint.
"""

from __future__ import annotations

from app.schemas.common import ApiModel, BoundingBox, TaskType


class ClassScore(ApiModel):
    class_id: int
    label: str
    probability: float


class SegmentationMask(ApiModel):
    """Run-length encoded integer label map (row-major, C order).

    `rle` alternates (value, run_length) pairs, flattened into one list. This
    is typically 50–500x smaller than the raw array for medical masks and is
    exactly reconstructible client-side.
    """

    height: int
    width: int
    num_classes: int
    class_labels: list[str]
    rle: list[int]
    #: Fraction of pixels assigned to each class index.
    class_pixel_fraction: list[float]
    #: URL of a PNG rendering of this mask (served by the media route).
    png_url: str | None = None


class Detection(ApiModel):
    id: str
    class_id: int
    label: str
    bbox: BoundingBox
    confidence: float


class PredictionResult(ApiModel):
    task: TaskType
    model_id: str
    #: Mean max-softmax over the image for dense tasks; top-1 probability for
    #: classification. Uncalibrated unless `calibration.applied` is true.
    confidence: float
    classes: list[ClassScore]
    segmentation: SegmentationMask | None = None
    detections: list[Detection] | None = None
    inference_time_ms: float
    metadata: dict = {}
