"""Model abstraction.

The contract every adapter must satisfy is: *return per-class probabilities*.
The entire uncertainty stack is built on that array and nothing else, which is
what makes PUMS model-agnostic — swapping a classical segmenter for a
foundation model changes only which class implements `predict_segmentation`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from app.core.errors import TaskNotSupportedError
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import ImplementationStatus, TaskType


@dataclass(slots=True)
class SegmentationOutput:
    """Dense per-class probabilities.

    `probabilities` has shape (C, H, W), sums to 1 along axis 0, and MUST be a
    genuine model output. Uncertainty is derived from it; fabricating it would
    invalidate every downstream number.
    """

    probabilities: np.ndarray
    class_labels: list[str]
    #: Optional per-class logits, when the adapter has them.
    logits: np.ndarray | None = None
    #: Channels the adapter declares to be "foreground" / structure of interest.
    #: The binary calibration and evaluation layers need a two-class reduction,
    #: and only the adapter knows which of its classes carry that meaning. When
    #: None, every non-zero channel is treated as foreground (channel 0 is
    #: background by PUMS convention).
    foreground_channels: list[int] | None = None

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 3:
            raise ValueError("probabilities must have shape (C, H, W)")
        if self.probabilities.shape[0] != len(self.class_labels):
            raise ValueError("class_labels length must match probabilities channels")

    @property
    def labels(self) -> np.ndarray:
        return np.argmax(self.probabilities, axis=0).astype(np.int32)

    @property
    def foreground_probability(self) -> np.ndarray:
        """(H, W) probability that a pixel belongs to the foreground.

        Summing the declared foreground channels is the correct reduction for a
        multiclass softmax: the events are mutually exclusive, so the
        probability of their union is the sum.
        """
        if self.foreground_channels is None:
            return np.clip(1.0 - self.probabilities[0], 0.0, 1.0)
        selected = self.probabilities[self.foreground_channels]
        return np.clip(selected.sum(axis=0), 0.0, 1.0)


@dataclass(slots=True)
class DetectionOutput:
    boxes: np.ndarray  # (N, 4) as x, y, w, h
    scores: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,)
    class_labels: list[str]
    #: (N, C) per-box class distribution when available — needed for real
    #: object-level entropy rather than a 1 - score proxy.
    class_probabilities: np.ndarray | None = None


@dataclass(slots=True)
class ClassificationOutput:
    probabilities: np.ndarray  # (C,)
    class_labels: list[str]


@dataclass(slots=True)
class ModelDescriptor:
    """Static, honest description of an adapter."""

    id: str
    name: str
    model_type: str
    tasks: list[TaskType]
    status: ImplementationStatus
    description: str
    trained_on: str
    #: True when the weights were fitted on real medical imaging data.
    is_medical: bool
    class_labels: list[str]
    #: ALWAYS False in this project. Being trained on medical data is not the
    #: same as being validated for clinical use, and conflating the two is the
    #: single easiest way for a research prototype to mislead.
    clinically_validated: bool = False
    input_channels: int = 1
    parameters: dict = field(default_factory=dict)
    #: Package imports required at runtime, checked by `availability()`.
    requires: list[str] = field(default_factory=list)


class BaseMedicalModel(abc.ABC):
    """Interface implemented by every PUMS inference adapter."""

    descriptor: ModelDescriptor

    def __init__(self) -> None:
        self._loaded = False

    # ------------------------------------------------------------ lifecycle -
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Idempotently prepare the adapter for inference."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self) -> None:
        """Adapter-specific setup. Default: nothing to do."""
        return None

    def availability(self) -> tuple[bool, str | None]:
        """Whether this adapter can run right now, and why not if it cannot."""
        import importlib.util

        missing = [
            package
            for package in self.descriptor.requires
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            return False, f"Missing optional dependencies: {', '.join(missing)}"
        return True, None

    def metadata(self) -> ModelDescriptor:
        return self.descriptor

    def supports(self, task: TaskType) -> bool:
        return task in self.descriptor.tasks

    # ----------------------------------------------------------- inference -
    def predict(
        self, image: PreprocessedImage, task: TaskType
    ) -> SegmentationOutput | DetectionOutput | ClassificationOutput:
        """Dispatch to the task-specific method, enforcing declared support."""
        if not self.supports(task):
            raise TaskNotSupportedError(
                f"Model '{self.descriptor.id}' does not support task '{task.value}'",
                detail={
                    "model_id": self.descriptor.id,
                    "supported_tasks": [t.value for t in self.descriptor.tasks],
                },
            )
        self.load()
        if task is TaskType.SEGMENTATION:
            return self.predict_segmentation(image)
        if task is TaskType.DETECTION:
            return self.predict_detection(image)
        return self.predict_classification(image)

    @abc.abstractmethod
    def predict_segmentation(self, image: PreprocessedImage) -> SegmentationOutput:
        """Return calibrated-or-not per-class probabilities of shape (C, H, W)."""

    def predict_detection(self, image: PreprocessedImage) -> DetectionOutput:
        raise TaskNotSupportedError(
            f"Detection adapter not configured for '{self.descriptor.id}'"
        )

    def predict_classification(self, image: PreprocessedImage) -> ClassificationOutput:
        raise TaskNotSupportedError(
            f"Classification adapter not configured for '{self.descriptor.id}'"
        )
