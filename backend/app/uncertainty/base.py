"""Uncertainty estimator interface.

An estimator consumes a `(C, H, W)` probability array and returns a dense map
in [0, 1]. Nothing else. That narrow contract is why the estimator can be
swapped (entropy -> evidential -> deep-ensemble variance) without touching the
calibration, shift or risk layers.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from app.schemas.common import ImplementationStatus


@dataclass(slots=True)
class UncertaintyMap:
    """A dense uncertainty field plus the provenance needed to interpret it."""

    #: (H, W) float64 in [0, 1]. 0 = maximally confident, 1 = maximally uncertain.
    values: np.ndarray
    method: str
    method_name: str
    status: ImplementationStatus
    #: Per-pixel max posterior — the model's own confidence, kept alongside so
    #: downstream code never has to re-derive it from the probability array.
    confidence: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


class BaseUncertaintyEstimator(abc.ABC):
    """Base class for all PUMS uncertainty estimators."""

    id: str = "base"
    name: str = "Base estimator"
    status: ImplementationStatus = ImplementationStatus.PLANNED

    @abc.abstractmethod
    def estimate(self, probabilities: np.ndarray) -> UncertaintyMap:
        """Return an uncertainty map derived from per-class probabilities."""

    @staticmethod
    def _validate(probabilities: np.ndarray) -> np.ndarray:
        if probabilities.ndim != 3:
            raise ValueError("probabilities must have shape (C, H, W)")
        if probabilities.shape[0] < 2:
            raise ValueError("at least two classes are required for uncertainty")
        probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
        return probs / probs.sum(axis=0, keepdims=True)
