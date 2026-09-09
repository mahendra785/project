"""Calibration method interface and catalogue.

The catalogue is the single source of truth for what PUMS has actually built
versus what the literature review identifies as future work. The frontend reads
it directly, which makes it impossible for the UI to advertise a method the
backend does not implement.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from app.calibration.datasets import CalibrationSet
from app.schemas.calibration import CalibrationMethodInfo, CalibrationResult
from app.schemas.common import ImplementationStatus

EXCHANGEABILITY = (
    "Exchangeability: the calibration samples and the test image must be "
    "exchangeable. Distribution shift violates this, and the coverage/risk "
    "guarantee does not hold when it is violated."
)
MONOTONE_LOSS = (
    "Monotone loss: the per-sample loss must be non-increasing in the "
    "threshold parameter lambda (Angelopoulos et al., ICLR 2024)."
)
MARGINAL_ONLY = (
    "Marginal, not conditional: the guarantee holds on average over images, "
    "not for any individual image, region or patient subgroup."
)
SYNTHETIC_SET = (
    "The shipped calibration set is synthetic (procedural phantoms). Numbers "
    "derived from it characterise phantom behaviour, not clinical performance."
)


@dataclass(slots=True)
class AppliedCalibration:
    """What the calibrated rule says about the *current* image."""

    abstention_rate: float
    calibrated_confidence: float
    #: (H, W) boolean — pixels the calibrated rule declines to commit to.
    abstain_mask: np.ndarray | None = None


class BaseCalibrationMethod(abc.ABC):
    """Interface for post-hoc calibration / risk-control methods."""

    id: str = "base"
    name: str = "Base"
    status: ImplementationStatus = ImplementationStatus.PLANNED
    description: str = ""
    reference: str | None = None
    assumptions: list[str] = []

    def info(self) -> CalibrationMethodInfo:
        return CalibrationMethodInfo(
            id=self.id,
            name=self.name,
            status=self.status,
            description=self.description,
            reference=self.reference,
            assumptions=self.assumptions,
        )

    @abc.abstractmethod
    def calibrate(
        self, calibration_set: CalibrationSet, *, target_risk: float, n_bins: int
    ) -> CalibrationResult:
        """Fit the calibration parameter and report its measured behaviour."""

    def apply(
        self, result: CalibrationResult, foreground_probability: np.ndarray
    ) -> AppliedCalibration:
        """Apply the fitted threshold to a new image."""
        raise NotImplementedError


class PlannedMethod(BaseCalibrationMethod):
    """Placeholder for a literature method that is NOT implemented.

    Instantiating one keeps the extension point visible in the API and the UI
    while making it impossible to accidentally run — `calibrate` raises.
    """

    def __init__(
        self,
        method_id: str,
        name: str,
        description: str,
        reference: str | None = None,
        assumptions: list[str] | None = None,
    ) -> None:
        self.id = method_id
        self.name = name
        self.status = ImplementationStatus.PLANNED
        self.description = description
        self.reference = reference
        self.assumptions = assumptions or []

    def calibrate(
        self, calibration_set: CalibrationSet, *, target_risk: float, n_bins: int
    ) -> CalibrationResult:
        raise NotImplementedError(
            f"'{self.id}' is a planned extension point and is not implemented."
        )
