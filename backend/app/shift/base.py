"""Distribution-shift detector interface and the reference distribution."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.shift import DistributionShiftResult


@dataclass(slots=True)
class ReferenceProfile:
    """The 'in-distribution' domain a test image is compared against.

    Built by aggregating the preprocessing statistics of a set of images. The
    shipped default is derived from the synthetic phantom set, so it defines
    'in-distribution' as *phantom-like*, not as any real clinical population.
    Swap it via `set_reference_profile()` once real reference data exists.
    """

    name: str
    mean: float
    std: float
    skewness: float
    kurtosis: float
    entropy: float
    edge_density: float
    histogram: np.ndarray
    #: Per-statistic spread across the reference set — used to standardise each
    #: indicator so a 1.0 normalised score means "one reference SD away".
    dispersion: dict[str, float]
    n_samples: int
    description: str


class BaseShiftDetector(abc.ABC):
    """Interface for OOD / distribution-shift detection."""

    id: str = "base"
    name: str = "Base detector"

    @abc.abstractmethod
    def detect(
        self,
        image: PreprocessedImage,
        reference: ReferenceProfile,
        *,
        thresholds: dict[str, float],
    ) -> DistributionShiftResult:
        """Score how far `image` is from `reference`."""
