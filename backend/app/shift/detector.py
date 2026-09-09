r"""Composite intensity-domain shift detector (BASELINE).

Method
------
Four interpretable, cheap indicators are computed between the test image and a
reference profile, each mapped monotonically into [0, 1] and combined by a
fixed weighted mean:

1. **Wasserstein-1 distance** between the intensity histograms. The earth-
   mover distance is the right choice here because it is sensitive to *how far*
   intensity mass moved, not merely that bins differ — exactly the behaviour of
   a scanner/protocol change.
2. **Jensen-Shannon divergence** between the same histograms. Bounded,
   symmetric, and sensitive to shape changes (new modes) that leave the mean
   intact.
3. **Standardised moment deviation**: |mean|, |std|, |skew|, |kurtosis|
   differences divided by the reference set's own dispersion, so "far" is
   measured in units the reference data itself defines.
4. **Texture/edge-density deviation**: mean gradient magnitude, a cheap proxy
   for reconstruction kernel and resolution differences.

What this is not
----------------
This is an **input-space** detector operating on intensity statistics. It will
catch modality, exposure, scanner and preprocessing changes. It will NOT catch
semantic shift that preserves intensity statistics (e.g. a new pathology with
familiar appearance) — that needs feature-space or model-internal detection,
which is listed as future work. The composite score is a heuristic, not a
calibrated OOD probability, and the band thresholds are configuration, not
findings.
"""

from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import DataProvenance, ImplementationStatus, ShiftStatus
from app.schemas.shift import (
    DistributionShiftResult,
    DistributionSummary,
    ShiftIndicator,
)
from app.shift.base import BaseShiftDetector, ReferenceProfile


class CompositeIntensityShiftDetector(BaseShiftDetector):
    """Weighted combination of histogram-distance and moment-deviation terms."""

    id = "composite_intensity_shift_v1"
    name = "Composite Intensity-Domain Shift Detector"

    #: Indicator weights. Histogram distances dominate because they use the
    #: full distribution; moments are a lower-resolution summary of the same
    #: information and would otherwise double-count it.
    WEIGHTS = {
        "wasserstein": 0.35,
        "jensen_shannon": 0.30,
        "moments": 0.20,
        "texture": 0.15,
    }

    #: Scale at which each raw statistic reaches a normalised score of ~1.
    #: Chosen so that a same-domain image scores well below the MILD band and
    #: a cross-modality image saturates. These are configuration constants, and
    #: they are echoed in the API response so results stay reproducible.
    SCALES = {
        "wasserstein": 0.25,
        "moments": 3.0,
        "texture": 0.04,
    }

    def detect(
        self,
        image: PreprocessedImage,
        reference: ReferenceProfile,
        *,
        thresholds: dict[str, float],
    ) -> DistributionShiftResult:
        current_hist = (
            image.histogram
            if image.histogram is not None
            else np.histogram(image.gray, bins=32, range=(0, 1), density=True)[0]
        )
        current_hist = np.asarray(current_hist, dtype=np.float64)
        current_hist = current_hist / max(current_hist.sum(), 1e-12)
        reference_hist = reference.histogram / max(reference.histogram.sum(), 1e-12)

        indicators = [
            self._wasserstein_indicator(current_hist, reference_hist),
            self._jensen_shannon_indicator(current_hist, reference_hist),
            self._moment_indicator(image.statistics, reference),
            self._texture_indicator(image.statistics, reference),
        ]

        # A degenerate input (constant intensity, empty histogram) can make an
        # individual statistic non-finite. Treat that as maximal deviation for
        # that indicator rather than letting NaN propagate into the score and
        # silently disable every downstream comparison.
        for indicator in indicators:
            if not np.isfinite(indicator.normalised):
                indicator.normalised = 1.0
                indicator.value = 0.0

        score = float(
            np.clip(sum(i.normalised * i.weight for i in indicators), 0.0, 1.0)
        )
        status = self._status(score, thresholds)

        return DistributionShiftResult(
            method=self.id,
            status_label=ImplementationStatus.BASELINE,
            shift_score=score,
            status=status,
            reference_domain=reference.name,
            current_domain=self._describe_current(image),
            thresholds=thresholds,
            indicators=indicators,
            reference_distribution=DistributionSummary(
                name=reference.name,
                mean=reference.mean,
                std=reference.std,
                skewness=reference.skewness,
                kurtosis=reference.kurtosis,
                entropy=reference.entropy,
                histogram=np.round(reference_hist, 6).tolist(),
                bin_edges=np.round(np.linspace(0, 1, reference_hist.size + 1), 4).tolist(),
            ),
            current_distribution=DistributionSummary(
                name="uploaded_image",
                mean=image.statistics.get("mean", 0.0),
                std=image.statistics.get("std", 0.0),
                skewness=image.statistics.get("skewness", 0.0),
                kurtosis=image.statistics.get("kurtosis", 0.0),
                entropy=image.statistics.get("entropy", 0.0),
                histogram=np.round(current_hist, 6).tolist(),
                bin_edges=np.round(np.linspace(0, 1, current_hist.size + 1), 4).tolist(),
            ),
            # Coupling heuristics, NOT measured degradation. Reported so the
            # risk engine has a shift-aware term; labelled everywhere as such.
            confidence_degradation=round(float(np.clip(score * 0.6, 0.0, 1.0)), 4),
            uncertainty_increase=round(float(np.clip(score * 0.75, 0.0, 1.0)), 4),
            provenance=DataProvenance.COMPUTED,
            notes=[
                "Input-space intensity statistics only; semantic shift that "
                "preserves intensity statistics will not be detected.",
                "confidence_degradation and uncertainty_increase are heuristic "
                "couplings of the shift score, NOT measured performance drops.",
                f"Reference domain '{reference.name}': {reference.description}",
            ],
        )

    # -------------------------------------------------------- indicators ---
    def _wasserstein_indicator(
        self, current: np.ndarray, reference: np.ndarray
    ) -> ShiftIndicator:
        from scipy.stats import wasserstein_distance

        centres = np.linspace(0.0, 1.0, current.size)
        distance = float(
            wasserstein_distance(centres, centres, u_weights=current, v_weights=reference)
        )
        return ShiftIndicator(
            id="wasserstein",
            name="Intensity Wasserstein-1 distance",
            value=round(distance, 6),
            normalised=round(min(distance / self.SCALES["wasserstein"], 1.0), 4),
            weight=self.WEIGHTS["wasserstein"],
            description=(
                "Earth-mover distance between the test and reference intensity "
                "histograms; measures how far intensity mass has moved."
            ),
        )

    def _jensen_shannon_indicator(
        self, current: np.ndarray, reference: np.ndarray
    ) -> ShiftIndicator:
        from scipy.spatial.distance import jensenshannon

        # jensenshannon returns the *distance* (sqrt of divergence), already in
        # [0, 1] for base-2, so it needs no ad-hoc rescaling.
        distance = float(jensenshannon(current, reference, base=2.0))
        if not np.isfinite(distance):
            distance = 1.0
        return ShiftIndicator(
            id="jensen_shannon",
            name="Jensen-Shannon distance",
            value=round(distance, 6),
            normalised=round(min(max(distance, 0.0), 1.0), 4),
            weight=self.WEIGHTS["jensen_shannon"],
            description=(
                "Bounded, symmetric divergence between the histograms; sensitive "
                "to new intensity modes that leave the mean unchanged."
            ),
        )

    def _moment_indicator(
        self, statistics: dict[str, float], reference: ReferenceProfile
    ) -> ShiftIndicator:
        keys = ("mean", "std", "skewness", "kurtosis")
        deviations = []
        for key in keys:
            spread = max(reference.dispersion.get(key, 0.0), 1e-3)
            deviations.append(
                abs(statistics.get(key, 0.0) - getattr(reference, key)) / spread
            )
        total = float(np.mean(deviations))
        return ShiftIndicator(
            id="moments",
            name="Standardised moment deviation",
            value=round(total, 4),
            normalised=round(min(total / self.SCALES["moments"], 1.0), 4),
            weight=self.WEIGHTS["moments"],
            description=(
                "Mean/std/skew/kurtosis distance from the reference, expressed "
                "in units of the reference set's own dispersion."
            ),
        )

    def _texture_indicator(
        self, statistics: dict[str, float], reference: ReferenceProfile
    ) -> ShiftIndicator:
        deviation = abs(statistics.get("edge_density", 0.0) - reference.edge_density)
        return ShiftIndicator(
            id="texture",
            name="Edge-density deviation",
            value=round(deviation, 6),
            normalised=round(min(deviation / self.SCALES["texture"], 1.0), 4),
            weight=self.WEIGHTS["texture"],
            description=(
                "Mean gradient-magnitude difference; a proxy for resolution and "
                "reconstruction-kernel differences between scanners."
            ),
        )

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _status(score: float, thresholds: dict[str, float]) -> ShiftStatus:
        if score < thresholds["mild"]:
            return ShiftStatus.ID
        if score < thresholds["moderate"]:
            return ShiftStatus.MILD_SHIFT
        if score < thresholds["severe"]:
            return ShiftStatus.MODERATE_SHIFT
        return ShiftStatus.SEVERE_SHIFT

    @staticmethod
    def _describe_current(image: PreprocessedImage) -> str:
        if image.is_dicom:
            modality = image.tags.get("Modality", "unknown modality")
            return f"uploaded DICOM ({modality})"
        return f"uploaded {image.image_format} ({image.original_channels}ch)"


def build_detector(settings: Settings) -> CompositeIntensityShiftDetector:
    del settings  # detector is stateless; thresholds are passed per call
    return CompositeIntensityShiftDetector()
