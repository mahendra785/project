"""Calibration data sources.

Conformal calibration needs (probability, ground-truth) pairs. PUMS ships no
annotated medical data, so the default source is a deterministic **synthetic
phantom** set: procedurally generated images whose foreground masks are known
exactly by construction.

READ THIS BEFORE QUOTING ANY NUMBER FROM IT
-------------------------------------------
The conformal procedure run on this set is the real algorithm, executed
honestly. The *data* is synthetic. Therefore:

  * lambda_hat calibrated here is valid for phantoms, not for patients;
  * the reported empirical risk / ECE describe the model's behaviour on
    phantoms and are NOT medical performance figures;
  * every result derived from it is tagged `provenance="synthetic"` and the
    UI is required to display that tag.

`register_calibration_source()` is the extension point: point it at a real
annotated split and every number becomes a genuine measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger
from app.models.base import BaseMedicalModel
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import DataProvenance

logger = get_logger(__name__)


@dataclass(slots=True)
class CalibrationSample:
    """One calibration point: foreground probability map + true mask."""

    foreground_probability: np.ndarray  # (H, W) in [0, 1]
    ground_truth: np.ndarray  # (H, W) boolean


@dataclass(slots=True)
class CalibrationSet:
    samples: list[CalibrationSample]
    name: str
    provenance: DataProvenance
    description: str

    def __len__(self) -> int:
        return len(self.samples)


def _make_phantom(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate one phantom image and its exact foreground mask.

    Structure: a smooth background field, 1-3 elliptical inclusions at higher
    intensity, blurred edges (so boundaries are genuinely ambiguous, which is
    what makes the calibration exercise non-trivial), and additive noise.
    """
    from scipy import ndimage

    yy, xx = np.mgrid[0:size, 0:size] / size

    # Smooth, low-frequency background so the task is not trivially thresholdable.
    background = 0.25 + 0.15 * np.sin(3.1 * xx + rng.uniform(0, 3)) * np.cos(
        2.7 * yy + rng.uniform(0, 3)
    )

    mask = np.zeros((size, size), dtype=bool)
    image = background.copy()
    for _ in range(int(rng.integers(1, 4))):
        cy, cx = rng.uniform(0.25, 0.75, size=2)
        # Radii are kept above the blur scale below, so inclusions survive
        # smoothing as detectable structures. Otherwise the smallest ellipses
        # are erased entirely and the calibration target becomes unreachable
        # for reasons that say nothing about the uncertainty method.
        ry, rx = rng.uniform(0.11, 0.24, size=2)
        angle = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        ry_rot = dy * np.cos(angle) + dx * np.sin(angle)
        rx_rot = -dy * np.sin(angle) + dx * np.cos(angle)
        ellipse = (ry_rot / ry) ** 2 + (rx_rot / rx) ** 2 <= 1.0
        mask |= ellipse
        image = np.where(ellipse, image + rng.uniform(0.28, 0.50), image)

    # Blur then add noise. The blur is deliberately smaller than the smallest
    # inclusion radius: it must create a genuine partial-volume boundary zone
    # (where uncertainty *should* be high) without destroying the structures.
    image = ndimage.gaussian_filter(image, sigma=rng.uniform(0.8, 1.8))
    image = image + rng.normal(0.0, rng.uniform(0.02, 0.07), size=image.shape)
    return np.clip(image, 0.0, 1.0), mask


def _as_preprocessed(gray: np.ndarray) -> PreprocessedImage:
    """Wrap a raw array in the pipeline's output type without re-encoding it."""
    rgb = np.dstack([gray, gray, gray])
    return PreprocessedImage(
        gray=gray,
        rgb=rgb,
        display=gray,
        original_height=gray.shape[0],
        original_width=gray.shape[1],
        original_channels=1,
        working_size=gray.shape[0],
        image_format="SYNTHETIC",
        is_dicom=False,
        checksum="synthetic",
        size_bytes=0,
    )


def build_synthetic_calibration_set(
    model: BaseMedicalModel, *, n_samples: int = 80, size: int = 96, seed: int = 1337
) -> CalibrationSet:
    """Run the *actual* model over deterministic phantoms to collect scores.

    The probabilities are real model outputs; only the images and masks are
    procedurally generated.
    """
    rng = np.random.default_rng(seed)
    samples: list[CalibrationSample] = []

    for index in range(n_samples):
        gray, mask = _make_phantom(rng, size)
        output = model.predict_segmentation(_as_preprocessed(gray))
        # The adapter declares which of its channels are foreground; the
        # calibration layer never hard-codes a class index, which is what keeps
        # it model-agnostic.
        foreground = output.foreground_probability
        samples.append(
            CalibrationSample(
                foreground_probability=np.clip(foreground, 0.0, 1.0),
                ground_truth=mask,
            )
        )
        if index and index % 25 == 0:
            logger.debug("calibration set: %d/%d phantoms", index, n_samples)

    return CalibrationSet(
        samples=samples,
        name="synthetic_phantom_v1",
        provenance=DataProvenance.SYNTHETIC,
        description=(
            f"{n_samples} procedurally generated {size}x{size} phantoms with exact "
            "foreground masks. Real model probabilities, synthetic ground truth. "
            "NOT a medical dataset; results are not clinical performance."
        ),
    )


#: Registry of calibration sources. Replace/extend to plug in real data.
_SOURCES: dict[str, CalibrationSet] = {}


def register_calibration_source(key: str, calibration_set: CalibrationSet) -> None:
    _SOURCES[key] = calibration_set


def get_calibration_set(
    model: BaseMedicalModel, *, n_samples: int = 80, seed: int = 1337
) -> CalibrationSet:
    """Cached per-model calibration set (built on first use).

    Resolution order:
      1. A source explicitly registered via `register_calibration_source()`.
      2. The configured REAL annotated dataset, if it has been downloaded.
      3. Procedural phantoms, clearly reported as synthetic.

    Falling back is never silent: the provenance tag on the returned set drives
    the badges the UI shows, so a phantom-backed result cannot be mistaken for
    a measurement on clinical data.
    """
    key = f"{model.metadata().id}:{n_samples}:{seed}"
    if key in _SOURCES:
        return _SOURCES[key]

    from app.core.config import get_settings

    settings = get_settings()
    if settings.use_real_dataset:
        try:
            from app.datasets.download import is_available
            from app.datasets.medical import build_calibration_set

            if is_available(settings.dataset_id):
                logger.info(
                    "Building calibration set from real dataset '%s' for %s",
                    settings.dataset_id,
                    model.metadata().id,
                )
                _SOURCES[key] = build_calibration_set(
                    model,
                    settings.dataset_id,
                    max_cases=settings.dataset_max_cases,
                    slices_per_case=settings.dataset_slices_per_case,
                    max_samples=n_samples,
                )
                return _SOURCES[key]
            logger.warning(
                "Dataset '%s' is not downloaded; falling back to synthetic "
                "phantoms. Run: python scripts/download_dataset.py %s",
                settings.dataset_id,
                settings.dataset_id,
            )
        except Exception as exc:  # a dataset problem must not break inference
            logger.warning(
                "Could not build a real calibration set (%s); falling back to "
                "synthetic phantoms.",
                exc,
            )

    logger.info("Building synthetic calibration set for %s", model.metadata().id)
    _SOURCES[key] = build_synthetic_calibration_set(model, n_samples=n_samples, seed=seed)
    return _SOURCES[key]
