"""Build PUMS calibration sets and reference domains from real medical data.

This is the module that changes what PUMS is allowed to claim. Once a real
annotated split is wired in here, λ̂, empirical risk, coverage, ECE, Brier,
Dice, IoU, AURC and the uncertainty-error correlation stop being phantom
statistics and become measurements on real anatomy with expert masks —
`provenance` flips from `synthetic` to `computed` automatically, because it is
read from the calibration set rather than hard-coded.

The conformal guarantee is then marginal over *this dataset's* distribution.
That is a real guarantee with a real scope, and the scope is stated.
"""

from __future__ import annotations

import numpy as np

from app.calibration.datasets import CalibrationSample, CalibrationSet
from app.core.config import get_settings
from app.core.logging import get_logger
from app.datasets.loaders import Slice2D, load_slices, split_by_case
from app.datasets.registry import get_spec
from app.models.base import BaseMedicalModel
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import DataProvenance
from app.shift.base import ReferenceProfile
from app.shift.reference import build_profile_from_images

logger = get_logger(__name__)


def resize_pair(
    image: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resize an (image, mask) pair to the working resolution.

    The image is interpolated; the mask is thresholded back to binary after
    interpolation so it stays a mask rather than becoming a soft edge map.
    """
    import cv2

    interpolation = cv2.INTER_AREA if image.shape[0] > size else cv2.INTER_LINEAR
    resized_image = cv2.resize(
        image.astype(np.float32), (size, size), interpolation=interpolation
    ).astype(np.float64)
    resized_mask = (
        cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        > 0.5
    )
    return np.clip(resized_image, 0.0, 1.0), resized_mask


def as_preprocessed(gray: np.ndarray) -> PreprocessedImage:
    """Wrap an already-normalised array in the pipeline's output type."""
    from app.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(get_settings())
    statistics, histogram = pipeline._summarise(gray)
    return PreprocessedImage(
        gray=gray,
        rgb=np.dstack([gray, gray, gray]),
        display=gray,
        original_height=gray.shape[0],
        original_width=gray.shape[1],
        original_channels=1,
        working_size=gray.shape[0],
        image_format="DATASET",
        is_dicom=False,
        checksum="dataset",
        size_bytes=0,
        statistics=statistics,
        histogram=histogram,
    )


def build_calibration_set(
    model: BaseMedicalModel,
    dataset_id: str,
    *,
    max_cases: int = 120,
    slices_per_case: int = 2,
    max_samples: int = 120,
) -> CalibrationSet:
    """Run the model over a real held-out calibration split.

    The split is patient-disjoint from whatever the model was trained on, which
    is what makes the resulting conformal threshold meaningful rather than an
    in-sample artefact.
    """
    spec = get_spec(dataset_id)
    settings = get_settings()

    slices = load_slices(dataset_id, max_cases=max_cases, slices_per_case=slices_per_case)
    _, calibration_slices, _ = split_by_case(slices)
    calibration_slices = calibration_slices[:max_samples]
    if not calibration_slices:
        raise ValueError(f"No calibration slices available for '{dataset_id}'")

    samples: list[CalibrationSample] = []
    for item in calibration_slices:
        image, mask = resize_pair(item.image, item.mask, settings.inference_size)
        output = model.predict_segmentation(as_preprocessed(image))
        samples.append(
            CalibrationSample(
                foreground_probability=np.clip(output.foreground_probability, 0.0, 1.0),
                ground_truth=mask,
            )
        )

    n_cases = len({s.case_id for s in calibration_slices})
    return CalibrationSet(
        samples=samples,
        name=f"{dataset_id}:calibration",
        # Real images, real expert masks, real model outputs.
        provenance=DataProvenance.COMPUTED,
        description=(
            f"{len(samples)} slices from {n_cases} patient-disjoint cases of "
            f"{spec.name} ({spec.modality.value}, target: {spec.target}). "
            f"Expert annotations. Licence: {spec.licence}. "
            "Conformal guarantees are marginal over this dataset's distribution."
        ),
    )


def build_reference_profile(
    dataset_id: str, *, max_cases: int = 80, slices_per_case: int = 2
) -> ReferenceProfile:
    """Define 'in-distribution' as this dataset's real acquisition domain."""
    spec = get_spec(dataset_id)
    settings = get_settings()

    slices = load_slices(dataset_id, max_cases=max_cases, slices_per_case=slices_per_case)
    train_slices, _, _ = split_by_case(slices)
    images = [
        resize_pair(item.image, item.mask, settings.inference_size)[0]
        for item in train_slices
    ]
    if not images:
        raise ValueError(f"No reference slices available for '{dataset_id}'")

    return build_profile_from_images(
        images,
        name=f"{dataset_id}:reference",
        description=(
            f"{len(images)} slices from the training split of {spec.name} "
            f"({spec.modality.value}). 'In-distribution' means: this modality, "
            "this acquisition protocol, this patient population. A different "
            f"scanner or modality is genuinely out-of-distribution. Licence: {spec.licence}."
        ),
    )


def evaluate_on_split(
    model: BaseMedicalModel, dataset_id: str, split: str = "test", *, max_samples: int = 60
) -> dict[str, float]:
    """Measure real segmentation and uncertainty quality on a held-out split.

    Returns measured values only — nothing here is estimated or assumed.
    """
    from app.calibration.metrics import expected_calibration_error
    from app.evaluation.metrics import (
        area_under_risk_coverage,
        dice_coefficient,
        iou_score,
        uncertainty_error_correlation,
    )

    settings = get_settings()
    slices = load_slices(dataset_id, max_cases=140, slices_per_case=2)
    train, calibration, test = split_by_case(slices)
    chosen = {"train": train, "calibration": calibration, "test": test}[split][:max_samples]

    dice_values, iou_values, correlations = [], [], []
    all_confidence, all_correct, all_error, all_entropy = [], [], [], []

    for item in chosen:
        image, mask = resize_pair(item.image, item.mask, settings.inference_size)
        output = model.predict_segmentation(as_preprocessed(image))
        p_fg = np.clip(output.foreground_probability, 1e-9, 1 - 1e-9)
        predicted = p_fg >= 0.5

        dice_values.append(dice_coefficient(predicted, mask))
        iou_values.append(iou_score(predicted, mask))

        entropy = -(p_fg * np.log(p_fg) + (1 - p_fg) * np.log(1 - p_fg)) / np.log(2)
        error = (predicted != mask).astype(np.float64)
        correlations.append(uncertainty_error_correlation(entropy, error))

        all_confidence.append(np.maximum(p_fg, 1 - p_fg).ravel())
        all_correct.append((predicted == mask).astype(np.float64).ravel())
        all_error.append(error.ravel())
        all_entropy.append(entropy.ravel())

    confidence = np.concatenate(all_confidence)
    correct = np.concatenate(all_correct)
    error = np.concatenate(all_error)
    entropy = np.concatenate(all_entropy)

    return {
        "n_slices": float(len(chosen)),
        "n_cases": float(len({s.case_id for s in chosen})),
        "dice": float(np.mean(dice_values)),
        "iou": float(np.mean(iou_values)),
        "pixel_accuracy": float(correct.mean()),
        "ece": expected_calibration_error(confidence, correct, settings.ece_bins),
        "uncertainty_error_correlation": float(np.mean(correlations)),
        "aurc": area_under_risk_coverage(error, entropy),
    }
