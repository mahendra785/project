"""Derive the distribution-shift band thresholds from measured data.

The ID / MILD / MODERATE / SEVERE boundaries must not be guessed constants.
This script measures the composite shift score on:

  (a) held-out slices from the SAME dataset and split family as the reference
      domain — the genuine in-distribution population; and
  (b) the same slices under controlled, increasing corruption.

It then reports operating points:

    ID ceiling      = p99 of the in-distribution scores
    MODERATE start  = midway between that ceiling and the weakest corruption
    SEVERE start    = just below the strongest corruption

Copy the printed values into `Settings` (or set PUMS_SHIFT_THRESHOLD_*).
Re-run whenever the reference domain or dataset changes — thresholds derived
against one domain say nothing about another.

    python scripts/calibrate_shift_thresholds.py
    python scripts/calibrate_shift_thresholds.py --synthetic   # phantom fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.datasets.download import is_available  # noqa: E402
from app.preprocessing.pipeline import PreprocessedImage, PreprocessingPipeline  # noqa: E402
from app.shift.detector import CompositeIntensityShiftDetector  # noqa: E402
from app.shift.reference import get_reference_profile  # noqa: E402


def _wrap(pipeline: PreprocessingPipeline, gray: np.ndarray) -> PreprocessedImage:
    statistics, histogram = pipeline._summarise(gray)
    return PreprocessedImage(
        gray=gray,
        rgb=np.dstack([gray] * 3),
        display=gray,
        original_height=gray.shape[0],
        original_width=gray.shape[1],
        original_channels=1,
        working_size=gray.shape[0],
        image_format="THRESHOLD-CALIBRATION",
        is_dicom=False,
        checksum="threshold-calibration",
        size_bytes=0,
        statistics=statistics,
        histogram=histogram,
    )


def _corrupt(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Scanner-like corruption: gamma + smooth bias field + noise.

    Chosen over intensity inversion because it is the artefact family a real
    protocol or scanner change actually produces, and it degrades smoothly with
    `strength` instead of collapsing to a constant image at the midpoint.
    """
    from scipy import ndimage

    gamma = 1.0 + 1.4 * strength
    out = np.power(np.clip(image, 0, 1), gamma)

    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width] / max(height, width)
    bias = 1.0 + strength * (0.8 * np.sin(2.2 * xx + 0.6) * np.cos(1.8 * yy + 0.3))
    out = out * ndimage.gaussian_filter(bias, sigma=8)

    out = out + rng.normal(0.0, 0.06 * strength, out.shape)
    return np.clip(out, 0.0, 1.0)


def _in_distribution_images(settings, use_synthetic: bool) -> tuple[list[np.ndarray], str]:
    if not use_synthetic and settings.use_real_dataset and is_available(settings.dataset_id):
        from app.datasets.loaders import load_slices, split_by_case
        from app.datasets.medical import resize_pair

        slices = load_slices(settings.dataset_id, max_cases=140, slices_per_case=2)
        _, calibration, test = split_by_case(slices)
        held_out = calibration + test
        images = [
            resize_pair(item.image, item.mask, settings.inference_size)[0]
            for item in held_out
        ]
        return images, f"{settings.dataset_id} (held-out calibration+test slices)"

    from app.calibration.datasets import _make_phantom

    rng = np.random.default_rng(20260909)
    return [_make_phantom(rng, 128)[0] for _ in range(80)], "synthetic phantoms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Force the phantom fallback")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = PreprocessingPipeline(settings)
    reference = get_reference_profile()
    detector = CompositeIntensityShiftDetector()
    thresholds = settings.shift_thresholds
    rng = np.random.default_rng(20260909)

    def score(gray: np.ndarray) -> float:
        return detector.detect(
            _wrap(pipeline, gray), reference, thresholds=thresholds
        ).shift_score

    images, source = _in_distribution_images(settings, args.synthetic)
    print(f"\nReference domain : {reference.name}  (n={reference.n_samples})")
    print(f"ID population    : {source}  (n={len(images)})\n")

    in_distribution = np.array([score(image) for image in images])
    print("In-distribution scores:")
    for label, value in (
        ("min", in_distribution.min()),
        ("p50", np.percentile(in_distribution, 50)),
        ("p95", np.percentile(in_distribution, 95)),
        ("p99", np.percentile(in_distribution, 99)),
        ("max", in_distribution.max()),
    ):
        print(f"  {label:>4}: {value:.4f}")

    print("\nControlled acquisition corruption (gamma + bias field + noise):")
    corruption_means: dict[float, float] = {}
    sample = images[: min(30, len(images))]
    for strength in (0.25, 0.5, 0.75, 1.0):
        values = np.array([score(_corrupt(image, strength, rng)) for image in sample])
        corruption_means[strength] = float(values.mean())
        print(
            f"  strength {strength:.2f}: mean {values.mean():.4f}  sd {values.std():.4f}"
        )

    # ID ceiling: accept ~99% of genuine in-distribution inputs as ID.
    mild = float(np.percentile(in_distribution, 99))
    # MODERATE starts above ANY observed in-distribution score, midway to the
    # weakest corruption that is reliably separable. Using the ID *maximum*
    # rather than a percentile here matters: the bands must not overlap the
    # observed ID range, or "MODERATE_SHIFT" would fire on healthy inputs.
    moderate = float((in_distribution.max() + corruption_means[0.5]) / 2)
    severe = float(corruption_means[1.0] * 0.95)

    # Guarantee a strictly increasing set of bands even if the populations
    # overlap badly on some future dataset.
    moderate = max(moderate, mild + 0.02)
    severe = max(severe, moderate + 0.02)

    print("\nRecommended thresholds:")
    print(f"  PUMS_SHIFT_THRESHOLD_MILD={mild:.2f}")
    print(f"  PUMS_SHIFT_THRESHOLD_MODERATE={moderate:.2f}")
    print(f"  PUMS_SHIFT_THRESHOLD_SEVERE={severe:.2f}")

    def band(value: float) -> str:
        if value < mild:
            return "ID"
        if value < moderate:
            return "MILD_SHIFT"
        if value < severe:
            return "MODERATE_SHIFT"
        return "SEVERE_SHIFT"

    print("\nWhere each corruption level lands under those thresholds:")
    for strength, mean_score in corruption_means.items():
        print(f"  strength {strength:.2f} (mean {mean_score:.3f}) -> {band(mean_score)}")

    undetectable = [s for s, m in corruption_means.items() if band(m) == "ID"]
    if undetectable:
        print(
            "\n  LIMITATION: corruption at strength "
            + ", ".join(f"{s:.2f}" for s in undetectable)
            + " scores inside the in-distribution range and is therefore NOT\n"
            "  detectable by this detector. Natural inter-patient variation in "
            "this\n  dataset is as large as that much acquisition change. Report "
            "this rather\n  than tightening the threshold until it 'works'."
        )

    print(
        "\nThese calibrate the detector's bands against its own reference domain. "
        "They are operating points, not a measurement of clinical OOD detection "
        "performance — that needs labelled multi-site data."
    )


if __name__ == "__main__":
    main()
