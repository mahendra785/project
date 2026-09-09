"""Derive the distribution-shift band thresholds from measured data.

The ID / MILD / MODERATE / SEVERE boundaries must not be guessed constants. This
script measures the composite shift score on (a) images drawn from the same
generator as the reference profile and (b) images corrupted by a controlled
amount, then reports operating points:

    ID upper bound        = 95th percentile of known-in-distribution scores
    MODERATE upper bound  = midpoint between mild corruption and severe
    SEVERE lower bound    = below the mean score of a fully inverted image

Copy the printed values into `Settings` (or set the PUMS_SHIFT_THRESHOLD_*
environment variables). Re-run whenever the reference profile changes.

Run:  python scripts/calibrate_shift_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.calibration.datasets import _make_phantom  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.preprocessing.pipeline import PreprocessedImage, PreprocessingPipeline  # noqa: E402
from app.shift.detector import CompositeIntensityShiftDetector  # noqa: E402
from app.shift.reference import get_reference_profile  # noqa: E402

N_SAMPLES = 80


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
        image_format="SYNTHETIC",
        is_dicom=False,
        checksum="threshold-calibration",
        size_bytes=0,
        statistics=statistics,
        histogram=histogram,
    )


def main() -> None:
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

    print(f"Reference domain: {reference.name} (n={reference.n_samples})\n")

    in_distribution = np.array(
        [score(_make_phantom(rng, 128)[0]) for _ in range(N_SAMPLES)]
    )
    print("In-distribution scores (same generator as the reference):")
    for label, value in (
        ("min", in_distribution.min()),
        ("p50", np.percentile(in_distribution, 50)),
        ("p95", np.percentile(in_distribution, 95)),
        ("p99", np.percentile(in_distribution, 99)),
        ("max", in_distribution.max()),
    ):
        print(f"  {label:>4}: {value:.4f}")

    print("\nControlled corruption (alpha-blend toward the intensity inverse):")
    corruption_means = {}
    for strength in (0.25, 0.5, 0.75, 1.0):
        values = np.array(
            [
                score(
                    np.clip(
                        (lambda g: g * (1 - strength) + strength * (1 - g))(
                            _make_phantom(rng, 128)[0]
                        ),
                        0,
                        1,
                    )
                )
                for _ in range(25)
            ]
        )
        corruption_means[strength] = float(values.mean())
        print(f"  strength {strength:.2f}: mean {values.mean():.4f} sd {values.std():.4f}")

    print("\nAdditive-noise corruption:")
    for sigma in (0.05, 0.15, 0.30):
        values = np.array(
            [
                score(
                    np.clip(_make_phantom(rng, 128)[0] + rng.normal(0, sigma, (128, 128)), 0, 1)
                )
                for _ in range(25)
            ]
        )
        print(f"  sigma {sigma:.2f}: mean {values.mean():.4f}")

    # ID ceiling: p99 keeps genuine in-distribution inputs out of the shift
    # bands. MILD/MODERATE split: midway between that ceiling and the weakest
    # corruption we can detect. SEVERE: just below full intensity inversion.
    # `strength 0.5` is excluded — blending an image with its own inverse at
    # 0.5 produces a constant field, a degenerate case rather than a shift.
    mild = float(np.percentile(in_distribution, 99))
    moderate = float((mild + corruption_means[0.25]) / 2)
    severe = float(corruption_means[1.0] * 0.95)

    print("\nRecommended thresholds:")
    print(f"  PUMS_SHIFT_THRESHOLD_MILD={mild:.2f}")
    print(f"  PUMS_SHIFT_THRESHOLD_MODERATE={moderate:.2f}")
    print(f"  PUMS_SHIFT_THRESHOLD_SEVERE={severe:.2f}")
    print(
        "\nThese are operating points on SYNTHETIC phantoms. They calibrate the "
        "detector's bands against its own reference domain and say nothing "
        "about clinical OOD detection performance."
    )


if __name__ == "__main__":
    main()
