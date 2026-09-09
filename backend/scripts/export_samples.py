"""Export real MRI slices from a downloaded dataset into `data/sample/`.

These replace the procedural phantoms as the images you drop into the Analyze
page, so the demo runs on genuine annotated anatomy.

    python scripts/export_samples.py --dataset msd-hippocampus

Slices are taken from the TEST split, so they are patient-disjoint from
anything the segmenter was trained on — demoing on training data would make
the model look better than it is.

An out-of-distribution variant is also written for each sample. It is a
*labelled simulation* (gamma + bias field + noise, the artefacts a different
scanner or protocol produces), not a real second site, and its filename and
the accompanying README say so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPO_ROOT, get_settings  # noqa: E402
from app.datasets.loaders import load_slices, split_by_case  # noqa: E402
from app.datasets.medical import resize_pair  # noqa: E402
from app.datasets.registry import get_spec  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "data" / "sample"


def simulate_acquisition_shift(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply scanner-like corruption: gamma, a smooth bias field, and noise.

    This is the classic MRI intensity-inhomogeneity model. It is a SIMULATION
    of cross-scanner shift, not a second real site — useful for exercising the
    detector deterministically, but not evidence about real OOD performance.
    """
    from scipy import ndimage

    gamma = 1.9
    shifted = np.power(np.clip(image, 0, 1), gamma)

    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width] / max(height, width)
    bias = 0.65 + 0.7 * np.sin(2.2 * xx + 0.6) * np.cos(1.8 * yy + 0.3)
    shifted = shifted * ndimage.gaussian_filter(bias, sigma=8)

    shifted = shifted + rng.normal(0.0, 0.05, shifted.shape)
    return np.clip(shifted, 0.0, 1.0)


def save(array: np.ndarray, name: str) -> None:
    path = OUTPUT_DIR / name
    Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="msd-hippocampus")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--keep-phantoms",
        action="store_true",
        help="Keep the old synthetic phantom samples alongside the real slices.",
    )
    args = parser.parse_args()

    spec = get_spec(args.dataset)
    settings = get_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.keep_phantoms:
        for stale in OUTPUT_DIR.glob("phantom_*.png"):
            stale.unlink()
            print(f"  removed {stale.name} (synthetic phantom)")

    slices = load_slices(args.dataset, max_cases=140, slices_per_case=1)
    _, _, test = split_by_case(slices)
    if not test:
        raise SystemExit(f"No test-split slices available for '{args.dataset}'")

    rng = np.random.default_rng(20260909)
    manifest = []
    print(f"\nExporting real slices from {spec.name} (test split):")

    for index, item in enumerate(test[: args.count]):
        image, mask = resize_pair(item.image, item.mask, settings.inference_size)

        base = f"{args.dataset}_{index + 1:02d}"
        save(image, f"{base}.png")
        save(mask.astype(float), f"{base}_groundtruth.png")
        save(simulate_acquisition_shift(image, rng), f"{base}_shifted.png")

        manifest.append(
            {
                "sample": f"{base}.png",
                "ground_truth": f"{base}_groundtruth.png",
                "shifted_variant": f"{base}_shifted.png",
                "dataset": spec.id,
                "dataset_name": spec.name,
                "modality": spec.modality.value,
                "target": spec.target,
                "case_id": item.case_id,
                "slice_index": item.slice_index,
                "split": "test",
                "foreground_fraction": round(float(mask.mean()), 4),
                "licence": spec.licence,
                "citation": spec.citation,
            }
        )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "note": (
                    "Real medical images from a patient-disjoint TEST split. "
                    "'_shifted' variants are SIMULATED acquisition shift, not a "
                    "real second site."
                ),
                "licence": spec.licence,
                "licence_url": spec.licence_url,
                "citation": spec.citation,
                "homepage": spec.homepage,
                "samples": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote data/sample/manifest.json ({len(manifest)} samples)")
    print(f"\nLicence: {spec.licence} — cite: {spec.citation}")


if __name__ == "__main__":
    main()
