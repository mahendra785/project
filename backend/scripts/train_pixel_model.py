"""Train the random-forest pixel segmenter on a real annotated dataset.

    python scripts/train_pixel_model.py --dataset msd-hippocampus

Trains on the patient-disjoint TRAIN split only, then reports measured quality
on the held-out TEST split. Those numbers are real measurements on real
anatomy — cite the dataset when quoting them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging  # noqa: E402
from app.datasets.loaders import load_slices, split_by_case  # noqa: E402
from app.datasets.medical import evaluate_on_split, resize_pair  # noqa: E402
from app.datasets.registry import get_spec  # noqa: E402
from app.models.adapters.rf_pixel_segmenter import RandomForestPixelSegmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="msd-hippocampus")
    parser.add_argument("--max-cases", type=int, default=140)
    parser.add_argument("--slices-per-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    spec = get_spec(args.dataset)
    from app.core.config import get_settings

    settings = get_settings()

    print(f"\nDataset : {spec.name}")
    print(f"Target  : {spec.target}")
    print(f"Licence : {spec.licence}\n")

    slices = load_slices(
        args.dataset, max_cases=args.max_cases, slices_per_case=args.slices_per_case
    )
    train, calibration, test = split_by_case(slices)
    print(
        f"Patient-wise split — train {len(train)} slices / "
        f"{len({s.case_id for s in train})} cases, "
        f"calibration {len(calibration)}, test {len(test)}"
    )

    images, masks = [], []
    for item in train:
        image, mask = resize_pair(item.image, item.mask, settings.inference_size)
        images.append(image)
        masks.append(mask)

    model = RandomForestPixelSegmenter(seed=args.seed)
    print("\nTraining random forest ...")
    report = model.train(
        images,
        masks,
        dataset_id=args.dataset,
        n_cases=len({s.case_id for s in train}),
    )

    print(
        f"  pixels sampled  : {report.n_pixels_sampled:,} "
        f"({report.foreground_fraction:.1%} foreground)"
    )
    print(f"  features        : {report.n_features}")
    print(f"  training time   : {report.train_seconds:.1f}s")
    print(f"  OOB accuracy    : {report.oob_accuracy:.4f}")
    print(f"  saved to        : {model.model_path}")

    print("\nTop feature importances:")
    for name, importance in model.feature_importances()[:6]:
        print(f"  {name:<26} {importance:.4f}")

    print("\nMeasured on the held-out TEST split (patient-disjoint):")
    metrics = evaluate_on_split(model, args.dataset, "test")
    for key in (
        "n_slices",
        "n_cases",
        "dice",
        "iou",
        "pixel_accuracy",
        "ece",
        "uncertainty_error_correlation",
        "aurc",
    ):
        value = metrics[key]
        print(f"  {key:<30} {value:.4f}" if value < 1000 else f"  {key:<30} {value:.0f}")

    print(
        "\nThese are real measurements on real annotated MRI. They are NOT a "
        "clinical validation, and this model is a research baseline."
    )


if __name__ == "__main__":
    main()
