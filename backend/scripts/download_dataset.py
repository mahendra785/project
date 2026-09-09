"""Download and extract a registered medical imaging dataset.

    python scripts/download_dataset.py                    # list what is available
    python scripts/download_dataset.py msd-hippocampus    # fetch one (~28 MB)
    python scripts/download_dataset.py msd-heart --yes    # skip the prompt

The licence and citation are printed before anything is written. Datasets
requiring a signed data-use agreement are listed but never auto-downloaded —
accepting those terms is the user's decision, not this script's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging  # noqa: E402
from app.datasets.download import ensure_dataset, is_available, licence_banner  # noqa: E402
from app.datasets.registry import DEFAULT_DATASET_ID, get_spec, list_specs  # noqa: E402


def print_catalogue() -> None:
    print("\nRegistered datasets\n" + "=" * 96)
    print(f"{'ID':<20} {'MODALITY':<10} {'SIZE':>9}  {'STATUS':<16} TARGET")
    print("-" * 96)
    for spec in list_specs():
        if is_available(spec.id):
            status = "downloaded"
        elif spec.requires_registration:
            status = "needs signup"
        else:
            status = "available"
        size = f"{spec.approx_mb:.0f} MB" if spec.approx_bytes else "—"
        print(
            f"{spec.id:<20} {spec.modality.value:<10} {size:>9}  {status:<16} {spec.target[:38]}"
        )
    print("-" * 96)
    print(f"Default: {DEFAULT_DATASET_ID}")
    print("\nFetch one with:  python scripts/download_dataset.py <ID>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", help="Dataset id (omit to list)")
    parser.add_argument("--yes", action="store_true", help="Accept the licence non-interactively")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract")
    args = parser.parse_args()

    configure_logging("INFO")

    if not args.dataset:
        print_catalogue()
        return

    try:
        spec = get_spec(args.dataset)
    except KeyError as exc:
        print(f"error: {exc}")
        print_catalogue()
        raise SystemExit(1) from exc

    print(licence_banner(spec))

    if spec.requires_registration:
        print(
            "This dataset requires you to register and accept a data-use agreement.\n"
            f"Obtain it from: {spec.homepage}\n"
            "Then extract it under data/datasets/"
            f"{spec.id}/ and PUMS will pick it up."
        )
        raise SystemExit(1)

    if not args.yes:
        answer = input("Accept the licence and download? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            raise SystemExit(1)

    root = ensure_dataset(args.dataset, force=args.force)
    print(f"\nReady: {root}")
    print("\nNext steps:")
    print(f"  python scripts/train_pixel_model.py --dataset {spec.id}")
    print(f"  python scripts/export_samples.py --dataset {spec.id}")


if __name__ == "__main__":
    main()
