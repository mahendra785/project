"""Turn a downloaded medical dataset into 2-D (image, mask) slice pairs.

PUMS's pipeline is 2-D, so volumetric datasets are sliced. Two decisions matter
for honesty of the resulting numbers:

*   **Slice selection is deterministic and stated.** We take the axial slices
    with the largest annotated area per volume, so the calibration set is not
    dominated by empty background slices where every method looks perfect.
    That biases the split toward "the structure is present", which is the
    regime the false-negative-rate loss is about — and it is disclosed rather
    than left implicit.
*   **Train/calibration/test splits are by PATIENT, never by slice.** Slices
    from one volume are highly correlated; splitting on slices would leak the
    patient across the boundary and inflate every reported metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.datasets.download import dataset_root, is_available
from app.datasets.registry import DatasetFormat, DatasetSpec, get_spec

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class Slice2D:
    """One 2-D image and its binary foreground mask, both in canonical form."""

    #: (H, W) float64 in [0, 1].
    image: np.ndarray
    #: (H, W) bool.
    mask: np.ndarray
    #: Identifier of the source volume/patient — used for group-wise splitting.
    case_id: str
    slice_index: int
    dataset_id: str


def _normalise(volume: np.ndarray) -> np.ndarray:
    """Robust percentile scaling to [0, 1].

    Medical intensities have arbitrary, modality-dependent ranges, so a plain
    min-max would let a single bright artefact crush the rest of the histogram.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float64)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi - lo < 1e-9:
        lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        return np.zeros_like(volume, dtype=np.float64)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float64)


def _foreground(labels: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    if spec.foreground_labels:
        return np.isin(labels, spec.foreground_labels)
    return labels > 0


def _load_msd_case(
    image_path: Path, label_path: Path, spec: DatasetSpec, slices_per_case: int
) -> list[Slice2D]:
    import nibabel as nib

    image_volume = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float64)
    label_volume = np.asarray(nib.load(str(label_path)).dataobj)

    # Multi-modal volumes (e.g. BraTS FLAIR/T1/T1c/T2) arrive as (H, W, D, C).
    # FLAIR — channel 0 in the MSD layout — is the sequence in which tumour is
    # most visible, so it is the single channel a 2-D grayscale pipeline uses.
    if image_volume.ndim == 4:
        image_volume = image_volume[..., 0]
    if image_volume.ndim != 3 or label_volume.ndim != 3:
        return []

    image_volume = _normalise(image_volume)
    foreground = _foreground(label_volume, spec)

    # Rank slices by annotated area and take the densest few.
    areas = foreground.sum(axis=(0, 1))
    if not areas.any():
        return []
    order = np.argsort(areas)[::-1]
    chosen = [int(i) for i in order[:slices_per_case] if areas[i] > 0]

    case_id = image_path.name.split(".")[0]
    return [
        Slice2D(
            image=np.ascontiguousarray(image_volume[:, :, index]),
            mask=np.ascontiguousarray(foreground[:, :, index]),
            case_id=case_id,
            slice_index=index,
            dataset_id=spec.id,
        )
        for index in chosen
    ]


def _load_msd(spec: DatasetSpec, max_cases: int, slices_per_case: int) -> list[Slice2D]:
    root = dataset_root(spec)
    images = sorted(root.rglob(f"{spec.images_dir}/*.nii.gz"))
    images = [p for p in images if not p.name.startswith("._")]
    if not images:
        raise FileNotFoundError(f"No NIfTI images under {root}/{spec.images_dir}")

    out: list[Slice2D] = []
    for image_path in images[:max_cases]:
        label_path = image_path.parent.parent / spec.labels_dir / image_path.name
        if not label_path.exists():
            continue
        try:
            out.extend(_load_msd_case(image_path, label_path, spec, slices_per_case))
        except Exception as exc:  # one bad file must not kill the whole load
            logger.warning("Skipping %s: %s", image_path.name, exc)
    return out


def _load_image_mask_pairs(spec: DatasetSpec, max_cases: int) -> list[Slice2D]:
    """Datasets shipped as paired image/mask directories (Kvasir-SEG, FracAtlas)."""
    from PIL import Image

    root = dataset_root(spec)
    mask_dirs = [
        d
        for d in root.rglob("*")
        if d.is_dir() and d.name.lower() in {"masks", "mask", "annotations"}
    ]
    if not mask_dirs:
        raise FileNotFoundError(f"No mask directory found under {root}")
    mask_dir = mask_dirs[0]
    image_dirs = [
        d
        for d in root.rglob("*")
        if d.is_dir() and d.name.lower() in {"images", "image", "img"}
    ]
    if not image_dirs:
        raise FileNotFoundError(f"No image directory found under {root}")
    image_dir = image_dirs[0]

    out: list[Slice2D] = []
    for mask_path in sorted(mask_dir.iterdir())[:max_cases]:
        if mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        candidates = [
            image_dir / mask_path.name,
            *(image_dir / f"{mask_path.stem}{ext}" for ext in IMAGE_EXTENSIONS),
        ]
        image_path = next((c for c in candidates if c.exists()), None)
        if image_path is None:
            continue

        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("L"), dtype=np.float64) / 255.0
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L")) > 127

        if mask.shape != image.shape or not mask.any():
            continue
        out.append(
            Slice2D(
                image=image,
                mask=mask,
                case_id=mask_path.stem,
                slice_index=0,
                dataset_id=spec.id,
            )
        )
    return out


def load_slices(
    dataset_id: str, *, max_cases: int = 120, slices_per_case: int = 2
) -> list[Slice2D]:
    """Load 2-D slice pairs from a downloaded dataset."""
    spec = get_spec(dataset_id)
    if not is_available(dataset_id):
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' is not downloaded. Run:\n"
            f"    python scripts/download_dataset.py {dataset_id}"
        )

    if spec.fmt is DatasetFormat.MSD_NIFTI:
        slices = _load_msd(spec, max_cases, slices_per_case)
    else:
        slices = _load_image_mask_pairs(spec, max_cases)

    logger.info(
        "Loaded %d slices from %d cases of %s",
        len(slices),
        len({s.case_id for s in slices}),
        dataset_id,
    )
    return slices


def split_by_case(
    slices: list[Slice2D], *, fractions: tuple[float, float, float] = (0.5, 0.3, 0.2), seed: int = 1337
) -> tuple[list[Slice2D], list[Slice2D], list[Slice2D]]:
    """Group-wise train / calibration / test split.

    Splitting by CASE rather than by slice is not a detail: adjacent slices of
    one patient are near-duplicates, so a slice-level split would place the same
    anatomy on both sides of the boundary and inflate every metric reported
    downstream — including the conformal coverage, which would then be measured
    on data the model had effectively already seen.
    """
    cases = sorted({s.case_id for s in slices})
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(n * fractions[0])
    n_calib = int(n * fractions[1])
    groups = {
        "train": set(cases[:n_train]),
        "calibration": set(cases[n_train : n_train + n_calib]),
        "test": set(cases[n_train + n_calib :]),
    }
    return (
        [s for s in slices if s.case_id in groups["train"]],
        [s for s in slices if s.case_id in groups["calibration"]],
        [s for s in slices if s.case_id in groups["test"]],
    )
