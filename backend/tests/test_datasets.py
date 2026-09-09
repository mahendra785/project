"""Dataset layer tests.

Tests that need the actual data skip when it is absent, so a fresh clone still
passes a full `pytest` run without a multi-hundred-megabyte download.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.datasets.download import _is_within, dataset_root, is_available
from app.datasets.loaders import Slice2D, split_by_case
from app.datasets.registry import DATASETS, DatasetFormat, get_spec, list_specs

DATASET_ID = "msd-hippocampus"
needs_data = pytest.mark.skipif(
    not is_available(DATASET_ID),
    reason=f"{DATASET_ID} not downloaded (python scripts/download_dataset.py {DATASET_ID})",
)


# ---------------------------------------------------------------- registry --
def test_every_dataset_declares_licence_and_citation():
    """A result whose data source cannot be cited is not a usable result."""
    for spec in list_specs():
        assert spec.licence, f"{spec.id} has no licence"
        assert spec.citation, f"{spec.id} has no citation"
        assert spec.homepage.startswith("http"), f"{spec.id} has no homepage"
        assert spec.use_case, f"{spec.id} does not state a clinical use case"


def test_registration_gated_datasets_are_not_auto_downloadable():
    """Accepting a data-use agreement must stay a human decision."""
    for spec in list_specs():
        if spec.requires_registration:
            assert not spec.downloadable, f"{spec.id} would bypass its DUA"


def test_unknown_dataset_raises_with_guidance():
    with pytest.raises(KeyError) as info:
        get_spec("not-a-dataset")
    assert "Available" in str(info.value)


def test_msd_datasets_declare_foreground_labels():
    for spec in list_specs():
        if spec.fmt is DatasetFormat.MSD_NIFTI and spec.downloadable:
            assert spec.foreground_labels, f"{spec.id} has no foreground labels"


# ---------------------------------------------------------------- security --
def test_archive_extraction_rejects_path_traversal():
    """A malicious archive must not be able to write outside its directory."""
    base = dataset_root(DATASETS[DATASET_ID])
    assert _is_within(base, base / "imagesTr" / "case.nii.gz")
    assert not _is_within(base, base / ".." / ".." / "evil.txt")
    assert not _is_within(base, base.parent.parent / "evil.txt")


# ------------------------------------------------------------------ splits --
def _fake_slices(n_cases: int = 10, per_case: int = 3) -> list[Slice2D]:
    return [
        Slice2D(
            image=np.zeros((8, 8)),
            mask=np.zeros((8, 8), dtype=bool),
            case_id=f"case_{c:03d}",
            slice_index=s,
            dataset_id="test",
        )
        for c in range(n_cases)
        for s in range(per_case)
    ]


def test_split_is_by_patient_not_by_slice():
    """The single most important guard against inflated metrics.

    Adjacent slices of one patient are near-duplicates. If a case appeared in
    both train and calibration, the conformal coverage would be measured on
    data the model had effectively already seen.
    """
    train, calibration, test = split_by_case(_fake_slices())

    train_cases = {s.case_id for s in train}
    calibration_cases = {s.case_id for s in calibration}
    test_cases = {s.case_id for s in test}

    assert not train_cases & calibration_cases
    assert not train_cases & test_cases
    assert not calibration_cases & test_cases


def test_split_covers_every_slice_exactly_once():
    slices = _fake_slices()
    train, calibration, test = split_by_case(slices)
    assert len(train) + len(calibration) + len(test) == len(slices)


def test_split_is_deterministic():
    a = split_by_case(_fake_slices(), seed=7)
    b = split_by_case(_fake_slices(), seed=7)
    assert [s.case_id for s in a[0]] == [s.case_id for s in b[0]]


# ------------------------------------------------------- real data (skips) --
@needs_data
def test_real_slices_load_with_sane_geometry():
    from app.datasets.loaders import load_slices

    slices = load_slices(DATASET_ID, max_cases=8, slices_per_case=2)
    assert slices, "no slices loaded"

    for item in slices:
        assert item.image.shape == item.mask.shape
        assert 0.0 <= item.image.min() and item.image.max() <= 1.0
        assert item.mask.dtype == bool
        # Slice selection targets annotated slices, so every one must have
        # foreground — otherwise the calibration split is padded with trivial
        # all-background cases that make any method look good.
        assert item.mask.any(), f"{item.case_id} slice {item.slice_index} is empty"


@needs_data
def test_resize_pair_keeps_the_mask_binary():
    from app.datasets.loaders import load_slices
    from app.datasets.medical import resize_pair

    item = load_slices(DATASET_ID, max_cases=2, slices_per_case=1)[0]
    image, mask = resize_pair(item.image, item.mask, 256)

    assert image.shape == (256, 256)
    assert mask.shape == (256, 256)
    assert mask.dtype == bool
    assert set(np.unique(mask)).issubset({True, False})
    assert mask.any(), "resizing must not erase the annotation"


@needs_data
def test_real_calibration_set_is_marked_computed_not_synthetic():
    """The whole point of wiring real data in: provenance must flip."""
    from app.datasets.medical import build_calibration_set
    from app.models.registry import get_registry

    model = get_registry().get("intensity-mixture-seg-v1")
    calibration_set = build_calibration_set(
        model, DATASET_ID, max_cases=6, slices_per_case=1, max_samples=4
    )

    assert calibration_set.provenance.value == "computed"
    assert DATASET_ID in calibration_set.name
    # The description must name the licence so results stay citable.
    assert "CC BY" in calibration_set.description

    for sample in calibration_set.samples:
        assert sample.foreground_probability.shape == sample.ground_truth.shape
        assert 0.0 <= sample.foreground_probability.min()
        assert sample.foreground_probability.max() <= 1.0


@needs_data
def test_real_reference_profile_describes_a_real_domain():
    from app.datasets.medical import build_reference_profile

    profile = build_reference_profile(DATASET_ID, max_cases=6, slices_per_case=1)
    assert profile.n_samples > 0
    assert DATASET_ID in profile.name
    assert "in-distribution" in profile.description.lower()
    assert profile.histogram.sum() > 0
