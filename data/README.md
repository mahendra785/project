# Data

## Layout

| Path | Contents | In git? |
| --- | --- | --- |
| `raw/` | Downloaded dataset archives | No — gitignored |
| `datasets/` | Extracted volumes and annotations | No — gitignored |
| `sample/` | A handful of real slices exported for the demo | Yes |

## `sample/` — real MRI slices

`sample/` holds **real MRI slices** exported from the held-out TEST split of
the configured dataset, together with their expert masks. `manifest.json`
records the case id, slice index, licence and citation for each one.

| File pattern | Contents |
| --- | --- |
| `<dataset>_NN.png` | A real MRI slice, patient-disjoint from anything the model was trained on. |
| `<dataset>_NN_groundtruth.png` | The expert annotation for that slice. |
| `<dataset>_NN_shifted.png` | **Simulated** acquisition shift (gamma + bias field + noise) — a *model* of cross-scanner change, not a real second site. |

Regenerate with:

```bash
cd backend
python scripts/export_samples.py --dataset msd-hippocampus
```

Demo slices come from the TEST split deliberately: demonstrating on training
data would make the model look better than it is.

## What is *not* here, and why

Dataset archives and extracted volumes are **gitignored and never vendored**.
Medical imaging datasets carry licensing and patient-privacy conditions that
forbid redistribution inside a project repository, and each has a citation
requirement that a silent copy would strip.

Fetch them explicitly — the licence and citation are printed and must be
accepted first:

```bash
python backend/scripts/download_dataset.py                  # list the catalogue
python backend/scripts/download_dataset.py msd-hippocampus  # ~28 MB, CC BY-SA 4.0
```

If no dataset is downloaded, PUMS falls back to procedural phantoms and tags
every derived figure `provenance: "synthetic"`, so a phantom-backed number can
never be mistaken for a measurement on clinical data.

## Adding a real dataset

The two extension points are small and deliberately independent of the
algorithms:

### 1. A real calibration set

Conformal calibration needs `(predicted probability, ground truth)` pairs.

```python
from app.calibration.datasets import (
    CalibrationSample, CalibrationSet, register_calibration_source,
)
from app.schemas.common import DataProvenance

samples = [
    CalibrationSample(
        foreground_probability=model_probability_map,  # (H, W) float in [0, 1]
        ground_truth=annotated_mask,                   # (H, W) bool
    )
    for ... in your_annotated_split
]

register_calibration_source(
    "intensity-mixture-seg-v1:100:1337",   # key format: "{model_id}:{n}:{seed}"
    CalibrationSet(
        samples=samples,
        name="acdc_val",
        provenance=DataProvenance.COMPUTED,
        description="ACDC validation split, expert annotations.",
    ),
)
```

Every number derived from it — λ̂, empirical risk, coverage, ECE, Brier, Dice,
IoU, AURC — becomes a genuine measurement, and the UI's `SYNTHETIC` badges
become `COMPUTED`. **No algorithm changes are required.**

### 2. A real reference domain

```python
from app.shift.reference import build_profile_from_images, set_reference_profile

set_reference_profile(
    build_profile_from_images(
        images,                       # list of (H, W) float arrays in [0, 1]
        name="site_a_ct",
        description="Site A, Siemens scanner, 2024 protocol.",
    )
)
```

Then re-derive the band thresholds against that domain:

```bash
python backend/scripts/calibrate_shift_thresholds.py
```

Until you do, "in-distribution" means *phantom-like* — which is what the UI
says.

## Supported input formats

| Format | Status | Notes |
| --- | --- | --- |
| PNG / JPEG / BMP / TIFF | Supported | Loaded via Pillow; scaled by /255. |
| DICOM (single-frame) | Supported | `pydicom`; rescale slope/intercept applied, robust 1st–99th percentile normalisation. |
| DICOM (multi-frame) | Partial | The middle slice is taken. Full 3-D is a documented extension point, not a claim. |
| NIfTI / NRRD | Not implemented | Add a loader in `backend/app/preprocessing/loaders.py`; nothing else changes. |
| Multi-channel (>4) | Not implemented | Rejected with an explicit error rather than silently mishandled. |
