# PUMS — API reference

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger) · `/redoc`
OpenAPI schema: `http://localhost:8000/openapi.json`

All payloads are `snake_case` JSON. Dense arrays are **never** sent as JSON
floats — see [Dense data transport](#dense-data-transport).

---

## Conventions

### Error format

Every error returns the same shape:

```json
{ "code": "invalid_image", "message": "File is not a recognised image format", "detail": {} }
```

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `task_not_supported` | The model has no adapter for the requested task |
| 404 | `model_not_found` / `analysis_not_found` | Unknown id |
| 413 | `payload_too_large` | Upload exceeds `PUMS_MAX_UPLOAD_BYTES` |
| 422 | `invalid_image` | Undecodable file, bad geometry, or malformed `options` |
| 501 | `not_implemented` | A `planned` method was invoked |
| 503 | `model_unavailable` | Adapter exists but its dependencies are missing |
| 500 | `internal_error` | Unhandled — details are logged, not returned |

### Research-integrity fields

These appear throughout and **must be rendered** by any client:

| Field | Values | Meaning |
| --- | --- | --- |
| `status` / `status_label` | `implemented`, `baseline`, `experimental`, `planned` | Maturity of the producing component |
| `provenance` | `computed`, `synthetic`, `mock` | Where the number came from |
| `assumptions` | `string[]` | Conditions the result's validity depends on |
| `validity_warnings` | `string[]` | Detected reasons not to trust the result |
| `disclaimer` | `string` | Non-diagnostic notice; travels with every risk payload |

### Coordinates

All masks, bounding boxes and uncertainty maps are in **inference-resolution**
pixels (default 256×256), never original-image pixels. `metadata.image` carries
both geometries.

---

## System

### `GET /health`

Liveness plus a capability report, so a client can tell *why* an adapter is
unavailable rather than discovering it at request time.

```json
{
  "status": "ok",
  "version": "0.4.0",
  "python": "3.13.0",
  "capabilities": { "torch": false, "pydicom": true, "opencv": true, "scikit_learn": true },
  "available_models": ["intensity-mixture-seg-v1", "region-proposal-det-v1"],
  "unavailable_models": { "deeplabv3-resnet50-voc": "Missing optional dependencies: torch, torchvision" },
  "warmup": { "status": "ready", "duration_s": 14.8, "error": null }
}
```

`warmup.status` is `cold` → `queued` → `warming` → `ready`. The synthetic
calibration set and reference profile are built on a background thread at
startup; until it reports `ready`, the first analysis is slower.

### `GET /`

Service banner with the non-medical-device disclaimer.

---

## Models

### `GET /api/models`

```json
{
  "models": [
    {
      "id": "intensity-mixture-seg-v1",
      "name": "Bayesian Intensity Mixture Segmenter",
      "model_type": "classical-probabilistic",
      "tasks": ["segmentation"],
      "status": "baseline",
      "available": true,
      "unavailable_reason": null,
      "uncertainty_method": "normalised_predictive_entropy",
      "calibration_methods": ["conformal_risk_control", "temperature_scaling", "baseline_threshold"],
      "ood_detector": "composite_intensity_shift_v1",
      "trained_on": "Nothing — fitted per image at inference time (unsupervised).",
      "is_medical": false,
      "class_labels": ["Background", "Low-intensity stratum", "Mid-intensity stratum", "High-intensity stratum"]
    }
  ],
  "default_model_id": "intensity-mixture-seg-v1"
}
```

`trained_on` and `is_medical` are mandatory and must be surfaced in any UI. No
adapter shipped in this phase is a validated medical model.

### `GET /api/models/{model_id}` → one `ModelInfo`

---

## Analysis

### `POST /api/analyze`

`multipart/form-data`:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `file` | binary | yes | PNG, JPEG, BMP, TIFF or DICOM |
| `options` | string | no | JSON-encoded `AnalysisOptions` |

`options` is a JSON *string* in a form field so the whole request stays one
round trip alongside the file.

```jsonc
{
  "model_id": null,                              // null → server default
  "task": "segmentation",                        // segmentation | detection | classification
  "calibration_method": "conformal_risk_control",
  "target_risk": 0.10,                           // α ∈ [0.001, 0.5]
  "uncertainty_threshold": 0.55,                 // ∈ [0, 1]
  "reference_domain": "synthetic_reference_v1",
  "apply_calibration": true,
  "detect_shift": true
}
```

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@data/sample/phantom_id_1.png;type=image/png" \
  -F 'options={"task":"segmentation","target_risk":0.15}'
```

#### Response (abridged)

```jsonc
{
  "analysis_id": "pums_75c9b7db3482",
  "status": "completed",

  "metadata": {
    "created_at": "2026-09-09T09:12:44.812Z",
    "duration_ms": 358.0,
    "model_id": "intensity-mixture-seg-v1",
    "task": "segmentation",
    "options": { },
    "image": {
      "filename": "phantom_id_1.png",
      "original_width": 256, "original_height": 256,
      "working_width": 256,  "working_height": 256,
      "format": "PNG", "is_dicom": false,
      "checksum": "sha256…",
      "preview_url": "/api/analyses/pums_75c9b7db3482/media/source.png"
    },
    "pipeline_version": "pums-pipeline-0.4.0",
    "seed": 1337
  },

  "prediction": {
    "task": "segmentation",
    "confidence": 0.6495,                 // mean max-posterior
    "classes": [ { "class_id": 0, "label": "Background", "probability": 0.41 } ],
    "segmentation": {
      "height": 256, "width": 256, "num_classes": 4,
      "rle": [0, 812, 1, 44, …],          // (value, run_length) pairs
      "class_pixel_fraction": [0.42, 0.21, 0.24, 0.13],
      "png_url": "/api/analyses/{id}/media/mask.png"
    },
    "detections": null,                   // populated only for task=detection
    "inference_time_ms": 214.9
  },

  "uncertainty": {
    "method": "normalised_predictive_entropy",
    "status": "implemented",
    "global": {                           // LEVEL 1
      "score": 0.5391,
      "statistics": { "mean": 0.539, "std": 0.21, "median": 0.55, "p95": 0.86, "maximum": 0.99 },
      "high_uncertainty_fraction": 0.41,
      "boundary_uncertainty": 0.71,       // must exceed interior on a working map
      "interior_uncertainty": 0.52,
      "provenance": "computed"
    },
    "objects": [ {                        // LEVEL 2
      "id": "region-003",
      "label": "Uncertain region · Mid-intensity stratum",
      "bbox": { "x": 96, "y": 40, "width": 63, "height": 58 },
      "area_px": 1204,
      "confidence": 0.61, "uncertainty": 0.78, "peak_uncertainty": 0.97,
      "calibrated_risk": 0.24,
      "risk_level": "REVIEW_REQUIRED",
      "source": "high_uncertainty_region"  // or "detection"
    } ],
    "pixel": {                            // LEVEL 3
      "height": 256, "width": 256,
      "encoding": "png-gray-u8-base64",
      "raw_url": "/api/analyses/{id}/media/uncertainty-raw.png",
      "heatmap_url": "/api/analyses/{id}/media/uncertainty-heatmap.png",
      "threshold": 0.55,
      "high_uncertainty_mask_fraction": 0.41,
      "preview_grid": [[0.12, …], …]      // 32×32 block means
    },
    "notes": ["Aleatoric-dominated: single-pass entropy conflates …"]
  },

  "calibration": {
    "method": "conformal_risk_control",
    "status": "implemented",
    "target_risk": 0.15,
    "threshold": 0.126,                   // λ̂ (or fitted T for temperature scaling)
    "empirical_risk": 0.138,
    "coverage": 0.862,
    "expected_calibration_error": 0.0532,
    "brier_score": 0.0571,
    "applied_abstention_rate": 0.061,     // ambiguity band on THIS image
    "applied": true,
    "reliability_bins": [ … ],
    "risk_coverage_curve": [ … ],
    "calibration_set_size": 100,
    "provenance": "synthetic",
    "assumptions": ["Exchangeability: …", "Monotone loss: …", "Marginal, not conditional: …"],
    "validity_warnings": ["Input flagged as MODERATE_SHIFT …"]
  },

  "distribution_shift": {
    "method": "composite_intensity_shift_v1",
    "status_label": "baseline",
    "shift_score": 0.6235,
    "status": "MODERATE_SHIFT",           // ID | MILD_SHIFT | MODERATE_SHIFT | SEVERE_SHIFT
    "reference_domain": "synthetic_reference_v1",
    "thresholds": { "mild": 0.42, "moderate": 0.50, "severe": 0.78 },
    "indicators": [ { "id": "wasserstein", "value": 0.19, "normalised": 0.76, "weight": 0.35 } ],
    "reference_distribution": { "histogram": [ … ], "mean": 0.31 },
    "current_distribution":   { "histogram": [ … ], "mean": 0.47 },
    "confidence_degradation": 0.374,      // HEURISTIC coupling, not measured
    "uncertainty_increase": 0.468,        // HEURISTIC coupling, not measured
    "provenance": "computed"
  },

  "risk_assessment": {
    "risk_score": 0.5555,
    "risk_level": "REVIEW_REQUIRED",      // LOW | MODERATE | HIGH | REVIEW_REQUIRED
    "referral_priority": "urgent_review", // routine | elevated | urgent_review
    "review_recommended": true,
    "reasons": ["Largest contribution: mean predictive uncertainty (0.216 …)", …],
    "factors": [ {
      "id": "uncertainty", "name": "Mean predictive uncertainty",
      "raw_value": 0.5391, "normalised": 0.5391, "weight": 0.40, "contribution": 0.2156,
      "rationale": "Mean normalised predictive entropy over all pixels. …"
    } ],
    "weights":    { "uncertainty": 0.4, "confidence": 0.25, "distribution_shift": 0.2, "calibration": 0.15 },
    "thresholds": { "low": 0.25, "moderate": 0.50, "high": 0.72 },
    "triggered_overrides": ["region_peak_uncertainty>=0.85 (observed 0.97)"],
    "disclaimer": "Research decision-support signal. Not a diagnostic result. …"
  },

  "warnings": ["'Bayesian Intensity Mixture Segmenter' is not a validated medical model …"]
}
```

`risk_score` is exactly `Σ factors[i].contribution` — verified by test. Nothing
is hidden.

### Retrieval

| Endpoint | Returns |
| --- | --- |
| `GET /api/analyses?limit=&offset=` | `{ items: AnalysisSummary[], total }` |
| `GET /api/analyses/{id}` | The full `AnalysisResponse` |
| `GET /api/analyses/{id}/uncertainty` | `HierarchicalUncertainty` |
| `GET /api/analyses/{id}/calibration` | `CalibrationResult \| null` |
| `GET /api/analyses/{id}/shift` | `DistributionShiftResult \| null` |
| `GET /api/analyses/{id}/risk` | `RiskAssessment` |
| `DELETE /api/analyses/{id}` | `204` |

`calibration` and `distribution_shift` are `null` when the stage was disabled or
failed. Clients must handle that rather than assume presence.

---

## Dense data transport

### `GET /api/analyses/{id}/media/{key}.png`

`key` ∈ `source` · `mask` · `uncertainty-raw` · `uncertainty-heatmap`
(allow-listed, so the path parameter can never read arbitrary files).

| Key | Content |
| --- | --- |
| `source` | Preprocessed grayscale input |
| `mask` | RGBA segmentation overlay, background transparent |
| `uncertainty-raw` | **8-bit grayscale — the numeric map.** `value / 255` recovers the original float |
| `uncertainty-heatmap` | RGBA colour-mapped overlay, alpha scaled by value |

Served with `Cache-Control: public, max-age=86400`.

**Why PNG.** A 256×256 float32 map is ~1.5 MB as JSON and ~15 kB as a PNG. Use
`uncertainty-raw` when you want numbers:

```ts
const blob = await (await fetch(rawUrl)).blob();
const bitmap = await createImageBitmap(blob);
// draw to canvas, then:
values[i] = imageData.data[i * 4] / 255;   // back to [0, 1]
```

Fetching into a Blob (rather than setting `img.src` cross-origin) keeps the
canvas untainted so `getImageData` works.

**Segmentation RLE.** `rle` is flattened `(value, run_length)` pairs, row-major
C order. Run lengths sum to exactly `height × width`.

---

## Calibration

### `GET /api/calibration/methods`

Returns the full catalogue **including planned methods**, each with its status
and assumptions. A client should filter to `implemented`/`baseline` for
selectable options — this is what makes it structurally impossible for the UI to
offer something the backend cannot do.

### `POST /api/calibration/run?model_id=&method=&target_risk=`

Fits a method on the calibration set independently of any image. Returns
`CalibrationResult`. Invoking a `planned` method returns **501**.

---

## Distribution shift

### `GET /api/shift/config`

Detector id, band thresholds, indicator weights, normalisation scales, the
registered reference domains (with their histograms), and an explicit
`limitations` array. The thresholds are configuration and are reported as such.

---

## Evaluation

### `GET /api/evaluation?include_calibration_run=true`

Two clearly separated kinds of output:

1. **Session metrics** — real aggregates over analyses this instance ran.
2. **Calibration-set metrics** — Dice, IoU, AURC and uncertainty–error
   correlation computed on the synthetic phantom set (masks are exact by
   construction), tagged `provenance: "synthetic"`.

Metrics needing annotated clinical data are returned with `value: null` and an
`unavailable_reason`. **That absence is the honest result** — no placeholder
number is invented.

### `GET /api/evaluation/self-test`

Runs each metric implementation against inputs with analytically known answers
(e.g. two 8-px masks overlapping in 4 px must give Dice = 0.5). This
demonstrates the evaluation *code* is correct without inventing dataset results.

### `GET /api/dashboard`

Aggregates over this instance's analyses only. An empty instance returns nulls
and zero counts — there are no seeded figures.

---

## CORS

Configured for `http://localhost:3000` and `http://127.0.0.1:3000` by default;
override with `PUMS_CORS_ORIGINS` (a JSON array). Methods: `GET, POST, DELETE,
OPTIONS`.
