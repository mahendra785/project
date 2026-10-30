# PUMS — Perception Uncertainty Management System

> **Reproducing the journal article** *Lesion-Level Joint Conformal Risk Control for Medical Image Segmentation*:
> see [REPRODUCING.md](REPRODUCING.md).

*A post-hoc hierarchical uncertainty calibration framework for dense medical
vision models under distribution shift.*

**BCSE497J Project-I · B.Tech · Vellore Institute of Technology**
Mahendra Sajjan Choudhary (23BCE0701) · Vansh Bhatiya (23BCE0631)
Faculty guide: Dr. Shashank Mouli Satapathy

> ### ⚠️ Research prototype — not a medical device
>
> PUMS trains and calibrates on **real, openly-licensed annotated medical MRI**
> (Medical Segmentation Decathlon), but it has **no clinical validation** and
> ships **no clinically validated model**. Training on medical data is not the
> same as being fit for clinical use. Its risk outputs describe how reliable the
> *model* is on a given input — never a patient's condition. Do not use any
> output for patient care.

---

## 1. Overview

Modern medical vision models produce confident predictions without reliable
confidence. PUMS attacks that with a **post-hoc, model-agnostic** layer that
sits on top of any dense predictor and produces:

- **statistically calibrated** uncertainty via split Conformal Risk Control,
- **hierarchical** uncertainty at pixel, region and image level,
- **distribution-shift detection** that says when the calibration guarantee has
  lapsed,
- a **transparent risk signal** with no hidden constants.

### What makes this repository unusual

Most student implementations of this topic show a pretty heatmap. This one is
built so that every displayed number can be defended:

| Property | How it is enforced |
| --- | --- |
| Results come from real annotated MRI | Calibration, the reference domain and the segmenter all use patient-disjoint splits of a real dataset; provenance flips to `computed` automatically because it is read from the calibration set, not hard-coded |
| Uncertainty is never fabricated | The map is a pure function of the model's posteriors; a test recomputes it independently and asserts an exact match |
| The UI cannot over-claim | It reads the backend's method registry; `planned` methods return HTTP 501 |
| Provenance is always visible | Every figure carries `computed` / `synthetic` / `mock` and is badged in the UI |
| No magic numbers | Shift thresholds are *derived* by a committed script; the risk score is the exact sum of itemised factors |
| Assumptions travel with results | Conformal outputs carry their exchangeability assumption and warn when shift invalidates it |
| Absence is reported honestly | Metrics needing clinical labels return `null` **with a reason**, not a placeholder |

---

## 2. Architecture

```
IMAGE → PREPROCESSING → MODEL → PREDICTION
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                     PIXEL UQ            GLOBAL UQ
                          └─────────┬─────────┘
                                    ▼
                              CALIBRATION
                                    ▼
                          DISTRIBUTION SHIFT
                                    ▼
                              RISK ENGINE
                                    ▼
                          FASTAPI → NEXT.JS
```

Full detail: [`docs/architecture.md`](docs/architecture.md) ·
API reference: [`docs/api.md`](docs/api.md)

```
pums/
├── backend/                    Python · FastAPI · the research engine
│   ├── app/
│   │   ├── main.py             App, CORS, error handlers
│   │   ├── api/routes/         health · analysis · models · calibration · shift · evaluation
│   │   ├── schemas/            Pydantic v2 — the API contract
│   │   ├── services/           AnalysisService (the single orchestrator), storage, warmup
│   │   ├── models/             BaseMedicalModel, registry, adapters/
│   │   ├── preprocessing/      PreprocessingPipeline, loaders (PNG/JPEG/DICOM)
│   │   ├── uncertainty/        Estimators + hierarchical engine
│   │   ├── calibration/        Conformal risk control, temperature scaling, metrics
│   │   ├── shift/              Composite OOD detector + reference profiles
│   │   ├── risk/               Transparent risk engine
│   │   ├── evaluation/         Dice, IoU, ECE, AURC, selective risk
│   │   └── utils/              RLE, PNG encoding, safe statistics
│   ├── datasets/           Registry, downloader, NIfTI loaders, real splits
│   ├── tests/                  107 pytest tests
│   ├── scripts/                Dataset download, training, sample export, thresholds
│   └── configs/                Annotated reference configuration
├── frontend/                   Next.js 16 · React 19 · TypeScript
│   ├── app/                    9 App-Router routes
│   ├── components/             ui/ · layout/ · viewer/ · charts/ · research/
│   ├── features/               analyze/ · results/
│   ├── hooks/                  TanStack Query hooks
│   ├── lib/                    api/ (client, Zod schemas, endpoints), utils
│   ├── stores/                 Zustand (client state only)
│   ├── types/                  Mirror of the Pydantic contract
│   └── tests/                  27 tests incl. a live contract fixture
├── shared/api-contracts/       How the two sides stay in sync
├── data/sample/                Real MRI slices exported from the test split
├── data/datasets/              Downloaded datasets (gitignored, never vendored)
├── docs/                       architecture.md · api.md
└── scripts/                    setup / dev / test-all (PowerShell + bash)
```

---

## 3. Tech stack

**Backend** — Python 3.11+ · FastAPI · Pydantic v2 · NumPy · SciPy ·
scikit-learn · scikit-image · OpenCV · Pillow · pydicom · nibabel · joblib · pytest
*Optional:* PyTorch · torchvision · MONAI (`requirements-torch.txt`)

**Frontend** — Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS 4 ·
shadcn/ui-style Radix primitives · Lucide · TanStack Query · Zustand · Recharts ·
Zod · React Hook Form

**Infrastructure** — Docker / docker-compose (optional) · `.env` configuration

> **Why PyTorch is optional.** The default segmenter is a random forest over a
> multiscale filter bank — genuinely supervised, trained on real annotated MRI
> in ~10 s on CPU, with no GPU and no pretrained weights. The whole research
> stack therefore runs on any laptop with a ~450 MB install instead of ~2.5 GB.
> Torch is needed only for the optional demonstration CNN adapter, which the
> registry reports as unavailable rather than failing.

---

## 4. Setup

### Prerequisites

- Python **3.11+** (developed on 3.13)
- Node.js **20+** (developed on 24)

### One-command setup

```powershell
# Windows / PowerShell
./scripts/setup.ps1
```

```bash
# macOS / Linux / Git Bash
./scripts/setup.sh
```

### Manual setup

<details>
<summary>Backend</summary>

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# --- Real medical data (recommended) ---
# ~28 MB, CC BY-SA 4.0. The licence and citation are printed and must be
# accepted before anything is downloaded.
python scripts/download_dataset.py msd-hippocampus

# Train the supervised segmenter on the patient-disjoint TRAIN split (~10 s)
python scripts/train_pixel_model.py --dataset msd-hippocampus

# Export real MRI slices from the held-out TEST split as demo samples
python scripts/export_samples.py --dataset msd-hippocampus

# Without the dataset PUMS still runs, on procedural phantoms, and labels
# every derived figure `synthetic`:
# python scripts/make_sample_images.py

# Optional: enable the torchvision CNN adapter (~2 GB)
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
```
</details>

<details>
<summary>Frontend</summary>

```bash
cd frontend
npm install
```
</details>

---

## 5. Running

### Both services

```powershell
./scripts/dev.ps1     # Windows — opens two terminals
```
```bash
./scripts/dev.sh      # macOS / Linux / Git Bash
```

### Individually

```bash
# Backend  → http://localhost:8000  (docs at /docs)
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend → http://localhost:3000
cd frontend
npm run dev
```

### Docker (optional — local development works without it)

```bash
docker compose up --build
```

### First run

The backend builds the calibration set and reference profile from the
downloaded dataset on a background thread at startup (~17 s). `GET /health` reports
`warmup.status`, and the header shows **Warming up** until it is `ready`.
Analyses run before then still work — they are just slower.

### Try it

1. Open <http://localhost:3000> → redirects to the dashboard.
2. Go to **Analyze**, drop in `data/sample/msd-hippocampus_01.png` (a real MRI
   slice from the held-out test split), press **Run analysis**.
3. Repeat with `msd-hippocampus_01_shifted.png` to watch the shift bands and
   risk overrides fire on simulated acquisition shift.
4. Hover the heatmap — the readout shows the **decoded numeric value** at that
   pixel, not a colour guess.

---

## 6. Environment variables

Backend (`backend/.env`, all prefixed `PUMS_`) — see
[`backend/.env.example`](backend/.env.example) for the full annotated list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PUMS_CORS_ORIGINS` | `["http://localhost:3000", …]` | Must include the frontend origin |
| `PUMS_STORAGE_BACKEND` | `json` | `json` persists to disk; `memory` is ephemeral |
| `PUMS_DEFAULT_MODEL_ID` | `intensity-mixture-seg-v1` | Adapter used when none is requested |
| `PUMS_INFERENCE_SIZE` | `256` | Working resolution for all maps and boxes |
| `PUMS_DEFAULT_TARGET_RISK` | `0.10` | Conformal α |
| `PUMS_SHIFT_THRESHOLD_{MILD,MODERATE,SEVERE}` | `0.42 / 0.50 / 0.78` | Shift bands (derived, see §8) |
| `PUMS_RISK_WEIGHT_*` | `0.40 / 0.25 / 0.20 / 0.15` | Risk factor weights (sum to 1.0) |

Frontend (`frontend/.env.local`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend base URL (inlined at build time) |

---

## 7. API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness, capabilities, warmup state |
| `GET` | `/api/models` | Adapter registry with provenance |
| `POST` | `/api/analyze` | **Run the full pipeline on one image** |
| `GET` | `/api/analyses` | List analyses |
| `GET` | `/api/analyses/{id}` | Full result |
| `GET` | `/api/analyses/{id}/{uncertainty\|calibration\|shift\|risk}` | Sub-resources |
| `GET` | `/api/analyses/{id}/media/{key}.png` | Dense maps as PNG |
| `GET` | `/api/calibration/methods` | Method catalogue with status |
| `POST` | `/api/calibration/run` | Fit a method standalone |
| `GET` | `/api/shift/config` | Detector config and limitations |
| `GET` | `/api/evaluation` | Metrics + honest unavailability |
| `GET` | `/api/evaluation/self-test` | Verify metric implementations |
| `GET` | `/api/dashboard` | Session aggregates |

Full reference with payloads: [`docs/api.md`](docs/api.md).

---

## 8. Research implementation status

| Objective | Status | What actually exists |
| --- | --- | --- |
| Statistically calibrated uncertainty | ✅ **Implemented** | Split Conformal Risk Control with the finite-sample `+B/(n+1)` correction (Angelopoulos et al., ICLR 2024), fitted on a **real patient-disjoint calibration split**. Tests verify loss monotonicity, minimal-λ selection, and that the correction is strictly more conservative than the uncorrected baseline. |
| Real annotated medical data | ✅ **Implemented** | Dataset registry with licences and citations, licence-gated downloader, NIfTI loaders, and **patient-wise** train/calibration/test splits. Provenance flips from `synthetic` to `computed` automatically. |
| Supervised model on real anatomy | 🟡 **Baseline** | Random-forest pixel classifier over a multiscale filter bank, trained on MSD Hippocampus. Measured on the held-out test split: **Dice 0.524, IoU 0.361, ECE 0.111, uncertainty–error correlation 0.512**. Modest, and well below deep-learning results on this task. |
| Hierarchical pixel + object uncertainty | ✅ **Implemented** | One dense field; Levels 2 and 1 are strict aggregations of it, so cross-level consistency is structural. |
| Localised uncertainty maps | ✅ **Implemented** | Per-pixel normalised entropy, PNG-transported, decoded back to a `Float32Array` in the browser. Verified end-to-end by a round-trip test. |
| Lightweight post-hoc estimation | ✅ **Implemented** | Single forward pass. No ensembling, MC dropout, or retraining. ~0.4 s end-to-end at 256×256 after warmup. |
| Calibration robustness (ECE, reliability) | ✅ **Implemented** | ECE, Brier, reliability diagrams, risk–coverage; plus temperature scaling as a second real method. Measured ECE on real MRI is **0.111**, i.e. genuine miscalibration for the calibration layer to correct. |
| Distribution-shift awareness | 🟡 **Baseline** | Composite input-space detector with band thresholds **derived from real held-out MRI** (`0.38 / 0.48 / 0.73`). It *detects* shift and invalidates the guarantee — it does not yet *restore* validity. Measured limitation: mild corruption is **not** separable from inter-patient variation. |
| Standardised evaluation | 🟡 **Baseline** | Dice, IoU, ECE, Brier, AURC, selective risk, uncertainty–error correlation — unit-tested against known answers **and** measured on a real dataset. Missing: evaluation across multiple benchmarks. |
| Object-level uncertainty | 🟡 **Baseline** | Region-derived by default; a segmentation-derived proposal adapter exists. No trained detector. |
| Clinical decision support | 🟠 **Experimental** | Transparent weighted risk engine with override rules, referral priority and triage queue. Not clinically validated. |
| Foundation-model reliability | ⚪ **Planned** | Model-agnostic interface is in place (proven by a second CNN adapter). No SAM/SAM2 work. |
| Cross-modality calibration robustness | ⚪ **Planned** | One real domain (mono-modal MRI) is wired in. `msd-spleen` (CT) is registered specifically to make cross-modality shift measurable, but has not been run. |

Live version of this table: the **Research status** page in the app.

### Methods implemented vs. catalogued

| Method | Status |
| --- | --- |
| Conformal Risk Control (split, FNR loss) | ✅ Implemented |
| Temperature scaling | ✅ Implemented |
| Empirical threshold (no-guarantee control) | 🟡 Baseline |
| Lesion-aware joint CRC (`lesion_aware_crc`): pixel + lesion + small-lesion risk, lesion-equalized sets | ✅ Implemented (Project-II) |
| Morphological prediction sets, joint risks (`morphological_conformal`) | 🟡 Baseline |
| Automatically Adaptive CRC · SACP · Localized Adaptive Risk Control · Evidential (DEviS) | ⚪ Planned — registry entries that return **501** |

---

## 8b. Project-II research: lesion-level joint conformal risk control

**Question.** CRC guarantees pixel FNR ≤ α. Does that protect *lesions*, and
can lesion-level risk be controlled jointly and efficiently?

**Contributions.**
1. Lesion-miss and small-lesion-miss losses on ground-truth components. They are
   monotone under any nested set family, and are controlled jointly with pixel
   FNR with no Bonferroni split.
2. *Lesion-equalized* prediction sets: each candidate lesion is peak-normalised.
   The sets are amplitude-invariant, exclude background above the candidate
   level, and keep the guarantee when tuned on a separate split.

**Pre-registered results** (FracAtlas + Kvasir-SEG, CPU random-forest backbone,
3 seeds × 1000 splits; full tables in
[`docs/research/results/summary.md`](docs/research/results/summary.md)):

| Finding | Outcome |
|---|---|
| Every method met every guarantee it claims | ✅ |
| Pixel CRC under-protects multi-lesion images | Kvasir **PROCEED** (miss rate 1.56α at α = 0.10); FracAtlas inconclusive |
| Cost of joint lesion-level control over pixel-only CRC | ≤ 0.7% set size |
| Joint thresholding vs Learn-Then-Test / morphological sets | 10–46% / 1–43% smaller sets |
| Lesion-equalized sets beat joint thresholding on real data | ❌ not supported (tie) |
| Simulation: lesion-equalized gain when small lesions are under-confident | up to −47%; ≈ 0 when lesions sink into the background (pre-registered rule: failed) |

**Where things live.** The pre-registration is in
[`docs/research/preregistration-mgcrc.md`](docs/research/preregistration-mgcrc.md),
the theory in [`docs/research/theory-memo-v2.md`](docs/research/theory-memo-v2.md),
and the related work in [`docs/research/related-work.md`](docs/research/related-work.md).
The paper source is in [`paper/`](paper/) (LaTeX, LNCS; numbers are generated in
`paper/generated/`). The VIT report and slides are in [`report/`](report/), and
the dashboard is at `/research` → *Lesion-level CRC*.

**Reproduce.** `./scripts/reproduce.ps1` (or `.sh`) runs everything on CPU, and
`-Fast` reuses the cached posteriors. A GPU U-Net backbone is available with
`-Backbone unet-r34 -Size 512` once PyTorch is installed. The 3-D MSD Task08
pipeline is `backend/scripts/experiments/train_backbone_msd.py`. Both are
implemented, but neither has been run yet.

---

## 9. Testing

```bash
./scripts/test-all.sh            # everything

cd backend  && pytest            # 107 tests
cd frontend && npm test          # 27 tests
cd frontend && npm run typecheck
```

Highlights:

- `test_uncertainty_derives_from_real_probabilities` — the integrity guarantee.
- `test_finite_sample_correction_is_conservative` — the conformal correction
  demonstrably does what the theorem says.
- `test_loss_curve_is_monotone_non_increasing` — the CRC precondition is
  verified, not assumed.
- `test_risk_score_equals_sum_of_declared_contributions` — no hidden terms.
- `test_uncertainty_png_round_trips_to_the_reported_mean` — the transported
  image really is the data.
- `contract.test.mts` — a **real captured backend response** validated against
  the frontend Zod schemas, catching cross-language drift.
- `test_split_is_by_patient_not_by_slice` — guards the single easiest way to
  inflate every metric in this project.
- `test_real_calibration_set_is_marked_computed_not_synthetic` — the provenance
  flip is asserted, not assumed.
- `test_archive_extraction_rejects_path_traversal` — a malicious dataset archive
  cannot write outside its directory.

Dataset-dependent tests **skip** when the data is not downloaded, so a fresh
clone passes a full run without a download.

---

## 9b. Datasets

PUMS **vendors no data**. Each dataset is registered with its licence and
citation, both printed before any download and surfaced in the API and UI.

| ID | Modality | Target | Size | Licence | Use case |
| --- | --- | --- | --- | --- | --- |
| `msd-hippocampus` *(default)* | MRI | Hippocampus head/body | 28 MB | CC BY-SA 4.0 | Hippocampal volumetry — an Alzheimer's / epilepsy biomarker |
| `msd-heart` | MRI | Left atrium | 435 MB | CC BY-SA 4.0 | AF ablation planning; 320×320 full-FOV slices |
| `msd-brain-tumour` | MRI | Glioma sub-regions | 7.3 GB | CC BY-SA 4.0 | Neuro-oncology; the flagship medical UQ benchmark |
| `msd-spleen` | CT | Spleen | 1.5 GB | CC BY-SA 4.0 | **Cross-modality shift partner** for the MRI domains |
| `fracatlas` | X-ray | Fracture site | 744 MB | CC BY 4.0 | **ED fracture triage** — clearest value for a calibrated escalate signal |
| `kvasir-seg` | Endoscopy | Colorectal polyp | 44 MB | CC BY 4.0 | Colonoscopy screening; missed-polyp rate *is* the FNR loss |
| `brats`, `mura` | MRI / X-ray | — | — | DUA required | Listed, never auto-downloaded |

```bash
python scripts/download_dataset.py                  # list the catalogue
python scripts/download_dataset.py msd-hippocampus  # licence shown, must be accepted
python scripts/train_pixel_model.py --dataset msd-hippocampus
python scripts/export_samples.py --dataset msd-hippocampus
python scripts/calibrate_shift_thresholds.py        # re-derive the shift bands
```

Switch datasets with `PUMS_DATASET_ID=msd-heart`, or disable real data entirely
with `PUMS_USE_REAL_DATASET=false` (results then report `synthetic`).

**Two methodological choices worth knowing:**

- **Splits are patient-disjoint, never slice-level.** Adjacent slices of one
  patient are near-duplicates; a slice split would put the same anatomy on both
  sides of the boundary and inflate every metric — including the conformal
  coverage, which would then be measured on data the model had already seen.
- **The densest annotated slices are selected per volume.** This avoids a
  calibration set dominated by empty background slices where every method looks
  perfect. It biases the split toward "the structure is present", which is the
  regime the false-negative-rate loss is about — stated rather than left
  implicit.

Registration-gated datasets (BraTS, MURA) are listed with their homepages but
are never fetched automatically: accepting a data-use agreement is your
decision, not the code's.

---

## 10. Known limitations

1. **No clinical validation.** The default adapter *is* trained on real
   annotated MRI, but training on medical data is not the same as being
   validated for clinical use. Every adapter reports
   `clinically_validated: false`.
2. **A modest baseline model.** Dice 0.524 on hippocampus MRI is a real but
   weak result — a random forest over filter responses has no anatomical prior
   and makes spatially incoherent errors a U-Net would not. It is a defensible
   research baseline, not a deployable segmenter.
3. **One dataset, one modality.** All measurements come from MSD Hippocampus
   (mono-modal MRI, 260 volumes). Nothing here demonstrates generalisation to
   other scanners, sites, modalities or populations.
4. **The `_shifted` samples are simulated.** Gamma + bias field + noise is a
   *model* of cross-scanner change, not a real second site. No genuine
   multi-site OOD evaluation has been done.
5. **Mild shift is measurably undetectable.** Corruption at strength 0.25
   scores 0.328, inside the in-distribution range (max 0.410). Natural
   inter-patient variation is as large as that much protocol change. Reported
   rather than hidden by tightening the threshold.
6. **Conformal guarantees are marginal and assume exchangeability**, which
   distribution shift violates. PUMS detects this and emits a validity warning —
   but restoring validity under shift is **not yet implemented**, and that is the
   project's principal remaining gap.
7. **Shift detection is input-space only.** Semantic shift that preserves
   intensity statistics will not be caught. `confidence_degradation` and
   `uncertainty_increase` are heuristic couplings, **not** measured drops.
8. **Per-region "calibrated risk" is a combination, not a guarantee.**
   Region-conditional control needs localized risk control (planned).
9. **Single-pass entropy is aleatoric-dominated** and does not isolate epistemic
   uncertainty.
10. **2-D only.** Volumes are sliced; the densest annotated slices are taken,
    which biases the split toward "structure present" (disclosed, and the
    regime the FNR loss is about). Multi-frame DICOM takes the middle slice.
11. **No database, no authentication, no multi-user.** Storage is JSON files
    behind a repository interface so a real store can be added later.

---

## 11. Future work

In the order that would add the most value:

1. **Implement adaptive / weighted conformal risk control** so the guarantee
   survives the shift the detector already identifies. *This is the actual
   research contribution and the largest remaining gap.*
2. **Download a second, cross-modality domain** (`msd-spleen`, CT) and measure
   real calibration degradation under genuine distribution shift — turning
   `confidence_degradation` from a heuristic into a measurement. The registry
   and loaders already support this; it needs one download and one script.
3. **Add a deep segmentation adapter** (nnU-Net or a MONAI bundle) behind
   `BaseMedicalModel` to replace the random-forest baseline. Dice should move
   from ~0.52 to ~0.85 on this task, and the calibration story gets stronger
   because deep networks are more overconfident, not less.
4. **Add the fracture use case** — `fracatlas` is registered (CC BY 4.0,
   X-ray, segmentation masks for the fractured subset) and is the setting where
   a calibrated "escalate this" signal has the clearest clinical value.
5. **Implement localized (region-conditional) risk control** so per-region risk
   becomes a guarantee.
6. Spatially-aware conformal prediction (SACP), evidential uncertainty as a
   single-pass alternative, and foundation-model UQ.

---

## 12. References

Key papers this implementation actually builds on (the full literature map,
including which papers are *not* yet implemented, is on the app's **Research
status** page):

- Angelopoulos, Bates, Fisch, Lei & Schuster. *Conformal Risk Control.* ICLR 2024.
- Bates et al. *Distribution-Free Risk-Controlling Prediction Sets.* JACM 2021.
- Guo, Pleiss, Sun & Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
- El-Yaniv & Wiener. *On the Foundations of Noise-Free Selective Classification.* JMLR 2010.
- Zhang, Brady & Smith. *Segmentation via a hidden MRF model and the EM algorithm.* IEEE TMI 2001.
- Lambert, Forbes, Doyle, Dehaene & Dojat. *Trustworthy Clinical AI: A Unified Review of UQ in Medical Image Analysis.* Artificial Intelligence in Medicine 2024.
- Blot et al. *Automatically Adaptive Conformal Risk Control.* 2024. *(planned)*
- Bereska, Karimi & Samavi. *SACP: Spatially-Aware Conformal Prediction.* MIDL 2025. *(planned)*
- Zecchin & Simeone. *Localized Adaptive Risk Control.* 2024. *(planned)*
- Zou et al. *DEviS: Reliable Medical Segmentation via Evidential Calibrated Uncertainty.* IEEE TCyb 2025. *(planned)*
