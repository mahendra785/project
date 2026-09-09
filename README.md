# PUMS — Perception Uncertainty Management System

*A post-hoc hierarchical uncertainty calibration framework for dense medical
vision models under distribution shift.*

**BCSE497J Project-I · B.Tech · Vellore Institute of Technology**
Mahendra Sajjan Choudhary (23BCE0701) · Vansh Bhatiya (23BCE0631)
Faculty guide: Dr. Shashank Mouli Satapathy

> ### ⚠️ Research prototype — not a medical device
>
> PUMS has **no clinical validation**, uses **no medical dataset**, and ships
> **no validated medical model**. Its risk outputs describe how reliable the
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
│   ├── tests/                  93 pytest tests
│   ├── scripts/                Sample generation, threshold derivation
│   └── configs/                Annotated reference configuration
├── frontend/                   Next.js 16 · React 19 · TypeScript
│   ├── app/                    9 App-Router routes
│   ├── components/             ui/ · layout/ · viewer/ · charts/ · research/
│   ├── features/               analyze/ · results/
│   ├── hooks/                  TanStack Query hooks
│   ├── lib/                    api/ (client, Zod schemas, endpoints), utils
│   ├── stores/                 Zustand (client state only)
│   ├── types/                  Mirror of the Pydantic contract
│   └── tests/                  25 tests incl. a live contract fixture
├── shared/api-contracts/       How the two sides stay in sync
├── data/sample/                Synthetic phantoms (NOT medical images)
├── docs/                       architecture.md · api.md
└── scripts/                    setup / dev / test-all (PowerShell + bash)
```

---

## 3. Tech stack

**Backend** — Python 3.11+ · FastAPI · Pydantic v2 · NumPy · SciPy ·
scikit-learn · scikit-image · OpenCV · Pillow · pydicom · pytest
*Optional:* PyTorch · torchvision · MONAI (`requirements-torch.txt`)

**Frontend** — Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS 4 ·
shadcn/ui-style Radix primitives · Lucide · TanStack Query · Zustand · Recharts ·
Zod · React Hook Form

**Infrastructure** — Docker / docker-compose (optional) · `.env` configuration

> **Why PyTorch is optional.** The default inference path is a classical
> probabilistic segmenter that produces genuine posteriors with no pretrained
> weights, so the whole research stack runs on any machine with a ~400 MB
> install instead of ~2.5 GB. Torch is only needed for the optional
> demonstration CNN adapter, and the registry reports it as unavailable rather
> than failing.

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
python scripts/make_sample_images.py     # generate the sample phantoms

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

The backend builds a synthetic calibration set and reference profile on a
background thread at startup (~15 s). `GET /health` reports
`warmup.status`, and the header shows **Warming up** until it is `ready`.
Analyses run before then still work — they are just slower.

### Try it

1. Open <http://localhost:3000> → redirects to the dashboard.
2. Go to **Analyze**, drop in `data/sample/phantom_id_1.png`, press **Run analysis**.
3. Repeat with `phantom_shift_gamma.png` and `phantom_shift_severe.png` to watch
   the shift bands and risk overrides fire.
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
| Statistically calibrated uncertainty | ✅ **Implemented** | Split Conformal Risk Control with the finite-sample `+B/(n+1)` correction (Angelopoulos et al., ICLR 2024). Tests verify loss monotonicity, minimal-λ selection, and that the correction is strictly more conservative than the uncorrected baseline. |
| Hierarchical pixel + object uncertainty | ✅ **Implemented** | One dense field; Levels 2 and 1 are strict aggregations of it, so cross-level consistency is structural. |
| Localised uncertainty maps | ✅ **Implemented** | Per-pixel normalised entropy, PNG-transported, decoded back to a `Float32Array` in the browser. Verified end-to-end by a round-trip test. |
| Lightweight post-hoc estimation | ✅ **Implemented** | Single forward pass. No ensembling, MC dropout, or retraining. ~0.4 s end-to-end at 256×256 after warmup. |
| Calibration robustness (ECE, reliability) | ✅ **Implemented** | ECE, Brier, reliability diagrams, risk–coverage; plus temperature scaling as a second real method. |
| Distribution-shift awareness | 🟡 **Baseline** | Composite input-space detector (Wasserstein + JS + moments + texture) with **measured** band thresholds. It *detects* shift and invalidates the guarantee — it does not yet *restore* validity. |
| Standardised evaluation | 🟡 **Baseline** | Dice, IoU, ECE, Brier, AURC, selective risk, uncertainty–error correlation, all unit-tested against known answers. Missing: standard **datasets**, not metrics. |
| Object-level uncertainty | 🟡 **Baseline** | Region-derived by default; a segmentation-derived proposal adapter exists. No trained detector. |
| Clinical decision support | 🟠 **Experimental** | Transparent weighted risk engine with override rules, referral priority and triage queue. Not clinically validated. |
| Foundation-model reliability | ⚪ **Planned** | Model-agnostic interface is in place (proven by a second CNN adapter). No SAM/SAM2 work. |
| Cross-modality calibration robustness | ⚪ **Planned** | Only one calibration domain exists; cannot be claimed or measured yet. |

Live version of this table: the **Research status** page in the app.

### Methods implemented vs. catalogued

| Method | Status |
| --- | --- |
| Conformal Risk Control (split, FNR loss) | ✅ Implemented |
| Temperature scaling | ✅ Implemented |
| Empirical threshold (no-guarantee control) | 🟡 Baseline |
| Automatically Adaptive CRC · SACP · Localized Adaptive Risk Control · Morphological conformal · Evidential (DEviS) | ⚪ Planned — registry entries that return **501** |

---

## 9. Testing

```bash
./scripts/test-all.sh            # everything

cd backend  && pytest            # 93 tests
cd frontend && npm test          # 25 tests
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

---

## 10. Known limitations

1. **No medical data, no medical model, no clinical validation.** The default
   adapter is an unsupervised intensity mixture; the optional CNN is trained on
   natural images. Both are labelled `is_medical: false`.
2. **The calibration and reference sets are synthetic phantoms.** The
   *algorithms* are real and correctly implemented; the *data* is not clinical.
   Every derived figure is tagged `synthetic`.
3. **Conformal guarantees are marginal and assume exchangeability**, which
   distribution shift violates. PUMS detects this and emits a validity warning —
   but restoring validity under shift is **not yet implemented**, and that is the
   project's principal remaining gap.
4. **Shift detection is input-space only.** Semantic shift that preserves
   intensity statistics will not be caught. `confidence_degradation` and
   `uncertainty_increase` are heuristic couplings, **not** measured drops.
5. **Per-region "calibrated risk" is a combination, not a guarantee.**
   Region-conditional control needs localized risk control (planned).
6. **Single-pass entropy is aleatoric-dominated** and does not isolate epistemic
   uncertainty.
7. **2-D only.** Multi-frame DICOM takes the middle slice. No NIfTI.
8. **No database, no authentication, no multi-user.** Storage is JSON files
   behind a repository interface so a real store can be added later.

---

## 11. Future work

In the order that would add the most value:

1. **Register a real annotated medical split** as the calibration source. Every
   `SYNTHETIC` badge becomes `COMPUTED` and every derived figure becomes a
   genuine measurement — *no algorithm changes needed*. See
   [`data/README.md`](data/README.md) for the exact hook.
2. **Add a trained medical segmentation adapter** (nnU-Net or a MONAI bundle)
   behind `BaseMedicalModel`.
3. **Implement adaptive / weighted conformal risk control** so the guarantee
   survives the shift the detector already identifies. *This is the actual
   research contribution and the largest remaining gap.*
4. **Build paired ID/OOD splits** so calibration degradation under shift becomes
   measured rather than heuristic.
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
