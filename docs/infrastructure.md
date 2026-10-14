# PUMS — Infrastructure and reproducibility

## 1. Environments

| Environment | Contents | Install |
|---|---|---|
| Backend core | Python 3.11+ (dev 3.13), FastAPI, NumPy/SciPy, scikit-learn/-image, OpenCV, nibabel | `pip install -r backend/requirements.txt` (~450 MB) |
| Backend GPU (optional) | PyTorch + torchvision (CUDA), segmentation-models-pytorch | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && pip install segmentation-models-pytorch` |
| Frontend | Node 20+, Next.js 16, React 19, TanStack Query, Zod, Recharts | `cd frontend && npm ci` |
| Paper | any TeX Live ≥ 2022 with `llncs`, `pgfplots` | `cd paper && latexmk -pdf main.tex` |

Everything that produces a paper number runs on CPU. A GPU only enables the
U-Net backbone (`--backbone unet-r34`).

Reference hardware for the reported runs: Windows 11, Python 3.13.0,
scikit-learn 1.9.0, CPU random-forest backbone. The machine also has an RTX 4060
Laptop (8 GB), CUDA 12.5 driver, which these runs did not use.

## 2. Data

| Dataset | Licence | Size | Fetch | Role |
|---|---|---|---|---|
| FracAtlas (Scientific Data 2023) | CC BY 4.0 | 338 MB | `python scripts/download_dataset.py fracatlas` | primary multi-lesion 2-D |
| Kvasir-SEG | research use, cite | 46 MB | `... kvasir-seg` | single-lesion control |
| MSD Task04 Hippocampus | CC BY-SA 4.0 | 28 MB | `... msd-hippocampus` | app demo / smoke test |
| MSD Task08 Hepatic vessel + tumour | CC BY-SA 4.0 | 9.35 GB | `... msd-hepatic-vessel` | 3-D multi-tumour (pipeline built, not yet run) |

Raw archives go in `data/raw/` and extracted data in `data/datasets/`. Both are
gitignored and never vendored. The download script prints each dataset's licence
and citation and requires you to accept them first.

## 3. Artefact chain

```
data/datasets/<id>/              (gitignored)
   └─ train_backbone.py ──► backend/var/posteriors/<id>/<backbone>_seed<k>.npz  + .json (sha256, env, fold Dice)
         └─ mgcrc.py   ──► backend/var/stats/*.pkl                               (cache; deterministic)
                       ──► docs/research/results/main_<id>_<bb>_seed<k>_tau<t>.json  + _reps.csv   (committed)
               └─ aggregate.py ──► docs/research/results/summary.md, mgcrc_summary.json, paper/data/*.csv
```

The committed result files are sufficient to rebuild every table and the paper
figures without the datasets. The posteriors are sufficient to rebuild the
result files without retraining.

## 4. Reproducibility checklist

- [x] Pre-registration committed before results (`git log -- docs/research/preregistration-mgcrc.md`
      predates `docs/research/results/`); amendments are dated and precede results.
- [x] All seeds fixed: fold assignment `1000+seed`, model `seed*100+fold`, splits `seed*7919+1`.
- [x] Posterior files hashed (sha256 in manifest); environment recorded per run.
- [x] One command per artefact; `scripts/reproduce.{sh,ps1}` runs them all.
- [x] Tests cover the guarantee: loss monotonicity for every family, Lemma 2
      versus brute force, a Monte-Carlo validity simulation, LTT ≥ CRC
      conservativeness, and API shape.
- [ ] GPU backbone (U-Net) results: the pipeline is ready, but the runs have not been done.
- [ ] Task08 3-D results: the pipeline is ready, but the data is not downloaded.

## 5. Continuous integration

`.github/workflows/ci.yml` runs on every push. The backend job runs pytest, which
includes the MG-CRC validity simulation. The frontend job runs typecheck, lint and
the node tests. Experiments are not run in CI because they need the datasets,
but the committed result files are checked by the API test.

## 6. Running the system

```bash
# backend
cd backend && uvicorn app.main:app --reload --port 8000     # docs at /docs
# frontend
cd frontend && npm run dev                                   # http://localhost:3000/research
# or both
docker compose up --build
```

The `/research` page's **Lesion-level CRC** tab reads `GET /api/research/mgcrc`,
which serves `mgcrc_summary.json` as it is. The page does not compute any numbers itself.
