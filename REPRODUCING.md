# Reproducing the article

This file explains how to reproduce the results of *Lesion-Level Joint Conformal Risk Control for Medical Image
Segmentation: Guarantees on Missed Lesions, Not Only Missed Pixels* (Mahendra Sajjan Choudhary, Vansh Bhatiya and
Shashank Mouli Satapathy, Vellore Institute of Technology). The rest of this repository, including `frontend/` and
most of `backend/app/`, is the research web application developed in the same project; it is not needed to
reproduce the article.

## What is where

| Path | Contents |
|---|---|
| `backend/app/calibration/` | losses (`losses.py`), score families (`families.py`), CRC, joint, max-form and LTT rules (`joint.py`, `lesion_crc.py`) |
| `backend/scripts/experiments/kaggle_train.py` | trains the frozen U-Net backbones and writes the pool posteriors |
| `backend/scripts/experiments/mgcrc_v2.py` | partitions, every calibration rule (comparisons C1–C3), clustered bootstrap |
| `backend/scripts/experiments/aggregate_v2.py` | seed aggregation, confidence intervals, hypothesis verdicts → `summary_all.json` |
| `backend/scripts/experiments/simulation.py`, `sim_diagnostic.py` | controlled simulation and its post-hoc diagnostic |
| `scripts/kaggle/` | the notebooks that ran every reported experiment ([HOW_TO_RUN.md](scripts/kaggle/HOW_TO_RUN.md)); they download the data from the official sources, verify checksums, train, calibrate and aggregate |
| `docs/research/preregistration-mgcrc.md` | pre-registration with Amendments 1–3 |
| `docs/research/results_v2/` | aggregated results (`summary_all.json`) and per-run manifests (environment, splits) |
| `docs/research/results/` | preliminary study and simulation results |
| `paper/journal/build.py` | regenerates every number, table and figure of the article and its supplement |

## 1. Tables and figures from the released results (a few minutes, no GPU)

```bash
pip install matplotlib numpy
python paper/journal/build.py
```

This writes `paper/journal/manuscript.tex`, `paper/journal/supplementary.tex` and the figures from the result files
in `docs/research/`.

## 2. All experiments from scratch (GPU)

Run the three session notebooks in `scripts/kaggle/` on a GPU (Kaggle T4), as described in
[HOW_TO_RUN.md](scripts/kaggle/HOW_TO_RUN.md). Unzip each result archive into `docs/research/`, then run
`backend/scripts/experiments/aggregate_v2.py` and `paper/journal/build.py`.

The notebooks embed the code version used for the reported runs. The repository version of `mgcrc_v2.py` and
`backend/app/calibration/joint.py` additionally contains the corrected empty-subgroup rule (a risk with no eligible
calibration unit falls back to the whole unit, for CRC and LTT), described in Section F, item (v), of the
supplementary material; the event has probability below 10⁻⁶ per partition in the reported configurations.

Python dependencies are listed in `backend/requirements.txt` (calibration and analysis) and
`backend/requirements-torch.txt` (training).

## Datasets

The datasets are not redistributed here; the notebooks download them from their official sources: Kvasir-SEG
(Jha et al., MMM 2020), FracAtlas (Abedeen et al., Scientific Data 2023) and the Medical Segmentation Decathlon
Task08 (Antonelli et al., Nature Communications 2022).

## Contact

Corresponding author: Shashank Mouli Satapathy, shashankmouli.s@vit.ac.in.
