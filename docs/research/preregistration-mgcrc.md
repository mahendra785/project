# Pre-registration — lesion-level joint conformal risk control (MG-CRC)

**Committed before any posterior from `train_backbone.py` existed** (see git
history: this file precedes every file under `docs/research/results/`). Nothing
below may be edited after results are seen; any later analysis is labelled
POST-HOC in the report.

## Hypotheses

**H1 (gap).** At the pixel-FNR CRC threshold $\hat\lambda^{pix}$, the lesion-miss
rate on multi-lesion units and the small-lesion miss rate exceed $\alpha$:
$G_{multi}=R^{les}_{K\ge2}(\hat\lambda^{pix})/\alpha$ and
$G_{small}=R^{small}(\hat\lambda^{pix})/\alpha$.

**H2 (efficiency — the method claim).** Under the *same* simultaneous guarantee
(pixel FNR, lesion miss and small-lesion miss each $\le\alpha$ in expectation),
lesion-equalized prediction sets with the family chosen on a tuning split
(`ours_union`) have a smaller set-to-truth ratio than thresholding with the
union threshold (`union_threshold`), even though the baseline is calibrated on
50% more data.

## Fixed protocol

| Item | Value |
|---|---|
| Datasets | FracAtlas (primary; 719 fractured radiographs, COCO polygons). Kvasir-SEG (control: 95% single-lesion). MSD Task08 tumours if the download completes (3-D units, 26-conn) |
| Backbone | U-Net, ResNet-34 ImageNet encoder, 40 epochs, BCE+Dice, flips+gamma, 5-fold cross-fitting, seed 0 (seeds 1, 2 if time permits — reported either way) |
| Working res | 512², top-left letterbox. Components labelled at native res (8-conn), min size 10 px native, nearest-resized |
| Lesion hit | GT component hit iff $\ge\tau=0.5$ of its pixels in $S_\lambda$. τ ∈ {0.25, 0.75} is an ablation, not primary |
| Small lesion | component area (at 512²) below the 33rd percentile of all GT component areas in the dataset (a label-only definition of the estimand) |
| Empty GT | excluded from all losses |
| λ grid | 1001 points on [0, 1]; $S_\lambda=\{s\ge1-\lambda\}$ |
| α | 0.05, 0.10, 0.15 |
| Repetitions | 1000 random partitions of the pool: tune 20% / cal 40% / test 40% |
| Baselines (calibrated on tune∪cal) | pixel CRC; union-threshold; max-threshold; LTT-threshold (HB p-values, fixed sequence, δ = 0.1); union-dilation (morphological sets); union-stratified (CCRA-S-style, 3 strata of total predicted mass) |
| Ours | lesion-equalized family, grid ρ ∈ {0.25, 0.5, 0.75, 1}, t₀ ∈ {0.05, 0.1, 0.2, 0.3}, plus plain thresholding; chosen on tune by smallest set-to-truth ratio under the union rule; calibrated on cal. `ours_max` is the same with the max rule |
| Primary metric | set-to-truth ratio $\sum|S|/\sum|Y|$ on test units with GT |
| Primary comparison | paired relative change `ours_union` vs `union_threshold`, per α, over the 1000 repetitions |

## Decision rules

- **H1** on FracAtlas: KILL if $G_{multi}\le1.1$ **and** $G_{small}\le1.1$ at every α.
  PROCEED if either is $\ge1.5$ at α = 0.10. Otherwise INCONCLUSIVE.
  $n_{cal}$ for multi-lesion units will be ≈ 95 per repetition (below the earlier
  n ≥ 100 bar); this is reported, not hidden.
- **H2** is supported at a given α if the mean paired relative change is < 0 and
  `ours_union` is smaller in ≥ 95% of repetitions, **and** its mean test risks
  for pixel, lesion and small-lesion loss are all ≤ α + 0.005 (Monte-Carlo slack).
  H2 fails if the change is ≥ 0. Anything in between is reported as partial.
- Validity is checked for **every** method: mean test risk ≤ α for each risk the
  method claims to control.

## Known threats, stated in advance

1. Cross-fitted posteriors come from five models; CRC assumes one. A fixed-model
   check (train on folds ≠ k, calibrate and test on fold k only) is reported.
2. FracAtlas has no patient id; splits are image-wise. Some images come from one
   study (`multiscan`), so exchangeability is approximate.
3. FracAtlas polygons are coarse fracture *regions*. "Lesion" means annotated
   fracture region.
4. Repetitions share units, so the 1000 repetitions are not independent samples;
   the percentile intervals describe split-to-split variability, not a sampling
   CI for the population. Independent evidence comes from training seeds and datasets.

---

## Amendment 1 — 2026-10-14, before any real posterior or result existed

The user stopped all large downloads (PyTorch/CUDA wheels and MSD Task08), so:

- **Backbone** for this round is the repository's CPU random-forest pixel
  segmenter (`RandomForestPixelSegmenter`, filter bank, 120 trees, depth 16),
  retrained per fold with the same 5-fold cross-fitting, seeds 0, 1, 2. It is a
  weak model, and the paper says so. The U-Net path (`--backbone unet-r34`) is
  kept and runs unchanged once PyTorch is installed.
- **Working resolution** is 256² instead of 512² (CPU cost). The W1 FracAtlas
  audit shows 0.0% of fracture components fall below 10 px at 256².
- **Datasets**: FracAtlas (primary) and Kvasir-SEG (control). Task08 is
  deferred; the 3-D pipeline (`train_backbone_msd.py`) is built but not run.
- Hypotheses, metrics, splits, baselines, grids and decision rules are
  **unchanged**.

---

## Amendment 2 — 2026-10-14, controlled simulation (E7), registered before it was run

**Disclosure.** Before this amendment, the harness was debugged on 120 units of
the blob generator in `tests/test_structured_crc.py` (20 repetitions). That run
showed the lesion-equalized method smaller than joint-threshold. Those numbers
were deleted and are not reported. A 5-repetition check on real FracAtlas seed-0
posteriors showed the RF backbone has out-of-fold Dice ≈ 0.06 and the methods
roughly tied. The main runs are unchanged by either.

**E7 purpose.** Test Proposition 2/3's mechanism under control: when lesion
confidence falls with lesion size, lesion-equalized sets should need smaller sets
than thresholding for the same joint guarantee; when confidence does not depend
on size, they should tie or lose (false-candidate cost).

**Generator** (`scripts/experiments/simulation.py`, fixed before running):
128² units; K ~ 1 + Poisson(1.5) lesions; radius r ~ U(2, 12) px; posterior on
lesion k is the Gaussian bump a_k·exp(−d²/2r²) with
a_k = clip(a_0·(r/12)^γ + N(0, 0.05), 0.05, 1), a_0 = 0.9; background
p ~ U(0, b) plus 3 Poisson noise bumps of amplitude U(0.1, 0.35). GT = disc
of radius r. γ ∈ {0, 0.5, 1, 1.5, 2} (γ = 0: amplitude independent of size),
b ∈ {0.05, 0.15}. 600 units per cell, seed 0, 500 repetitions, same split, rules,
families and metric as the main protocol.

**Decision rule.** The mechanism is supported if `ours_union` vs
`union_threshold` is < 0 with P(smaller) ≥ 0.95 at γ ≥ 1 for both b, **and**
the advantage at γ = 2 is larger than at γ = 0. It fails otherwise.

---

## Amendment 3 — 2026-10-20, revision protocol (registered before any revised result)

**Trigger.** An external pre-submission review (2026-10-19) required: (i) a protocol
whose assumptions match the conformal guarantee exactly; (ii) a competent
deep-learning backbone, a second backbone, and pre-CRC segmentation metrics;
(iii) fair baselines with equal calibration data; (iv) statistically valid
uncertainty for overlapping splits; (v) the multi-lesion analysis either labelled
exploratory or validated under a pre-specified protocol. Everything in this
amendment is fixed before any posterior from the revised pipeline exists. All
earlier results (random forest, cross-fitting) are kept, unchanged, and reported
as the *preliminary* study.

**Disclosure.** No revised posterior, statistic or figure has been computed when
this amendment is written. PyTorch had just been installed; the only GPU runs
before the main runs will be ≤ 2-epoch smoke tests on ≤ 40 units, whose outputs
are deleted and not reported.

### P1. Frozen-model protocol (primary)

For each dataset, architecture and model seed m ∈ {0, 1, 2}:

1. Units are partitioned once at random (seed m) into **D_train (40%)** and
   **D_pool (60%)**. For MSD Task08 the unit is the patient volume (split is
   patient-disjoint). Kvasir-SEG and FracAtlas have no patient identifiers; their
   splits are image-wise (limitation, stated).
2. The backbone is trained on D_train only (10% of D_train held out for
   checkpoint selection by validation Dice), then **frozen**.
3. The frozen model predicts every unit of D_pool. No D_pool label is used
   before calibration.
4. The small-lesion size cut is the 33rd percentile of ground-truth component
   sizes in **D_train** (not the pool), so the loss is fixed before calibration.
5. Within D_pool, R = 1000 random partitions: tune 20% / cal 40% / test 40%.

Under P1 the calibration and test units are i.i.d. given the frozen model, so the
CRC guarantee holds exactly (conditional on D_train). Cross-fitting (the
preliminary study) is analysed theoretically and kept as a secondary protocol.

### Datasets and backbones

| | |
|---|---|
| Datasets | Kvasir-SEG (all 1000), FracAtlas (719 fractured; non-fractured images used only for training), MSD Task08 liver tumours (3-D volumes, 26-connected tumours) **if the archive completes**; if it does not, this is reported |
| Backbone A | U-Net, ResNet-34 ImageNet encoder (segmentation_models_pytorch) |
| Backbone B | Transformer encoder: U-Net decoder with MiT-B2 (SegFormer) ImageNet encoder; if unavailable in the installed library, FPN with EfficientNet-B3 (stated) |
| Training | AdamW 3e-4, BCE + soft Dice, flips + gamma, 60 epochs (2-D) / 25 epochs (Task08 slices), batch 8, AMP, best validation-Dice checkpoint |
| Resolution | 352² (Kvasir-SEG, FracAtlas, letterbox as before), 256² (Task08 slices) |
| Pre-CRC metrics (on D_pool) | Dice, IoU (per unit, threshold 0.5), lesion-wise recall (GT component with ≥ 50% of its pixels at p ≥ 0.5), predicted-component precision, false-positive components per unit |

### Comparisons (all on the same partitions; paired)

- **C1 equal calibration data (primary).** Every method calibrates on *cal* only.
  The tune split is used only by the lesion-equalized family choice; baselines
  ignore it. Methods: pixel CRC; joint (union) thresholding — **reference**;
  max-form thresholding; joint dilation (morphological family under CRC);
  joint CCRA-S-style strata; ours (union and max). Learn-Then-Test (δ = 0.1) is
  reported in a separate block because its guarantee is high-probability.
- **C2 equal total data.** Baselines on tune ∪ cal (the original design).
- **C3 no tuning.** Lesion-equalized with ρ = 0.5, t₀ = 0.2 fixed now, calibrated
  on tune ∪ cal — same data as C2 baselines.
- Set sizes are compared only after the validity check for the risks each
  method claims.

### Hypotheses and decision rules (per dataset × backbone, α ∈ {0.05, 0.10, 0.15})

- **H1** unchanged (thresholds 1.5 / 1.1 on G_multi or G_small at α = 0.10),
  now with a bootstrap 95% CI. In addition to the point rule, the paper reports
  whether the CI lower bound exceeds 1.
- **H2** unchanged in form, evaluated under C1: supported if the mean paired
  relative change of `ours_union` vs `union_threshold` is < 0, `ours_union` is
  smaller in ≥ 95% of partitions, the bootstrap 95% CI of the mean change is
  entirely below 0, and all three mean test risks are ≤ α + 0.005.
- **H3 (new, was post-hoc E8).** Adding the K ≥ 2 lesion risk to the joint rule
  brings the mean test lesion-miss rate on K ≥ 2 units to ≤ α + 0.005. Reported
  with its set-size and background (false-positive area) cost. Evaluated only
  where the pool has ≥ 30 multi-lesion units; otherwise reported as
  under-powered. The earlier E8 numbers stay labelled post hoc.

### Statistics

- Model seeds are the independent replicates (fresh D_train/D_pool and
  initialisation). Results are given per seed and as mean ± s.d. over seeds.
- Within a seed, the 1000 partitions describe split variability only
  (2.5–97.5% percentiles; never called a CI).
- **CIs**: a unit-level (patient-level for Task08) bootstrap of D_pool, B = 200
  resamples × 100 partitions each; percentile 95% CI of the mean paired change,
  of G, and of each mean test risk. This accounts for overlapping partitions and
  for the finite pool.
- Counts reported for every risk: number of calibration and test units on which
  it is defined (all, with a small lesion, with K ≥ 2).

### Ablations and analyses (secondary, all reported)

τ ∈ {0.25, 0.5, 0.75}; small-lesion percentile ∈ {25, 33, 50}; candidate
connectivity 4 vs 8 (2-D); each of the 17 fixed (ρ, t₀) members on tune ∪ cal;
calibration size n_cal ∈ {50, 100, 200, all}. Error analysis of lesions missed at
the joint threshold (size, peak posterior, whether a candidate exists, peak vs
99th-percentile background). These are descriptive and carry no decision rule.
