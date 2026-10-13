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
