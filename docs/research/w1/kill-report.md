# Gate A kill report — Week 1

**Claim under test.** Pixel-level conformal risk control leaves lesions unprotected:
at the pixel-calibrated threshold $\hat\lambda^{\mathrm{pix}}$, the lesion-level miss rate
$R^{\mathrm{les}}$ substantially exceeds $\alpha$.

**Pre-registered statistic.** $G = R^{\mathrm{les}}(\hat\lambda^{\mathrm{pix}})/\alpha$ on the
full calibration split. Bands: KILL $G\le1.1$ · INCONCLUSIVE $1.1<G<1.5$ ·
PROCEED $G\ge1.5$ with $n_{\mathrm{cal}}\ge100$ and K3 passing.

---

## 1. Protocol (fixed before any result; unchanged after)

| | |
|---|---|
| Hit criterion | ground-truth component hit iff $\ge\tau=0.5$ of its pixels are in $S_\lambda$; GT components only |
| Connectivity | 8-connectivity, 2-D |
| Minimum lesion size | $m=10$ px at 256² working resolution — chosen from the K3 histogram before the gap test (§3) |
| Empty GT | excluded from both losses |
| Prediction set | $S_\lambda=\{\hat p\ge1-\lambda\}$, 200-point grid |
| Selection | $(n\hat R_n+1)/(n+1)\le\alpha$, first feasible — identical to `ConformalRiskControl` |
| $\alpha$ | 0.05, 0.10, 0.15 |
| Model | `RandomForestPixelSegmenter`, retrained per dataset and seed on the train split (the hippocampus-trained forest is meaningless on endoscopy). Kvasir Dice at argmax: 0.40–0.43 |
| Splits | `split_by_case`, 50/30/20, seeds {1337, 2024, 7}. **Kvasir has no patient IDs → image-wise; see §6** |
| Script | `backend/scripts/experiments/kill_test.py`; raw rows in `kill_kvasir-seg.{csv,json}` |

Nothing above was changed after seeing results. One thing was **added** after seeing
results and is labelled as such throughout: the stratified analysis in §4.

## 2. Kill table — Kvasir-SEG (pilot), mean ± sd over 3 seeds

| α | n_cal | multi-lesion frac | λ̂_pix | R_pix | R_les(λ̂_pix) | **G** | R_les (held-out test) | λ̂_les | λ̂_union | infl pix | infl les | infl union |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 300 | 0.048 ± 0.007 | 0.761 ± 0.027 | 0.046 | 0.021 ± 0.004 | **0.41 ± 0.08** | 0.016 ± 0.003 | 0.646 ± 0.027 | 0.761 | 1.58 | 1.33 | 1.58 |
| 0.10 | 300 | 0.048 ± 0.007 | 0.645 ± 0.022 | 0.095 | 0.046 ± 0.007 | **0.47 ± 0.07** | 0.062 ± 0.017 | 0.548 ± 0.023 | 0.645 | 1.32 | 1.10 | 1.32 |
| 0.15 | 300 | 0.048 ± 0.007 | 0.563 ± 0.016 | 0.146 | 0.080 ± 0.008 | **0.54 ± 0.05** | 0.094 ± 0.010 | 0.492 ± 0.011 | 0.563 | 1.14 | 0.98 | 1.14 |

Readings:

- **K1 (the gap).** $G=0.41$–$0.54$, tight across seeds. On the full split, pixel
  control *over*-protects lesions by roughly 2×. Falls in the KILL band at every α.
- **K2 (does the lesion constraint bind?).** $\hat\lambda^{\mathrm{les}}<\hat\lambda^{\mathrm{pix}}$
  at every α and seed, so $\hat\lambda^{\cup}=\hat\lambda^{\mathrm{pix}}$. The lesion
  constraint never binds on Kvasir. Abandon-signal for this dataset.
- **Held-out check.** Test-split $R^{\mathrm{les}}$ at $\hat\lambda^{\mathrm{pix}}$ (0.016 /
  0.062 / 0.094) tracks the calibration value and stays under α. The CRC guarantee is
  doing what it says.
- **Feasibility.** All thresholds feasible; set inflation 1.1–1.6×; no empty argmax
  masks. No floor problem under thresholding, as expected.
- The α = 0.15 row gives $\hat\lambda^{\mathrm{pix}}=0.563$ on Kvasir versus 0.392 on
  hippocampus in Project-I — different dataset, different model; not comparable
  beyond "same procedure".

## 3. K3 data audit

### Kvasir-SEG

| | native | 256² |
|---|---|---|
| units (all non-empty) | 1000 | 1000 |
| split train / cal / test | 500 / 300 / 200 | 500 / 300 / 200 |
| multi-lesion fraction (K ≥ 2) | 0.180 (inflated by speckle) | **0.049** |
| mean K | 1.21 | 1.06 |
| components | 1206 | 1064 |
| components < 1% image area | 0.144 | **0.031** |
| size p1 / p50 / p95 (px) | 1.0 / 31 321 / 134 454 | 309 / 6 788 / 26 200 |

Native-resolution histogram: **143 components < 10 px, zero in [10, 200), real polyps
from 200 px.** The sub-10 band is JPEG mask speckle (Kvasir masks are lossy JPEG).
That bimodality is the justification for $m=10$: it removes exactly the artefact
band, and any $m\in[10,200)$ is equivalent. At 256² only one speckle component
survives resizing.

**K3 verdict for Kvasir: FAIL as a decisive dataset.** 4.9% multi-lesion frames
(≈14 per calibration split); 3.1% small components. The gap's precondition —
several lesions of unequal size per unit — is essentially absent.

### msd-hippocampus (smoke test only)

780 slices, 390/234/156. At 256², $m=10$: multi-"lesion" fraction 0.101, mean K
1.11. Verified directly on six label volumes: anterior (label 1) and posterior
(label 2) **touch in every slice**, so merged foreground is a single component;
K = 2 arises only where a sliver detaches. Not a lesion task.

### FracAtlas — audit only (gap test deliberately not run)

Download resolved via the figshare API (article 22363012 v7: `FracAtlas.zip`,
338 MB, CC BY 4.0 — my registry entry had a wrong file id and size; fixed).
Segmentation ships as **COCO polygons** (`Annotations/COCO JSON/COCO_fracture_masks.json`),
not PNG masks; PASCAL VOC / YOLO files are boxes. The repo loader cannot read this
yet (W3).

| | native resolution |
|---|---|
| fractured images with ≥ 1 polygon | 719 |
| polygon annotations | 924 |
| **multi-lesion fraction (K ≥ 2)** | **0.238** — K dist {1: 548, 2: 146, 3: 17, 4: 7, 5: 1} |
| mean K | 1.29 |
| **components < 1% of image area** | **0.939** (< 0.1%: 0.081) |
| area fraction p5 / p50 / p95 | 0.0008 / 0.0026 / 0.0112 |
| size p5 / p50 / p95 (px) | 154 / 509 / 25 474 |
| bbox fill ratio, median | 0.61 (blob-like, not thin lines) |
| image resolutions | **373×454: 531 (74%)** · 2304×2880: 90 · 454×373: 83 · others: 15 |
| equivalent size at 256² working res, p5 / p50 / p95 | 45 / 142 / 603 px — **0.0% below 10 px** |
| split files | `Utilities/Fracture Split/{train,valid,test}.csv`; `dataset.csv` has a `multiscan` flag but **no patient id** |

Contact sheet: `docs/research/w1/fracatlas_contact_sheet.jpg` (50 random
fractured images, polygon in red, K per tile). Eyeballed: **coarse region
polygons around the fracture site, 4–8 vertices, not tight fracture-line
masks.** Several visible K = 2 tiles. Heavy letterboxing and scanner markers.

**Two corrections to what I said before the audit:**

1. *"256² erases a 4-px fracture."* Wrong for this dataset. 74% of images are
   373×454, and the polygons are blobs, so the median annotation is 142 px at
   256² and none fall below the m = 10 speckle floor. The resolution constraint
   is much weaker than I claimed. A per-dataset working size is still right —
   384–512 is sensible and cheap — but 256 would not have destroyed the labels.
2. *"744 MB, file 41725323."* Both wrong; corrected in the registry.

**K3 verdict for FracAtlas: PASS on the criteria that matter.** 23.8%
multi-lesion (just under the 25% bar, comfortably above Kvasir's 4.9%), 94% of
components in the small-lesion tail, 719 units → ~215 in a 30% calibration
split, of which ~50 would be multi-lesion. That last number is **below the
n ≥ 100 subpopulation bar** set in §7; Task08 remains necessary for a decisive
result, but FracAtlas is a far better second dataset than Kvasir.

**Caveats carried into W3:**
- "Lesion" here means *annotated fracture region*. Pixel FNR against a coarse
  polygon is not fracture-line recall; the paper must say which it measures.
- No patient id. `multiscan` marks images from the same study, so
  exchangeability is again approximate; group by the `multiscan` flag where
  possible.
- The RF pixel model sees mostly normal bone inside each polygon. Expect a weak
  segmenter, a high dilation-infeasibility floor (§1 of the memo), and possibly
  a *large* gap — which would be the hypothesis's regime, but must not be
  confused with "the model is bad".
- Loader work needed: COCO polygon → mask, `Fractured/` subfolder, split CSVs.

## 4. Hippocampus smoke test — execution and interpretation

| α | λ̂_pix | R_pix | R_les(λ̂_pix) | G | λ̂_les | infl pix |
|---|---|---|---|---|---|---|
| 0.05 | 0.658 | 0.046 | 0.013 | **0.26** | 0.327 | 1.18 |
| 0.10 | 0.497 | 0.096 | 0.013 | **0.13** | 0.286 | 1.00 |
| 0.15 | 0.417 | 0.143 | 0.017 | **0.11** | 0.266 | 0.89 |

You expected $G\approx1$ and said a different value would mean the labelling is
wrong. **The labelling is correct; the expectation was.** With one component per
image, "component missed at τ = 0.5" ⟺ "that image's pixel FNR > 0.5", which is a
far rarer event than the *mean* FNR being α. $G\approx1$ would require τ → 1
(missed iff any pixel missed). Two checks support this:

- Direct inspection of six label volumes (above): K = 1 in 5/6 densest slices.
- Brute-force lesion loss versus the order-statistic shortcut on 12 calibration
  slices × 7 λ values: **max |diff| = 0.0**.

The script runs end to end on a NIfTI dataset; that was the point of the smoke test.

## 5. POST-HOC stratified analysis — not pre-registered, read as exploratory

The pre-registered $G$ averages over a population that is 95% single large polyps,
where the effect cannot exist by construction (§4's argument). The hypothesis
does make a prediction about the *other* 5%, so I computed it. Same models, same
$\hat\lambda^{\mathrm{pix}}$, same protocol; only the subpopulation changes.

| seed | α | Dice | K ≥ 2: n | R_les \| K ≥ 2 | **G \| K ≥ 2** | per-lesion pooled R | G pooled | small (<1%): n | small miss rate |
|---|---|---|---|---|---|---|---|---|---|
| 1337 | 0.05 | 0.402 | 12 | 0.083 | **1.67** | 0.026 | 0.51 | 7 | 0.286 |
| 1337 | 0.10 | 0.402 | 12 | 0.306 | **3.06** | 0.067 | 0.67 | 7 | 0.429 |
| 1337 | 0.15 | 0.402 | 12 | 0.347 | **2.31** | 0.086 | 0.58 | 7 | 0.429 |
| 2024 | 0.05 | 0.402 | 14 | 0.071 | **1.43** | 0.025 | 0.51 | 9 | 0.333 |
| 2024 | 0.10 | 0.402 | 14 | 0.143 | **1.43** | 0.044 | 0.44 | 9 | 0.444 |
| 2024 | 0.15 | 0.402 | 14 | 0.179 | **1.19** | 0.095 | 0.63 | 9 | 0.444 |
| 7 | 0.05 | 0.429 | 17 | 0.029 | 0.59 | 0.016 | 0.31 | 5 | 0.000 |
| 7 | 0.10 | 0.429 | 17 | 0.127 | **1.27** | 0.050 | 0.50 | 5 | 0.000 |
| 7 | 0.15 | 0.429 | 17 | 0.216 | **1.44** | 0.084 | 0.56 | 5 | 0.000 |

What it shows, and what it cannot:

- **Where the precondition holds, the gap opens.** On multi-polyp frames,
  $G\mid K\ge2$ exceeds 1.1 in 8 of 9 cells and reaches 1.3–3.1 at α = 0.10.
  Small lesions are missed outright ~40% of the time in two of three seeds.
  Direction and mechanism are exactly the hypothesis: the pixel average is
  carried by large lesions; small ones are sacrificed.
- **It cannot pass any bar.** n ≈ 14 multi-lesion frames per seed; the small-lesion
  cell has n = 5–9 and one seed shows 0.0. Confidence intervals on $G\mid K\ge2$
  span roughly 0.6–3. This is a hypothesis-preserving signal, not evidence.
- **Pooled per-lesion $G$ stays below 1** because the pool is 95% large singletons.
  A per-lesion estimand does not fix the dataset problem.

## 6. Infrastructure flags encountered

- **Kvasir has no patient identifiers.** `split_by_case` is image-wise; the repo's
  patient-wise test passes trivially and proves nothing here. Frames from one
  colonoscopy would violate exchangeability and shrink the effective $n$. Reported;
  not fixable with this dataset.
- **Densest-slice sampler** (`loaders._load_msd_case`) affected the hippocampus
  smoke test and will affect Task08; it drops exactly the slices where small
  lesions appear alone. Did not affect Kvasir. Must be replaced before W6.
- **Global 256² resize** is why FracAtlas was not gap-tested. Per-dataset working
  size is W3.
- **Components labelled post-resize.** Kvasir lost 1206 → 1064 components, all
  speckle; acceptable here, not for fractures.
- **Registry URL for FracAtlas was wrong** (guessed file id; figshare returned
  202/empty). Resolved via the API and corrected.
- **Task08 (9.35 GB confirmed) at ~300 KB/s needs hours**; the harness-bound
  download was killed at the 10-minute limit and restarted detached with resume.
  On disk by W3 is plausible; monitor `data/raw/msd-hepatic-vessel.tar`.

## 7. Gate A verdict

**By the pre-registered criterion: KILL.** $G=0.41$–$0.54\le1.1$ on every seed and
every α. The lesion constraint never binds (K2). Under the rule as written, this
routes to fallback F1.

**My recommendation — disagree with it if you think the pre-registration should
bind regardless: reclassify as INCONCLUSIVE and move Gate A to W3 (FracAtlas at
512) and W6 (Task08), with a corrected pre-registration.**

Reasons, in order of weight:

1. **The pre-registered statistic was mis-specified for this dataset, and you said
   so in advance.** Your own brief states Kvasir "structurally cannot open the gap"
   and "can only kill or clearly greenlight". But a population-average $G$ on 95%
   single large lesions is *guaranteed* to fall below 1 by the τ = 0.5 argument
   in §4 — so on this dataset the KILL band was reachable by construction, not by
   the hypothesis being false. A KILL that the design guaranteed is not a
   falsification.
2. **K3 failed.** The gap's precondition (≥ 25% multi-lesion units, or a
   small-lesion tail) is absent: 4.9% and 3.1%. The brief made PROCEED conditional
   on K3 passing; symmetry says a KILL is not decisive when K3 fails either.
3. **The stratified signal points the hypothesis's way** (§5), on too few units
   to count. That is precisely what "the pilot cannot decide" looks like.

What I would *not* do is round this to PROCEED. Nothing here has $n\ge100$ in the
relevant subpopulation, and Kvasir never will (≈48 multi-polyp frames in total).

**Corrected pre-registration for the W3/W6 decisive test:**

- Compute $G$ on the subpopulation with $K_i\ge2$ (and separately on components
  below 1% of unit area), not on the full split.
- Require $n\ge100$ units in that subpopulation. Task08 (multiple liver tumours per
  volume, 303 volumes) should meet this at the volume level; FracAtlas is to be
  checked in the audit.
- Keep τ = 0.5, 8/26-connectivity, exclusion rule, and the selection rule as is.
  Re-choose $m$ from each dataset's own histogram before its gap test.
- Bands unchanged: KILL ≤ 1.1 · INCONCLUSIVE (1.1, 1.5) · PROCEED ≥ 1.5.

If you hold to the original pre-registration, F1 on Kvasir is a real (if thin)
result — "on a single-large-lesion task, pixel-level CRC over-protects lesions by
~2× at τ = 0.5" — but a reviewer will call it expected, and it would be.

## 8. What was and was not built this week

Built: `scripts/experiments/kill_test.py` (audit + gap phases, no `app/` changes),
this report, `theory-memo.md`, the `msd-hepatic-vessel` registry entry, the
FracAtlas URL fix. Not built, per instruction: `calibration/losses.py`,
`JointConformalRiskControl`, any parameterisation module. Nothing from this week
needs undoing if Gate A is ultimately failed.
