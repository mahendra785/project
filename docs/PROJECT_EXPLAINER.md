# PUMS / Lesion-Level Conformal Risk Control — Complete Project Explainer

> **How to use this file.** Paste the whole file into ChatGPT, or another
> assistant, and ask it to explain the project. The file is self-contained: it
> covers what the project is, why it exists, what is new, how every part works,
> every result (including the negative ones), how the code is organised, and the
> limits of what may be claimed. Every number below comes from the committed
> result files in `docs/research/results/`.
>
> **Instruction to the assistant reading this:** explain the project at whatever
> depth the user asks for. Keep the honesty rules in §12. Never state that the
> new set family "beats state of the art" on real data. It did not, and this
> file says so explicitly.

---

## 0. One-paragraph summary

PUMS (Perception Uncertainty Management System) is a B.Tech research project at
VIT Vellore (students: Mahendra Sajjan Choudhary 23BCE0701, Vansh Bhatiya
23BCE0631; guide: Dr. Shashank Mouli Satapathy). It adds statistically
guaranteed uncertainty to medical image segmentation models, after training and
without retraining them.

Project-II asks a specific question. The standard guarantee, *conformal risk
control* (CRC), promises that on average at most α of the lesion **pixels** are
missed. Does that also protect **lesions**, especially small ones? The answer on
real data is no: images with several lesions miss lesions at up to 1.56× the
target rate.

The project therefore does two things. First, it defines lesion-level risks and
controls them **jointly** with the pixel risk, which costs almost nothing.
Second, it proposes a new way to build prediction sets, **lesion-equalized
sets**, intended to make that joint guarantee cheaper.

All guarantees held in testing. The joint formulation beat the existing
multi-risk methods (Learn-Then-Test, morphological prediction sets) on set size.
The new set family gave up to 47% smaller sets in a controlled simulation but
only **tied** the simpler approach on the two real datasets. The likely reason
is the weak CPU backbone used for these runs.

---

## 1. Background concepts (plain language)

**Segmentation.** A model looks at a medical image (X-ray, endoscopy frame, MRI
slice) and labels each pixel as lesion or background. It outputs a probability
per pixel, the **posterior** `p̂(u)`.

**Prediction set.** A prediction set does not commit to one mask. It is a region
`S` that should contain the lesion, typically every pixel whose score exceeds a
cutoff. A bigger set catches more lesion but flags more healthy tissue. The
**set size** is the cost.

**Conformal risk control (CRC).** Angelopoulos et al., ICLR 2024. It chooses the
cutoff using a held-out *calibration* set, so that the expected loss on a new
image is at most a target α:

```
λ̂ = smallest λ such that (n · R̂_n(λ) + 1) / (n + 1) ≤ α
```

Here `R̂_n(λ)` is the average loss over the n calibration images at setting λ.
The `+1` term makes the guarantee exact at finite sample size. It needs only
**exchangeability** (calibration and test images are like random draws from the
same pool) and a loss that **does not increase** as the set grows. The guarantee
is **marginal**: an average over images, not a promise for every image.

**Pixel FNR.** Pixel false-negative rate is the fraction of true lesion pixels
outside the set. Standard CRC for segmentation controls this.

**Why pixels are the wrong unit.** Big lesions contain most of the lesion pixels.
Missing a small lesion entirely barely changes the pixel FNR, so the pixel
guarantee can hold while small lesions are routinely missed. Clinically, a missed
small lesion (an early tumour, a hairline fracture, a small polyp) is often the
error that matters most.

---

## 2. The research question and hypotheses

- **Question.** Can we guarantee, with the same finite-sample rigour as CRC,
  that few *lesions* are missed, and small lesions separately, while still
  guaranteeing pixel FNR? And can we do it without inflating the prediction sets?
- **H1 (gap).** At the pixel-CRC threshold, the lesion-miss rate on multi-lesion
  images, or the small-lesion miss rate, exceeds α.
- **H2 (efficiency of the new family).** Under the *same* joint guarantee,
  lesion-equalized sets are smaller than plain thresholding.
- **E7 (mechanism, simulation).** The gain should appear when small lesions are
  less confident than large ones, and grow with that coupling.

All three were **pre-registered**. The decision rules were committed to git
before any result existed (`docs/research/preregistration-mgcrc.md`).

---

## 3. Novelty: what is new and how strong each claim is

A literature search was run on 14 October 2026. It covered arXiv and the web,
from 2021 onward, with twelve query strings such as "lesion-wise conformal
segmentation", "multi-risk conformal segmentation" and "Learn then Test
segmentation".

| Prior work | What it controls | Guarantee | Several risks? |
|---|---|---|---|
| Conformal Risk Control (ICLR 2024) | any monotone loss; pixel FNR in practice | expectation | no |
| Learn-Then-Test (arXiv 2110.01052) | any loss | high-probability (FWER) | yes, but conservative |
| Conformal Lesion Segmentation, CLS (arXiv 2510.17897) | **voxel** FNR per volume, despite the "lesion" name | high-probability | no |
| CRA / CCRA / CCRA-S (arXiv 2504.07611) | pixel FNR per image, strata of images | expectation per stratum | no |
| ReCIRC (arXiv 2609.38112, 2026) | pixel FNR per image | marginal + approx. conditional | no |
| Morphological prediction sets (MICCAI 2025) | ground-truth mask containment (dilation) | high-probability | no |
| Per-organ recall (arXiv 2608.18193, 2026) | voxel FNR per organ | high-probability | Bonferroni |
| FDR instance segmentation (arXiv 2504.04482) | pixel FDR **or** FNR | expectation | one at a time |
| Andéol et al., COPA 2023 (object detection) | box-wise **or** pixel recall | expectation | one at a time |

**Novelty verdict, stated carefully:**

1. **Lesion-level risk — new, supported.** No published segmentation
   risk-control method defines its loss on ground-truth connected components
   (lesions), or on small lesions separately.
2. **Joint control — new but simple.** Controlling pixel, lesion and
   small-lesion risk at once follows from a short monotonicity argument (a
   corollary of CRC). It is a new *problem formulation with a measured gap*, not
   a deep theorem.
3. **Lesion-equalized prediction sets — new, not yet supported on real data.**
   This is the most original part. It won clearly in simulation and tied on both
   real datasets.

The search was thorough but not exhaustive. If a reviewer finds prior work on
item 3, items 1 and 2 plus the measured gap still stand.

---

## 4. The method in detail

### 4.1 Notation

- Image (unit) `i`, posterior `p̂_i(u) ∈ [0,1]` over pixels `u`.
- Ground truth `Y_i`, split into connected components `Y_i^1 … Y_i^K` (8-connected
  in 2-D, 26-connected in 3-D). These are the **lesions**.
- A **score family** is a fixed transform `s = T(p̂)` that does not look at
  labels. Prediction sets are `S_λ = { u : s(u) ≥ 1 − λ }`, so a larger λ gives a
  larger set. The sets are **nested** by construction.

### 4.2 The three losses

A lesion counts as **found** if at least τ = 50% of its pixels are inside the set.

- Pixel FNR: `L_pix = 1 − |S ∩ Y| / |Y|`
- Lesion miss: `L_les = (1/K) · #{lesions with < 50% coverage}`
- Small-lesion miss: the same, restricted to lesions below the dataset's 33rd
  percentile of lesion area. This definition uses labels only and was fixed
  before testing.

Images with no lesion are excluded, and images with no small lesion are excluded
from the small-lesion loss. Each guarantee is therefore *conditional on the
structure being present*.

### 4.3 Theory

- **Lemma 1 (monotonicity).** If the sets are nested and the loss is a
  non-increasing function of `S ∩ Y`, the loss never increases with λ. All three
  losses qualify, under *every* family, so CRC applies to each.
- **Lemma 2 (one number per lesion).** Lesion k is found at cutoff c exactly when
  its ⌈τ·|Y^k|⌉-th largest score is at least c. The lesion loss therefore needs
  one sort per image, the same cost as the pixel loss. The test suite checks this
  against brute force.
- **Joint threshold (Corollary).** Compute the CRC threshold `λ̂^r` for each risk
  and take `λ̂ = max_r λ̂^r`. By monotonicity every risk stays ≤ α in expectation
  at once. No Bonferroni split is needed, because each bound is an expectation
  bound controlled on its own.
- **Max-form variant.** Apply CRC to `max(L_pix, L_les, L_small)` per image. This
  bounds the per-image *worst* risk, which is stronger, at a larger threshold.
- **Proposition 1 (validity with a tuned family).** If the family is chosen
  using a *tuning split* disjoint from the calibration split, the guarantee is
  unchanged. Conditional on the tuning data the transform is fixed. Choosing the
  family on the calibration data would break the guarantee.

### 4.4 Lesion-equalized prediction sets (the new family)

**The problem with thresholding (`s = p̂`).** A faint lesion is found only if the
global cutoff drops below its confidence. The lesion constraint is therefore set
by the *least confident* lesion, which is usually a small one. At that low
cutoff, every background pixel above it joins every image's set.

**The fix.** Find candidate regions `C_j` = connected components of
`{p̂ ≥ t₀}`, each with peak `P_j`. Then:

```
s(u) = p̂(u) / P_j^ρ    if u is inside candidate C_j
s(u) = p̂(u)           otherwise           (ρ ∈ [0, 1])
```

**Properties:**
- **Lemma 3.** Scores stay in [0, 1], every candidate pixel outranks every
  non-candidate pixel, and ρ = 0 gives exactly thresholding, so the baseline is
  inside the family.
- **Proposition 2 (amplitude invariance).** If a candidate's posterior is
  `a · g` (confidence a times a shape g), then at ρ = 1 the score is `g / max g`.
  Whether a lesion is found no longer depends on the model's confidence, only on
  the shape of its response.
- **Proposition 3 (background exclusion).** At any cutoff ≥ t₀, the set lies
  inside the candidates. Thresholding has no such containment.
- **The cost.** False candidates (noise blobs above t₀) are also rescaled and
  join the set. The pair (ρ, t₀) sets this trade-off. It is chosen on the tuning
  split from a 17-member grid: ρ ∈ {0.25, 0.5, 0.75, 1} × t₀ ∈ {0.05, 0.1, 0.2,
  0.3}, plus plain thresholding.
- **Not claimed.** This is a sufficient mechanism, not an optimal family. The
  best nested family for a lesion loss is an open question.

---

## 5. Experimental protocol

- **Pre-registration:** committed before any result. **Amendment 1** recorded
  the backbone change: PyTorch and the 9 GB Task08 dataset were not downloaded,
  so a CPU random forest was used. **Amendment 2** added the simulation and
  disclosed two debug runs seen beforehand.
- **Datasets:**
  - **FracAtlas** (Scientific Data 2023, CC BY 4.0): 719 fractured X-rays, with
    fracture polygons rasterised to components. 159 have ≥ 2 lesions, and 234
    have a small lesion (< 104 px).
  - **Kvasir-SEG**: 1000 colonoscopy polyp images. Only 48 have ≥ 2 polyps, so
    it serves as a near single-lesion *control*. 312 have a small lesion
    (< 3774 px).
- **Backbone:** the repository's supervised random forest over a multiscale
  filter bank, at 256×256. It is trained with **5-fold cross-fitting**, so every
  image's posterior comes from a model that never saw it. Out-of-fold Dice was
  **0.057** on FracAtlas, where the model is nearly failing, and **0.413** on
  Kvasir-SEG.
- **Repetitions:** 3 training seeds × 1000 random partitions of each dataset into
  tune 20% / calibrate 40% / test 40%.
- **Fairness:** the baselines need no tuning, so they calibrate on
  tune + calibrate (60%). Our method calibrates on 40% only, which biases the
  comparison against us.
- **Baselines:**
  1. Pixel CRC (controls pixel FNR only)
  2. Joint CRC with thresholding (our formulation, standard family; **the reference**)
  3. Max-form with thresholding
  4. Learn-Then-Test (Hoeffding–Bentkus p-values, fixed sequence, δ = 0.1)
  5. Morphological (dilation) prediction sets, joint
  6. CCRA-S-style strata on total predicted mass, joint
- **Ours:** lesion-equalized, joint (union rule) and max-form.
- **Metric:** set-to-truth ratio `Σ|S| / Σ|Y|` on test images at a threshold
  covering all three risks. Lower is better. Validity: mean test risk ≤ α
  (+0.005 Monte-Carlo slack) for every risk a method claims.
- **Decision rules:**
  - **H1.** PROCEED if G ≥ 1.5 at α = 0.10, where G = lesion-miss ÷ α on the
    multi-lesion or small-lesion subpopulation. KILL if ≤ 1.1 everywhere.
  - **H2.** Supported if the change is < 0 and our method is smaller in ≥ 95% of
    splits, with validity intact.

---

## 6. Results

### 6.1 Validity

**Every method met every guarantee it claims on held-out data**, at every α, on
both datasets.

### 6.2 H1 — the gap (pixel CRC threshold, test split)

| Dataset | α | G (all images) | G, multi-lesion [95% split interval] | G, small lesions |
|---|---|---|---|---|
| FracAtlas | 0.05 | 0.64 | 0.83 [0.22, 1.64] | 0.72 |
| FracAtlas | 0.10 | 0.76 | 1.14 [0.59, 1.76] | 0.73 |
| FracAtlas | 0.15 | 0.77 | 1.26 [0.78, 1.77] | 0.81 |
| Kvasir-SEG | 0.05 | 0.40 | 0.95 [0.00, 2.24] | 0.73 |
| Kvasir-SEG | 0.10 | 0.51 | **1.56** [0.66, 2.52] | 0.82 |
| Kvasir-SEG | 0.15 | 0.62 | 1.49 [0.72, 2.28] | 0.81 |

**Verdicts:** Kvasir-SEG **PROCEED**; FracAtlas **INCONCLUSIVE**. The intervals
are wide because only about 29 (Kvasir) or 95 (FracAtlas) multi-lesion images
fall in each calibration split.

### 6.3 Main comparison — Kvasir-SEG, α = 0.10

Mean over 3 seeds × 1000 splits. Δ is the paired change in set size versus
joint thresholding.

| Method | pixel FNR | lesion miss | small miss | miss, K≥2 | set/truth | Δ |
|---|---|---|---|---|---|---|
| Pixel CRC | 0.099 | 0.051 | 0.082 | 0.156 | 3.08 | −0.1% |
| Joint CRC, thresholding (reference) | 0.098 | 0.050 | 0.081 | 0.155 | 3.09 | — |
| Joint max-form | 0.084 | 0.039 | 0.067 | 0.126 | 3.24 | +5.0% |
| Learn-Then-Test | 0.071 | 0.032 | 0.056 | 0.098 | 3.40 | +10.2% |
| Morphological sets | 0.093 | 0.050 | 0.081 | 0.147 | 3.14 | +1.6% |
| CCRA-S strata | 0.092 | 0.037 | 0.066 | 0.160 | 3.06 | −0.8% |
| **Lesion-equalized (ours)** | 0.096 | 0.049 | 0.080 | 0.153 | 3.11 | +0.7% |

**Across all datasets and α:**
- Adding lesion-level control costs **at most 0.7%** in set size over pixel-only CRC.
- **Learn-Then-Test** needs **10–46%** larger sets. It gives a stronger,
  high-probability guarantee, so the comparison is not perfectly like-for-like.
- **Morphological sets** need **1–43%** larger sets.
- **CCRA-S** ranges from **−2% to +16%**.
- **H2 FAILED at every α on both datasets.** Lesion-equalized sets tied joint
  thresholding (+0.3% to +3.6%, never smaller in ≥ 95% of splits).

**Why H2 failed on real data.** With this weak backbone, the **pixel**
constraint is the binding one: pixel CRC and joint thresholding differ by under
1%. When the lesion constraint doesn't bind, there's nothing for equalization to
save. The tuning split often picks plain thresholding itself (on FracAtlas,
thresholding was chosen most often at α = 0.05).

### 6.4 E7 — controlled simulation

Lesion confidence scales as `a ∝ radius^γ`. At γ = 0 confidence is independent
of size; at γ = 2 small lesions are very faint. Two background levels b; 600
scans and 500 splits per cell. The table shows the change in set size,
lesion-equalized vs joint thresholding (negative = ours smaller).

| b, α | γ=0 | γ=0.5 | γ=1 | γ=1.5 | γ=2 |
|---|---|---|---|---|---|
| 0.05, 0.05 | −4.0% | −25.7% | −19.8% | +0.7% | +0.8% |
| 0.05, 0.10 | −2.1% | −41.8% | −30.7% | −0.5% | −0.5% |
| 0.05, 0.15 | −1.3% | −40.0% | −45.0% | −0.7% | +0.3% |
| 0.15, 0.05 | −3.6% | −33.5% | −0.2% | +0.7% | +1.8% |
| 0.15, 0.10 | −2.6% | −47.2% | −2.1% | +1.2% | +0.9% |
| 0.15, 0.15 | −2.0% | −38.1% | +4.0% | +1.3% | +0.8% |

**Verdict under the pre-registered rule: FAILED.** The rule required a gain at
every γ ≥ 1 that grows with γ.

The gain is real but sits in a window: up to −47% at moderate coupling, and about
0 at strong coupling. A **post-hoc diagnostic** explains why. At γ ≥ 1.5, 76–92%
of small lesions have a peak posterior that does not rise above the image's
99th-percentile background. No transform of the posterior alone can then
separate them from noise, so Proposition 3's precondition fails. The method
falls back to thresholding (worst cell +4.0%) instead of doing worse.

### 6.5 E8 — post-hoc: protecting multi-lesion images (not pre-registered)

The joint guarantee averages lesion misses over all images, so multi-lesion
images can still exceed α. A lesion loss restricted to images with K ≥ 2 is a
valid CRC risk, for the same reason as the small-lesion risk: it lives on a
subpopulation defined by labels.

| Dataset | α | miss, K≥2 (joint) | miss, K≥2 (+ conditional risk) | set/truth change |
|---|---|---|---|---|
| Kvasir-SEG | 0.05 | 0.048 | 0.024 | +28% |
| Kvasir-SEG | 0.10 | 0.155 | 0.076 | +16% |
| Kvasir-SEG | 0.15 | 0.224 | 0.125 | +24% |
| FracAtlas | 0.05 | 0.040 | 0.038 | +2% |
| FracAtlas | 0.10 | 0.113 | 0.092 | +7% |
| FracAtlas | 0.15 | 0.187 | 0.143 | +13% |

The conditional risk roughly halves the multi-lesion miss rate on Kvasir and
brings it under α, at a 16–28% larger set.

---

## 7. What the results mean (the honest story)

1. Pixel guarantees really do hide missed lesions in multi-lesion images.
2. Lesion-level guarantees are cheap to add, and the formulation is more
   efficient than the existing multi-risk tools.
3. The new set family works exactly in the regime its theory predicts: small
   lesions present, under-confident, but still above the background. Such a
   regime was not present on the real data with this weak model.
4. The decisive open experiment is a **strong backbone** (GPU U-Net) and **3-D
   multi-tumour data** (MSD Task08). Both pipelines are implemented but have not
   been run.

---

## 8. System architecture (where everything lives)

The method sits inside PUMS: a FastAPI backend (Python) and a Next.js 16
dashboard (React 19, TypeScript).

```
Dataset loaders ─► train_backbone.py (5-fold cross-fitting) ─► posteriors .npz + sha256 manifest
                                                               │
          families.py ─► losses.py ─► joint.py  ◄──────────────┘
          (score maps)   (grid counts) (CRC / union / max / LTT / strata / family selection)
                                   │
          mgcrc.py · simulation.py · posthoc_multi.py ─► docs/research/results/*.json
                                   │
          aggregate.py (pre-registered verdicts) ─► summary.md, mgcrc_summary.json
          paper_macros.py ─► paper/generated/*.tex (every number in the paper)
                                   │
          FastAPI GET /api/research/mgcrc ─► Next.js /research tab
          Next.js /showcase reads the result files directly (live demo + every case)
          calibration/lesion_crc.py ─► registry: lesion_aware_crc, morphological_conformal
```

| Module | Path | Role |
|---|---|---|
| Score families | `backend/app/calibration/families.py` | threshold, dilation, lesion-equalized |
| Losses | `backend/app/calibration/losses.py` | per-image grid counts for the three risks (Lemma 2) |
| Selection rules | `backend/app/calibration/joint.py` | CRC, union, max-form, LTT, CCRA-S strata, conditional K≥2 risk, family selection, evaluation |
| App method | `backend/app/calibration/lesion_crc.py` | exposes the method through the existing calibration API |
| Data | `backend/app/datasets/loaders.py` | new FracAtlas COCO-polygon loader |
| Backbones | `backend/scripts/experiments/train_backbone.py`, `train_backbone_msd.py` | 2-D cross-fitting (RF or U-Net); 3-D Task08 pipeline |
| Experiments | `mgcrc.py`, `simulation.py`, `sim_diagnostic.py`, `posthoc_multi.py` | E1–E8 |
| Reporting | `aggregate.py`, `paper_macros.py` | verdicts, tables, LaTeX macros |
| API | `backend/app/api/routes/research.py` | `GET /api/research/mgcrc` |
| UI | `frontend/app/research`, `frontend/app/showcase` | results tab; full project demo with live in-browser pipeline |

**Engineering standards carried over from Project-I:**
- Every number carries provenance (computed / synthetic / mock).
- Planned methods return HTTP 501.
- No magic constants.
- Typed contracts: Pydantic in the backend, Zod in the frontend.

---

## 9. Testing, reproducibility and infrastructure

- **Tests:** 123 backend tests (pytest) and 29 frontend tests.
  - Loss monotonicity for every family.
  - Lemma 2 against brute force.
  - A Monte-Carlo check that the joint guarantee holds on exchangeable data.
  - LTT at least as conservative as CRC.
  - API shape, and a contract test against a real captured API response.
- **CI:** GitHub Actions runs pytest, typecheck, lint and the node tests.
- **One command:** `./scripts/reproduce.ps1` (Windows) or `.sh` regenerates
  everything: backbones, experiments, simulation, post-hoc, verdicts and paper
  numbers. `-Fast` reuses cached posteriors.
- **Integrity:**
  - Posterior files carry sha256 hashes and environment records.
  - Seeds are fixed throughout.
  - The pre-registration is dated in git before its results.
- **Hardware:** every reported number was produced on CPU. A GPU path exists:
  `-Backbone unet-r34 -Size 512` after installing PyTorch.

---

## 10. Deliverables in the repository

| Deliverable | Path |
|---|---|
| Pre-registration (+2 amendments) | `docs/research/preregistration-mgcrc.md` |
| Theory memo v2 (proofs) | `docs/research/theory-memo-v2.md` |
| Related work / novelty check | `docs/research/related-work.md` |
| Generated results summary | `docs/research/results/summary.md` |
| Paper draft (LaTeX, LNCS / MICCAI) | `paper/` |
| VIT Project-II report and Review-3 slides | `report/` |
| Architecture (C4 diagrams) and infrastructure | `docs/architecture.md` §10, `docs/infrastructure.md` |
| Standalone showcase pages | `docs/showcase/` |
| In-app demo tab | `/showcase` ("Project demo" in the sidebar) |

---

## 11. Limitations

1. **Weak backbone.** FracAtlas results describe a near-failing segmenter
   (Dice 0.057). H2 has not been tested where it is most plausible: a strong
   model whose remaining misses are faint small lesions.
2. **Cross-fitting.** Posteriors come from five fold models, so exchangeability
   is approximate.
3. **No patient IDs** in either dataset, so splits are image-wise.
4. **Small multi-lesion subpopulation** (29–95 images per calibration split),
   so the gap intervals are wide.
5. **Marginal guarantees,** conditional on a lesion being present. Not per
   patient.
6. **FracAtlas "lesions" are coarse fracture regions,** not fracture lines.
7. **No clinical validation of any kind.** This is a research prototype, not a
   medical device.

---

## 12. Honesty rules for explaining this project

- ✅ Say: "first lesion-level, jointly controlled risk guarantee for
  segmentation, as far as our October 2026 search found."
- ✅ Say: "joint lesion-level control costs under 1% over pixel-only control and
  beats Learn-Then-Test and morphological sets at the same guarantee."
  Note that LTT's guarantee is high-probability, which is stronger.
- ✅ Say: "lesion-equalized sets cut set size by up to 47% in simulation when
  small lesions are under-confident."
- ❌ Do not say: "the new method beats state of the art on real data". It tied.
- ❌ Do not say the joint guarantee is a deep theoretical result. It is a short
  corollary; the novelty is the formulation and the measurement.
- ❌ Do not make clinical claims.
- Mention that E8 is post-hoc whenever it is discussed.

---

## 13. Glossary

- **α (alpha):** the allowed average miss rate, e.g. 0.10 = 10%.
- **λ (lambda):** the knob that grows the prediction set. λ̂ is the calibrated value.
- **Exchangeability:** calibration and test images behave like random draws from the same pool.
- **Marginal guarantee:** holds on average over images and calibration draws, not for each image.
- **Calibration / tuning / test split:** disjoint data partitions. Tuning picks the family, calibration picks λ, test measures.
- **Connected component:** a group of touching lesion pixels. This project's definition of a "lesion".
- **τ (tau) = 0.5:** a lesion is "found" if at least half of it is in the set.
- **Set-to-truth ratio:** total predicted-set area ÷ total true-lesion area. Lower is better.
- **G (gap):** lesion-miss rate ÷ α at the pixel-CRC threshold. Above 1 means under-protected.
- **Cross-fitting:** each image is predicted by a model trained on the other folds.
- **LTT (Learn-Then-Test):** multiple-testing approach to risk control with a high-probability guarantee.
- **CCRA-S:** per-stratum CRC thresholds from Luo & Zhou (2025).
- **Morphological sets:** prediction sets made by dilating the predicted mask (Mossina & Friedrich, MICCAI 2025).
- **Pre-registration:** writing down hypotheses and decision rules before seeing results.

---

## 14. Suggested ways to explain it

- **To a non-technical audience (30 seconds):** "AI models that outline tumours
  come with a safety promise about pixels, not tumours. We show the promise can
  hold while small tumours are missed. We add a promise about tumours themselves
  that costs almost nothing, and we tested a new way to make it even cheaper.
  That new way works in controlled tests but not yet on real data."
- **To an ML audience (2 minutes):** cover these points in order:
  1. CRC with a pixel FNR loss is dominated by large components.
  2. Define component-level and small-component losses; Lemma 1 gives
     monotonicity under any nested family.
  3. A union threshold gives simultaneous expectation control.
  4. Thresholding is the Neyman–Pearson ranking for the pixel loss but not for
     the lesion loss.
  5. Peak-normalised candidate scores are amplitude-invariant and contain
     thresholding.
  6. Tune the family on a disjoint split for validity.
  7. Results: valid everywhere; beats LTT and morphological sets on efficiency;
     the new family ties on real data and wins in simulation within a predicted
     regime.
- **To a viva or review panel:** lead with the pre-registration, the validity
  results and the negative H2 result, then explain why it failed and what
  experiment would decide it.
