# Multi-granularity conformal risk control — theory memo (W1)

**Scope.** Everything a reviewer needs to check before believing the gap experiment.
Written before the Kvasir gap numbers existed; the "assumptions as used" section
records exactly what the kill test ran with.

**Position, stated first.** The mechanism below is a *corollary* of monotonicity,
not a contribution. Learn-Then-Test [LTT] already controls multiple risks in general.
What this line of work can contribute is the **measured gap** — how badly a
pixel-level guarantee protects lesions — and the **price of closing it**. The memo
exists so the measurement rests on stated assumptions, not on intuition.

---

## 1. Setup

Pixel grid $\Omega$. Ground truth $Y_i\subseteq\Omega$ with connected components
$\mathcal C(Y_i)=\{Y_i^1,\dots,Y_i^{K_i}\}$. A *nested* family of prediction sets:
$\lambda\le\lambda' \Rightarrow S_\lambda(X_i)\subseteq S_{\lambda'}(X_i)$.

Split conformal risk control [CRC] with a loss $L_i(\lambda)\in[0,B]$ that is
non-increasing in $\lambda$:

$$
\hat\lambda=\inf\Big\{\lambda:\ \frac{n\,\hat R_n(\lambda)+B}{n+1}\le\alpha\Big\},
\qquad \hat R_n(\lambda)=\frac1n\sum_{i=1}^n L_i(\lambda),
\qquad \mathbb E\big[L_{n+1}(\hat\lambda)\big]\le\alpha .
$$

The guarantee is **marginal** (in expectation over calibration set and test point)
and requires the calibration units and the test unit to be **exchangeable**.

## 2. Both parameterisations are score-map thresholds

$$
S^{\mathrm{thr}}_\lambda=\{u:\hat p(u)\ge 1-\lambda\},\qquad
S^{\mathrm{dil}}_\lambda=\{u:d(u,\hat M)\le\lambda\}=\hat M\oplus B_\lambda,\quad
\hat M=\{\hat p\ge\tfrac12\}.
$$

Both have the form $S_\lambda=\{u:s(u)\ge c(\lambda)\}$ with $c$ decreasing:
$(s,c)=(\hat p,\,1-\lambda)$ for thresholding, $(s,c)=(-d(\cdot,\hat M),\,-\lambda)$
for dilation, where $d(\cdot,\hat M)$ is the Euclidean distance transform of the
argmax mask. A one-knob hybrid $S^\rho_\lambda=\{\hat p-\rho\,d(\cdot,\hat M)\ge 1-\lambda\}$
is also nested for any fixed $\rho\ge0$.

**Consequence.** Dilation by radius $\lambda$ is a threshold on the EDT. One EDT per
image, and the whole family is free. The existing `_loss_curve` (one `searchsorted`
over foreground scores) therefore serves both parameterisations unchanged.

## 3. Losses

$$
L^{\mathrm{pix}}_i(\lambda)=1-\frac{|S_\lambda\cap Y_i|}{|Y_i|},\qquad
L^{\mathrm{les}}_i(\lambda;\tau)=\frac1{K_i}\sum_{k=1}^{K_i}
\mathbf 1\!\left[\frac{|S_\lambda\cap Y_i^k|}{|Y_i^k|}<\tau\right],\quad \tau\in(0,1].
$$

$$
L^{\max}_i=\max\big(L^{\mathrm{pix}}_i,L^{\mathrm{les}}_i\big),\qquad
L^{w}_i=w\,L^{\mathrm{pix}}_i+(1-w)\,L^{\mathrm{les}}_i .
$$

Units with $Y_i=\emptyset$ (after the size filter of §6) are **excluded** from both
losses: the losses are undefined there, and scoring them 0 would dilute the risk
with free wins. The guarantee is therefore *conditional on the structure being
present*.

## 4. Lemma 1 — set-monotone losses are monotone under any nested family

**Lemma 1.** Let $\{S_\lambda\}$ be nested and $L_i(\lambda)=f(S_\lambda\cap Y_i)$ where
$f$ is non-increasing under set inclusion. Then $\lambda\mapsto L_i(\lambda)$ is
non-increasing.

*Proof.* $\lambda\le\lambda'\Rightarrow S_\lambda\cap Y_i\subseteq S_{\lambda'}\cap Y_i
\Rightarrow f(S_{\lambda'}\cap Y_i)\le f(S_\lambda\cap Y_i)$. $\square$

$L^{\mathrm{pix}}$ and $L^{\mathrm{les}}(\cdot;\tau)$ both have this form: enlarging
$S\cap Y_i^k$ can only raise the covered fraction of $Y_i^k$, so each indicator can
only flip $1\to0$. Both are bounded in $[0,1]$, so $B=1$. The max and weighted
forms inherit boundedness and monotonicity.

**This holds for thresholding, dilation and the hybrid alike.** The choice between
them is therefore a question of *efficiency* (set size at a given risk), not of
validity. The premise that thresholding might break lesion-level monotonicity
was wrong; what breaks it is stated in §7.

## 5. Corollary 1 — the trivial joint threshold, and what the max form adds

**Corollary 1.** Let $\hat\lambda^{\mathrm{pix}},\hat\lambda^{\mathrm{les}}$ be CRC
thresholds at level $\alpha$ for each loss separately, and
$\hat\lambda^{\cup}=\max(\hat\lambda^{\mathrm{pix}},\hat\lambda^{\mathrm{les}})$. Then

$$
\mathbb E\big[L^{\mathrm{pix}}_{n+1}(\hat\lambda^{\cup})\big]\le\alpha
\quad\text{and}\quad
\mathbb E\big[L^{\mathrm{les}}_{n+1}(\hat\lambda^{\cup})\big]\le\alpha .
$$

*Proof.* By Lemma 1, $L^{\mathrm{pix}}_{n+1}(\hat\lambda^{\cup})\le
L^{\mathrm{pix}}_{n+1}(\hat\lambda^{\mathrm{pix}})$ pointwise; take expectations and
apply the CRC guarantee. Identically for lesions. $\square$

No Bonferroni split is needed: these are two **marginal expectation bounds**,
each controlled on its own, not a joint probability statement.

**What the max form buys — name the guarantee.** Because
$\hat R^{\max}_n(\lambda)\ge\max\big(\hat R^{\mathrm{pix}}_n(\lambda),\hat R^{\mathrm{les}}_n(\lambda)\big)$
pointwise, the max-form feasible set is contained in the intersection of the two
individual feasible sets, so $\hat\lambda^{\max}\ge\hat\lambda^{\cup}$. In exchange,

$$
\mathbb E\big[\max(L^{\mathrm{pix}}_{n+1},L^{\mathrm{les}}_{n+1})(\hat\lambda^{\max})\big]\le\alpha
$$

bounds the expected **per-image worse** of the two rates. That is strictly
stronger than two marginal bounds, at a strictly larger $\lambda$. It is not
correct to say $\hat\lambda^{\cup}$ is "never more conservative" without adding
*for the two-marginal guarantee*; for the per-image-worst-case guarantee,
$\hat\lambda^{\cup}$ does not qualify at all.

**What the weighted form does not buy.** $\mathbb E[L^w]\le\alpha$ implies only
$\mathbb E[L^{\mathrm{pix}}]\le\alpha/w$ and $\mathbb E[L^{\mathrm{les}}]\le\alpha/(1-w)$.
It is not simultaneous control. Keep it as an ablation showing how a scalarised
objective silently under-protects one granularity; do not headline it.

## 6. Lemma 2 — order-statistic reduction

**Lemma 2.** For component $Y_i^k$ let $s^{(k)}_\tau$ be the
$\lceil\tau|Y_i^k|\rceil$-th largest score in $Y_i^k$. Then $Y_i^k$ is hit at cutoff
$c$ iff $s^{(k)}_\tau\ge c$, and

$$
L^{\mathrm{les}}_i(\lambda;\tau)=\frac1{K_i}\,\#\{k:\ s^{(k)}_\tau<c(\lambda)\}.
$$

*Proof.* At least $\tau|Y_i^k|$ pixels of $Y_i^k$ have score $\ge c$ iff the
$\lceil\tau|Y_i^k|\rceil$-th largest does. $\square$

So the lesion loss is one `searchsorted` over the sorted multiset
$\{s^{(k)}_\tau\}_k$ per image — the same cost as the pixel loss. Verified in
`kill_test.py` against a brute-force implementation on 12 real slices × 7 λ values:
max absolute difference 0.0.

## 7. Where the proof breaks, and the fallback

**Any hit criterion that references predicted components.** IoU with a matched
predicted component, Hungarian or greedy matching, false-positive component counts:
under thresholding, predicted islands appear and merge as $\lambda$ grows and the
matching changes; under dilation they merge and bloat and matched IoU can *fall*.
Neither parameterisation rescues these. FP-component counts are non-*decreasing*
in $\lambda$ — they are the cost, not a controllable risk.

**Fallback.** If a reviewer insists on IoU-based detection, use Learn-Then-Test on
the finite $\lambda$ grid [LTT]: no monotonicity required, multiple risks
controlled with family-wise error control, but a *high-probability* guarantee
rather than an expectation bound, and more conservative. It is built as a baseline
in W5 regardless.

**Feasibility floors.**
*Dilation:* if $\hat M=\emptyset$ then $S^{\mathrm{dil}}_\lambda=\emptyset$ for every
$\lambda$ and the unit contributes loss 1 forever. If the fraction of such
calibration units exceeds $\alpha$ no $\lambda$ is feasible; CRC must return the
boundary with a warning, and the infeasibility rate is a result. A weak segmenter
makes this likely. Mitigation: seed the dilation from the top-$q$ pixels rather than
the argmax mask so the seed is never empty, or use the hybrid.
*Thresholding:* $S^{\mathrm{thr}}_1=\Omega$, always feasible, always trivially so.

## 8. Assumptions as actually used in the W1 kill test

| Assumption | Value used | Why |
|---|---|---|
| Hit criterion | $\tau=0.5$, on **ground-truth components only** | Fixed before any result; predicted-component criteria excluded by §7 |
| Connectivity | 8-connectivity, 2-D | Fixed before any result |
| Minimum lesion size $m$ | **10 px at 256² working resolution** | Chosen from the K3 histogram *before* the gap test: native-res components are bimodal (143 below 10 px = JPEG mask speckle; zero in [10, 200); real polyps from 200 px). At 256² the smallest real polyp is 309 px. Any $m\in[10,200)$ is equivalent; 10 removes exactly the one residual artefact |
| Empty-GT rule | excluded from both losses | §3 |
| Prediction set | thresholding, $\lambda$ grid of 200 on $[0,1]$ | Dilation and hybrid are W2+ |
| Selection | $(n\hat R_n+1)/(n+1)\le\alpha$, first feasible | Identical to `ConformalRiskControl.calibrate` |
| $\alpha$ | $\{0.05,\,0.10,\,0.15\}$ | 0.15 matches Project-I's $\hat\lambda=0.392$ |
| Working resolution | 256² | Adequate for polyps; **not** for fractures (§9) |
| Units / splits | one image = one case; `split_by_case`, seeds {1337, 2024, 7} | See §9 on Kvasir exchangeability |

Nothing in this table was changed after the gap numbers were seen.

## 9. Infrastructure caveats that bear on the guarantee

- **Kvasir has no patient identifiers.** `split_by_case` is image-wise there; the
  repository's patient-wise test passes trivially and proves nothing for this
  dataset. If several frames come from one colonoscopy the effective $n$ is smaller
  than the nominal one and exchangeability is only approximate. Reported, not
  fixable.
- **Densest-slice sampler** (`loaders._load_msd_case`) selects the slices with
  the largest annotated area. For volumetric datasets it drops exactly the slices
  where small lesions appear alone. Affects the hippocampus smoke test and Task08
  later; does **not** affect Kvasir (natively 2-D). Must be replaced before any 3-D
  lesion experiment.
- **Global 256² resize** erases thin structures *in principle*. In practice the
  FracAtlas audit (kill-report §3) showed 74% of its images are 373×454 and the
  polygons are blob-like, so the median annotation is 142 px at 256² and none
  fall below the speckle floor. The constraint is real for genuinely thin,
  high-resolution masks; it is not the reason FracAtlas was audit-only this
  week (that was the pre-declared rule). A per-dataset working size (384–512)
  remains the right fix and is W3.
- **Components are labelled post-resize.** Bilinear + 0.5 threshold can erase or
  merge small components. Acceptable for polyps (measured: 1206 native components
  → 1064 at 256², the loss being speckle); fatal for fractures.
- **2-D `CalibrationSample` cannot express a 3-D lesion.** For MSD volumes the
  calibration unit must be the volume with 26-connectivity on stacked slice
  predictions; $n$ becomes the volume count.

## 10. Prior work this must be positioned against

- **[CRC]** Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.*
  ICLR 2024. The guarantee used here, verbatim.
- **[LTT]** Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test:
  Calibrating Predictive Algorithms to Achieve Risk Control.* arXiv:2110.01052,
  2021. Controls **multiple** risks simultaneously via multiple-hypothesis testing
  with FWER control; needs no monotonicity; yields high-probability guarantees.
  Corollary 1 is the special case where every loss is monotone and only
  expectation control is needed — simpler, tighter, but narrower. Any claim of
  "joint control" must be stated relative to LTT.
- Mossina & Friedrich, *Morphological prediction sets*, MICCAI 2025 — dilation as
  the nested family. Check whether they report a component-level loss.
- Luo & Zhou, *Conditional Conformal Risk Adaptation* (CRA/CCRA/CCRA-S),
  arXiv:2504.07611 — group-stratified thresholds; W5 baseline.
- *Conformal Lesion Segmentation for 3D Medical Images*, arXiv:2510.17897 — check
  for any lesion-level or size-stratified loss before claiming the gap is new.
- Andéol et al. — image-wise CRC with box-wise and pixel-wise recall losses, one
  at a time. Check whether they ever take the max.

**Search strings to run before claiming novelty** (Scholar + arXiv, 2021→):
`"conformal" "lesion-level" OR "instance-level" segmentation "false negative"` ·
`"Learn then Test" segmentation` · `"conformal" "multiple risks" segmentation` ·
`"morphological" "conformal" segmentation dilation` ·
`"small lesion" "conformal" OR "risk control" size stratified`.
