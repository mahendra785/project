# Lesion-level joint conformal risk control — theory memo v2

Extends `w1/theory-memo.md` (notation, Lemmas 1–2, Corollary 1 are reused
verbatim). v1 established that lesion-level CRC is *valid* under any nested
family. v2 is about **efficiency**: which nested family to use, and why the
standard one is the wrong choice for a lesion-level loss.

---

## 1. Problem

Unit $i$ has posterior $\hat p_i:\Omega\to[0,1]$, ground truth $Y_i$ with components
$Y_i^1..Y_i^{K_i}$. A *family* is a label-free map $T$ from posteriors to scores
$s_i=T(\hat p_i)\in[0,1]^\Omega$, with sets $S_\lambda(i)=\{s_i\ge1-\lambda\}$. We want, simultaneously,

$$
\mathbb E[L^{pix}_{n+1}]\le\alpha,\qquad \mathbb E[L^{les}_{n+1}]\le\alpha,\qquad
\mathbb E[L^{small}_{n+1}\mid\text{small lesion present}]\le\alpha,
$$

with the smallest expected set $\mathbb E|S(n{+}1)|$.

## 2. Validity with a data-chosen family

**Proposition 1.** Let $\mathcal T$ be a finite set of families and
$D_{tune},D_{cal}$ disjoint, with $D_{cal}\cup\{n+1\}$ exchangeable conditionally on
$D_{tune}$. Let $\hat T=\Psi(D_{tune})\in\mathcal T$ be *any* selection rule, and let
$\hat\lambda^r$ be the CRC threshold for risk $r$ computed on $D_{cal}$ with scores
$\hat T(\hat p)$, on the subpopulation where $L^r$ is defined. Then with
$\hat\lambda^\cup=\max_r\hat\lambda^r$, for every $r$,
$\mathbb E[L^r_{n+1}(\hat\lambda^\cup)]\le\alpha$.

*Proof.* Condition on $D_{tune}$; $\hat T$ is then a fixed label-free transform, so
the sets are nested (Lemma 1 applies to $L^{pix},L^{les},L^{small}$) and the CRC
theorem gives $\mathbb E[L^r_{n+1}(\hat\lambda^r)\mid D_{tune}]\le\alpha$ for each $r$
over its own exchangeable subpopulation. Monotonicity gives
$L^r_{n+1}(\hat\lambda^\cup)\le L^r_{n+1}(\hat\lambda^r)$ pointwise. Take expectations
over $D_{tune}$. $\square$

Selecting the family **on the calibration split** would break the conditioning.
The fix would be to treat the (family, λ) pair as an LTT hypothesis grid, which
costs a high-probability guarantee and a multiplicity correction. We avoid that
by spending 20% of the data on tuning.

## 3. Lesion-equalized scores

Candidates $C_1..C_J$ are the components of $\{\hat p\ge t_0\}$, with peaks
$P_j=\max_{C_j}\hat p$:

$$
s_{\rho,t_0}(u)=\hat p(u)\,P_j^{-\rho}\ \ (u\in C_j),\qquad s_{\rho,t_0}(u)=\hat p(u)\ \ (\text{otherwise}).
$$

**Lemma 3 (well-posed, ordered, nests thresholding).** $s\in[0,1]$. Every
candidate pixel scores $\ge t_0$, and every non-candidate pixel scores $<t_0$.
Setting $\rho=0$ gives $s=\hat p$.
*Proof.* Inside $C_j$, $t_0\le\hat p\le P_j\le1$. For $\rho\in[0,1]$ we have
$P_j\le P_j^\rho\le1$, hence $t_0\le\hat p\le\hat p/P_j^\rho\le\hat p/P_j\le1$. Outside, $s=\hat p<t_0$
by definition. $\square$

**Proposition 2 (amplitude invariance).** Suppose on candidate $C_j$ the posterior
factorises as $\hat p=a_j\,g_j$ with amplitude $a_j\in(0,1]$ and profile $g_j$.
Then $s_{1,t_0}=g_j/\max_{C_j}g_j$ on $C_j$. So whether lesion $k\subseteq C_j$ is hit at
cutoff $c$ — that is, whether $q_\tau(g_j|_{Y^k})/\max g_j\ge c$ — does not depend on $a_j$.
*Proof.* $P_j=a_j\max g_j$; substitute. Lemma 2 gives the hit condition. $\square$

**Interpretation.** Under thresholding, lesion $k$ is hit iff
$a_j\,q_\tau(g_j|_{Y^k})\ge c$. The lesion constraint therefore forces $c$ below the
$\alpha$-quantile of $a\cdot q$ across lesions, a quantile governed by the
**least confident** lesions. Small lesions are disproportionately low-confidence,
because receptive fields average them with background. At $\rho=1$ the constraint
involves only profile shape, so the confidence amplitude stops setting the cutoff.

## 4. Why this saves set size

**Proposition 3 (background exclusion).** Under $s_{\rho,t_0}$ with any $\rho$, for
every cutoff $c\ge t_0$, $S=\{s\ge c\}\subseteq\bigcup_jC_j$. Under thresholding,
$\{\hat p\ge c\}$ contains every pixel with $\hat p\ge c$.

So whenever the joint constraint needs a thresholding cutoff $c_{thr}<t_0$ (a faint
lesion below the candidate level), thresholding admits the whole background
region $\{c_{thr}\le\hat p<t_0\}$. That region is the low-confidence halo around every
structure plus scattered noise, and it grows like the area of the posterior's
tail. The equalized family can instead satisfy the same constraint at some
$c_{eq}\ge t_0$ whenever every lesion that must be hit lies inside a candidate.
Its set is then confined to candidates.

**The price.** Every false candidate (a noise component of $\{\hat p\ge t_0\}$) is
also peak-normalised and enters the set at a high cutoff. The trade-off is set
by $(\rho,t_0)$: a large $t_0$ admits fewer false candidates but more lesions fall
outside every candidate, and a small $\rho$ moves the family back toward
thresholding. This is why the pair is chosen on a tuning split, with
$\rho=0$ (thresholding) kept in the grid. By construction the method then
contains the baseline, and on tuning data it can only tie or improve.

**What is not claimed.** Nothing here says equalization is optimal. The
lesion-loss-optimal nested family depends on the joint law of
$(\hat p, Y)$, and characterising it is open. Proposition 3 is a sufficient
mechanism, not an optimality result. The empirical question (H2 of the
pre-registration) is whether the mechanism dominates the false-candidate cost
on real data.

## 5. Relation to existing guarantees

| Method | Loss granularity | Guarantee | Multiple risks | Family |
|---|---|---|---|---|
| CRC (Angelopoulos+ 2024) | any monotone loss, one at a time | expectation | no | thresholding |
| CLS (Tan+ 2025, arXiv 2510.17897) | voxel FNR per volume | high-prob. per-sample | no | thresholding |
| CRA / CCRA / CCRA-S (Luo & Zhou 2025) | pixel FNR per image | expectation, per stratum | no | thresholding (adaptive score) |
| ReCIRC (2026, arXiv 2609.38112) | pixel FNR per image | marginal + approx. conditional | no | thresholding, per-input budget |
| Morphological sets (Mossina & Friedrich, MICCAI 2025) | GT containment | high-prob. | no | dilation |
| FDR instance-seg (2025, arXiv 2504.04482) | pixel FDR or FNR | expectation | no (one at a time) | thresholding |
| Per-organ recall (2026, arXiv 2608.18193) | voxel FNR per organ | high-prob. (RCPS) | Bonferroni across organs | thresholding |
| Andéol+ 2023 (detection) | box-wise recall *or* pixel recall | expectation | no | box margins |
| LTT (Angelopoulos+ 2021) | any | high-prob., FWER | yes | any |
| **This work** | **pixel + GT-lesion + small-lesion** | **expectation, simultaneous** | **yes (union / max form)** | **lesion-equalized, tuned** |

The one-sentence novelty: *the first segmentation risk-control method whose
guarantee is stated on ground-truth lesions — every lesion, and small ones
separately — simultaneously with pixel FNR, plus a nested family designed for
that loss rather than inherited from the pixel loss.*
