# Review 3 — Slide Content (Project-II)

## 1. Title
Lesion-Level Joint Conformal Risk Control with Lesion-Equalized Prediction Sets
Mahendra Sajjan Choudhary (23BCE0701) · Vansh Bhatiya (23BCE0631) · Guide: Dr. Shashank Mouli Satapathy

## 2. Problem, in one picture
- CRC guarantees **pixel** FNR ≤ α, and large lesions dominate that average.
- A missed small lesion barely moves the number.
- Measured at the pixel-CRC threshold on Kvasir: multi-lesion images miss lesions at **1.56α**.

## 3. Gap in the literature (searched Oct 2026)
- CRC, CLS, CCRA-S, ReCIRC, per-organ recall: pixel/voxel losses, one risk.
- Morphological sets: a different family, but still a pixel/containment loss.
- Learn-Then-Test: multiple risks, but high-probability and conservative.
- **Nobody controls ground-truth-lesion risk jointly with pixel FNR.**

## 4. Contribution 1: lesion-level joint guarantee
- Lesion-miss and small-lesion-miss losses are defined on GT components (τ = 50%).
- Both are monotone under any nested family (Lemma 1); Lemma 2 reduces each lesion to one order statistic.
- λ̂ = max of the per-risk CRC thresholds gives every risk ≤ α, with no Bonferroni split.

## 5. Contribution 2: lesion-equalized prediction sets
- Within each candidate component, s = p̂ / peak^ρ.
- At ρ = 1 the score is amplitude-invariant: a faint lesion is hit at the same cutoff as a confident one of the same shape.
- No background above t₀ enters the set; ρ = 0 recovers thresholding.
- (ρ, t₀) is chosen on a tuning split, so the guarantee stays exact.

## 6. Protocol
- Pre-registered before any result, with dated amendments.
- FracAtlas (719) and Kvasir-SEG (1000); 3 seeds × 1000 tune/cal/test splits.
- Six baselines: pixel CRC, max-form, LTT, morphological, CCRA-S, and joint thresholding as the reference.

## 7. Results: validity and cost
- All guarantees held on held-out data.
- Joint lesion control costs ≤ 0.7% set size over pixel-only CRC.
- At the same lesion guarantee, LTT needs +10–46% and morphological sets +1–43% larger sets.

## 8. Results: honest negatives
- H2 (lesion-equalized smaller than joint thresholding on real data): **FAILED**; a tie (+0.7% / +1.8%).
- Why: with the CPU backbone the pixel constraint binds, so there is nothing to save.
- FracAtlas backbone Dice is 0.057, so those numbers describe a near-failing segmenter.

## 9. Simulation: when it helps
- Up to **−47%** set size when small lesions are moderately under-confident (γ = 0.5–1).
- About 0 when small lesions sink into the background (γ ≥ 1.5), where it falls back to thresholding.
- The pre-registered rule therefore FAILED, but the mechanism is confirmed in its regime.

## 10. Post-hoc: closing the multi-lesion gap
- Adding a risk conditioned on K ≥ 2: Kvasir multi-lesion miss rate 0.155 → 0.076 at α = 0.10.
- Price: 16–28% larger sets.

## 11. System
- New modules: `families.py`, `losses.py`, `joint.py`, `lesion_crc.py`, and the FracAtlas loader.
- API `GET /api/research/mgcrc` feeds the dashboard's Lesion-level CRC tab.
- 123 backend and 29 frontend tests, CI, and a one-command `scripts/reproduce`.

## 12. Next
- GPU U-Net and 3-D MSD Task08 (both implemented, not yet run).
- Weighted conformal under shift.
- Submission target: MICCAI UNSURE workshop.
