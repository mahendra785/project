# Related work and novelty check (searched 2026-10-14)

Search strings run (arXiv + web, 2021→2026-10): the five strings in
`w1/theory-memo.md` §10, plus "lesion-wise / instance-wise conformal
segmentation", "per-lesion detection guarantee conformal", "multi-risk conformal
segmentation", "Learn then Test segmentation".

| Paper | Loss granularity | Guarantee | Multiple risks jointly? | Size-stratified? | How this work differs |
|---|---|---|---|---|---|
| Angelopoulos et al., *Conformal Risk Control*, ICLR 2024 | any monotone loss; demo uses pixel FNR | expectation | no | no | we use CRC as the engine; contribution is the losses + family |
| Angelopoulos et al., *Learn then Test*, arXiv 2110.01052 | any | high-prob., FWER | yes | no | baseline in our harness; ours gives expectation control at smaller sets (measured) |
| Tan et al., *Conformal Lesion Segmentation for 3D Medical Images*, arXiv 2510.17897 | **voxel** FNR per volume (despite the "lesion-level" wording) | high-prob. per sample (ECR) | no | no; small lesions motivate the paper but the method does not treat them | our loss counts GT **components**; a missed small lesion costs 1/K, not its voxel share |
| Luo & Zhou, *Conditional Conformal Risk Adaptation* (CRA/CCRA/CCRA-S), arXiv 2504.07611 | pixel FNR per image | expectation, per stratum | no | strata = total predicted probability per image | baseline (`union_stratified`); strata are images, not lesions |
| *ReCIRC: Rectified CRC*, arXiv 2609.38112 (2026) | pixel FNR per image | marginal + approx. conditional | no | no | orthogonal (per-input budget); could be combined with our family |
| Mossina & Friedrich, *Morphological prediction sets*, MICCAI 2025 (arXiv 2503.05618) | GT-mask containment | high-prob. | no | no | dilation family is a baseline (`union_dilation`) |
| *Controlling False Positives via CP*, arXiv 2511.15406 | pixel FP / containment (shrunk masks) | high-prob. | no | no | opposite direction (FP); not lesion-level |
| *FDR in Medical Instance Segmentation via CRC*, arXiv 2504.04482 | pixel FDR **or** FNR | expectation | no (one at a time) | no | ours is component-level and simultaneous |
| *Bound-Aware Per-Organ Recall Risk Control*, arXiv 2608.18193 (2026) | voxel FNR per organ | high-prob. (RCPS) | Bonferroni over organs | no | organs ≠ lesions; no component loss |
| *sem-CRC*, MICCAI 2025 | per anatomical structure, image reconstruction | expectation | per structure | no | different task (reconstruction uncertainty) |
| *Conformal Prediction Sets for Instance Segmentation*, arXiv 2602.10045 | IoU of instance for a queried pixel | per query | no | no | instance seg. output sets; not semantic lesion miss control |
| Andéol et al., *Confident Object Detection via CP and CRC*, COPA 2023 | box-wise recall **or** pixel recall | expectation | no | no | the nearest object-level loss, but detection and one at a time |
| *CRC for Pulmonary Nodule Detection*, arXiv 2412.20167 | detection sensitivity | expectation | no | no | detection, not segmentation sets |

## Verdict

Not found anywhere:
1. a risk-control guarantee on **ground-truth lesion components** for semantic segmentation
   prediction sets;
2. **simultaneous** control of pixel FNR, lesion miss and small-lesion miss;
3. a nested family designed for a lesion loss (**lesion-equalized scores**).

(1)–(2) are an easy corollary once posed (theory memo v1, Corollary 1); the paper
states this and presents them as a *problem formulation plus measured gap*, not
as a deep theorem. (3) and the measured efficiency gain are the method
contribution. If a reviewer surfaces prior work on (3), the paper's fallback is
(1)+(2)+the empirical gap analysis, which still stand.
