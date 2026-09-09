# Review 2 Presentation Content

This content follows the provided 13-slide Review 2 template. Template placeholders and instructions have been replaced with project-specific content.

## Slide 1 - Title

**Programme:** B.Tech

**Course Code - Course Title:** BCSE497J - Project-I

**Project Title:** PUMS: Perception Uncertainty Management System

**Subtitle:** A Post-hoc Hierarchical Uncertainty Calibration Framework for Dense Medical Vision Models under Distribution Shift

**Team:**
- Vansh Bhatiya (23BCE0631)
- Mahendra Sajjan Choudhary (23BCE0701)

**Faculty Guide:** Dr. Shashank Mouli Satapathy

School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology

## Slide 2 - Approval Mail From Guide

**Approval requested for:**

- Project title and problem statement.
- Use of the Medical Segmentation Decathlon Task 04 Hippocampus MRI dataset.
- Post-hoc hierarchical uncertainty estimation at pixel, region, and image levels.
- Split Conformal Risk Control and temperature scaling as calibration methods.
- Input-space distribution-shift detection and transparent risk scoring.
- Use of a cautious novelty candidate concerning uncertainty propagation and semantic residual uncertainty.
- Research-prototype scope with no clinical or diagnostic claims.

**Approval status:** To be confirmed by the faculty guide through the required email process.

## Slide 3 - Aim

**Aim**

To develop a transparent, post-hoc, and model-agnostic framework that estimates and calibrates uncertainty in medical-image segmentation at pixel, region, and global levels, while detecting distribution shift and communicating when a prediction should be reviewed.

**Central idea**

~~~text
Model probabilities -> Pixel uncertainty -> Region/global summaries
                    -> Calibration -> Shift warning -> Explainable risk signal
~~~

## Slide 4 - Abstract

Medical segmentation models can be confidently wrong when an image is ambiguous or differs from the development distribution. PUMS adds a reliability layer on top of a dense predictor. It computes normalized predictive entropy from pixel-wise probabilities, aggregates one uncertainty field into local regions and global statistics, applies split Conformal Risk Control, detects intensity-domain shift, and produces an itemized review signal.

The prototype uses patient-disjoint MRI data from MSD Task04 Hippocampus and a CPU-compatible random-forest segmentation baseline for the reported evaluation. At runtime, PUMS prefers the trained random-forest adapter when available and otherwise falls back to an intensity-mixture adapter. Current held-out results are Dice 0.524, IoU 0.361, ECE 0.111, and uncertainty-error correlation 0.512. A FastAPI backend and Next.js interface make the result inspectable. The system is a research prototype, not a clinically validated medical device.

## Slide 5 - Literature Review

**Existing research areas**

- Medical image segmentation: probabilistic models, CNNs, U-Net-style systems, and foundation models.
- Uncertainty quantification: Bayesian approximations, ensembles, Monte Carlo dropout, evidential methods, entropy, margin, and confidence measures.
- Calibration: temperature scaling, reliability diagrams, ECE, Brier score, selective prediction, and conformal methods.
- Risk control: split Conformal Risk Control for finite-sample control of a chosen loss under exchangeability.
- Spatial UQ: boundary uncertainty, region-wise analysis, spatially-aware conformal prediction, and hierarchical summaries.
- Distribution shift: histogram, feature-space, and model-based out-of-distribution detection.
- Trustworthy clinical AI: explicit assumptions, provenance, uncertainty communication, and human review.

**Key references:** Angelopoulos et al. (2024), Guo et al. (2017), Kendall and Gal (2017), Lakshminarayanan et al. (2017), Lambert et al. (2024), Antonelli et al. (2022), and planned SACP/adaptive risk-control extensions.

## Slide 6 - Research Gap

Existing approaches often address one part of the problem:

- A heatmap localizes uncertainty but may not be calibrated.
- Calibration may assume exchangeability and may not warn about shift.
- A global score can hide a small but important uncertain region.
- Region-level values may be produced independently from pixel-level values, making consistency difficult to verify.

**Project gap:** an integrated and transparent pipeline that starts with one probability-derived uncertainty field, derives consistent pixel/region/global summaries, calibrates a risk quantity, detects shift, and exposes assumptions and provenance.

**Novelty status:** cautious candidate only.

> Object uncertainty = transported pixel uncertainty + semantic residual.

The semantic residual is the proposed future research object. A targeted search across semantic hierarchies, optimal transport, medical and non-medical literature, and patents is still required before making a definitive novelty claim.

## Slide 7 - Objectives, SDGs, and Outcomes

### Objectives

1. Build a reproducible medical-image preprocessing and segmentation pipeline.
2. Estimate pixel uncertainty directly from model probabilities.
3. Aggregate uncertainty at region and image levels.
4. Implement conformal calibration and temperature scaling.
5. Detect input-space distribution shift and emit validity warnings.
6. Generate an itemized and explainable review risk signal.
7. Evaluate the prototype on patient-disjoint hippocampus MRI.
8. Provide a usable API and web interface.

### Relevant Sustainable Development Goals

- **SDG 3: Good Health and Well-Being** - supports research into safer medical AI and medical-image analysis.
- **SDG 9: Industry, Innovation and Infrastructure** - develops an extensible and reproducible AI software framework.
- **SDG 4: Quality Education** - demonstrates responsible AI engineering, uncertainty, calibration, and research integrity.

### Expected outcomes

- Open and reproducible Project-I prototype.
- Calibrated, hierarchical uncertainty report for a segmentation analysis.
- Shift-aware warnings and transparent review recommendation.
- Evaluation and test suite with explicit data provenance.
- Refined Project-II direction for transport geometry and semantic residual uncertainty.

## Slide 8 - Framework / Architecture / Block Diagram

~~~text
Image Upload
    |
    v
Validation and Preprocessing
    |
    v
Segmentation Model Adapter
    |
    v
Class Probabilities and Mask
    |
    +-----------------------+
    |                       |
    v                       v
Pixel Uncertainty      Global Statistics
    |
    v
Hierarchical Aggregation
    |
    +-----------------------+
    |                       |
    v                       v
High-UQ Regions        Image-Level UQ
    |                       |
    +-----------+-----------+
                v
         Calibration Layer
                |
                v
      Distribution-Shift Detector
                |
                v
           Risk Engine
                |
                v
       FastAPI -> Next.js UI
~~~

**Core design rule:** Levels 1 and 2 are aggregations of the Level 3 map, so the hierarchy is internally consistent by construction.

## Slide 9 - Functional Requirements

- Upload and validate supported medical-image formats.
- Normalize images to a reproducible 256 x 256 inference resolution.
- Run a registered segmentation model and return class probabilities.
- Calculate normalized predictive entropy, margin, or least-confidence uncertainty.
- Display pixel uncertainty as a numeric heatmap.
- Identify connected high-uncertainty regions.
- Report global uncertainty, boundary/interior uncertainty, and high-UQ fraction.
- Apply calibration methods and report ECE, Brier, coverage, and risk-coverage values.
- Detect intensity-domain shift using interpretable indicators.
- Generate a weighted risk score with review overrides.
- Display provenance, assumptions, validity warnings, and disclaimers.
- Store and retrieve analyses through the API.

## Slide 10 - Modules

**Backend modules**

- preprocessing: validation, image loading, normalization, resizing, DICOM handling.
- models: adapter interface, registry, random-forest segmentation baseline, region proposal adapter, optional CNN adapter.
- uncertainty: entropy, margin, least-confidence, and hierarchical aggregation.
- calibration: Conformal Risk Control, temperature scaling, empirical threshold baseline, reliability metrics.
- shift: Wasserstein, Jensen-Shannon, moments, texture, and shift bands.
- risk: weighted factors, thresholds, override rules, referral priority.
- evaluation: Dice, IoU, ECE, Brier, AURC, selective risk, uncertainty-error correlation.
- api: health, models, analysis, calibration, shift, evaluation, and media routes.

**Frontend modules**

- Analysis upload and configuration.
- Image viewer and prediction overlay.
- Uncertainty heatmap and legend.
- Region uncertainty panel.
- Calibration and shift summaries.
- Research status and provenance badges.

## Slide 11 - Experiments and Results

### Dataset

- Medical Segmentation Decathlon Task04 Hippocampus.
- MRI modality; target: hippocampus head/anterior and body/posterior.
- 260 training volumes and 130 test volumes in the dataset metadata.
- Patient-wise train, calibration, and test separation.
- CC BY-SA 4.0 licence with citation recorded in the project manifest.

### Current baseline results

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Dice | 0.524 | Modest segmentation baseline |
| IoU | 0.361 | Consistent with a non-deep baseline |
| ECE | 0.111 | Measurable calibration error |
| Uncertainty-error correlation | 0.512 | Uncertainty is related to error locations |

### System checks

- Uncertainty map is recomputed from returned probabilities in a dedicated test.
- Hierarchical levels are derived from one dense map.
- Conformal finite-sample correction is tested to be conservative relative to the uncorrected baseline.
- Risk score equals the sum of declared factor contributions.
- Uncertainty PNG round-trips to the reported numeric summary.
- Patient-wise splitting is tested to avoid slice-level leakage.
- Frontend API responses are checked against runtime schemas.

### Shift experiment

Original test slices and shifted variants can be compared in the interface. The shifted variants use simulated gamma, bias-field, and noise corruption. They demonstrate the detector and warning path but are not evidence from a real external scanner site.

## Slide 12 - Conclusion

PUMS demonstrates a complete prototype for uncertainty management in dense medical-image predictions. Its main engineering result is an end-to-end path from model probabilities to pixel heatmaps, localized regions, global uncertainty, calibration, shift warnings, and explainable review risk.

The implementation is deliberately transparent: provenance, assumptions, limitations, and factor contributions travel with each result. The current baseline is useful for research validation but is not a clinical model. The main remaining research gap is to formalize uncertainty propagation across semantic levels and measure a semantic residual that cannot be explained by transported lower-level uncertainty.

**Next steps**

- Perform the targeted prior-art and patent search.
- Implement a formal transport/push-forward operator.
- Define and validate a semantic residual uncertainty measure.
- Add localized or spatially conditional calibration.
- Evaluate on additional datasets, modalities, and real external domains.
- Replace the baseline segmenter with a stronger medical segmentation model.

## Slide 13 - References

[1] M. Antonelli, A. Reinke, S. Bakas, et al., "The Medical Segmentation Decathlon," Nature Communications, 2022.

[2] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and J. Schuster, "Conformal Risk Control," ICLR, 2024.

[3] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, 2017.

[4] A. Kendall and Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS, 2017.

[5] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," NeurIPS, 2017.

[6] S. Lambert, E. Forbes, C. Doyle, A. Dehaene, and J. Dojat, "Trustworthy Clinical AI: A Unified Review of Uncertainty Quantification in Medical Image Analysis," Artificial Intelligence in Medicine, 2024.

[7] A. Bereska, R. Karimi, and J. Samavi, "Spatially-Aware Conformal Prediction," MIDL, 2025. Planned extension.

[8] J. Zhang, M. Brady, and S. Smith, "Segmentation of Brain MR Images through a Hidden Markov Random Field Model and the Expectation-Maximization Algorithm," IEEE TMI, 2001.
