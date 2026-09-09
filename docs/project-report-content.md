# BCSE497J Project-I Report Content

## Cover Page

**BCSE497J - PROJECT-I**

**PUMS: PERCEPTION UNCERTAINTY MANAGEMENT SYSTEM**

**A Post-hoc Hierarchical Uncertainty Calibration Framework for Dense Medical Vision Models under Distribution Shift**

**23BCE0631**  
**VANSH BHATIYA**

**23BCE0701**  
**MAHENDRA SAJJAN CHOUDHARY**

Under the supervision of

**Dr. Shashank Mouli Satapathy**  
Faculty Guide  
School of Computer Science and Engineering (SCOPE)

B.Tech. in Computer Science and Engineering  
Vellore Institute of Technology  
September 2026

---

## Abstract

Medical image segmentation systems can produce confident predictions even when the input image is ambiguous or differs from the data used to develop the model. In clinical decision-support settings, a segmentation mask without a reliable indication of uncertainty is therefore incomplete. This project presents PUMS, a post-hoc and model-agnostic framework for estimating, calibrating, and communicating uncertainty in dense medical-image predictions.

PUMS processes an uploaded image through preprocessing, segmentation, predictive uncertainty estimation, hierarchical aggregation, calibration, distribution-shift detection, and transparent risk assessment. Pixel-level uncertainty is calculated from the model's predictive probabilities using normalized predictive entropy. The same dense uncertainty field is then aggregated into localized high-uncertainty regions and global image-level statistics. Split Conformal Risk Control is used to provide a finite-sample marginal risk-control mechanism under its exchangeability assumption. A composite intensity-domain detector identifies deviations from a reference image distribution and reports when the calibration assumption may no longer hold. The final risk signal is an itemized weighted combination of uncertainty, model confidence, shift, and calibration error, with explicit review overrides.

The prototype is evaluated using the Medical Segmentation Decathlon Task 04 Hippocampus MRI dataset. The measured supervised evaluation baseline is a random-forest pixel classifier using multiscale image features. The current measured baseline on the held-out test split is Dice 0.524, IoU 0.361, ECE 0.111, and uncertainty-error correlation 0.512. At runtime, PUMS prefers the trained random-forest adapter when its artifact is available and otherwise falls back to the intensity-mixture adapter. The system also provides a Next.js interface for image upload, segmentation visualization, uncertainty heatmaps, calibration summaries, shift warnings, and risk explanations. PUMS is a research prototype and is not a clinically validated medical device.

**Keywords:** medical image segmentation, uncertainty quantification, uncertainty calibration, conformal risk control, distribution shift, predictive entropy, decision support, MRI.

---

## 1. INTRODUCTION

### 1.1 Background

Medical image analysis increasingly uses machine-learning models to segment anatomical structures and other clinically relevant regions. Segmentation systems are useful because they convert complex images into structured masks that can support measurement, visualization, and downstream analysis. However, a predicted mask does not by itself indicate whether the model is reliable for a particular image. A model may be highly confident on an image that contains acquisition artifacts, unusual anatomy, poor contrast, or a distribution shift from the development data.

Uncertainty quantification addresses this problem by estimating how unsure a model is about its prediction. In medical imaging, uncertainty can be examined at multiple levels. Pixel-level uncertainty localizes ambiguous boundaries and difficult regions. Object- or region-level uncertainty summarizes the reliability of a structure or connected component. Global uncertainty describes the reliability of the complete image. Calibration is required so that a numerical confidence or risk value corresponds reasonably to observed errors. Distribution-shift detection is also important because many statistical guarantees rely on the test input being exchangeable with the calibration data.

PUMS is designed as a post-hoc layer that consumes model probabilities and adds these reliability signals without requiring a new segmentation architecture. The present implementation focuses on 2-D slices from hippocampus MRI and provides a transparent research workflow rather than a clinical diagnosis.

### 1.2 Motivation

The motivation for this project is the gap between prediction and trustworthy prediction. A segmentation model that returns only a mask makes it difficult for a reviewer to distinguish a reliable result from one that needs inspection. A single image-level confidence score is also insufficient because errors are often concentrated around boundaries or small regions. Furthermore, calibration results may be misleading when the input distribution has changed.

The project is motivated by four practical needs:

1. Produce a numerical uncertainty map directly from the model's predictive probabilities.
2. Summarize the same uncertainty information consistently at pixel, region, and global levels.
3. Calibrate uncertainty and communicate the assumptions behind the calibration result.
4. Identify distribution shift and expose when a calibration guarantee may no longer remain valid.

The system is also motivated by research integrity. Every result should state its implementation status, data provenance, assumptions, and validity warnings. Synthetic, computed, and unavailable values should not be presented as equivalent evidence.

### 1.3 Scope of the Project

The project covers the design and implementation of a prototype framework for post-hoc uncertainty management in 2-D medical-image segmentation. The primary dataset is the Medical Segmentation Decathlon Task 04 Hippocampus dataset, which contains MRI volumes and expert annotations for hippocampus segmentation. The current workflow uses patient-disjoint train, calibration, and test splits. The measured supervised baseline is a random-forest pixel classifier trained on multiscale image features; the runtime also includes an unsupervised intensity-mixture fallback for environments where the trained artifact is unavailable.

The scope includes image preprocessing, PNG/JPEG/BMP/TIFF and single-frame DICOM loading, segmentation probabilities, normalized predictive entropy, top-2 margin and least-confidence alternatives, hierarchical uncertainty aggregation, split Conformal Risk Control, temperature scaling, input-space shift detection, transparent risk scoring, evaluation metrics, a FastAPI backend, and a Next.js frontend.

The scope does not include clinical validation, deployment for patient care, a clinical-grade deep segmentation model, multi-site clinical data, a guarantee under distribution shift, full 3-D inference, authentication, multi-user storage, or a claim that the current method is definitely novel. The shifted examples included in the prototype are simulated acquisition changes and do not represent a real second hospital or scanner site.

---

## 2. PROJECT DESCRIPTION AND GOALS

### 2.1 Literature Review

**Medical image segmentation and reliability.** Medical segmentation systems have progressed from classical probabilistic and graphical approaches to deep convolutional and transformer-based architectures. Classical models such as hidden Markov and Markov random field methods provide probabilistic formulations but may have limited representational capacity. Modern systems generally achieve better segmentation accuracy, but high predictive confidence does not automatically imply correctness. This has motivated the study of uncertainty quantification, calibration, selective prediction, and out-of-distribution detection for medical imaging.

**Uncertainty quantification.** Uncertainty is commonly separated into aleatoric uncertainty, arising from ambiguity or noise in the data, and epistemic uncertainty, arising from limited model knowledge. Bayesian approximations, deep ensembles, Monte Carlo dropout, evidential methods, and predictive-distribution statistics are common approaches. PUMS uses single-pass post-hoc estimators because the project prioritizes lightweight inference and model-agnostic integration. Normalized predictive entropy is the default estimator, with margin uncertainty and least confidence available as alternatives.

**Calibration.** Calibration measures whether predicted probabilities or confidence values correspond to observed frequencies or errors. Temperature scaling is a common post-hoc calibration technique. Conformal prediction and Conformal Risk Control provide distribution-free risk-control procedures under exchangeability assumptions. PUMS implements split Conformal Risk Control using a false-negative-rate loss for foreground segmentation. The finite-sample correction is included explicitly, and the uncorrected empirical threshold is retained as a comparison baseline.

**Spatial and hierarchical uncertainty.** Medical segmentation uncertainty is spatially structured. Boundary pixels are often more ambiguous than interior pixels, and a small high-risk region can be hidden by a benign image-level average. Existing approaches study spatially-aware conformal prediction, region-wise uncertainty, hierarchical uncertainty, and uncertainty-aware segmentation. PUMS represents uncertainty with one dense field and derives region-level and global summaries from that same field, which makes internal consistency testable.

**Distribution shift.** Differences in scanners, acquisition protocols, preprocessing, patient populations, and pathology can change the input distribution. Under distribution shift, calibration performance and conformal guarantees may degrade. PUMS therefore includes an interpretable composite detector based on intensity histogram Wasserstein distance, Jensen-Shannon distance, standardized moments, and edge-density deviation. The detector reports a shift band and validity warning; it does not claim to restore calibration under shift.

**Decision support and research integrity.** A useful uncertainty system should produce more than a heatmap. It should explain which factors contribute to a review recommendation, carry assumptions alongside the result, and distinguish computed measurements from synthetic or mock values. PUMS applies these principles in its API schemas and frontend. Planned methods are listed but cannot be selected as implemented methods, and metrics that require unavailable clinical annotations are returned as unavailable rather than replaced by invented values.

### 2.2 Research Gap and Novelty Status

The project addresses a practical gap between separate uncertainty components. Many existing works estimate uncertainty at one level, calibrate confidence, use spatial information, or detect distribution shift. These components are not always represented in a single transparent pipeline that links pixel evidence to region and image summaries while also reporting calibration assumptions and shift warnings.

The proposed contribution is framed cautiously as a **novelty candidate**, not as an established novelty claim. A targeted prior-art search remains necessary before making a definitive claim in a thesis or paper. The current working formulation is:

> The project investigates uncertainty propagation across semantic abstraction levels by mapping pixel-level uncertainty into region- and image-level summaries and quantifying uncertainty that is not explained by lower-level evidence.

The strongest future research direction is a formal transport or push-forward formulation:

> **Object uncertainty = transported pixel uncertainty + semantic residual.**

In this formulation, the potentially novel quantity is the semantic residual: uncertainty that cannot be explained by the transport of lower-level pixel uncertainty into a higher-level object representation. Optimal transport may be used as the mathematical tool for defining or measuring this mapping, rather than being presented as the novelty by itself.

The following claims are not made:

- Multi-level uncertainty estimation alone is not claimed as novel.
- Use of optimal transport for uncertainty alone is not claimed as novel.
- Uncertainty propagation alone is not claimed as novel.
- The current prototype does not implement the full transport geometry or conservation law.
- The literature search does not prove that no prior work has used a similar mechanism.

The current confidence assessment is approximately 7/10 for uncertainty transport geometry and 8/10 as a research hypothesis for uncertainty conservation or creation. These values are planning judgments, not experimental results. A second search should cover uncertainty propagation across semantic hierarchies, optimal-transport terminology, medical and non-medical literature, and patents.

### 2.3 Objectives

**Primary objective:** Develop and evaluate a post-hoc, model-agnostic framework that estimates, calibrates, localizes, and communicates segmentation uncertainty at pixel, region, and image levels under possible distribution shift.

**Specific objectives:**

1. Build a preprocessing pipeline for medical images and normalize inputs to a reproducible inference resolution.
2. Implement a baseline segmentation model that produces pixel-wise class probabilities on hippocampus MRI.
3. Calculate normalized predictive entropy and alternative uncertainty estimators directly from model probabilities.
4. Derive region-level uncertainty from high-uncertainty connected components or detection regions.
5. Aggregate the same uncertainty field into global statistics, including mean, percentiles, boundary uncertainty, interior uncertainty, and high-uncertainty area.
6. Implement split Conformal Risk Control with an explicit finite-sample correction and a false-negative-rate loss.
7. Implement temperature scaling and evaluation metrics including ECE, Brier score, AURC, Dice, IoU, and uncertainty-error correlation.
8. Detect input-space distribution shift using interpretable intensity and texture indicators.
9. Produce an itemized risk signal with visible factor contributions, assumptions, validity warnings, and review overrides.
10. Provide a web interface and API for analyzing images and inspecting uncertainty maps.
11. Evaluate the prototype on patient-disjoint real MRI data and disclose limitations honestly.
12. Formulate the future transport/residual-uncertainty hypothesis without making an unsupported definitive novelty claim.

### 2.4 Problem Statement

Medical image segmentation models may output confident predictions even when their predictions are wrong, spatially ambiguous, or affected by distribution shift. Existing workflows often display a segmentation mask or a single confidence value without linking pixel-level uncertainty to localized regions, global reliability, calibration validity, and shift warnings. This makes it difficult for a reviewer to identify which part of an image requires attention and whether a numerical confidence value remains trustworthy.

The problem addressed by this project is the design of an integrated and transparent uncertainty-management layer for dense medical-image predictions. Given an image and a segmentation model's predictive probabilities, the system should produce a spatial uncertainty map, consistent region and global summaries, a calibrated risk-control result, a distribution-shift assessment, and an explainable review recommendation. The system must also state its assumptions and distinguish computed evidence from synthetic or unavailable evidence.

### 2.5 Project Plan

| Phase | Activities | Deliverable |
| --- | --- | --- |
| 1 | Requirements, literature review, dataset and licence review | Problem definition and scope |
| 2 | Dataset downloader, patient-wise split handling, preprocessing | Reproducible data pipeline |
| 3 | Baseline segmentation model and probability output | Working model adapter |
| 4 | Pixel uncertainty and hierarchical aggregation | Pixel, region, and global UQ |
| 5 | Conformal calibration and evaluation metrics | Calibration service |
| 6 | Shift detector and transparent risk engine | Shift-aware risk report |
| 7 | FastAPI endpoints and Next.js interface | End-to-end prototype |
| 8 | Testing, held-out evaluation, limitation analysis | Project-I evaluation |
| 9 | Targeted novelty search and transport formulation | Project-II research direction |

---

## 3. TECHNICAL SPECIFICATION

### 3.1 Requirements

#### 3.1.1 Functional

- Accept PNG, JPEG, BMP, TIFF, and single-frame DICOM images.
- Reject unsupported, malformed, oversized, or geometrically invalid input with a structured error.
- Normalize inputs to the configured 256 x 256 inference resolution.
- Execute a registered segmentation adapter and return class probabilities and a point prediction.
- Calculate pixel uncertainty in [0, 1].
- Derive region and global summaries from the same dense uncertainty field.
- Support implemented calibration methods and report their assumptions.
- Compare an uploaded image with a reference domain and report a shift band.
- Produce an itemized risk score with factor contributions and review overrides.
- Display the source image, segmentation, uncertainty heatmap, region summaries, calibration, shift, and risk information.
- Calculate supported evaluation metrics and return a reason when required data is unavailable.
- Expose implementation status, data provenance, assumptions, and validity warnings.
- Provide API endpoints for health, models, analyses, calibration, shift, evaluation, and media.
- Record model id, configuration, seed, checksum, pipeline version, and thresholds.

#### 3.1.2 Non-Functional

- Uncertainty values shall be derived from model probabilities and not fabricated for visualization.
- Repeated analyses with the same image, configuration, model, and seed shall be reproducible within documented tolerances.
- Every risk score shall expose its factors, weights, normalized values, and contributions.
- Missing optional dependencies and disabled stages shall be handled explicitly.
- The interface shall be usable by a researcher without requiring raw JSON inspection.
- Backend modules shall remain separated by responsibility.
- New model adapters and datasets shall be addable without changing downstream UQ logic.
- The CPU baseline shall support single-image inference at 256 x 256 after warm-up.
- The application shall state that it is a research decision-support prototype, not a diagnostic device.
- Dataset licences and citations shall be surfaced and archives shall not be silently redistributed.

### 3.2 Feasibility Study

#### 3.2.1 Technical Feasibility

The project is technically feasible because it uses mature open-source components and a modular model interface. Python, NumPy, SciPy, scikit-learn, scikit-image, OpenCV, Pillow, pydicom, FastAPI, Next.js, React, TypeScript, Zod, and pytest provide the required capabilities. The measured supervised baseline is CPU-compatible, and the runtime includes an intensity-mixture fallback, so a GPU is optional. The API contract is defined with Pydantic schemas and checked in the frontend with Zod schemas.

The dataset is downloaded explicitly and is not bundled in the repository. Patient-disjoint splitting prevents adjacent slices from the same patient from appearing on both sides of an evaluation boundary. The modular design allows a future U-Net, nnU-Net, MONAI, or foundation-model adapter to provide probabilities to the same uncertainty and calibration stack.

Technical risks include the modest accuracy of the baseline model, the limited 2-D scope, missing clinical validation, and the fact that the current shift detector operates on input statistics rather than learned semantic features. These risks do not prevent a Project-I prototype.

#### 3.2.2 Economic Feasibility

The prototype can be developed using open-source software and a standard development laptop. The default workflow does not require paid cloud infrastructure or a GPU. The primary dataset is openly licensed under CC BY-SA 4.0, subject to its terms and required citation. Optional deep-learning dependencies may increase installation size and computational requirements but are not necessary for the baseline.

The expected economic benefit is methodological: the system provides a low-cost platform for studying calibrated uncertainty before investing in larger models or clinical infrastructure. It is not currently a product or clinical service, so no financial return is claimed.

#### 3.2.3 Social Feasibility

The project is socially feasible as a research and educational prototype when its limitations are made visible. The system supports human review rather than replacing clinical judgment. It does not report a diagnosis or a patient's health status. It reports the estimated reliability of the model for the uploaded image.

The main social risks are over-trust in numerical confidence, bias caused by limited data, and misuse of simulated shift results as evidence of clinical robustness. These risks are addressed through disclaimers, provenance badges, validity warnings, explicit assumptions, conservative review overrides, and a statement that no clinical validation has been performed.

### 3.3 System Specification

#### 3.3.1 Hardware Specification

Replace these recommended values with the actual machine used for experiments.

| Component | Specification |
| --- | --- |
| Processor | Modern 64-bit multi-core CPU; 4 cores or more recommended |
| Memory | 8 GB RAM recommended; 4 GB minimum for the CPU baseline |
| Storage | At least 10 GB free for environment, dataset, logs, and outputs |
| GPU | Optional; not required for the default baseline |
| Display | 1366 x 768 or higher |
| Network | Required for dependencies and licensed dataset download |

#### 3.3.2 Software Specification

| Category | Specification |
| --- | --- |
| Operating system | Windows, Linux, or macOS |
| Backend | Python 3.11+, FastAPI, Pydantic v2 |
| Frontend | Next.js 16, React 19, TypeScript |
| Scientific libraries | NumPy, SciPy, scikit-learn, scikit-image, OpenCV, Pillow, pydicom, nibabel, joblib |
| Frontend libraries | Tailwind CSS, Radix-style primitives, Lucide, TanStack Query, Zustand, Recharts, Zod |
| Testing | pytest, Node test runner, TypeScript type checking |
| Storage | JSON or in-memory prototype storage |
| API documentation | FastAPI OpenAPI, Swagger UI, ReDoc |
| Optional | PyTorch, torchvision, MONAI |

---

## 4. DESIGN APPROACH AND DETAILS

### 4.1 System Architecture

~~~text
Uploaded Image
      |
      v
Preprocessing and Validation
      |
      v
Model Adapter -> Class Probabilities and Point Prediction
      |
      +--------------------+
      |                    |
      v                    v
Pixel Uncertainty     Global Statistics
      |
      v
Hierarchical Engine
      |
      +--------------------+
      |                    |
      v                    v
Region Uncertainty   Image-Level Uncertainty
      |                    |
      +----------+---------+
                 |
                 v
        Calibration Layer
                 |
                 v
       Distribution Shift Detector
                 |
                 v
            Risk Engine
                 |
                 v
       FastAPI Response and PNG Media
                 |
                 v
        Next.js Visualization UI
~~~

### 4.2 Design

#### 4.2.1 Data Flow Diagram

**External entities:** Researcher/Reviewer, Image File, Dataset and Calibration Store.

**Processes:** Upload and validate, preprocess, generate probabilities, estimate uncertainty, aggregate hierarchy, calibrate, detect shift, compute risk, visualize and export.

~~~text
Researcher -> Upload Image -> Validate and Preprocess
                                      |
                                      v
                              Model Inference
                                      |
                       probabilities, mask, metadata
                                      |
                +---------------------+---------------------+
                |                     |                     |
                v                     v                     v
       Pixel UQ            Calibration             Shift Detector
                |                     |                     |
                +----------+----------+----------+----------+
                           v
                    Risk Engine
                           |
                           v
                 Analysis Response
                           |
                           v
                     Web UI
~~~

#### 4.2.2 Use Case Diagram Content

**Researcher/Reviewer:** upload an MRI image; select analysis options; run segmentation; view the predicted mask; inspect the pixel heatmap; adjust the threshold; inspect uncertain regions; review global uncertainty; inspect calibration assumptions; inspect shift warnings; read the risk explanation; compare original and simulated-shifted images; retrieve previous analyses.

**Developer/Administrator:** register a model adapter; register a dataset and licence; configure thresholds; run evaluation and self-tests; inspect API health and optional capabilities.

#### 4.2.3 Class Diagram Content

~~~text
BaseMedicalModel
  |-- RandomForestPixelSegmenter
  |-- RegionProposalDetector
  |-- Optional CNN Adapter

BaseUncertaintyEstimator
  |-- PredictiveEntropyEstimator
  |-- MarginConfidenceEstimator
  |-- LeastConfidenceEstimator

BaseCalibrationMethod
  |-- ConformalRiskControl
  |-- TemperatureScaling
  |-- BaselineThresholdCalibration

BaseShiftDetector
  |-- CompositeIntensityShiftDetector

AnalysisService
  --> PreprocessingPipeline
  --> ModelRegistry
  --> HierarchicalUncertaintyEngine
  --> CalibrationRegistry
  --> ShiftDetector
  --> RiskAssessmentService
~~~

#### 4.2.4 Sequence Diagram Content

~~~text
Researcher -> Frontend: Select image and options
Frontend -> Backend: POST /api/analyze
Backend -> Preprocessor: Validate, load, normalize, resize
Preprocessor -> Model Adapter: Preprocessed image
Model Adapter -> Backend: Probabilities and prediction
Backend -> UQ Estimator: Probability tensor
UQ Estimator -> Hierarchical Engine: Dense uncertainty map
Hierarchical Engine -> Backend: Pixel, region, global summaries
Backend -> Calibration: Apply selected method
Backend -> Shift Detector: Compare with reference profile
Backend -> Risk Engine: Combine factors and overrides
Risk Engine -> Backend: Risk assessment and warnings
Backend -> Frontend: Response and media URLs
Frontend -> Researcher: Render results
~~~

### Mathematical Formulation

For model probabilities p_c(x) at pixel x, the default uncertainty estimator is normalized predictive entropy:

H_norm(x) = - sum_c p_c(x) log(p_c(x)) / log(C)

The value lies in [0, 1]. The global score is the mean of the dense map. A high-uncertainty mask is obtained by thresholding it. Connected components form region-level candidates, and their mean and peak values are reported.

For Conformal Risk Control:

S_lambda(X) = { x : p_foreground(x) >= 1 - lambda }

The selected threshold uses the finite-sample corrected risk:

(n * empirical_risk(lambda) + B) / (n + 1) <= alpha

The project risk score is:

risk = w_u * uncertainty + w_c * low_confidence + w_s * distribution_shift + w_k * calibration_error

The current default weights are 0.40, 0.25, 0.20, and 0.15. Overrides force REVIEW_REQUIRED for a large high-uncertainty area, a high peak uncertainty in any region, or severe distribution shift.

---

## 5. REFERENCES

[1] M. Antonelli, A. Reinke, S. Bakas, et al., "The Medical Segmentation Decathlon," Nature Communications, vol. 13, article 4128, 2022.

[2] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and J. Schuster, "Conformal Risk Control," in International Conference on Learning Representations, 2024.

[3] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. Springer, 2005.

[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proceedings of ICML, 2017, pp. 1321-1330.

[5] A. Kendall and Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" in Advances in Neural Information Processing Systems, 2017.

[6] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," in Advances in Neural Information Processing Systems, 2017.

[7] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in Proceedings of ICML, 2016, pp. 1050-1059.

[8] S. Lambert, E. Forbes, C. Doyle, A. Dehaene, and J. Dojat, "Trustworthy Clinical AI: A Unified Review of Uncertainty Quantification in Medical Image Analysis," Artificial Intelligence in Medicine, 2024.

[9] J. Blot et al., "Automatically Adaptive Conformal Risk Control," 2024. Planned extension referenced by the project implementation.

[10] A. Bereska, R. Karimi, and J. Samavi, "Spatially-Aware Conformal Prediction," in Medical Imaging with Deep Learning, 2025. Planned extension referenced by the project implementation.

[11] A. Zecchin and D. Simeone, "Localized Adaptive Risk Control," 2024. Planned extension referenced by the project implementation.

[12] Y. Zou et al., "DEviS: Reliable Medical Segmentation via Evidential Calibrated Uncertainty," IEEE Transactions on Cybernetics, 2025. Planned extension referenced by the project implementation.

[13] J. Zhang, M. Brady, and S. Smith, "Segmentation of Brain MR Images through a Hidden Markov Random Field Model and the Expectation-Maximization Algorithm," IEEE Transactions on Medical Imaging, vol. 20, no. 1, pp. 45-57, 2001.

[14] PUMS project repository, README.md, docs/architecture.md, docs/api.md, and implementation modules, accessed September 2026.

---

## Claims to Keep Explicit in the Final Report

- PUMS is a research prototype and not a medical device.
- The measured supervised model is a modest random-forest baseline, and the runtime also has an intensity-mixture fallback; neither is clinically validated.
- Current results are based primarily on one MRI dataset and one modality.
- The shifted samples are simulated acquisition changes, not a real external site.
- Conformal guarantees are marginal and rely on exchangeability; shift can invalidate them.
- The current object-level risk is a region-derived summary, not a region-conditional guarantee.
- The transport/residual formulation is a future research direction and a cautious novelty candidate, not a confirmed original contribution.
