"""Evaluation service.

Two kinds of output, kept strictly separate:

1. **Session metrics** — computed over analyses this instance has actually run.
   These are real but describe *this session*, not a benchmark.
2. **Verification self-tests** — the metric implementations executed on inputs
   with analytically known answers. This demonstrates the evaluation code is
   correct without inventing dataset results.

Metrics needing ground truth (Dice, IoU, uncertainty-error correlation) are
reported as unavailable at inference time with the reason stated, because no
annotated data is present. That absence is the honest result.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.calibration.datasets import get_calibration_set
from app.core.config import Settings
from app.core.logging import get_logger
from app.evaluation.metrics import (
    area_under_risk_coverage,
    dice_coefficient,
    iou_score,
    risk_coverage_from_uncertainty,
    selective_risk,
    uncertainty_error_correlation,
)
from app.models.registry import ModelRegistry, resolve_default_model_id
from app.schemas.common import AnalysisStatus, DataProvenance, ImplementationStatus
from app.schemas.evaluation import (
    EvaluationReport,
    MetricDefinition,
    MetricValue,
    RiskCoverageSeriesPoint,
    SelfTestResult,
)
from app.services.storage import AnalysisRepository

logger = get_logger(__name__)

NO_GROUND_TRUTH = (
    "Requires annotated ground truth. No labelled dataset is registered in this "
    "instance, so this metric is deliberately not reported."
)

DEFINITIONS = [
    MetricDefinition(
        id="dice",
        name="Dice coefficient",
        category="segmentation",
        status=ImplementationStatus.IMPLEMENTED,
        description="2|A∩B| / (|A|+|B|). Implemented and unit-tested; needs labels.",
        requires_ground_truth=True,
        higher_is_better=True,
    ),
    MetricDefinition(
        id="iou",
        name="Intersection over Union",
        category="segmentation",
        status=ImplementationStatus.IMPLEMENTED,
        description="|A∩B| / |A∪B|. Implemented and unit-tested; needs labels.",
        requires_ground_truth=True,
        higher_is_better=True,
    ),
    MetricDefinition(
        id="ece",
        name="Expected Calibration Error",
        category="calibration",
        status=ImplementationStatus.IMPLEMENTED,
        description="Equal-width binned |accuracy - confidence| gap.",
        requires_ground_truth=True,
        higher_is_better=False,
    ),
    MetricDefinition(
        id="brier",
        name="Brier score",
        category="calibration",
        status=ImplementationStatus.IMPLEMENTED,
        description="Mean squared error of the predicted probability.",
        requires_ground_truth=True,
        higher_is_better=False,
    ),
    MetricDefinition(
        id="aurc",
        name="Area under risk-coverage curve",
        category="uncertainty",
        status=ImplementationStatus.IMPLEMENTED,
        description="Summarises selective prediction quality across coverages.",
        requires_ground_truth=True,
        higher_is_better=False,
    ),
    MetricDefinition(
        id="unc_error_corr",
        name="Uncertainty-error correlation",
        category="uncertainty",
        status=ImplementationStatus.IMPLEMENTED,
        description=(
            "Correlation between the uncertainty map and per-pixel error. The "
            "core sanity check that an uncertainty map is informative."
        ),
        requires_ground_truth=True,
        higher_is_better=True,
    ),
    MetricDefinition(
        id="session_mean_uncertainty",
        name="Session mean uncertainty",
        category="uncertainty",
        status=ImplementationStatus.IMPLEMENTED,
        description="Mean uncertainty over analyses run on this instance.",
        requires_ground_truth=False,
        higher_is_better=False,
    ),
    MetricDefinition(
        id="session_mean_shift",
        name="Session mean shift score",
        category="distribution_shift",
        status=ImplementationStatus.BASELINE,
        description="Mean composite shift score over analyses on this instance.",
        requires_ground_truth=False,
        higher_is_better=False,
    ),
    MetricDefinition(
        id="ood_calibration_degradation",
        name="OOD calibration degradation",
        category="distribution_shift",
        status=ImplementationStatus.PLANNED,
        description=(
            "Change in ECE between an in-distribution and an out-of-distribution "
            "split. Requires paired ID/OOD labelled datasets; not implemented."
        ),
        requires_ground_truth=True,
        higher_is_better=False,
    ),
    MetricDefinition(
        id="detection_map",
        name="Detection mAP",
        category="detection",
        status=ImplementationStatus.PLANNED,
        description="Requires a trained detection adapter and labelled boxes.",
        requires_ground_truth=True,
        higher_is_better=True,
    ),
]


class EvaluationService:
    def __init__(
        self,
        settings: Settings,
        repository: AnalysisRepository,
        registry: ModelRegistry,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.registry = registry

    # ------------------------------------------------------------ report ---
    def report(self, include_calibration_run: bool = True) -> EvaluationReport:
        records, _ = self.repository.list(limit=200)
        completed = [r for r in records if r.status is AnalysisStatus.COMPLETED]

        metrics: list[MetricValue] = []
        notes: list[str] = [
            "PUMS reproduces no published benchmark result. Figures here are "
            "measured by this instance, on the data named below.",
        ]

        metrics.extend(self._session_metrics(completed))

        risk_coverage: list[RiskCoverageSeriesPoint] = []
        if include_calibration_run:
            calibration_metrics, risk_coverage, source_note = self._calibration_set_metrics()
            metrics.extend(calibration_metrics)
            notes.append(source_note)

        for definition in DEFINITIONS:
            if any(m.id == definition.id for m in metrics):
                continue
            metrics.append(
                MetricValue(
                    id=definition.id,
                    name=definition.name,
                    category=definition.category,
                    value=None,
                    sample_size=0,
                    provenance=DataProvenance.COMPUTED,
                    unavailable_reason=(
                        NO_GROUND_TRUTH
                        if definition.requires_ground_truth
                        else "Not implemented."
                    )
                    if definition.status is not ImplementationStatus.PLANNED
                    else "Planned — not implemented in this phase.",
                )
            )

        return EvaluationReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            session_sample_size=len(completed),
            metrics=metrics,
            definitions=DEFINITIONS,
            risk_coverage_curve=risk_coverage,
            uncertainty_vs_confidence=[
                {
                    "analysis_id": r.analysis_id,
                    "uncertainty": round(r.uncertainty.global_.score, 4),
                    "confidence": round(r.prediction.confidence, 4),
                    "shift": round(r.distribution_shift.shift_score, 4)
                    if r.distribution_shift
                    else None,
                    "risk": r.risk_assessment.risk_score,
                    "risk_level": r.risk_assessment.risk_level.value,
                }
                for r in completed
            ],
            notes=notes,
        )

    # ---------------------------------------------------------- internals --
    def _session_metrics(self, completed: list) -> list[MetricValue]:
        if not completed:
            return []
        uncertainties = np.array([r.uncertainty.global_.score for r in completed])
        shifts = np.array(
            [
                r.distribution_shift.shift_score
                for r in completed
                if r.distribution_shift
            ]
        )
        out = [
            MetricValue(
                id="session_mean_uncertainty",
                name="Session mean uncertainty",
                category="uncertainty",
                value=round(float(uncertainties.mean()), 4),
                sample_size=len(completed),
                provenance=DataProvenance.COMPUTED,
            )
        ]
        if shifts.size:
            out.append(
                MetricValue(
                    id="session_mean_shift",
                    name="Session mean shift score",
                    category="distribution_shift",
                    value=round(float(shifts.mean()), 4),
                    sample_size=int(shifts.size),
                    provenance=DataProvenance.COMPUTED,
                )
            )
        return out

    def _calibration_set_metrics(
        self,
    ) -> tuple[list[MetricValue], list[RiskCoverageSeriesPoint], str]:
        """Dice / IoU / AURC / correlation on the registered calibration split.

        These are genuine metric computations either way; what changes with the
        data source is what they DESCRIBE. Provenance is read from the
        calibration set rather than assumed, so a real annotated dataset reports
        `computed` and the phantom fallback reports `synthetic` automatically.
        """
        try:
            model = self.registry.get_available(
                resolve_default_model_id(
                    self.settings.default_model_id, self.settings.fallback_model_id
                )
            )
            calibration_set = get_calibration_set(
                model, n_samples=self.settings.calibration_set_size // 2
            )
        except Exception as exc:
            logger.warning("Could not build calibration metrics: %s", exc)
            return [], [], "Calibration-set metrics are unavailable in this environment."

        dice_values, iou_values, correlations = [], [], []
        errors_all, uncertainty_all = [], []

        for sample in calibration_set.samples:
            p_fg = sample.foreground_probability
            predicted = p_fg >= 0.5
            truth = sample.ground_truth
            dice_values.append(dice_coefficient(predicted, truth))
            iou_values.append(iou_score(predicted, truth))

            # Binary predictive entropy of the foreground probability.
            p = np.clip(p_fg, 1e-9, 1 - 1e-9)
            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
            error = (predicted != truth).astype(np.float64)
            correlations.append(uncertainty_error_correlation(entropy, error))
            errors_all.append(error.ravel())
            uncertainty_all.append(entropy.ravel())

        errors = np.concatenate(errors_all)
        uncertainties = np.concatenate(uncertainty_all)
        n = len(calibration_set)

        provenance = calibration_set.provenance
        metrics = [
            MetricValue(
                id="dice",
                name="Dice coefficient",
                category="segmentation",
                value=round(float(np.mean(dice_values)), 4),
                sample_size=n,
                provenance=provenance,
                unavailable_reason=None,
            ),
            MetricValue(
                id="iou",
                name="Intersection over Union",
                category="segmentation",
                value=round(float(np.mean(iou_values)), 4),
                sample_size=n,
                provenance=provenance,
            ),
            MetricValue(
                id="unc_error_corr",
                name="Uncertainty-error correlation",
                category="uncertainty",
                value=round(float(np.mean(correlations)), 4),
                sample_size=n,
                provenance=provenance,
            ),
            MetricValue(
                id="aurc",
                name="Area under risk-coverage curve",
                category="uncertainty",
                value=round(area_under_risk_coverage(errors, uncertainties), 4),
                sample_size=n,
                provenance=provenance,
            ),
        ]

        curve = [
            RiskCoverageSeriesPoint(
                coverage=round(coverage, 4),
                selective_risk=round(risk, 5),
                threshold=round(coverage, 4),
            )
            for coverage, risk in risk_coverage_from_uncertainty(
                errors, uncertainties, n_points=20
            )
        ]
        source_note = (
            f"Segmentation and uncertainty metrics below are measured on "
            f"'{calibration_set.name}' ({n} samples). {calibration_set.description}"
        )
        return metrics, curve, source_note

    # --------------------------------------------------------- self-tests --
    @staticmethod
    def self_test() -> list[SelfTestResult]:
        """Verify the metric implementations against known-answer inputs."""
        results: list[SelfTestResult] = []

        a = np.zeros((4, 4), dtype=bool)
        a[:2, :] = True  # 8 pixels
        b = np.zeros((4, 4), dtype=bool)
        b[1:3, :] = True  # 8 pixels, overlap 4

        results.append(
            SelfTestResult(
                metric="dice",
                scenario="two 8-px masks overlapping in 4 px -> 2*4/16",
                expected=0.5,
                actual=round(dice_coefficient(a, b), 6),
                passed=abs(dice_coefficient(a, b) - 0.5) < 1e-9,
            )
        )
        expected_iou = 4 / 12
        results.append(
            SelfTestResult(
                metric="iou",
                scenario="same masks -> 4/12",
                expected=round(expected_iou, 6),
                actual=round(iou_score(a, b), 6),
                passed=abs(iou_score(a, b) - expected_iou) < 1e-9,
            )
        )
        empty = np.zeros((4, 4), dtype=bool)
        results.append(
            SelfTestResult(
                metric="dice",
                scenario="both masks empty -> 1.0 by convention",
                expected=1.0,
                actual=dice_coefficient(empty, empty),
                passed=dice_coefficient(empty, empty) == 1.0,
            )
        )

        # Selective risk: errors concentrated in the most uncertain half.
        uncertainty = np.linspace(0.0, 1.0, 100)
        error = (uncertainty > 0.5).astype(np.float64)
        actual = selective_risk(error, uncertainty, coverage=0.5)
        results.append(
            SelfTestResult(
                metric="selective_risk",
                scenario="errors only in the most-uncertain half; keep 50% -> 0.0",
                expected=0.0,
                actual=round(actual, 6),
                passed=abs(actual) < 1e-9,
            )
        )
        corr = uncertainty_error_correlation(uncertainty, uncertainty)
        results.append(
            SelfTestResult(
                metric="unc_error_corr",
                scenario="uncertainty perfectly predicts error -> 1.0",
                expected=1.0,
                actual=round(corr, 6),
                passed=abs(corr - 1.0) < 1e-9,
            )
        )
        return results
