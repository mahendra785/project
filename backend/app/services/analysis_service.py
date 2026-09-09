"""`AnalysisService` — the single orchestrator of the PUMS pipeline.

    bytes -> preprocess -> model -> prediction
                                 -> uncertainty (pixel -> object -> global)
                                 -> calibration
                                 -> distribution shift
                                 -> risk
                                 -> persisted, typed response

Routes call exactly one method here. Keeping the orchestration in one place is
what stops the research logic from leaking into HTTP handlers.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import numpy as np

from app.calibration.datasets import get_calibration_set
from app.calibration.registry import get_method
from app.core.config import Settings
from app.core.errors import PumsError, TaskNotSupportedError
from app.core.logging import get_logger
from app.models.base import (
    BaseMedicalModel,
    DetectionOutput,
    SegmentationOutput,
)
from app.models.registry import ModelRegistry
from app.preprocessing.pipeline import PreprocessedImage, PreprocessingPipeline
from app.risk.engine import RiskAssessmentService, RiskInputs
from app.schemas.analysis import (
    AnalysisMetadata,
    AnalysisOptions,
    AnalysisResponse,
    ImageMetadata,
)
from app.schemas.calibration import CalibrationResult
from app.schemas.common import AnalysisStatus, ShiftStatus, TaskType
from app.schemas.prediction import (
    ClassScore,
    Detection,
    PredictionResult,
    SegmentationMask,
)
from app.schemas.shift import DistributionShiftResult
from app.schemas.uncertainty import HierarchicalUncertainty
from app.services.storage import AnalysisRepository
from app.shift.detector import CompositeIntensityShiftDetector
from app.shift.reference import get_reference_profile
from app.uncertainty.estimators import get_estimator
from app.uncertainty.hierarchical import HierarchicalUncertaintyEngine
from app.utils import encoding
from app.utils.arrays import rle_encode

logger = get_logger(__name__)

PIPELINE_VERSION = "pums-pipeline-0.4.0"


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        repository: AnalysisRepository,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.repository = repository
        self.preprocessor = PreprocessingPipeline(settings)
        self.hierarchy = HierarchicalUncertaintyEngine(settings)
        self.shift_detector = CompositeIntensityShiftDetector()
        self.risk_service = RiskAssessmentService(settings)

    # ------------------------------------------------------------- public --
    def analyze(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str,
        options: AnalysisOptions,
    ) -> AnalysisResponse:
        started = time.perf_counter()
        analysis_id = f"pums_{uuid.uuid4().hex[:12]}"
        warnings: list[str] = []

        image = self.preprocessor.run(
            data, content_type=content_type, filename=filename
        )

        model_id = options.model_id or self.settings.default_model_id
        model = self.registry.get_available(model_id)

        # The requested task must be one the adapter declares. Without this the
        # pipeline would fall through to segmentation and label the result with
        # the requested task — reporting, say, a "classification" that no
        # classifier produced. Refusing is the only honest outcome.
        if not model.supports(options.task):
            raise TaskNotSupportedError(
                f"Model '{model.metadata().id}' does not support task "
                f"'{options.task.value}'",
                detail={
                    "model_id": model.metadata().id,
                    "requested_task": options.task.value,
                    "supported_tasks": [t.value for t in model.metadata().tasks],
                },
            )

        # --- prediction -----------------------------------------------------
        # Segmentation always runs: it is the dense probability substrate the
        # uncertainty hierarchy is built from, whatever the requested task.
        inference_started = time.perf_counter()
        segmentation = model.predict_segmentation(image)
        detections: DetectionOutput | None = None
        if options.task is TaskType.DETECTION:
            detections = model.predict_detection(image)
        inference_ms = (time.perf_counter() - inference_started) * 1000

        prediction = self._build_prediction(
            analysis_id, model, segmentation, detections, options.task, inference_ms
        )

        # --- uncertainty (levels 3 -> 2 -> 1) -------------------------------
        estimator = get_estimator()
        uncertainty_map = estimator.estimate(segmentation.probabilities)
        hierarchy = self.hierarchy.build(
            uncertainty_map,
            segmentation.probabilities,
            segmentation.labels,
            segmentation.class_labels,
            threshold=options.uncertainty_threshold,
            detections=detections,
        )

        # --- distribution shift ---------------------------------------------
        shift: DistributionShiftResult | None = None
        if options.detect_shift:
            reference = get_reference_profile(options.reference_domain)
            shift = self.shift_detector.detect(
                image, reference, thresholds=self.settings.shift_thresholds
            )

        # --- calibration -----------------------------------------------------
        calibration: CalibrationResult | None = None
        if options.apply_calibration:
            calibration, calibration_warnings = self._calibrate(
                model, segmentation, options, shift
            )
            warnings.extend(calibration_warnings)

        uncertainty = self._attach_media_urls(analysis_id, hierarchy.payload)
        uncertainty = self._attach_calibrated_region_risk(uncertainty, calibration)

        # --- risk -------------------------------------------------------------
        risk = self.risk_service.assess(
            RiskInputs(
                uncertainty=uncertainty,
                confidence=prediction.confidence,
                calibration=calibration,
                shift=shift,
            )
        )

        # --- media ------------------------------------------------------------
        self._store_media(analysis_id, image, hierarchy.uncertainty_map, segmentation)

        duration_ms = (time.perf_counter() - started) * 1000
        response = AnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.COMPLETED,
            metadata=AnalysisMetadata(
                created_at=datetime.now(timezone.utc),
                duration_ms=round(duration_ms, 2),
                model_id=model.metadata().id,
                model_name=model.metadata().name,
                task=options.task,
                options=options,
                image=self._image_metadata(analysis_id, image, filename, content_type),
                pipeline_version=PIPELINE_VERSION,
                seed=self.settings.random_seed,
            ),
            prediction=prediction,
            uncertainty=uncertainty,
            calibration=calibration,
            distribution_shift=shift,
            risk_assessment=risk,
            warnings=warnings + self._model_warnings(model),
        )

        self.repository.save(response)
        logger.info(
            "analysis %s completed in %.0f ms (model=%s risk=%s)",
            analysis_id,
            duration_ms,
            model_id,
            risk.risk_level.value,
        )
        return response

    # -------------------------------------------------------- prediction ---
    def _build_prediction(
        self,
        analysis_id: str,
        model: BaseMedicalModel,
        segmentation: SegmentationOutput,
        detections: DetectionOutput | None,
        task: TaskType,
        inference_ms: float,
    ) -> PredictionResult:
        probabilities = segmentation.probabilities
        labels = segmentation.labels
        n_classes = probabilities.shape[0]

        mean_distribution = probabilities.reshape(n_classes, -1).mean(axis=1)
        pixel_fraction = [
            float((labels == class_id).mean()) for class_id in range(n_classes)
        ]

        mask = SegmentationMask(
            height=int(labels.shape[0]),
            width=int(labels.shape[1]),
            num_classes=n_classes,
            class_labels=segmentation.class_labels,
            rle=rle_encode(labels),
            class_pixel_fraction=[round(v, 6) for v in pixel_fraction],
            png_url=f"/api/analyses/{analysis_id}/media/mask.png",
        )

        detection_items: list[Detection] | None = None
        if detections is not None:
            detection_items = [
                Detection(
                    id=f"det-{index:03d}",
                    class_id=int(detections.class_ids[index]),
                    label=detections.class_labels[int(detections.class_ids[index])],
                    bbox={
                        "x": int(detections.boxes[index][0]),
                        "y": int(detections.boxes[index][1]),
                        "width": int(detections.boxes[index][2]),
                        "height": int(detections.boxes[index][3]),
                    },
                    confidence=round(float(detections.scores[index]), 4),
                )
                for index in range(len(detections.boxes))
            ]

        return PredictionResult(
            task=task,
            model_id=model.metadata().id,
            # Mean max-posterior: the model's average per-pixel commitment.
            confidence=round(float(probabilities.max(axis=0).mean()), 4),
            classes=[
                ClassScore(
                    class_id=index,
                    label=segmentation.class_labels[index],
                    probability=round(float(mean_distribution[index]), 6),
                )
                for index in range(n_classes)
            ],
            segmentation=mask,
            detections=detection_items,
            inference_time_ms=round(inference_ms, 2),
            metadata={
                "n_classes": n_classes,
                "model_type": model.metadata().model_type,
                "is_medical_model": model.metadata().is_medical,
                "trained_on": model.metadata().trained_on,
            },
        )

    # ------------------------------------------------------- calibration ---
    def _calibrate(
        self,
        model: BaseMedicalModel,
        segmentation: SegmentationOutput,
        options: AnalysisOptions,
        shift: DistributionShiftResult | None,
    ) -> tuple[CalibrationResult | None, list[str]]:
        warnings: list[str] = []
        try:
            method = get_method(options.calibration_method)
        except KeyError:
            warnings.append(
                f"Unknown calibration method '{options.calibration_method}'; "
                "falling back to conformal_risk_control."
            )
            method = get_method("conformal_risk_control")

        try:
            calibration_set = get_calibration_set(
                model, n_samples=self.settings.calibration_set_size // 2
            )
            result = method.calibrate(
                calibration_set,
                target_risk=options.target_risk,
                n_bins=self.settings.ece_bins,
            )
        except NotImplementedError as exc:
            warnings.append(str(exc))
            return None, warnings
        except Exception as exc:  # calibration must never fail the analysis
            logger.exception("Calibration failed")
            warnings.append(f"Calibration failed and was skipped: {exc}")
            return None, warnings

        foreground = segmentation.foreground_probability
        applied = method.apply(result, foreground)
        result.applied_abstention_rate = round(applied.abstention_rate, 4)
        result.calibrated_confidence = round(applied.calibrated_confidence, 4)
        result.applied = True

        # The exchangeability assumption is the whole basis of the guarantee.
        # If the shift detector says the input is off-domain, say so here.
        if shift and shift.status is not ShiftStatus.ID:
            result.validity_warnings.append(
                f"Input flagged as {shift.status.value} (score {shift.shift_score:.2f}). "
                "Exchangeability with the calibration set is therefore in doubt and "
                "the conformal risk guarantee should NOT be considered valid for "
                "this image."
            )
        return result, warnings

    def _attach_calibrated_region_risk(
        self, uncertainty: HierarchicalUncertainty, calibration: CalibrationResult | None
    ) -> HierarchicalUncertainty:
        """Attach a per-region calibrated risk when calibration ran.

        This is an interpretable *combination* of the calibrated empirical risk
        with the region's own uncertainty, not a region-conditional conformal
        guarantee — localized risk control is listed as PLANNED for exactly
        this reason.
        """
        if calibration is None:
            return uncertainty
        base = calibration.empirical_risk
        for obj in uncertainty.objects:
            obj.calibrated_risk = round(
                float(np.clip(base + (1.0 - base) * obj.uncertainty, 0.0, 1.0)), 4
            )
        return uncertainty

    # ------------------------------------------------------------- media ---
    def _attach_media_urls(
        self, analysis_id: str, uncertainty: HierarchicalUncertainty
    ) -> HierarchicalUncertainty:
        base = f"/api/analyses/{analysis_id}/media"
        uncertainty.pixel.heatmap_url = f"{base}/uncertainty-heatmap.png"
        uncertainty.pixel.raw_url = f"{base}/uncertainty-raw.png"
        return uncertainty

    def _store_media(
        self,
        analysis_id: str,
        image: PreprocessedImage,
        uncertainty_map: np.ndarray,
        segmentation: SegmentationOutput,
    ) -> None:
        self.repository.put_media(
            analysis_id, "source", encoding.image_png_bytes(image.gray)
        )
        self.repository.put_media(
            analysis_id, "uncertainty-raw", encoding.gray_png_bytes(uncertainty_map)
        )
        self.repository.put_media(
            analysis_id,
            "uncertainty-heatmap",
            encoding.heatmap_png_bytes(uncertainty_map),
        )
        self.repository.put_media(
            analysis_id, "mask", encoding.mask_png_bytes(segmentation.labels)
        )

    # ----------------------------------------------------------- metadata --
    @staticmethod
    def _image_metadata(
        analysis_id: str,
        image: PreprocessedImage,
        filename: str,
        content_type: str,
    ) -> ImageMetadata:
        return ImageMetadata(
            filename=filename,
            content_type=content_type,
            size_bytes=image.size_bytes,
            original_width=image.original_width,
            original_height=image.original_height,
            original_channels=image.original_channels,
            working_width=image.gray.shape[1],
            working_height=image.gray.shape[0],
            format=image.image_format,
            is_dicom=image.is_dicom,
            checksum=image.checksum,
            tags=image.tags,
            preview_url=f"/api/analyses/{analysis_id}/media/source.png",
        )

    @staticmethod
    def _model_warnings(model: BaseMedicalModel) -> list[str]:
        descriptor = model.metadata()
        if descriptor.is_medical:
            return []
        return [
            f"'{descriptor.name}' is not a validated medical model "
            f"({descriptor.trained_on}). Outputs demonstrate the PUMS pipeline "
            "and carry no clinical meaning."
        ]


def unwrap_error(exc: Exception) -> PumsError:
    return exc if isinstance(exc, PumsError) else PumsError(str(exc))
