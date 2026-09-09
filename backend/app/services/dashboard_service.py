"""Dashboard aggregation.

Every figure here is an aggregate over analyses THIS INSTANCE actually ran.
There are no seeded, illustrative or benchmark numbers: an empty instance
reports nulls and zero counts, and the frontend renders an empty state.
"""

from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.schemas.analysis import DashboardStats
from app.schemas.common import AnalysisStatus
from app.services.storage import AnalysisRepository, summarise


class DashboardService:
    #: Bins used for the uncertainty distribution chart.
    HISTOGRAM_BINS = 10

    def __init__(self, settings: Settings, repository: AnalysisRepository) -> None:
        self.settings = settings
        self.repository = repository

    def stats(self, recent_limit: int = 8) -> DashboardStats:
        records, total = self.repository.list(limit=200)
        completed = [r for r in records if r.status is AnalysisStatus.COMPLETED]

        uncertainties = np.array(
            [r.uncertainty.global_.score for r in completed], dtype=np.float64
        )
        confidences = np.array(
            [r.prediction.confidence for r in completed], dtype=np.float64
        )
        shift_scores = np.array(
            [
                r.distribution_shift.shift_score
                for r in completed
                if r.distribution_shift is not None
            ],
            dtype=np.float64,
        )
        eces = np.array(
            [
                r.calibration.expected_calibration_error
                for r in completed
                if r.calibration is not None
            ],
            dtype=np.float64,
        )

        risk_breakdown: dict[str, int] = {}
        shift_breakdown: dict[str, int] = {}
        for record in completed:
            level = record.risk_assessment.risk_level.value
            risk_breakdown[level] = risk_breakdown.get(level, 0) + 1
            if record.distribution_shift:
                status = record.distribution_shift.status.value
                shift_breakdown[status] = shift_breakdown.get(status, 0) + 1

        histogram = (
            np.histogram(uncertainties, bins=self.HISTOGRAM_BINS, range=(0.0, 1.0))[0]
            .astype(int)
            .tolist()
            if uncertainties.size
            else [0] * self.HISTOGRAM_BINS
        )

        return DashboardStats(
            total_analyses=total,
            completed_analyses=len(completed),
            failed_analyses=sum(
                1 for r in records if r.status is AnalysisStatus.FAILED
            ),
            active_model_id=self._active_model_id(),
            active_model_name=self._active_model_name(),
            mean_uncertainty=float(uncertainties.mean()) if uncertainties.size else None,
            mean_confidence=float(confidences.mean()) if confidences.size else None,
            high_risk_count=risk_breakdown.get("HIGH", 0)
            + risk_breakdown.get("REVIEW_REQUIRED", 0),
            review_required_count=risk_breakdown.get("REVIEW_REQUIRED", 0),
            shift_flagged_count=sum(
                count for status, count in shift_breakdown.items() if status != "ID"
            ),
            mean_shift_score=float(shift_scores.mean()) if shift_scores.size else None,
            mean_ece=float(eces.mean()) if eces.size else None,
            uncertainty_histogram=histogram,
            risk_breakdown=risk_breakdown,
            shift_breakdown=shift_breakdown,
            recent=[summarise(record) for record in records[:recent_limit]],
        )

    def _active_model_id(self) -> str:
        from app.models.registry import resolve_default_model_id

        return resolve_default_model_id(
            self.settings.default_model_id, self.settings.fallback_model_id
        )

    def _active_model_name(self) -> str:
        from app.models.registry import get_registry

        try:
            return get_registry().get(self._active_model_id()).metadata().name
        except Exception:
            return self._active_model_id()
