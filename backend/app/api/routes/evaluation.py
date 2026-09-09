"""Evaluation and dashboard routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.dependencies import DashboardServiceDep, EvaluationServiceDep
from app.schemas.analysis import DashboardStats
from app.schemas.common import ApiModel
from app.schemas.evaluation import EvaluationReport, SelfTestResult

router = APIRouter(tags=["evaluation"])


class SelfTestResponse(ApiModel):
    results: list[SelfTestResult]
    all_passed: bool
    note: str


@router.get(
    "/evaluation",
    response_model=EvaluationReport,
    summary="Evaluation report over this instance's analyses",
)
def evaluation(
    service: EvaluationServiceDep,
    include_calibration_run: bool = Query(True),
) -> EvaluationReport:
    return service.report(include_calibration_run=include_calibration_run)


@router.get(
    "/evaluation/self-test",
    response_model=SelfTestResponse,
    summary="Verify metric implementations against known-answer inputs",
)
def self_test(service: EvaluationServiceDep) -> SelfTestResponse:
    results = service.self_test()
    return SelfTestResponse(
        results=results,
        all_passed=all(r.passed for r in results),
        note=(
            "These verify the metric CODE is correct on constructed inputs. They "
            "are not dataset results and say nothing about model performance."
        ),
    )


@router.get(
    "/dashboard",
    response_model=DashboardStats,
    summary="Aggregate statistics over this instance's analyses",
)
def dashboard(service: DashboardServiceDep) -> DashboardStats:
    return service.stats()
