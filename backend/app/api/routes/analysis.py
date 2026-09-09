"""Analysis routes: run the pipeline, retrieve results, serve dense media."""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import Response

from app.api.dependencies import AnalysisServiceDep, RepositoryDep
from app.core.errors import AnalysisNotFoundError, InvalidImageError
from app.core.logging import get_logger
from app.schemas.analysis import (
    AnalysisListResponse,
    AnalysisOptions,
    AnalysisResponse,
)
from app.schemas.calibration import CalibrationResult
from app.schemas.risk import RiskAssessment
from app.schemas.shift import DistributionShiftResult
from app.schemas.uncertainty import HierarchicalUncertainty
from app.services.storage import summarise

logger = get_logger(__name__)
router = APIRouter(tags=["analysis"])

#: Media keys the API is willing to serve, so a path parameter can never be
#: used to read arbitrary files.
ALLOWED_MEDIA = {"source", "uncertainty-raw", "uncertainty-heatmap", "mask"}


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Run the full PUMS pipeline on one image",
)
async def analyze(
    service: AnalysisServiceDep,
    file: UploadFile = File(..., description="PNG, JPEG, TIFF, BMP or DICOM"),
    options: str | None = Form(
        None, description="JSON-encoded AnalysisOptions object"
    ),
) -> AnalysisResponse:
    """Upload an image and receive the unified hierarchical analysis.

    `options` is sent as a JSON string in a multipart field so the whole
    request stays a single round trip alongside the file.
    """
    parsed = AnalysisOptions()
    if options:
        try:
            parsed = AnalysisOptions.model_validate(json.loads(options))
        except json.JSONDecodeError as exc:
            raise InvalidImageError(f"`options` is not valid JSON: {exc}") from exc

    data = await file.read()
    return service.analyze(
        data,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        options=parsed,
    )


@router.get(
    "/analyses",
    response_model=AnalysisListResponse,
    summary="List analyses run on this instance",
)
def list_analyses(
    repository: RepositoryDep,
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AnalysisListResponse:
    records, total = repository.list(limit=limit, offset=offset)
    return AnalysisListResponse(
        items=[summarise(record) for record in records], total=total
    )


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Retrieve one complete analysis",
)
def get_analysis(analysis_id: str, repository: RepositoryDep) -> AnalysisResponse:
    return repository.get(analysis_id)


@router.get(
    "/analyses/{analysis_id}/uncertainty",
    response_model=HierarchicalUncertainty,
    summary="Hierarchical uncertainty for one analysis",
)
def get_uncertainty(
    analysis_id: str, repository: RepositoryDep
) -> HierarchicalUncertainty:
    return repository.get(analysis_id).uncertainty


@router.get(
    "/analyses/{analysis_id}/calibration",
    response_model=CalibrationResult | None,
    summary="Calibration result for one analysis",
)
def get_calibration(
    analysis_id: str, repository: RepositoryDep
) -> CalibrationResult | None:
    return repository.get(analysis_id).calibration


@router.get(
    "/analyses/{analysis_id}/shift",
    response_model=DistributionShiftResult | None,
    summary="Distribution-shift result for one analysis",
)
def get_shift(
    analysis_id: str, repository: RepositoryDep
) -> DistributionShiftResult | None:
    return repository.get(analysis_id).distribution_shift


@router.get(
    "/analyses/{analysis_id}/risk",
    response_model=RiskAssessment,
    summary="Risk assessment for one analysis",
)
def get_risk(analysis_id: str, repository: RepositoryDep) -> RiskAssessment:
    return repository.get(analysis_id).risk_assessment


@router.get(
    "/analyses/{analysis_id}/media/{key}.png",
    summary="Dense map as PNG (source, mask, uncertainty)",
    response_class=Response,
)
def get_media(analysis_id: str, key: str, repository: RepositoryDep) -> Response:
    if key not in ALLOWED_MEDIA:
        raise AnalysisNotFoundError(
            f"Unknown media key '{key}'", detail={"allowed": sorted(ALLOWED_MEDIA)}
        )
    data = repository.get_media(analysis_id, key)
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.delete("/analyses/{analysis_id}", status_code=204, summary="Delete an analysis")
def delete_analysis(analysis_id: str, repository: RepositoryDep) -> Response:
    repository.get(analysis_id)  # 404 if missing
    repository.delete(analysis_id)
    return Response(status_code=204)
