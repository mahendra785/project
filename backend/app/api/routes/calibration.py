"""Calibration routes: method catalogue and standalone calibration runs."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.dependencies import RegistryDep, SettingsDep
from app.calibration.datasets import get_calibration_set
from app.calibration.registry import get_method, list_methods
from app.core.errors import PumsError
from app.schemas.calibration import CalibrationMethodInfo, CalibrationResult
from app.schemas.common import ApiModel

router = APIRouter(tags=["calibration"])


class CalibrationMethodsResponse(ApiModel):
    methods: list[CalibrationMethodInfo]
    default_method: str
    default_target_risk: float


@router.get(
    "/calibration/methods",
    response_model=CalibrationMethodsResponse,
    summary="Catalogue of calibration methods with implementation status",
)
def methods(settings: SettingsDep) -> CalibrationMethodsResponse:
    return CalibrationMethodsResponse(
        methods=[method.info() for method in list_methods()],
        default_method="conformal_risk_control",
        default_target_risk=settings.default_target_risk,
    )


@router.post(
    "/calibration/run",
    response_model=CalibrationResult,
    summary="Fit a calibration method on the registered calibration set",
)
def run_calibration(
    settings: SettingsDep,
    registry: RegistryDep,
    model_id: str | None = Query(None),
    method: str = Query("conformal_risk_control"),
    target_risk: float = Query(0.10, ge=0.001, le=0.5),
) -> CalibrationResult:
    """Run calibration independently of an image analysis.

    This powers the calibration dashboard's "recalibrate" control and lets the
    effect of alpha on lambda_hat be explored directly.
    """
    model = registry.get_available(model_id or settings.default_model_id)
    try:
        calibration_method = get_method(method)
    except KeyError as exc:
        raise PumsError(str(exc)) from exc

    calibration_set = get_calibration_set(
        model, n_samples=settings.calibration_set_size // 2
    )
    try:
        return calibration_method.calibrate(
            calibration_set, target_risk=target_risk, n_bins=settings.ece_bins
        )
    except NotImplementedError as exc:
        error = PumsError(str(exc))
        error.status_code = 501
        error.code = "not_implemented"
        raise error from exc
