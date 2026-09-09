"""Health and capability reporting."""

from __future__ import annotations

import importlib.util
import platform

from fastapi import APIRouter

from app.api.dependencies import RegistryDep, SettingsDep
from app.schemas.common import ApiModel
from app.services import warmup as warmup_service

router = APIRouter(tags=["system"])


class HealthResponse(ApiModel):
    status: str
    version: str
    environment: str
    python: str
    #: Which optional capabilities are actually present in this environment.
    capabilities: dict[str, bool]
    available_models: list[str]
    unavailable_models: dict[str, str]
    #: cold | queued | warming | ready | failed — the calibration/reference
    #: warmup runs in the background at startup.
    warmup: dict


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@router.get("/health", response_model=HealthResponse, summary="Liveness and capabilities")
def health(settings: SettingsDep, registry: RegistryDep) -> HealthResponse:
    models = registry.list_models()
    return HealthResponse(
        status="ok",
        version=settings.version,
        environment=settings.environment,
        python=platform.python_version(),
        capabilities={
            "torch": _has("torch"),
            "torchvision": _has("torchvision"),
            "monai": _has("monai"),
            "pydicom": _has("pydicom"),
            "opencv": _has("cv2"),
            "scikit_learn": _has("sklearn"),
        },
        available_models=[m.id for m in models if m.available],
        unavailable_models={
            m.id: m.unavailable_reason or "unknown" for m in models if not m.available
        },
        warmup=warmup_service.status(),
    )
