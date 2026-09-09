"""Model registry routes."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import RegistryDep, SettingsDep
from app.schemas.models import ModelInfo, ModelListResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelListResponse, summary="List model adapters")
def list_models(registry: RegistryDep, settings: SettingsDep) -> ModelListResponse:
    return ModelListResponse(
        models=registry.list_models(), default_model_id=settings.default_model_id
    )


@router.get("/models/{model_id}", response_model=ModelInfo, summary="Describe one model")
def get_model(model_id: str, registry: RegistryDep) -> ModelInfo:
    return registry.describe(model_id)
