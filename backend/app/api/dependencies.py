"""FastAPI dependency providers.

Services are singletons built once per process; `Depends` gives routes typed
access to them without importing globals, which keeps the routes trivially
testable with overrides.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.evaluation.service import EvaluationService
from app.models.registry import ModelRegistry, get_registry
from app.services.analysis_service import AnalysisService
from app.services.dashboard_service import DashboardService
from app.services.storage import AnalysisRepository, build_repository

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache(maxsize=1)
def _repository() -> AnalysisRepository:
    settings = get_settings()
    return build_repository(settings.storage_backend, settings.storage_dir)


def get_repository() -> AnalysisRepository:
    return _repository()


def get_model_registry() -> ModelRegistry:
    return get_registry(seed=get_settings().random_seed)


@lru_cache(maxsize=1)
def _analysis_service() -> AnalysisService:
    return AnalysisService(get_settings(), get_model_registry(), _repository())


def get_analysis_service() -> AnalysisService:
    return _analysis_service()


@lru_cache(maxsize=1)
def _dashboard_service() -> DashboardService:
    return DashboardService(get_settings(), _repository())


def get_dashboard_service() -> DashboardService:
    return _dashboard_service()


@lru_cache(maxsize=1)
def _evaluation_service() -> EvaluationService:
    return EvaluationService(get_settings(), _repository(), get_model_registry())


def get_evaluation_service() -> EvaluationService:
    return _evaluation_service()


RepositoryDep = Annotated[AnalysisRepository, Depends(get_repository)]
RegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
DashboardServiceDep = Annotated[DashboardService, Depends(get_dashboard_service)]
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]


def reset_dependency_cache() -> None:
    """Used by tests to rebuild services against a temporary storage dir."""
    for cached in (
        _repository,
        _analysis_service,
        _dashboard_service,
        _evaluation_service,
    ):
        cached.cache_clear()
    get_settings.cache_clear()
