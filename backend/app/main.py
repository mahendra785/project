"""PUMS FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    analysis,
    calibration,
    datasets,
    evaluation,
    health,
    models,
    research,
    samples,
    shift,
)
from app.core.config import get_settings
from app.core.errors import PumsError
from app.core.logging import configure_logging, get_logger
from app.services import warmup

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

DESCRIPTION = """
**Perception Uncertainty Management System** — a post-hoc hierarchical
uncertainty calibration framework for dense medical vision models under
distribution shift.

### Research integrity
This API returns **no clinically validated results**. Every component reports an
`ImplementationStatus` (`implemented` / `baseline` / `experimental` / `planned`)
and every figure reports a `DataProvenance` (`computed` / `synthetic` / `mock`).
The shipped calibration and reference sets are **synthetic phantoms**; risk
outputs are a research decision-support signal about model reliability and are
**not a diagnosis**.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.app_name, settings.version)
    if settings.storage_backend != "memory":
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Analysis storage: %s", settings.storage_dir)
    warmup.start(settings)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(PumsError)
async def pums_error_handler(request: Request, exc: PumsError) -> JSONResponse:
    logger.warning("%s -> %s: %s", request.url.path, exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message, "detail": exc.detail},
    )


@app.exception_handler(NotImplementedError)
async def not_implemented_handler(
    request: Request, exc: NotImplementedError
) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"code": "not_implemented", "message": str(exc), "detail": {}},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "An unexpected error occurred while processing the request.",
            "detail": {"type": type(exc).__name__},
        },
    )


app.include_router(health.router)
for module in (analysis, models, calibration, shift, evaluation, datasets, samples, research):
    app.include_router(module.router, prefix=settings.api_prefix)


@app.get("/", tags=["system"], summary="Service banner")
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/health",
        "api_prefix": settings.api_prefix,
        "disclaimer": (
            "Research prototype. Not a medical device. No clinical validation. "
            "Outputs must not be used for patient care."
        ),
    }
