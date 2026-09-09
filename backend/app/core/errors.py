"""Domain exceptions mapped to HTTP responses in `app.main`."""

from __future__ import annotations


class PumsError(Exception):
    """Base class for all recoverable PUMS errors."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class InvalidImageError(PumsError):
    status_code = 422
    code = "invalid_image"


class PayloadTooLargeError(PumsError):
    status_code = 413
    code = "payload_too_large"


class ModelNotFoundError(PumsError):
    status_code = 404
    code = "model_not_found"


class ModelUnavailableError(PumsError):
    """Adapter exists in the registry but its dependencies are missing."""

    status_code = 503
    code = "model_unavailable"


class TaskNotSupportedError(PumsError):
    status_code = 400
    code = "task_not_supported"


class AnalysisNotFoundError(PumsError):
    status_code = 404
    code = "analysis_not_found"
