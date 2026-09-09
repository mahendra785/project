"""Model registry — the single place adapters are discovered and instantiated.

Adapters are constructed lazily and cached, so an unavailable optional adapter
(e.g. torchvision missing) costs nothing and never breaks startup.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.errors import ModelNotFoundError, ModelUnavailableError
from app.core.logging import get_logger
from app.models.adapters.intensity_mixture import IntensityMixtureSegmenter
from app.models.adapters.region_proposal import RegionProposalDetector
from app.models.adapters.rf_pixel_segmenter import RandomForestPixelSegmenter
from app.models.adapters.torchvision_segmenter import TorchvisionSegmenter
from app.models.base import BaseMedicalModel
from app.schemas.models import ModelInfo

logger = get_logger(__name__)

#: Research components each adapter is currently wired to. Kept beside the
#: registry so the UI never has to guess what a model is connected to.
_WIRING = {
    "rf-pixel-seg-v1": {
        "uncertainty_method": "normalised_predictive_entropy",
        "calibration_methods": [
            "conformal_risk_control",
            "temperature_scaling",
            "baseline_threshold",
        ],
        "ood_detector": "composite_intensity_shift_v1",
    },
    "intensity-mixture-seg-v1": {
        "uncertainty_method": "normalised_predictive_entropy",
        "calibration_methods": [
            "conformal_risk_control",
            "temperature_scaling",
            "baseline_threshold",
        ],
        "ood_detector": "composite_intensity_shift_v1",
    },
    "region-proposal-det-v1": {
        "uncertainty_method": "normalised_predictive_entropy",
        "calibration_methods": [
            "conformal_risk_control",
            "temperature_scaling",
            "baseline_threshold",
        ],
        "ood_detector": "composite_intensity_shift_v1",
    },
    "deeplabv3-resnet50-voc": {
        "uncertainty_method": "normalised_predictive_entropy",
        "calibration_methods": [
            "conformal_risk_control",
            "temperature_scaling",
            "baseline_threshold",
        ],
        "ood_detector": "composite_intensity_shift_v1",
    },
}


class ModelRegistry:
    """Maps model ids to lazily-instantiated adapters."""

    def __init__(self, seed: int = 1337) -> None:
        self._factories: dict[str, Callable[[], BaseMedicalModel]] = {
            "rf-pixel-seg-v1": lambda: RandomForestPixelSegmenter(seed=seed),
            "intensity-mixture-seg-v1": lambda: IntensityMixtureSegmenter(seed=seed),
            "region-proposal-det-v1": lambda: RegionProposalDetector(seed=seed),
            "deeplabv3-resnet50-voc": TorchvisionSegmenter,
        }
        self._instances: dict[str, BaseMedicalModel] = {}

    def register(self, model_id: str, factory: Callable[[], BaseMedicalModel]) -> None:
        """Add an adapter at runtime (used by tests and future plugins)."""
        self._factories[model_id] = factory
        self._instances.pop(model_id, None)

    def ids(self) -> list[str]:
        return list(self._factories)

    def get(self, model_id: str) -> BaseMedicalModel:
        if model_id not in self._factories:
            raise ModelNotFoundError(
                f"Unknown model '{model_id}'",
                detail={"available": self.ids()},
            )
        if model_id not in self._instances:
            self._instances[model_id] = self._factories[model_id]()
        return self._instances[model_id]

    def get_available(self, model_id: str) -> BaseMedicalModel:
        """Like `get`, but refuses adapters whose dependencies are missing."""
        model = self.get(model_id)
        available, reason = model.availability()
        if not available:
            raise ModelUnavailableError(
                f"Model '{model_id}' is not runnable in this environment: {reason}",
                detail={"model_id": model_id, "reason": reason},
            )
        return model

    def describe(self, model_id: str) -> ModelInfo:
        model = self.get(model_id)
        descriptor = model.metadata()
        available, reason = model.availability()
        wiring = _WIRING.get(model_id, {})
        return ModelInfo(
            id=descriptor.id,
            name=descriptor.name,
            model_type=descriptor.model_type,
            tasks=descriptor.tasks,
            status=descriptor.status,
            available=available,
            unavailable_reason=reason,
            uncertainty_method=wiring.get("uncertainty_method", "unknown"),
            calibration_methods=wiring.get("calibration_methods", []),
            ood_detector=wiring.get("ood_detector", "unknown"),
            trained_on=descriptor.trained_on,
            is_medical=descriptor.is_medical,
            clinically_validated=descriptor.clinically_validated,
            trained_dataset=descriptor.parameters.get("trained_dataset"),
            description=descriptor.description,
            parameters=descriptor.parameters,
            input_channels=descriptor.input_channels,
            class_labels=descriptor.class_labels[:24],
        )

    def list_models(self) -> list[ModelInfo]:
        return [self.describe(model_id) for model_id in self._factories]


_registry: ModelRegistry | None = None


def resolve_default_model_id(preferred: str, fallback: str, seed: int = 1337) -> str:
    """Pick the preferred adapter when it can actually run, else the fallback.

    The supervised adapter reports itself unavailable until it has been
    trained, so a fresh clone with no dataset still serves requests instead of
    failing every analysis with a 503.
    """
    registry = get_registry(seed=seed)
    try:
        available, _ = registry.get(preferred).availability()
        if available:
            return preferred
    except Exception:
        pass
    logger.info("Default model '%s' is unavailable; using '%s'", preferred, fallback)
    return fallback


def get_registry(seed: int = 1337) -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(seed=seed)
    return _registry
