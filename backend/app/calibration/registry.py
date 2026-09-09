"""Catalogue of calibration methods — implemented and planned.

The PLANNED entries are the methods the literature review identifies as the
natural next steps for PUMS. They are listed so the roadmap is visible in the
product, and they raise if invoked so they can never be mistaken for working
code.
"""

from __future__ import annotations

from app.calibration.base import (
    EXCHANGEABILITY,
    MARGINAL_ONLY,
    BaseCalibrationMethod,
    PlannedMethod,
)
from app.calibration.conformal import BaselineThresholdCalibration, ConformalRiskControl
from app.calibration.temperature import TemperatureScaling

_METHODS: dict[str, BaseCalibrationMethod] = {}


def _register(method: BaseCalibrationMethod) -> BaseCalibrationMethod:
    _METHODS[method.id] = method
    return method


_register(ConformalRiskControl())
_register(TemperatureScaling())
_register(BaselineThresholdCalibration())

_register(
    PlannedMethod(
        "adaptive_conformal_risk_control",
        "Automatically Adaptive Conformal Risk Control",
        "Adapts lambda to the data instead of fixing it on a static calibration "
        "split. Target of the distribution-shift objective; not implemented.",
        reference="Blot et al., Automatically Adaptive Conformal Risk Control, 2024",
        assumptions=[MARGINAL_ONLY],
    )
)
_register(
    PlannedMethod(
        "spatially_aware_conformal",
        "Spatially-Aware Conformal Prediction (SACP)",
        "Conformal prediction sets that respect image geometry rather than "
        "treating pixels independently. Not implemented.",
        reference="Bereska, Karimi & Samavi, SACP, MIDL 2025",
        assumptions=[EXCHANGEABILITY],
    )
)
_register(
    PlannedMethod(
        "localized_adaptive_risk_control",
        "Localized Adaptive Risk Control",
        "Input-conditional (region-level) rather than marginal risk control, so "
        "guarantees can hold inside clinically critical zones. Not implemented.",
        reference="Zecchin & Simeone, Localized Adaptive Risk Control, 2024",
        assumptions=[],
    )
)
_register(
    PlannedMethod(
        "morphological_conformal",
        "Conformal Segmentation via Morphological Prediction Sets",
        "Wraps a black-box segmenter by dilating its mask until the risk target "
        "is met, requiring no logits. Not implemented.",
        reference="Mossina et al., MICCAI 2025",
        assumptions=[EXCHANGEABILITY],
    )
)
_register(
    PlannedMethod(
        "evidential_calibration",
        "Evidential Calibrated Uncertainty (DEviS-style)",
        "Single-pass evidential head giving calibrated uncertainty without a "
        "calibration split. Not implemented.",
        reference="Zou et al., DEviS, IEEE TCyb 2025",
        assumptions=[],
    )
)


def get_method(method_id: str) -> BaseCalibrationMethod:
    if method_id not in _METHODS:
        raise KeyError(f"Unknown calibration method '{method_id}'")
    return _METHODS[method_id]


def list_methods() -> list[BaseCalibrationMethod]:
    return list(_METHODS.values())


def implemented_ids() -> list[str]:
    from app.schemas.common import ImplementationStatus

    return [
        method.id
        for method in _METHODS.values()
        if method.status
        in (ImplementationStatus.IMPLEMENTED, ImplementationStatus.BASELINE)
    ]
