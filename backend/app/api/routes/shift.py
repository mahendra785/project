"""Distribution-shift routes: detector configuration and reference domains."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import SettingsDep
from app.schemas.common import ApiModel, ImplementationStatus
from app.shift.detector import CompositeIntensityShiftDetector
from app.shift.reference import get_reference_profile, list_reference_domains

router = APIRouter(tags=["distribution-shift"])


class IndicatorSpec(ApiModel):
    id: str
    name: str
    weight: float
    description: str


class ReferenceDomainInfo(ApiModel):
    name: str
    description: str
    n_samples: int
    mean: float
    std: float
    entropy: float
    histogram: list[float]


class ShiftConfigResponse(ApiModel):
    detector: str
    detector_name: str
    status: ImplementationStatus
    thresholds: dict[str, float]
    indicators: list[IndicatorSpec]
    scales: dict[str, float]
    reference_domains: list[ReferenceDomainInfo]
    limitations: list[str]


@router.get(
    "/shift/config",
    response_model=ShiftConfigResponse,
    summary="Shift-detector configuration, thresholds and reference domains",
)
def shift_config(settings: SettingsDep) -> ShiftConfigResponse:
    detector = CompositeIntensityShiftDetector()
    domains = []
    for name in list_reference_domains():
        profile = get_reference_profile(name)
        histogram = profile.histogram / max(profile.histogram.sum(), 1e-12)
        domains.append(
            ReferenceDomainInfo(
                name=profile.name,
                description=profile.description,
                n_samples=profile.n_samples,
                mean=round(profile.mean, 4),
                std=round(profile.std, 4),
                entropy=round(profile.entropy, 4),
                histogram=[round(float(v), 6) for v in histogram],
            )
        )

    return ShiftConfigResponse(
        detector=detector.id,
        detector_name=detector.name,
        status=ImplementationStatus.BASELINE,
        thresholds=settings.shift_thresholds,
        indicators=[
            IndicatorSpec(
                id="wasserstein",
                name="Intensity Wasserstein-1 distance",
                weight=detector.WEIGHTS["wasserstein"],
                description="Earth-mover distance between intensity histograms.",
            ),
            IndicatorSpec(
                id="jensen_shannon",
                name="Jensen-Shannon distance",
                weight=detector.WEIGHTS["jensen_shannon"],
                description="Bounded symmetric divergence; catches new modes.",
            ),
            IndicatorSpec(
                id="moments",
                name="Standardised moment deviation",
                weight=detector.WEIGHTS["moments"],
                description="Mean/std/skew/kurtosis distance in reference-SD units.",
            ),
            IndicatorSpec(
                id="texture",
                name="Edge-density deviation",
                weight=detector.WEIGHTS["texture"],
                description="Gradient-magnitude proxy for resolution/kernel change.",
            ),
        ],
        scales=detector.SCALES,
        reference_domains=domains,
        limitations=[
            "Input-space intensity statistics only — semantic shift with "
            "unchanged intensity statistics is not detected.",
            "The composite score is a heuristic, not a calibrated OOD probability.",
            "Band thresholds are configuration values, not experimentally "
            "derived operating points.",
            "The default reference domain is synthetic; 'in-distribution' means "
            "phantom-like until a real reference set is registered.",
        ],
    )
