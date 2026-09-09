"""Hierarchical uncertainty schemas (Level 1 global / Level 2 object / Level 3 pixel)."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import (
    ApiModel,
    BoundingBox,
    DataProvenance,
    ImplementationStatus,
    RiskLevel,
)


class UncertaintyStatistics(ApiModel):
    mean: float
    std: float
    minimum: float
    maximum: float
    median: float
    p90: float
    p95: float
    p99: float


class GlobalUncertainty(ApiModel):
    """Level 1 — one number (plus context) for the whole image."""

    #: Mean normalised predictive entropy over all pixels, in [0, 1].
    score: float
    statistics: UncertaintyStatistics
    #: Fraction of pixels above `high_uncertainty_threshold`.
    high_uncertainty_fraction: float
    #: Shannon entropy of the image-level averaged class distribution, in [0,1].
    aggregate_distribution_entropy: float
    #: Uncertainty concentrated at predicted class boundaries vs. interiors.
    boundary_uncertainty: float
    interior_uncertainty: float
    method: str
    provenance: DataProvenance = DataProvenance.COMPUTED


class ObjectUncertainty(ApiModel):
    """Level 2 — connected uncertain regions / detected objects."""

    id: str
    label: str
    #: Set when the region came from a detector; null for region-derived items.
    class_id: int | None = None
    bbox: BoundingBox
    area_px: int
    #: Mean model confidence inside the region.
    confidence: float
    #: Mean normalised entropy inside the region.
    uncertainty: float
    peak_uncertainty: float
    #: Calibrated per-region risk from the calibration module, if available.
    calibrated_risk: float | None = None
    risk_level: RiskLevel
    #: "high_uncertainty_region" | "detection" — how this object was produced.
    source: str
    review_status: str = "pending"


class PixelUncertainty(ApiModel):
    """Level 3 — the dense map itself, transported efficiently."""

    height: int
    width: int
    #: Quantised map: uint8 values 0-255 mapping linearly onto [0, 1].
    #: Base64 of a PNG (grayscale) so payload stays small and cacheable.
    encoding: str = "png-gray-u8-base64"
    data: str | None = None
    #: Endpoint returning a colour-mapped heatmap PNG for direct <img> use.
    heatmap_url: str | None = None
    #: Endpoint returning the raw grayscale PNG (for client-side thresholding).
    raw_url: str | None = None
    #: Threshold used to derive `high_uncertainty_mask_fraction`.
    threshold: float
    high_uncertainty_mask_fraction: float
    #: Coarse (32x32) downsample of the map, for charts without image decoding.
    preview_grid: list[list[float]] = []


class HierarchicalUncertainty(ApiModel):
    method: str
    status: ImplementationStatus
    #: Serialised as `global` (a Python keyword, hence the trailing underscore).
    global_: GlobalUncertainty = Field(alias="global")
    objects: list[ObjectUncertainty]
    pixel: PixelUncertainty
    #: Explains what is and is not implemented at each level.
    notes: list[str] = []
