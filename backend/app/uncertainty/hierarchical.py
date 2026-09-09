"""Hierarchical uncertainty aggregation (Levels 1-3).

The proposal's central structural claim is that pixel-level and object-level
uncertainty should live in ONE representation rather than being estimated by
separate methods. This module implements that: a single dense field is computed
once, and Levels 2 and 1 are strict aggregations of it. Consistency between
levels is therefore guaranteed by construction, not by post-hoc reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.models.base import DetectionOutput
from app.schemas.common import BoundingBox, DataProvenance, ImplementationStatus, RiskLevel
from app.schemas.uncertainty import (
    GlobalUncertainty,
    HierarchicalUncertainty,
    ObjectUncertainty,
    PixelUncertainty,
    UncertaintyStatistics,
)
from app.uncertainty.base import UncertaintyMap
from app.utils.arrays import downsample_grid, summarise


@dataclass(slots=True)
class HierarchicalResult:
    """Schema payload plus the raw arrays later stages still need."""

    payload: HierarchicalUncertainty
    uncertainty_map: np.ndarray
    confidence_map: np.ndarray
    high_uncertainty_mask: np.ndarray
    boundary_mask: np.ndarray


class HierarchicalUncertaintyEngine:
    """Builds the three-level uncertainty representation from one dense map."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------- public --
    def build(
        self,
        uncertainty: UncertaintyMap,
        probabilities: np.ndarray,
        labels: np.ndarray,
        class_labels: list[str],
        *,
        threshold: float | None = None,
        detections: DetectionOutput | None = None,
    ) -> HierarchicalResult:
        values = np.clip(uncertainty.values, 0.0, 1.0)
        confidence = (
            uncertainty.confidence
            if uncertainty.confidence is not None
            else probabilities.max(axis=0)
        )
        threshold = (
            self.settings.high_uncertainty_threshold if threshold is None else threshold
        )

        boundary_mask = self._boundary_mask(labels)
        high_mask = values >= threshold

        global_level = self._global_level(
            values, probabilities, boundary_mask, high_mask, uncertainty
        )
        objects = self._object_level(
            values, confidence, high_mask, labels, class_labels, detections
        )
        pixel_level = self._pixel_level(values, threshold, high_mask)

        payload = HierarchicalUncertainty(
            method=uncertainty.method,
            status=uncertainty.status,
            **{"global": global_level},
            objects=objects,
            pixel=pixel_level,
            notes=uncertainty.notes
            + [
                "Levels 1 and 2 are aggregations of the same Level 3 field, so the "
                "hierarchy is internally consistent by construction.",
                "Level 2 regions are connected components of the high-uncertainty "
                "mask unless a detection adapter supplied real objects.",
            ],
        )
        return HierarchicalResult(
            payload=payload,
            uncertainty_map=values,
            confidence_map=confidence,
            high_uncertainty_mask=high_mask,
            boundary_mask=boundary_mask,
        )

    # ------------------------------------------------------------ level 1 --
    def _global_level(
        self,
        values: np.ndarray,
        probabilities: np.ndarray,
        boundary_mask: np.ndarray,
        high_mask: np.ndarray,
        uncertainty: UncertaintyMap,
    ) -> GlobalUncertainty:
        stats = summarise(values)
        interior_mask = ~boundary_mask

        # Entropy of the spatially-averaged class distribution: distinct from
        # the mean of per-pixel entropies, and high when the image as a whole
        # has no dominant class rather than when individual pixels are unsure.
        mean_distribution = probabilities.reshape(probabilities.shape[0], -1).mean(axis=1)
        mean_distribution = np.clip(mean_distribution, 1e-12, 1.0)
        aggregate_entropy = float(
            -(mean_distribution * np.log(mean_distribution)).sum()
            / np.log(probabilities.shape[0])
        )

        return GlobalUncertainty(
            score=stats["mean"],
            statistics=UncertaintyStatistics(**stats),
            high_uncertainty_fraction=float(high_mask.mean()),
            aggregate_distribution_entropy=aggregate_entropy,
            boundary_uncertainty=float(values[boundary_mask].mean())
            if boundary_mask.any()
            else 0.0,
            interior_uncertainty=float(values[interior_mask].mean())
            if interior_mask.any()
            else 0.0,
            method=uncertainty.method,
            provenance=DataProvenance.COMPUTED,
        )

    # ------------------------------------------------------------ level 2 --
    def _object_level(
        self,
        values: np.ndarray,
        confidence: np.ndarray,
        high_mask: np.ndarray,
        labels: np.ndarray,
        class_labels: list[str],
        detections: DetectionOutput | None,
    ) -> list[ObjectUncertainty]:
        objects: list[ObjectUncertainty] = []

        if detections is not None and len(detections.boxes) > 0:
            objects.extend(self._objects_from_detections(values, confidence, detections))

        objects.extend(
            self._objects_from_regions(values, confidence, high_mask, labels, class_labels)
        )
        objects.sort(key=lambda item: item.uncertainty, reverse=True)
        return objects[: self.settings.max_reported_regions]

    def _objects_from_detections(
        self, values: np.ndarray, confidence: np.ndarray, detections: DetectionOutput
    ) -> list[ObjectUncertainty]:
        out: list[ObjectUncertainty] = []
        for index in range(len(detections.boxes)):
            x, y, w, h = (int(v) for v in detections.boxes[index])
            region_u = values[y : y + h, x : x + w]
            region_c = confidence[y : y + h, x : x + w]
            if region_u.size == 0:
                continue
            mean_u = float(region_u.mean())
            out.append(
                ObjectUncertainty(
                    id=f"det-{index:03d}",
                    label=detections.class_labels[int(detections.class_ids[index])],
                    class_id=int(detections.class_ids[index]),
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    area_px=int(w * h),
                    confidence=float(region_c.mean()),
                    uncertainty=mean_u,
                    peak_uncertainty=float(region_u.max()),
                    risk_level=self._region_risk(mean_u, float(region_c.mean())),
                    source="detection",
                )
            )
        return out

    def _objects_from_regions(
        self,
        values: np.ndarray,
        confidence: np.ndarray,
        high_mask: np.ndarray,
        labels: np.ndarray,
        class_labels: list[str],
    ) -> list[ObjectUncertainty]:
        from scipy import ndimage

        components, count = ndimage.label(high_mask)
        out: list[ObjectUncertainty] = []
        for component_id in range(1, count + 1):
            mask = components == component_id
            area = int(mask.sum())
            if area < self.settings.min_region_area:
                continue
            ys, xs = np.nonzero(mask)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())

            region_values = values[mask]
            region_confidence = confidence[mask]
            mean_u = float(region_values.mean())
            mean_c = float(region_confidence.mean())

            # Name the region by the class the model assigns to most of it.
            region_labels = labels[mask]
            dominant = int(np.bincount(region_labels).argmax())
            dominant_name = (
                class_labels[dominant] if dominant < len(class_labels) else f"class {dominant}"
            )

            out.append(
                ObjectUncertainty(
                    id=f"region-{component_id:03d}",
                    label=f"Uncertain region · {dominant_name}",
                    class_id=dominant,
                    bbox=BoundingBox(x=x0, y=y0, width=x1 - x0 + 1, height=y1 - y0 + 1),
                    area_px=area,
                    confidence=mean_c,
                    uncertainty=mean_u,
                    peak_uncertainty=float(region_values.max()),
                    risk_level=self._region_risk(mean_u, mean_c),
                    source="high_uncertainty_region",
                )
            )
        return out

    def _region_risk(self, uncertainty: float, confidence: float) -> RiskLevel:
        """Provisional per-region band; the risk engine owns the final call."""
        thresholds = self.settings.risk_thresholds
        score = 0.65 * uncertainty + 0.35 * (1.0 - confidence)
        if score >= thresholds["high"]:
            return RiskLevel.REVIEW_REQUIRED
        if score >= thresholds["moderate"]:
            return RiskLevel.HIGH
        if score >= thresholds["low"]:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    # ------------------------------------------------------------ level 3 --
    def _pixel_level(
        self, values: np.ndarray, threshold: float, high_mask: np.ndarray
    ) -> PixelUncertainty:
        height, width = values.shape
        return PixelUncertainty(
            height=height,
            width=width,
            threshold=threshold,
            high_uncertainty_mask_fraction=float(high_mask.mean()),
            preview_grid=downsample_grid(values, size=32),
        )

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _boundary_mask(labels: np.ndarray) -> np.ndarray:
        """Pixels whose 4-neighbourhood contains a different predicted class."""
        mask = np.zeros_like(labels, dtype=bool)
        mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
        mask[1:, :] |= labels[:-1, :] != labels[1:, :]
        mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        mask[:, 1:] |= labels[:, :-1] != labels[:, 1:]
        return mask
