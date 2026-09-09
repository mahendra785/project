"""Region-proposal detection adapter.

HONEST DESCRIPTION: this is **not a learned object detector**. It derives box
proposals from the connected components of a probabilistic segmentation, and
reports each region's *real* mean class posterior as its class distribution.
Every number it emits is computed from actual model probabilities; none is
invented. It exists so the object-level (Level 2) branch of the hierarchy has a
working, inspectable implementation while a trained detector adapter (Faster
R-CNN / DETR / nnDetection) remains a drop-in replacement.
"""

from __future__ import annotations

import numpy as np

from app.models.adapters.intensity_mixture import IntensityMixtureSegmenter
from app.models.base import (
    BaseMedicalModel,
    DetectionOutput,
    ModelDescriptor,
    SegmentationOutput,
)
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import ImplementationStatus, TaskType


class RegionProposalDetector(BaseMedicalModel):
    """Connected-component proposals over a probabilistic segmentation."""

    #: Proposals smaller than this (working-resolution pixels) are discarded as
    #: segmentation speckle rather than reported as objects.
    MIN_AREA = 60
    MAX_PROPOSALS = 20

    def __init__(self, seed: int = 1337) -> None:
        super().__init__()
        self._segmenter = IntensityMixtureSegmenter(seed=seed)
        self.descriptor = ModelDescriptor(
            id="region-proposal-det-v1",
            name="Segmentation-Derived Region Proposals",
            model_type="classical-probabilistic",
            tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
            status=ImplementationStatus.BASELINE,
            description=(
                "Connected components of the non-background segmentation classes, "
                "reported as boxes with their true mean class posterior. NOT a "
                "learned object detector — no trained detection head, no anatomy "
                "classes, no localisation training."
            ),
            trained_on="Nothing — derived per image from the intensity mixture segmenter.",
            is_medical=False,
            class_labels=self._segmenter.descriptor.class_labels,
            input_channels=1,
            parameters={
                "min_area": self.MIN_AREA,
                "max_proposals": self.MAX_PROPOSALS,
                "seed": seed,
            },
            requires=["sklearn", "scipy"],
        )

    def predict_segmentation(self, image: PreprocessedImage) -> SegmentationOutput:
        return self._segmenter.predict_segmentation(image)

    def predict_detection(self, image: PreprocessedImage) -> DetectionOutput:
        from scipy import ndimage

        segmentation = self._segmenter.predict_segmentation(image)
        probabilities = segmentation.probabilities
        labels = segmentation.labels

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        class_probs: list[np.ndarray] = []

        # Class 0 is background by construction in the mixture adapter.
        for class_id in range(1, probabilities.shape[0]):
            components, count = ndimage.label(labels == class_id)
            for component_id in range(1, count + 1):
                mask = components == component_id
                area = int(mask.sum())
                if area < self.MIN_AREA:
                    continue
                ys, xs = np.nonzero(mask)
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())

                mean_distribution = probabilities[:, mask].mean(axis=1)
                boxes.append([x0, y0, x1 - x0 + 1, y1 - y0 + 1])
                scores.append(float(mean_distribution[class_id]))
                class_ids.append(class_id)
                class_probs.append(mean_distribution)

        if not boxes:
            return DetectionOutput(
                boxes=np.zeros((0, 4), dtype=np.int32),
                scores=np.zeros((0,), dtype=np.float64),
                class_ids=np.zeros((0,), dtype=np.int32),
                class_labels=segmentation.class_labels,
                class_probabilities=np.zeros((0, probabilities.shape[0])),
            )

        order = np.argsort(scores)[::-1][: self.MAX_PROPOSALS]
        return DetectionOutput(
            boxes=np.asarray(boxes, dtype=np.int32)[order],
            scores=np.asarray(scores, dtype=np.float64)[order],
            class_ids=np.asarray(class_ids, dtype=np.int32)[order],
            class_labels=segmentation.class_labels,
            class_probabilities=np.asarray(class_probs, dtype=np.float64)[order],
        )
