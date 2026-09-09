"""Optional torchvision DeepLabV3 adapter.

This is a REAL trained CNN and a real dense-softmax path, which makes it a
useful second data point for the model-agnostic claim. It is, however, trained
on PASCAL VOC **natural images** — it has no medical validity whatsoever and is
labelled `is_medical=False` everywhere it surfaces. Do not read its class
outputs as anatomy.

Requires `pip install -r requirements-torch.txt`; weights (~160 MB) download on
first use into the torch hub cache. If either is missing, the registry reports
the adapter as unavailable rather than failing a request.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.errors import ModelUnavailableError
from app.core.logging import get_logger
from app.models.base import BaseMedicalModel, ModelDescriptor, SegmentationOutput
from app.preprocessing.pipeline import PreprocessedImage
from app.schemas.common import ImplementationStatus, TaskType

logger = get_logger(__name__)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

#: ImageNet channel statistics used by all torchvision segmentation weights.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class TorchvisionSegmenter(BaseMedicalModel):
    """DeepLabV3-ResNet50 dense softmax, exposed through the PUMS interface."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self.descriptor = ModelDescriptor(
            id="deeplabv3-resnet50-voc",
            name="DeepLabV3-ResNet50 (PASCAL VOC)",
            model_type="cnn",
            tasks=[TaskType.SEGMENTATION],
            status=ImplementationStatus.EXPERIMENTAL,
            description=(
                "Pretrained torchvision DeepLabV3-ResNet50. A genuine deep dense "
                "softmax, included to demonstrate that the uncertainty stack is "
                "model-agnostic. Trained on natural images (PASCAL VOC) — it has "
                "NO medical validity and its classes are not anatomy."
            ),
            trained_on="COCO subset with PASCAL VOC classes (natural images).",
            is_medical=False,
            class_labels=VOC_CLASSES,
            input_channels=3,
            parameters={"weights": "DeepLabV3_ResNet50_Weights.DEFAULT"},
            requires=["torch", "torchvision"],
        )

    def _load(self) -> None:
        available, reason = self.availability()
        if not available:
            raise ModelUnavailableError(reason or "torch/torchvision not installed")

        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            deeplabv3_resnet50,
        )

        logger.info("Loading DeepLabV3-ResNet50 weights (first run downloads ~160 MB)")
        self._model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self._model.eval()
        torch.manual_seed(1337)

    def predict_segmentation(self, image: PreprocessedImage) -> SegmentationOutput:
        import torch

        if self._model is None:
            self._load()

        normalised = (image.rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(
            np.transpose(normalised, (2, 0, 1)).astype(np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(tensor)["out"][0]
            probabilities = torch.softmax(logits, dim=0).numpy().astype(np.float64)

        # VOC has 21 classes but a medical/greyscale input activates very few.
        # Keeping all 21 would make entropy dominated by 19 near-zero channels,
        # so we retain the classes that carry real mass plus background.
        mass = probabilities.reshape(probabilities.shape[0], -1).mean(axis=1)
        keep = np.unique(np.concatenate([[0], np.flatnonzero(mass > 0.01)]))
        kept = probabilities[keep]
        kept /= np.clip(kept.sum(axis=0, keepdims=True), 1e-12, None)

        return SegmentationOutput(
            probabilities=kept,
            class_labels=[VOC_CLASSES[i] for i in keep.tolist()],
            logits=logits.numpy().astype(np.float64),
        )
