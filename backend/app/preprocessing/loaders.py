"""Image loading with an extension point for medical formats.

`LoadedImage` is the single hand-off type between I/O and the pipeline. Adding
DICOM/NIfTI support means adding a loader here — no other module changes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.errors import InvalidImageError
from app.core.logging import get_logger

logger = get_logger(__name__)

PNG_JPEG_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
DICOM_TYPES = {"application/dicom", "application/octet-stream"}


@dataclass(slots=True)
class LoadedImage:
    """Decoded pixel data in its native geometry and dynamic range."""

    #: float64, shape (H, W) for single-channel or (H, W, C).
    array: np.ndarray
    format: str
    is_dicom: bool = False
    tags: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.array.ndim == 2 else int(self.array.shape[2])


def _looks_like_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def load_dicom(data: bytes) -> LoadedImage:
    """Load a DICOM file. Applies rescale slope/intercept when present.

    Windowing is intentionally NOT applied here — the pipeline's normalisation
    step owns dynamic-range handling so behaviour is identical across formats.
    """
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise InvalidImageError("DICOM support requires the `pydicom` package") from exc

    try:
        dataset = pydicom.dcmread(io.BytesIO(data), force=True)
        pixels = dataset.pixel_array.astype(np.float64)
    except Exception as exc:
        raise InvalidImageError(f"Could not decode DICOM pixel data: {exc}") from exc

    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept

    if pixels.ndim == 3 and pixels.shape[0] > 1 and pixels.shape[-1] not in (3, 4):
        # Multi-frame / volumetric: take the middle slice for the 2-D pipeline.
        # 3-D support is a documented extension point, not a claim.
        pixels = pixels[pixels.shape[0] // 2]

    tags = {
        key: str(getattr(dataset, key))
        for key in ("Modality", "StudyDescription", "SeriesDescription", "BodyPartExamined")
        if getattr(dataset, key, None) is not None
    }
    return LoadedImage(array=pixels, format="DICOM", is_dicom=True, tags=tags)


def load_standard_image(data: bytes) -> LoadedImage:
    """Load PNG/JPEG/BMP/TIFF via Pillow."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img_format = img.format or "UNKNOWN"
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode not in ("L", "RGB", "I;16", "I", "F"):
                img = img.convert("RGB")
            array = np.asarray(img).astype(np.float64)
    except UnidentifiedImageError as exc:
        raise InvalidImageError("File is not a recognised image format") from exc
    except Exception as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc

    return LoadedImage(array=array, format=img_format)


def load_image(data: bytes, content_type: str | None = None) -> LoadedImage:
    """Dispatch to the correct loader. Sniffs content rather than trusting MIME."""
    if not data:
        raise InvalidImageError("Uploaded file is empty")

    if _looks_like_dicom(data) or (content_type or "").lower() == "application/dicom":
        logger.debug("Loading upload as DICOM")
        return load_dicom(data)

    return load_standard_image(data)
