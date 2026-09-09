"""`PreprocessingPipeline` — the single, testable path from bytes to tensors.

No API route is allowed to do image work itself. Everything a model adapter
needs (working-resolution grayscale, RGB, and an optional torch tensor) is
produced here, exactly once per request.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.errors import InvalidImageError, PayloadTooLargeError
from app.core.logging import get_logger
from app.preprocessing.loaders import LoadedImage, load_image
from app.utils.arrays import safe_moments

logger = get_logger(__name__)

MAX_DIMENSION = 8192
MIN_DIMENSION = 16


@dataclass(slots=True)
class PreprocessedImage:
    """Everything downstream stages are allowed to see about the input."""

    #: (H, W) float64 in [0, 1] at working resolution — the canonical form.
    gray: np.ndarray
    #: (H, W, 3) float64 in [0, 1] at working resolution.
    rgb: np.ndarray
    #: (H, W) float64 in [0, 1] at *original* resolution, for display.
    display: np.ndarray
    original_height: int
    original_width: int
    original_channels: int
    working_size: int
    image_format: str
    is_dicom: bool
    checksum: str
    size_bytes: int
    tags: dict = field(default_factory=dict)
    #: Intensity statistics of the working-resolution grayscale image; consumed
    #: by the distribution-shift detector.
    statistics: dict[str, float] = field(default_factory=dict)
    histogram: np.ndarray | None = None

    def as_torch(self) -> Any:
        """Optional NCHW float32 tensor for torch-based adapters.

        Returns None when PyTorch is not installed, so the classical path keeps
        working in a torch-free environment.
        """
        try:
            import torch
        except ImportError:
            return None
        chw = np.transpose(self.rgb, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(chw).unsqueeze(0)


class PreprocessingPipeline:
    """Validate -> load -> to-grayscale -> normalise -> resize -> summarise."""

    #: Number of bins in the intensity histogram handed to the shift detector.
    HISTOGRAM_BINS = 32

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------- public --
    def run(
        self,
        data: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> PreprocessedImage:
        self.validate(data, filename=filename)
        loaded = load_image(data, content_type)
        self._validate_geometry(loaded)

        display = self._to_grayscale(loaded.array)
        display = self._normalise(display, is_dicom=loaded.is_dicom)

        size = self.settings.inference_size
        gray = self._resize(display, size, size)
        rgb = self._to_rgb(loaded, size)

        stats, histogram = self._summarise(gray)

        return PreprocessedImage(
            gray=gray,
            rgb=rgb,
            display=display,
            original_height=loaded.height,
            original_width=loaded.width,
            original_channels=loaded.channels,
            working_size=size,
            image_format=loaded.format,
            is_dicom=loaded.is_dicom,
            checksum=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            tags=loaded.tags,
            statistics=stats,
            histogram=histogram,
        )

    def validate(self, data: bytes, *, filename: str | None = None) -> None:
        if not data:
            raise InvalidImageError("Uploaded file is empty")
        if len(data) > self.settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"File exceeds the {self.settings.max_upload_bytes // (1024 * 1024)} MB limit",
                detail={"size_bytes": len(data)},
            )

    # ------------------------------------------------------------ helpers --
    def _validate_geometry(self, loaded: LoadedImage) -> None:
        h, w = loaded.height, loaded.width
        if h < MIN_DIMENSION or w < MIN_DIMENSION:
            raise InvalidImageError(
                f"Image is too small ({w}x{h}); minimum is {MIN_DIMENSION}px per side"
            )
        if h > MAX_DIMENSION or w > MAX_DIMENSION:
            raise InvalidImageError(
                f"Image is too large ({w}x{h}); maximum is {MAX_DIMENSION}px per side"
            )
        if loaded.array.ndim == 3 and loaded.channels not in (1, 3, 4):
            raise InvalidImageError(
                f"Unsupported channel count: {loaded.channels}. "
                "Multi-channel (>4) volumes are a planned extension."
            )

    @staticmethod
    def _to_grayscale(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return array.astype(np.float64)
        if array.shape[2] == 1:
            return array[:, :, 0].astype(np.float64)
        # ITU-R BT.601 luma — the standard perceptual grayscale conversion.
        weights = np.array([0.299, 0.587, 0.114])
        return array[:, :, :3].astype(np.float64) @ weights

    @staticmethod
    def _normalise(gray: np.ndarray, *, is_dicom: bool) -> np.ndarray:
        """Map to [0, 1].

        Medical data (and DICOM in particular) has arbitrary, modality-dependent
        dynamic range, so we use robust percentile scaling (1st/99th) which is
        insensitive to a handful of extreme voxels. Ordinary 8-bit images are
        simply divided by 255 so their appearance is preserved exactly.
        """
        if not is_dicom and gray.max() <= 255.0 and gray.min() >= 0.0:
            return np.clip(gray / 255.0, 0.0, 1.0)

        lo, hi = np.percentile(gray, [1.0, 99.0])
        if hi - lo < 1e-9:
            lo, hi = float(gray.min()), float(gray.max())
        if hi - lo < 1e-9:
            return np.zeros_like(gray)
        return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
        """Area-weighted resize (anti-aliased); falls back to PIL if cv2 absent."""
        try:
            import cv2

            interpolation = (
                cv2.INTER_AREA
                if array.shape[0] > height or array.shape[1] > width
                else cv2.INTER_LINEAR
            )
            return cv2.resize(
                array.astype(np.float32), (width, height), interpolation=interpolation
            ).astype(np.float64)
        except ImportError:  # pragma: no cover - cv2 is a hard requirement
            from PIL import Image

            img = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            return np.asarray(img).astype(np.float64) / 255.0

    def _to_rgb(self, loaded: LoadedImage, size: int) -> np.ndarray:
        if loaded.array.ndim == 3 and loaded.channels >= 3:
            channels = [
                self._resize(
                    self._normalise(
                        loaded.array[:, :, c].astype(np.float64), is_dicom=loaded.is_dicom
                    ),
                    size,
                    size,
                )
                for c in range(3)
            ]
            return np.clip(np.dstack(channels), 0.0, 1.0)
        gray = self._resize(
            self._normalise(self._to_grayscale(loaded.array), is_dicom=loaded.is_dicom),
            size,
            size,
        )
        return np.dstack([gray, gray, gray])

    def _summarise(self, gray: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        flat = gray.ravel()
        hist, _ = np.histogram(flat, bins=self.HISTOGRAM_BINS, range=(0.0, 1.0))
        density = hist / max(hist.sum(), 1)

        # Shannon entropy of the intensity histogram, normalised to [0, 1].
        nz = density[density > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(self.HISTOGRAM_BINS))

        skewness, kurtosis = safe_moments(flat)
        summary = {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "entropy": entropy,
            # Mean gradient magnitude: a cheap proxy for sharpness/texture that
            # is sensitive to scanner and reconstruction-kernel differences.
            "edge_density": float(
                np.abs(np.gradient(gray, axis=0)).mean()
                + np.abs(np.gradient(gray, axis=1)).mean()
            ),
        }
        return summary, density
