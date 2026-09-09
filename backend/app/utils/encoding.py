"""Image encoding helpers.

Dense maps leave the backend as PNGs, never as JSON float arrays: a 256x256
float32 map is ~1.5 MB as JSON but ~15 kB as a grayscale PNG, and the browser
can decode it natively.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

#: A perceptually-ordered blue -> cyan -> yellow -> red ramp for uncertainty.
#: Deliberately not a rainbow: it is monotonic in luminance so low/high reads
#: correctly in greyscale printouts too.
_HEATMAP_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (12, 24, 58)),
    (0.25, (24, 96, 160)),
    (0.50, (46, 168, 152)),
    (0.75, (238, 186, 62)),
    (1.00, (206, 52, 48)),
]


def _build_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    positions = np.array([s[0] for s in _HEATMAP_STOPS])
    colours = np.array([s[1] for s in _HEATMAP_STOPS], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, 256)
    for channel in range(3):
        lut[:, channel] = np.interp(xs, positions, colours[:, channel]).astype(np.uint8)
    return lut


_LUT = _build_lut()


def quantise_map(values: np.ndarray) -> np.ndarray:
    """Map a float array in [0, 1] onto uint8 0-255 (clipped, not rescaled).

    Clipping rather than rescaling matters: the absolute uncertainty value must
    survive the round-trip so the client can threshold on real numbers.
    """
    arr = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.round(arr * 255.0).astype(np.uint8)


def gray_png_bytes(values: np.ndarray) -> bytes:
    """Grayscale PNG of a [0, 1] map — the lossless-enough numeric transport."""
    img = Image.fromarray(quantise_map(values), mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def heatmap_png_bytes(values: np.ndarray, alpha_from_value: bool = True) -> bytes:
    """Colour-mapped RGBA PNG for direct overlay in the browser.

    When `alpha_from_value`, low-uncertainty pixels become transparent so the
    heatmap can be laid over the source image without hiding anatomy.
    """
    quantised = quantise_map(values)
    rgb = _LUT[quantised]
    if alpha_from_value:
        alpha = quantised.astype(np.uint16)
        # Gentle floor so mid-range uncertainty is still visible.
        alpha = np.clip((alpha * 0.85 + 20), 0, 255).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        img = Image.fromarray(rgba, mode="RGBA")
    else:
        img = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


#: Distinct, colour-blind-friendly palette for segmentation class overlays.
_CLASS_COLOURS: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (232, 92, 74),
    (56, 152, 214),
    (86, 186, 138),
    (233, 176, 62),
    (156, 110, 200),
    (222, 124, 178),
    (108, 190, 200),
]


def mask_png_bytes(labels: np.ndarray, background_class: int = 0) -> bytes:
    """RGBA PNG of an integer label map; the background class is transparent."""
    arr = np.asarray(labels, dtype=np.int32)
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id in np.unique(arr):
        colour = _CLASS_COLOURS[int(class_id) % len(_CLASS_COLOURS)]
        mask = arr == class_id
        rgba[mask, 0:3] = colour
        rgba[mask, 3] = 0 if int(class_id) == background_class else 190
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_png_bytes(array: np.ndarray) -> bytes:
    """Encode a HxW (float 0-1) or HxWx3 (uint8) array as PNG."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        img = Image.fromarray(quantise_map(arr), mode="L")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def class_colour(class_id: int) -> str:
    r, g, b = _CLASS_COLOURS[class_id % len(_CLASS_COLOURS)]
    return f"#{r:02x}{g:02x}{b:02x}"
