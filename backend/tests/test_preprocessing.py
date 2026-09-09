"""Preprocessing pipeline tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core.errors import InvalidImageError, PayloadTooLargeError
from app.preprocessing.pipeline import PreprocessingPipeline
from app.utils.arrays import rle_decode, rle_encode


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    mode = "RGB" if array.ndim == 3 else "L"
    Image.fromarray(array.astype(np.uint8), mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def test_output_geometry_and_range(settings, phantom_png):
    result = PreprocessingPipeline(settings).run(phantom_png, content_type="image/png")

    assert result.gray.shape == (settings.inference_size, settings.inference_size)
    assert result.rgb.shape == (settings.inference_size, settings.inference_size, 3)
    assert 0.0 <= result.gray.min() and result.gray.max() <= 1.0
    assert result.original_width == 128 and result.original_height == 128
    assert len(result.checksum) == 64


def test_rgb_input_is_converted_to_grayscale(settings):
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # pure red
    result = PreprocessingPipeline(settings).run(_png(rgb), content_type="image/png")

    assert result.original_channels == 3
    # BT.601 luma of pure red is 0.299.
    assert result.gray.mean() == pytest.approx(0.299, abs=0.01)


def test_statistics_are_populated(settings, phantom_png):
    result = PreprocessingPipeline(settings).run(phantom_png, content_type="image/png")

    for key in ("mean", "std", "skewness", "kurtosis", "entropy", "edge_density"):
        assert key in result.statistics
        assert np.isfinite(result.statistics[key])
    assert result.histogram is not None
    assert result.histogram.sum() == pytest.approx(1.0, abs=1e-6)


def test_empty_upload_rejected(settings):
    with pytest.raises(InvalidImageError):
        PreprocessingPipeline(settings).run(b"")


def test_non_image_rejected(settings):
    with pytest.raises(InvalidImageError):
        PreprocessingPipeline(settings).run(b"this is not an image at all")


def test_oversized_upload_rejected(settings):
    payload = b"x" * (settings.max_upload_bytes + 1)
    with pytest.raises(PayloadTooLargeError):
        PreprocessingPipeline(settings).run(payload)


def test_tiny_image_rejected(settings):
    with pytest.raises(InvalidImageError):
        PreprocessingPipeline(settings).run(
            _png(np.zeros((8, 8), dtype=np.uint8)), content_type="image/png"
        )


def test_normalisation_is_deterministic(settings, phantom_png):
    pipeline = PreprocessingPipeline(settings)
    first = pipeline.run(phantom_png, content_type="image/png")
    second = pipeline.run(phantom_png, content_type="image/png")
    np.testing.assert_array_equal(first.gray, second.gray)


def test_rle_round_trip():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=(37, 53)).astype(np.int32)
    decoded = rle_decode(rle_encode(labels), 37, 53)
    np.testing.assert_array_equal(labels, decoded)


def test_rle_is_compact_for_smooth_masks():
    labels = np.zeros((256, 256), dtype=np.int32)
    labels[64:192, 64:192] = 1
    # A raw list would be 65536 entries; RLE must be dramatically smaller.
    assert len(rle_encode(labels)) < 1000
