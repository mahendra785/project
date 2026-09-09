"""Shared pytest fixtures.

Tests run against an in-memory repository and a *small* calibration set so the
suite stays fast; the algorithms exercised are identical to production.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("PUMS_STORAGE_BACKEND", "memory")
os.environ.setdefault("PUMS_CALIBRATION_SET_SIZE", "24")
os.environ.setdefault("PUMS_LOG_LEVEL", "WARNING")
# Tests must not depend on a downloaded dataset: they pin the deterministic
# phantom source so the suite passes on a fresh clone. `test_datasets.py`
# covers the real-dataset path separately and skips when it is absent.
os.environ.setdefault("PUMS_USE_REAL_DATASET", "false")
os.environ.setdefault("PUMS_DEFAULT_MODEL_ID", "intensity-mixture-seg-v1")


@pytest.fixture(scope="session")
def settings():
    from app.core.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def registry():
    from app.models.registry import get_registry

    return get_registry()


@pytest.fixture(scope="session")
def segmenter(registry):
    return registry.get("intensity-mixture-seg-v1")


@pytest.fixture(scope="session")
def phantom() -> np.ndarray:
    """A single deterministic phantom in [0, 1]."""
    from app.calibration.datasets import _make_phantom

    image, _ = _make_phantom(np.random.default_rng(7), 128)
    return image


@pytest.fixture(scope="session")
def phantom_png(phantom: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((phantom * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def calibration_set(segmenter):
    from app.calibration.datasets import build_synthetic_calibration_set

    return build_synthetic_calibration_set(segmenter, n_samples=12, size=64)


@pytest.fixture(scope="session")
def preprocessed(settings, phantom_png):
    from app.preprocessing.pipeline import PreprocessingPipeline

    return PreprocessingPipeline(settings).run(phantom_png, content_type="image/png")
