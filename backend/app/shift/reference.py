"""Reference-domain construction.

'In-distribution' is only meaningful relative to a reference set. PUMS builds
its default reference from the same procedural phantom generator used for
calibration, which makes the default meaning of ID explicit and honest:
*phantom-like*. It is not a claim about any hospital, scanner or population.

`set_reference_profile()` replaces it with a profile built from real images.
"""

from __future__ import annotations

import numpy as np

from app.calibration.datasets import _make_phantom
from app.core.logging import get_logger
from app.shift.base import ReferenceProfile
from app.utils.arrays import safe_moments

logger = get_logger(__name__)

_PROFILES: dict[str, ReferenceProfile] = {}

HISTOGRAM_BINS = 32


def _statistics(gray: np.ndarray) -> dict[str, float]:
    flat = gray.ravel()
    hist, _ = np.histogram(flat, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
    density = hist / max(hist.sum(), 1)
    nz = density[density > 0]
    skewness, kurtosis = safe_moments(flat)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy": float(-(nz * np.log(nz)).sum() / np.log(HISTOGRAM_BINS)),
        "edge_density": float(
            np.abs(np.gradient(gray, axis=0)).mean()
            + np.abs(np.gradient(gray, axis=1)).mean()
        ),
    }


def build_profile_from_images(
    images: list[np.ndarray], *, name: str, description: str
) -> ReferenceProfile:
    """Aggregate grayscale images in [0, 1] into a reference profile."""
    if not images:
        raise ValueError("At least one reference image is required")

    per_image = [_statistics(image) for image in images]
    histograms = np.stack(
        [
            np.histogram(image, bins=HISTOGRAM_BINS, range=(0.0, 1.0))[0]
            / max(image.size, 1)
            for image in images
        ]
    )

    def column(key: str) -> np.ndarray:
        return np.array([stats[key] for stats in per_image])

    dispersion = {
        key: float(max(column(key).std(), 1e-3))
        for key in ("mean", "std", "skewness", "kurtosis", "edge_density")
    }

    return ReferenceProfile(
        name=name,
        mean=float(column("mean").mean()),
        std=float(column("std").mean()),
        skewness=float(column("skewness").mean()),
        kurtosis=float(column("kurtosis").mean()),
        entropy=float(column("entropy").mean()),
        edge_density=float(column("edge_density").mean()),
        histogram=histograms.mean(axis=0),
        dispersion=dispersion,
        n_samples=len(images),
        description=description,
    )


def _build_synthetic_reference(n_samples: int = 64, size: int = 128, seed: int = 4242):
    rng = np.random.default_rng(seed)
    images = [_make_phantom(rng, size)[0] for _ in range(n_samples)]
    return build_profile_from_images(
        images,
        name="synthetic_reference_v1",
        description=(
            f"{n_samples} procedural phantoms (seed {seed}). Defines 'in-distribution' "
            "as phantom-like. NOT derived from any clinical dataset, hospital or "
            "scanner; replace via set_reference_profile() before drawing conclusions."
        ),
    )


def set_reference_profile(profile: ReferenceProfile) -> None:
    _PROFILES[profile.name] = profile


DEFAULT_REFERENCE = "synthetic_reference_v1"


def _try_real_reference() -> ReferenceProfile | None:
    """Build the reference domain from the configured real dataset, if present."""
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.use_real_dataset:
        return None
    try:
        from app.datasets.download import is_available
        from app.datasets.medical import build_reference_profile

        if not is_available(settings.dataset_id):
            return None
        logger.info("Building reference domain from real dataset '%s'", settings.dataset_id)
        return build_reference_profile(
            settings.dataset_id,
            max_cases=settings.dataset_max_cases,
            slices_per_case=settings.dataset_slices_per_case,
        )
    except Exception as exc:
        logger.warning("Could not build a real reference profile (%s)", exc)
        return None


def get_reference_profile(name: str = DEFAULT_REFERENCE) -> ReferenceProfile:
    """Resolve a reference domain, preferring real data over phantoms.

    `synthetic_reference_v1` is treated as "the default domain" rather than
    "the phantom domain": when a real dataset is available it resolves to that
    dataset's acquisition domain, so `ID` means *this scanner and protocol*
    instead of *phantom-like*. The returned profile's `name` and `description`
    always state which it actually is.
    """
    if name in _PROFILES:
        return _PROFILES[name]

    if name == DEFAULT_REFERENCE:
        real = _try_real_reference()
        if real is not None:
            _PROFILES[DEFAULT_REFERENCE] = real
            _PROFILES[real.name] = real
            return real
        logger.info("Building default synthetic reference profile")
        _PROFILES[name] = _build_synthetic_reference()
        return _PROFILES[name]

    logger.warning(
        "Reference domain '%s' is not registered; falling back to the default.", name
    )
    return get_reference_profile()


def list_reference_domains() -> list[str]:
    get_reference_profile()  # ensure the default exists
    return list(_PROFILES)
