"""Centralised runtime configuration for the PUMS backend.

Every tunable number that influences a *research* result (calibration target
risk, distribution-shift thresholds, risk-engine weights) lives here rather
than being buried as a literal inside an algorithm. This keeps the reported
behaviour of the framework auditable and reproducible.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Application settings, overridable via environment or `backend/.env`."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env"),
        env_prefix="PUMS_",
        extra="ignore",
    )

    # ---------------------------------------------------------------- app ---
    app_name: str = "PUMS — Perception Uncertainty Management System"
    version: str = "0.4.0"
    environment: str = "development"
    log_level: str = "INFO"

    # ---------------------------------------------------------------- api ---
    api_prefix: str = "/api"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # ------------------------------------------------------------ storage ---
    #: Directory used by the JSON-file analysis repository. In-memory storage is
    #: used when `storage_backend == "memory"`.
    storage_backend: str = "json"
    storage_dir: Path = BACKEND_ROOT / "var" / "analyses"
    max_upload_bytes: int = 25 * 1024 * 1024

    # --------------------------------------------------------- inference ---
    #: Preferred adapter. `rf-pixel-seg-v1` is supervised and reports itself
    #: unavailable until trained, in which case the registry-backed fallback in
    #: `resolve_default_model_id()` selects the unsupervised mixture instead.
    default_model_id: str = "rf-pixel-seg-v1"
    fallback_model_id: str = "intensity-mixture-seg-v1"
    #: Every image is resized to this square working resolution before
    #: inference so that pixel-level maps have a predictable, transferable size.
    inference_size: int = 256
    random_seed: int = 1337

    # ------------------------------------------------------- uncertainty ---
    #: Pixels whose normalised predictive entropy exceeds this are counted in
    #: the "high uncertainty" mask and used to seed region-level uncertainty.
    high_uncertainty_threshold: float = 0.55
    #: Minimum connected-component area (in pixels, at inference resolution)
    #: for a high-uncertainty region to be reported as a Level-2 object.
    min_region_area: int = 24
    max_reported_regions: int = 12

    # ----------------------------------------------------------- dataset ---
    #: Real annotated dataset backing calibration and the reference domain.
    #: When it is downloaded, every derived figure is a genuine measurement and
    #: is reported with provenance "computed". When it is absent, PUMS falls
    #: back to procedural phantoms and reports "synthetic" — it never pretends
    #: the fallback is real data.
    dataset_id: str = "msd-hippocampus"
    use_real_dataset: bool = True
    #: Cases/slices drawn when building calibration and reference splits.
    dataset_max_cases: int = 140
    dataset_slices_per_case: int = 3

    # ------------------------------------------------------- calibration ---
    #: Target expected risk alpha for conformal risk control.
    default_target_risk: float = 0.10
    #: Number of bins used by the Expected Calibration Error estimator.
    ece_bins: int = 15
    #: Size of the synthetic-but-deterministic calibration set generated when
    #: no user-supplied calibration split is registered. Clearly reported as
    #: `source="synthetic"` everywhere it is surfaced.
    calibration_set_size: int = 200

    # ------------------------------------------------------------- shift ---
    # Upper bounds on the composite shift score for each status band.
    #
    # DERIVED FROM REAL DATA, NOT GUESSED. Produced by
    # `scripts/calibrate_shift_thresholds.py` against the msd-hippocampus MRI
    # reference domain, using held-out patient slices as the in-distribution
    # population:
    #   * real ID slices score p50=0.125, p95=0.263, p99=0.377, max=0.410
    #     -> ID band ends at 0.38, accepting ~99% of genuine ID inputs;
    #   * MODERATE begins above the observed ID MAXIMUM (not a percentile), so
    #     the bands cannot overlap the natural inter-patient range;
    #   * strong acquisition corruption averages 0.764 -> SEVERE just below it.
    #
    # MEASURED LIMITATION: mild acquisition corruption (strength 0.25) scores
    # 0.328, which lies inside the in-distribution range and is therefore NOT
    # detectable. Natural inter-patient variation in this dataset is as large as
    # that much protocol change. This is reported rather than hidden by
    # tightening the threshold.
    #
    # Operating points against one reference domain — NOT a measurement of
    # clinical OOD detection performance. Re-run the script after changing the
    # dataset or reference profile.
    shift_threshold_mild: float = 0.38
    shift_threshold_moderate: float = 0.48
    shift_threshold_severe: float = 0.73

    # -------------------------------------------------------------- risk ---
    risk_weight_uncertainty: float = 0.40
    risk_weight_confidence: float = 0.25
    risk_weight_shift: float = 0.20
    risk_weight_calibration: float = 0.15
    risk_threshold_low: float = 0.25
    risk_threshold_moderate: float = 0.50
    risk_threshold_high: float = 0.72

    @property
    def shift_thresholds(self) -> dict[str, float]:
        return {
            "mild": self.shift_threshold_mild,
            "moderate": self.shift_threshold_moderate,
            "severe": self.shift_threshold_severe,
        }

    @property
    def risk_weights(self) -> dict[str, float]:
        return {
            "uncertainty": self.risk_weight_uncertainty,
            "confidence": self.risk_weight_confidence,
            "distribution_shift": self.risk_weight_shift,
            "calibration": self.risk_weight_calibration,
        }

    @property
    def risk_thresholds(self) -> dict[str, float]:
        return {
            "low": self.risk_threshold_low,
            "moderate": self.risk_threshold_moderate,
            "high": self.risk_threshold_high,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (also used as a FastAPI dependency)."""
    return Settings()
