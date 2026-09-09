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
    default_model_id: str = "intensity-mixture-seg-v1"
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
    # DERIVED, NOT GUESSED. Produced by `scripts/calibrate_shift_thresholds.py`
    # against the synthetic reference domain:
    #   * known-in-distribution phantoms score p50=0.236, p95=0.386, p99=0.421
    #     -> ID band ends at 0.42, so ~99% of genuine ID inputs are called ID;
    #   * the weakest controlled corruption averages 0.57, so the MILD/MODERATE
    #     split sits midway between the ID ceiling and that point;
    #   * full intensity inversion averages 0.82, so SEVERE begins just below it.
    #
    # These are operating points on synthetic phantoms. They calibrate the
    # detector against its own reference domain and are NOT a measurement of
    # clinical OOD detection performance. Re-run the script after changing the
    # reference profile.
    shift_threshold_mild: float = 0.42
    shift_threshold_moderate: float = 0.50
    shift_threshold_severe: float = 0.78

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
