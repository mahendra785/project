"""Background warmup.

The synthetic reference profile and calibration set are built by running the
real model over ~100 phantoms, which takes 10-20 s. Doing that lazily inside
the first `/api/analyze` request makes the first analysis look 30x slower than
it is. Warming up on a daemon thread at startup keeps the server responsive
immediately while moving the cost off the request path.
"""

from __future__ import annotations

import threading
import time

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_state = {"status": "cold", "duration_s": None, "error": None}
_lock = threading.Lock()


def status() -> dict:
    with _lock:
        return dict(_state)


def _run(settings: Settings) -> None:
    from app.calibration.datasets import get_calibration_set
    from app.models.registry import get_registry
    from app.shift.reference import get_reference_profile

    started = time.perf_counter()
    try:
        with _lock:
            _state["status"] = "warming"

        get_reference_profile()
        model = get_registry(seed=settings.random_seed).get_available(
            settings.default_model_id
        )
        get_calibration_set(model, n_samples=settings.calibration_set_size // 2)

        elapsed = time.perf_counter() - started
        with _lock:
            _state.update(status="ready", duration_s=round(elapsed, 2))
        logger.info("Warmup complete in %.1fs", elapsed)
    except Exception as exc:  # warmup failure must never stop the server
        logger.warning("Warmup failed (results will be computed lazily): %s", exc)
        with _lock:
            _state.update(status="failed", error=str(exc))


def start(settings: Settings) -> None:
    """Kick off warmup on a daemon thread. Safe to call once per process."""
    with _lock:
        if _state["status"] != "cold":
            return
        _state["status"] = "queued"
    threading.Thread(target=_run, args=(settings,), daemon=True, name="pums-warmup").start()
