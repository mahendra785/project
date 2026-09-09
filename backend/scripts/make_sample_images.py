"""Generate the deterministic sample images shipped in `data/sample/`.

These are SYNTHETIC PHANTOMS, not medical images. They exist so the pipeline
can be demonstrated and tested without redistributing patient data, and so that
in-distribution vs out-of-distribution behaviour can be exercised on inputs
whose relationship to the reference domain is known by construction.

Run:  python scripts/make_sample_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.calibration.datasets import _make_phantom  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


def _save(array: np.ndarray, name: str) -> Path:
    path = OUTPUT_DIR / name
    Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parents[1])}")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    print("Generating synthetic sample phantoms...")

    # In-distribution: same generator and parameters as the reference profile.
    for i in range(3):
        image, _ = _make_phantom(rng, 256)
        _save(image, f"phantom_id_{i + 1}.png")

    # Shifted: a gamma/contrast change of the kind a different scanner protocol
    # produces. Same content, different intensity distribution.
    base, _ = _make_phantom(rng, 256)
    _save(np.clip(base**2.2 * 1.4, 0, 1), "phantom_shift_gamma.png")

    # Strongly shifted: inverted polarity plus heavy noise — an easy severe case.
    inverted = 1.0 - base
    noisy = np.clip(inverted + rng.normal(0, 0.18, inverted.shape), 0, 1)
    _save(noisy, "phantom_shift_severe.png")

    print("Done. These are synthetic phantoms — not medical images.")


if __name__ == "__main__":
    main()
