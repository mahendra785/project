"""Array helpers: run-length encoding, safe statistics, downsampling."""

from __future__ import annotations

import numpy as np


def rle_encode(labels: np.ndarray) -> list[int]:
    """Flatten an integer label map into alternating (value, run_length) pairs.

    Lossless and exactly invertible by :func:`rle_decode`.
    """
    flat = np.asarray(labels, dtype=np.int32).ravel(order="C")
    if flat.size == 0:
        return []

    # Indices where the value changes.
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [flat.size]))

    out: list[int] = []
    for value, length in zip(flat[starts].tolist(), (ends - starts).tolist()):
        out.append(int(value))
        out.append(int(length))
    return out


def rle_decode(rle: list[int], height: int, width: int) -> np.ndarray:
    """Inverse of :func:`rle_encode`."""
    values = np.asarray(rle[0::2], dtype=np.int32)
    lengths = np.asarray(rle[1::2], dtype=np.int64)
    flat = np.repeat(values, lengths)
    if flat.size != height * width:
        raise ValueError(
            f"RLE decodes to {flat.size} elements, expected {height * width}"
        )
    return flat.reshape((height, width))


def normalise_01(array: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; constant arrays map to all-zeros."""
    arr = np.asarray(array, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def downsample_grid(array: np.ndarray, size: int = 32) -> list[list[float]]:
    """Block-mean downsample a 2-D map to `size x size` for lightweight charts."""
    arr = np.asarray(array, dtype=np.float64)
    h, w = arr.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    out = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
        for j in range(size):
            x0, x1 = xs[j], max(xs[j + 1], xs[j] + 1)
            block = arr[y0:y1, x0:x1]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return np.round(out, 4).tolist()


def safe_moments(values: np.ndarray) -> tuple[float, float]:
    """Skewness and kurtosis that degrade gracefully on constant input.

    `scipy.stats.skew` divides by the standard deviation, so a flat image (a
    blank scan, a saturated field, or an averaged corruption that happens to
    cancel) yields NaN and silently poisons every downstream score. A constant
    signal has no asymmetry and no tail, so 0.0 is both the correct answer and
    a safe one.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size < 4 or float(flat.std()) < 1e-9:
        return 0.0, 0.0

    from scipy import stats as sp_stats

    skewness = float(sp_stats.skew(flat))
    kurtosis = float(sp_stats.kurtosis(flat))
    if not np.isfinite(skewness):
        skewness = 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 0.0
    return skewness, kurtosis


def summarise(array: np.ndarray) -> dict[str, float]:
    """Standard descriptive statistics used by the uncertainty schemas."""
    arr = np.asarray(array, dtype=np.float64).ravel()
    if arr.size == 0:
        return dict.fromkeys(
            ("mean", "std", "minimum", "maximum", "median", "p90", "p95", "p99"), 0.0
        )
    q90, q95, q99 = np.percentile(arr, [90, 95, 99])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "minimum": float(arr.min()),
        "maximum": float(arr.max()),
        "median": float(np.median(arr)),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
    }
