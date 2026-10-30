"""Build the data bundle for the live lesion-level CRC demo (/api/demo, frontend /live-demo).

Input: the frozen-model pool posteriors written by experiments/kaggle_train.py
    var/posteriors_frozen/<dataset>/<arch>_seed<m>.npz (+ .json manifest)
and the original images, letterboxed exactly as in training (square zero padding at the
bottom/right, then area resize), so the colour image lines up pixel for pixel with the
posterior the model produced from its grey version.

Output: var/demo/<dataset>_<arch>_seed<m>.npz with prob, labels, valid, ids, JPEG bytes and
the manifest, which app/services/lesion_demo.py loads.

    python scripts/build_demo_bundle.py --data-root var/demo_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BACKEND_ROOT  # noqa: E402

POSTERIORS = BACKEND_ROOT / "var" / "posteriors_frozen"
OUT = BACKEND_ROOT / "var" / "demo"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def letterbox_rgb(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    canvas[:h, :w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kvasir-seg")
    ap.add_argument("--arch", default="unet-r34")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=str(BACKEND_ROOT / "var" / "demo_data"))
    args = ap.parse_args()

    src = POSTERIORS / args.dataset / f"{args.arch}_seed{args.seed}.npz"
    d = np.load(src)
    manifest = json.loads(src.with_suffix(".json").read_text())
    prob, labels, valid, ids = d["prob"], d["labels"], d["valid"], [str(i) for i in d["ids"]]
    size = prob.shape[-1]

    image_dir = next(p for p in sorted(Path(args.data_root, args.dataset).rglob("*"))
                     if p.is_dir() and p.name.lower() in {"images", "image", "img"})
    blobs, offsets = [], [0]
    for uid in ids:
        path = next(image_dir / f"{uid}{e}" for e in EXTS if (image_dir / f"{uid}{e}").exists())
        rgb = letterbox_rgb(cv2.imread(str(path), cv2.IMREAD_COLOR), size)  # BGR stays BGR for imencode
        ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok, path
        blobs.append(buf.tobytes())
        offsets.append(offsets[-1] + len(blobs[-1]))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.dataset}_{args.arch}_seed{args.seed}.npz"
    np.savez_compressed(
        out, prob=prob, labels=labels, valid=valid, ids=np.asarray(ids),
        jpeg=np.frombuffer(b"".join(blobs), dtype=np.uint8), jpeg_offsets=np.asarray(offsets, dtype=np.int64),
        manifest=np.asarray(json.dumps({k: manifest[k] for k in manifest if not k.endswith("_ids")})),
    )
    print(f"saved {out} ({out.stat().st_size / 2**20:.1f} MB): {len(ids)} units at {size}x{size}")


if __name__ == "__main__":
    main()
