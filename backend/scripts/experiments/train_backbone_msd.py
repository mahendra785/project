"""MSD Task08 (liver tumours): 2-D U-Net on axial slices -> 3-D volume posteriors.

The calibration unit is the VOLUME and lesions are 26-connected 3-D components,
fixing the two W1 infrastructure flags (densest-slice sampler; 2-D units cannot
express a 3-D lesion). Every slice of every volume is predicted; nothing is
sampled by label.

Cross-fitting as in train_backbone.py (5 folds over volumes). Output, one file
per volume so 3-D posteriors never have to sit in memory together:
    backend/var/posteriors/msd-hepatic-vessel/unet-r34_seed<k>/<case>.npz
        prob [D, S, S] float16, labels [D, S, S] uint8, valid [D, S, S] bool

    python scripts/experiments/train_backbone_msd.py --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_backbone import OUT_ROOT, predict, train_fold  # noqa: E402

from app.calibration.losses import gt_components  # noqa: E402
from app.datasets.download import dataset_root  # noqa: E402
from app.datasets.registry import get_spec  # noqa: E402

DATASET = "msd-hepatic-vessel"
#: CT window for liver (HU). Fixed a priori from standard abdominal windowing.
HU_LO, HU_HI = -100.0, 250.0
#: Minimum 3-D tumour size in voxels at native resolution.
MIN_COMPONENT_VOX = 10
#: Fraction of tumour-free slices kept per training volume (negatives).
NEG_SLICE_FRACTION = 0.1


def load_volume(image_path: Path, label_path: Path, size: int):
    import cv2
    import nibabel as nib

    img = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(str(label_path)).dataobj)
    tumour = lab == 2  # label 1 = vessel (ignored), 2 = tumour
    comps = gt_components(tumour, MIN_COMPONENT_VOX)
    assert comps.max() < 256, 'more than 255 tumours in one volume'
    comps = comps.astype(np.uint8)
    img = np.clip((img - HU_LO) / (HU_HI - HU_LO), 0, 1)
    d = img.shape[2]
    x = np.empty((d, size, size), np.uint8)
    y = np.empty((d, size, size), np.uint8)
    for z in range(d):
        x[z] = (cv2.resize(img[:, :, z], (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        y[z] = cv2.resize(comps[:, :, z], (size, size), interpolation=cv2.INTER_NEAREST)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-cases", type=int, default=10_000)
    args = ap.parse_args()
    import segmentation_models_pytorch as smp
    import torch

    spec = get_spec(DATASET)
    root = dataset_root(spec)
    images = sorted(p for p in root.rglob("imagesTr/*.nii.gz") if not p.name.startswith("._"))[: args.max_cases]
    cases = [(p, p.parent.parent / "labelsTr" / p.name) for p in images]
    cases = [c for c in cases if c[1].exists()]
    rng = np.random.default_rng(1000 + args.seed)
    fold = rng.permutation(np.arange(len(cases)) % args.folds)
    out_dir = OUT_ROOT / DATASET / f"unet-r34_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    vols = [load_volume(i, l, args.size) for i, l in cases]
    print(f"loaded {len(vols)} volumes in {time.time()-t0:.0f}s")
    fold_dice = []
    for f in range(args.folds):
        tf = time.time()
        xs, ys = [], []
        for v, (x, y) in enumerate(vols):
            if fold[v] == f:
                continue
            pos = np.flatnonzero(y.reshape(len(y), -1).max(1) > 0)
            neg = np.setdiff1d(np.arange(len(y)), pos)
            neg = rng.choice(neg, int(np.ceil(NEG_SLICE_FRACTION * len(neg))), replace=False) if len(neg) else neg
            keep = np.concatenate([pos, neg])
            xs.append(x[keep])
            ys.append(y[keep])
        model, mean, std = train_fold(np.concatenate(xs), np.concatenate(ys), np.empty((0,)),
                                      epochs=args.epochs, seed=args.seed * 100 + f, device="cuda",
                                      batch=args.batch)
        inter = tot = 0.0
        for v, (x, y) in enumerate(vols):
            if fold[v] != f:
                continue
            p = predict(model, x, mean, std, "cuda", batch=32)
            inter += float(((p >= 0.5) & (y > 0)).sum())
            tot += float((p >= 0.5).sum() + (y > 0).sum())
            np.savez_compressed(out_dir / f"{cases[v][0].name.split('.')[0]}.npz",
                                prob=p.astype(np.float16), labels=y, valid=np.ones(y.shape, bool),
                                fold=f)
        fold_dice.append(2 * inter / max(tot, 1))
        print(f"  fold {f}: global Dice {fold_dice[-1]:.3f} ({time.time()-tf:.0f}s)")
        del model
        torch.cuda.empty_cache()

    files = sorted(out_dir.glob("*.npz"))
    digest = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()
    manifest = dict(dataset=DATASET, backbone="unet-resnet34-imagenet (2-D slices, 3-D units)", seed=args.seed,
                    folds=args.folds, epochs=args.epochs, size=args.size, n_units=len(files),
                    hu_window=[HU_LO, HU_HI], min_component_vox_native=MIN_COMPONENT_VOX,
                    neg_slice_fraction=NEG_SLICE_FRACTION, fold_dice_global=fold_dice,
                    mean_dice=float(np.mean(fold_dice)), sha256_of_file_hashes=digest,
                    seconds=round(time.time() - t0),
                    env=dict(python=platform.python_version(), torch=torch.__version__, smp=smp.__version__,
                             cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    (out_dir.parent / f"unet-r34_seed{args.seed}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
