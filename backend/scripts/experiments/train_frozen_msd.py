"""Frozen-model protocol for MSD Task08 liver tumours (pre-registration Amendment 3, P1).

The unit is the PATIENT VOLUME, so every split is patient-disjoint. Volumes are
split once per model seed into D_train (40%) and D_pool (60%); 10% of the
D_train volumes are held out to pick the checkpoint. A 2-D model is trained on
axial slices of D_train volumes only, frozen, and run on every slice of every
pool volume; lesions are 26-connected 3-D tumour components.

    python scripts/experiments/train_frozen_msd.py --arch unet-r34 --seed 0

Output: backend/var/posteriors_frozen/msd-hepatic-vessel/<arch>_seed<m>/<case>.npz
        (prob [D,S,S] float16, labels [D,S,S] uint8, valid) + <arch>_seed<m>.json
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

from train_backbone_msd import NEG_SLICE_FRACTION, load_volume  # noqa: E402
from train_frozen import ARCHS, OUT_ROOT, SMALL_QUANTILES, TRAIN_FRACTION, VAL_FRACTION  # noqa: E402
from train_frozen import lesion_metrics, predict, train  # noqa: E402

from app.datasets.download import dataset_root  # noqa: E402
from app.datasets.registry import get_spec  # noqa: E402

DATASET = "msd-hepatic-vessel"


def slices(vols, idx, rng, neg_fraction):
    xs, ys = [], []
    for v in idx:
        x, y = vols[v]
        pos = np.flatnonzero(y.reshape(len(y), -1).max(1) > 0)
        neg = np.setdiff1d(np.arange(len(y)), pos)
        if len(neg) and neg_fraction > 0:
            neg = rng.choice(neg, int(np.ceil(neg_fraction * len(neg))), replace=False)
        keep = np.concatenate([pos, neg]).astype(int)
        xs.append(x[keep])
        ys.append(y[keep])
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default="unet-r34")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=10_000, help="smoke tests only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import segmentation_models_pytorch as smp
    import torch

    env = dict(torch=torch.__version__, smp=smp.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), python=platform.python_version())
    root = dataset_root(get_spec(DATASET))
    images = sorted(p for p in root.rglob("imagesTr/*.nii.gz") if not p.name.startswith("._"))
    cases = [(p, p.parent.parent / "labelsTr" / p.name) for p in images]
    cases = [c for c in cases if c[1].exists()][: args.max_cases]
    t0 = time.time()
    vols = [load_volume(i, lab, args.size) for i, lab in cases]
    ids = [c[0].name.split(".")[0] for c in cases]
    n = len(vols)
    rng = np.random.default_rng(5000 + args.seed)
    perm = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    tr_all, pool = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    vperm = rng.permutation(tr_all)
    n_val = max(1, int(round(VAL_FRACTION * len(tr_all))))
    val, tr = np.sort(vperm[:n_val]), np.sort(vperm[n_val:])
    x_tr, y_tr = slices(vols, tr, rng, NEG_SLICE_FRACTION)
    x_val, y_val = slices(vols, val, rng, NEG_SLICE_FRACTION)
    print(f"{DATASET} {args.arch} seed {args.seed}: {n} volumes -> train {len(tr)} / val {len(val)} / pool "
          f"{len(pool)} (patient-disjoint); {len(x_tr)} training slices; load {time.time() - t0:.0f}s", flush=True)
    model, mean, std, curve, best_val = train(args.arch, x_tr, y_tr, x_val, y_val, np.ones(y_val.shape, bool),
                                              np.empty((0,)), epochs=args.epochs, seed=args.seed,
                                              batch=args.batch)
    out_dir = OUT_ROOT / DATASET / f"{args.arch}{args.tag}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs, labs = [], []
    for v in pool:
        x, y = vols[v]
        p = predict(model, x, mean, std, "cuda", batch=32).astype(np.float16)
        np.savez_compressed(out_dir / f"{ids[v]}.npz", prob=p, labels=y, valid=np.ones(y.shape, bool))
        probs.append(p.astype(np.float32))
        labs.append(y)
    metrics = lesion_metrics(probs, labs, [np.ones(y.shape, bool) for y in labs])
    sizes = np.concatenate([np.bincount(vols[v][1].ravel())[1:] for v in tr_all if vols[v][1].max()])
    sizes = sizes[sizes > 0]
    cuts = {f"q{q:g}": float(np.percentile(sizes, q)) for q in SMALL_QUANTILES}
    files = sorted(out_dir.glob("*.npz"))
    digest = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()
    manifest = dict(
        protocol="P1 frozen model, patient-disjoint (pre-registration Amendment 3)", dataset=DATASET,
        arch=args.arch, seed=args.seed, epochs=args.epochs, size=args.size, n_units=n, n_train=len(tr),
        n_val=len(val), n_pool=len(pool), train_ids=[ids[i] for i in tr], val_ids=[ids[i] for i in val],
        pool_ids=[ids[i] for i in pool], best_val_dice=best_val, curve=curve, pool_metrics=metrics,
        small_cuts_from_train=cuts, sha256_of_file_hashes=digest, seconds=round(time.time() - t0), env=env,
    )
    (out_dir.parent / f"{args.arch}{args.tag}_seed{args.seed}.json").write_text(json.dumps(manifest, indent=2))
    print(f"  pool metrics: {json.dumps(metrics)}\n  small cuts: {cuts}\n  saved {out_dir} "
          f"({manifest['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
