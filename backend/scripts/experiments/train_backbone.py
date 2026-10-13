"""Train a U-Net backbone with K-fold cross-fitting and export out-of-fold posteriors.

Every unit receives a posterior from the one fold model that never saw it during
training, so calibration experiments can use the whole annotated set rather than
a small held-out remainder. Validity caveat (stated in the pre-registration and
paper): CRC assumes ONE fixed predictor; with cross-fitting the (score, label)
pairs come from K models trained on near-identically distributed folds, so
exchangeability is approximate. A fixed-model check is reported alongside.

    python scripts/experiments/train_backbone.py --dataset fracatlas --seed 0
    python scripts/experiments/train_backbone.py --dataset kvasir-seg --seed 0

Output: backend/var/posteriors/<dataset>/unet-r34_seed<k>.npz with
    prob   [N, S, S] float16  foreground posterior at the working resolution
    labels [N, S, S] uint8    GT components labelled at NATIVE resolution, then
                              nearest-resized (component ids preserved)
    valid  [N, S, S] bool     pixels that belong to the image (not letterbox pad)
    ids, fold
and a JSON manifest with per-fold Dice, the environment and the file hash.
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

from app.calibration.losses import gt_components  # noqa: E402
from app.core.config import BACKEND_ROOT  # noqa: E402
from app.datasets.loaders import _load_coco_polygons, load_slices  # noqa: E402
from app.datasets.registry import get_spec  # noqa: E402

OUT_ROOT = BACKEND_ROOT / "var" / "posteriors"
#: Minimum GT component size at NATIVE resolution. 10 px removes Kvasir's JPEG
#: mask speckle (W1 audit: bimodal, nothing in [10, 200)); FracAtlas polygons
#: have no component below 10 px, so the filter is inert there.
MIN_COMPONENT_PX = 10


def letterbox(arr: np.ndarray, size: int, *, nearest: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pad to a square (top-left anchored) then resize to size x size."""
    import cv2

    h, w = arr.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=arr.dtype)
    canvas[:h, :w] = arr
    valid = np.zeros((side, side), dtype=np.uint8)
    valid[:h, :w] = 1
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    out = cv2.resize(canvas, (size, size), interpolation=interp)
    v = cv2.resize(valid, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return out, v


def prepare(dataset: str, size: int, *, negatives: bool = True, max_units: int = 100_000):
    if dataset == "fracatlas":
        pos = _load_coco_polygons(get_spec(dataset), max_units)
        neg = [s for s in _load_coco_polygons(get_spec(dataset), max_units, include_negatives=True)
               if not s.mask.any()] if negatives else []
    else:
        pos, neg = load_slices(dataset, max_cases=max_units), []
    images, labels, valids, ids = [], [], [], []
    for s in pos:
        lab = gt_components(s.mask, MIN_COMPONENT_PX).astype(np.uint16)
        img, v = letterbox(s.image.astype(np.float32), size, nearest=False)
        lab_r, _ = letterbox(lab, size, nearest=True)
        images.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        labels.append(lab_r.astype(np.uint8))
        valids.append(v)
        ids.append(s.case_id)
    neg_images = [(np.clip(letterbox(s.image.astype(np.float32), size, nearest=False)[0], 0, 1) * 255)
                  .astype(np.uint8) for s in neg]
    return (np.stack(images), np.stack(labels), np.stack(valids), ids,
            np.stack(neg_images) if neg_images else np.empty((0, size, size), np.uint8))


def train_fold(x_tr, y_tr, x_neg, *, epochs, seed, device, batch):
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(x_tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_tr))
        for b in range(steps_per_epoch):
            idx = order[b * batch:(b + 1) * batch]
            xb, yb = x_tr[idx], (y_tr[idx] > 0).astype(np.float32)
            if len(x_neg):  # one negative image per two positives
                nidx = rng.integers(0, len(x_neg), max(1, len(idx) // 2))
                xb = np.concatenate([xb, x_neg[nidx]])
                yb = np.concatenate([yb, np.zeros((len(nidx),) + yb.shape[1:], np.float32)])
            # augmentation: flips + gamma; geometry-preserving so labels stay exact
            if rng.random() < 0.5:
                xb, yb = xb[:, :, ::-1], yb[:, :, ::-1]
            if rng.random() < 0.5:
                xb, yb = xb[:, ::-1], yb[:, ::-1]
            xb = (xb.astype(np.float32) / 255.0) ** rng.uniform(0.7, 1.4)
            xt = torch.from_numpy(np.ascontiguousarray(xb)).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
            yt = torch.from_numpy(np.ascontiguousarray(yb)).to(device).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model((xt - mean) / std)
                prob = torch.sigmoid(logits)
                inter = (prob * yt).sum((2, 3))
                dice = 1 - (2 * inter + 1) / (prob.sum((2, 3)) + yt.sum((2, 3)) + 1)
                loss = F.binary_cross_entropy_with_logits(logits, yt) + dice.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model, mean, std


def predict(model, x, mean, std, device, batch=8):
    import torch

    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(x), batch):
            xt = torch.from_numpy(x[b:b + batch].astype(np.float32) / 255.0).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
            xt = (xt - mean) / std
            # test-time flip averaging: cheap, label-free, keeps the score fixed
            p = torch.sigmoid(model(xt)) + torch.sigmoid(model(xt.flip(-1))).flip(-1)
            out.append((p / 2).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def train_predict_rf(x_tr, y_tr, x_te, *, seed, model_dir):
    """CPU backbone: the repository's supervised random forest over a filter bank.

    Same adapter the app serves (RandomForestPixelSegmenter), retrained per fold.
    Used when PyTorch is unavailable; a weaker model than the U-Net, reported as such.
    """
    from app.datasets.medical import as_preprocessed
    from app.models.adapters.rf_pixel_segmenter import RandomForestPixelSegmenter

    path = model_dir / f"rf_fold_seed{seed}.joblib"
    if path.exists():
        path.unlink()
    model = RandomForestPixelSegmenter(model_path=path, seed=seed)
    model.train([im.astype(np.float64) / 255.0 for im in x_tr], [m > 0 for m in y_tr],
                dataset_id="crossfit", n_cases=len(x_tr))
    model.load()
    out = np.stack([model.predict_segmentation(as_preprocessed(im.astype(np.float64) / 255.0))
                    .foreground_probability for im in x_te])
    path.unlink(missing_ok=True)
    return np.clip(out, 0, 1)


def dice(p, y):
    a, b = p >= 0.5, y > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--backbone", choices=["unet-r34", "rf"], default="unet-r34")
    ap.add_argument("--max-units", type=int, default=100_000, help="smoke tests only")
    ap.add_argument("--tag", default="", help="suffix for smoke-test outputs")
    args = ap.parse_args()
    if args.backbone == "unet-r34":
        import segmentation_models_pytorch as smp
        import torch
        env = dict(torch=torch.__version__, smp=smp.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0))
    else:
        import sklearn
        env = dict(sklearn=sklearn.__version__, device="cpu")
    env["python"] = platform.python_version()

    t0 = time.time()
    x, y, valid, ids, x_neg = prepare(args.dataset, args.size, negatives=args.backbone == "unet-r34",
                                      max_units=args.max_units)
    n = len(x)
    rng = np.random.default_rng(1000 + args.seed)
    fold = rng.permutation(np.arange(n) % args.folds)
    prob = np.zeros((n, args.size, args.size), np.float16)
    fold_dice = []
    model_dir = BACKEND_ROOT / "var" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}: {n} units, {len(x_neg)} negatives, {args.folds} folds, "
          f"backbone {args.backbone}, prep {time.time()-t0:.0f}s", flush=True)
    for f in range(args.folds):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        tf = time.time()
        if args.backbone == "unet-r34":
            model, mean, std = train_fold(x[tr], y[tr], x_neg, epochs=args.epochs,
                                          seed=args.seed * 100 + f, device="cuda", batch=args.batch)
            p = predict(model, x[te], mean, std, "cuda")
            del model
            torch.cuda.empty_cache()
        else:
            p = train_predict_rf(x[tr], y[tr], x[te], seed=args.seed * 100 + f, model_dir=model_dir)
        prob[te] = p.astype(np.float16)
        d = [dice(p[i], y[te][i]) for i in range(len(te))]
        fold_dice.append(float(np.mean(d)))
        print(f"  fold {f}: train {len(tr)} test {len(te)}  Dice {fold_dice[-1]:.3f}  ({time.time()-tf:.0f}s)",
              flush=True)

    out_dir = OUT_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.backbone}{args.tag}_seed{args.seed}.npz"
    np.savez_compressed(path, prob=prob, labels=y, valid=valid, ids=np.asarray(ids), fold=fold)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(
        dataset=args.dataset, backbone=args.backbone, seed=args.seed, folds=args.folds,
        epochs=args.epochs, size=args.size, n_units=n, n_negatives=len(x_neg),
        min_component_px_native=MIN_COMPONENT_PX, fold_dice=fold_dice,
        mean_dice=float(np.mean(fold_dice)), sha256=digest, seconds=round(time.time() - t0),
        env=env,
    )
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
