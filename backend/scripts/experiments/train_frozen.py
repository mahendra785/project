"""Frozen-model protocol (pre-registration Amendment 3, P1): train once, freeze, predict the pool.

Units are split once per model seed into D_train (40%) and D_pool (60%). The
backbone sees D_train only (10% of it held out to pick the checkpoint), is frozen,
and predicts every D_pool unit. Calibration and test units are then i.i.d. given
the frozen model, which is exactly the setting of the conformal risk control
theorem: no cross-fitting, no approximate exchangeability.

The small-lesion cut is taken from D_train ground truth, so the loss is fixed
before any pool label is looked at.

    python scripts/experiments/train_frozen.py --dataset kvasir-seg --arch unet-r34 --seed 0

Output: backend/var/posteriors_frozen/<dataset>/<arch>_seed<m>.npz
    prob [N_pool, S, S] float16, labels [N_pool, S, S] uint8, valid, ids
and <arch>_seed<m>.json with the split, pre-CRC metrics, small-lesion cuts,
training curve, environment and file hash.
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_backbone import prepare  # noqa: E402

from app.calibration.families import connectivity  # noqa: E402
from app.core.config import BACKEND_ROOT  # noqa: E402

OUT_ROOT = BACKEND_ROOT / "var" / "posteriors_frozen"
TRAIN_FRACTION = 0.4
VAL_FRACTION = 0.1  # of D_train, for checkpoint selection only
SMALL_QUANTILES = (25.0, 33.0, 50.0)
ARCHS = ("unet-r34", "unet-mitb2", "fpn-effb3")


def build(arch: str):
    import segmentation_models_pytorch as smp

    if arch == "unet-r34":
        return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    if arch == "unet-mitb2":
        return smp.Unet("mit_b2", encoder_weights="imagenet", in_channels=3, classes=1)
    if arch == "fpn-effb3":
        return smp.FPN("efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=1)
    raise ValueError(arch)


def to_tensor(xb, device, mean, std):
    import torch

    xt = torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32)).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
    return (xt - mean) / std


def predict(model, x, mean, std, device, batch=8):
    """Sigmoid posterior with horizontal-flip averaging (label-free, fixed)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(x), batch):
            xt = to_tensor(x[b:b + batch].astype(np.float32) / 255.0, device, mean, std)
            p = torch.sigmoid(model(xt)) + torch.sigmoid(model(xt.flip(-1))).flip(-1)
            out.append((p / 2).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,) + x.shape[1:], np.float32)


def mean_dice(prob, labels, valid):
    d = []
    for p, y, v in zip(prob, labels, valid):
        a, b = (p >= 0.5) & v, y > 0
        d.append(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    return float(np.mean(d))


def train(arch, x_tr, y_tr, x_val, y_val, v_val, x_neg, *, epochs, seed, batch, device="cuda", log=print):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(seed)
    model = build(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = int(np.ceil(len(x_tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler("cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    best, best_state, curve = -1.0, None, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(x_tr))
        losses = []
        for b in range(steps):
            idx = order[b * batch:(b + 1) * batch]
            xb, yb = x_tr[idx], (y_tr[idx] > 0).astype(np.float32)
            if len(x_neg):  # one lesion-free image per two positives (FracAtlas)
                nidx = rng.integers(0, len(x_neg), max(1, len(idx) // 2))
                xb = np.concatenate([xb, x_neg[nidx]])
                yb = np.concatenate([yb, np.zeros((len(nidx),) + yb.shape[1:], np.float32)])
            if rng.random() < 0.5:
                xb, yb = xb[:, :, ::-1], yb[:, :, ::-1]
            if rng.random() < 0.5:
                xb, yb = xb[:, ::-1], yb[:, ::-1]
            xb = (xb.astype(np.float32) / 255.0) ** rng.uniform(0.7, 1.4)
            xt = to_tensor(xb, device, mean, std)
            yt = torch.from_numpy(np.ascontiguousarray(yb)).to(device).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xt)
                prob = torch.sigmoid(logits)
                inter = (prob * yt).sum((2, 3))
                dice = 1 - (2 * inter + 1) / (prob.sum((2, 3)) + yt.sum((2, 3)) + 1)
                loss = F.binary_cross_entropy_with_logits(logits, yt) + dice.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
        vd = mean_dice(predict(model, x_val, mean, std, device), y_val, v_val)
        curve.append(dict(epoch=ep + 1, loss=float(np.mean(losses)), val_dice=vd))
        if vd > best:
            best = vd
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        log(f"    epoch {ep + 1:3d}  loss {np.mean(losses):.4f}  val Dice {vd:.3f}{'  *' if vd == best else ''}")
    model.load_state_dict(best_state)
    return model, mean, std, curve, best


def lesion_metrics(prob, labels, valid, tau=0.5):
    """Pre-CRC segmentation and lesion-detection metrics at p >= 0.5."""
    dice, iou, rec_hits, rec_n, prec_hits, prec_n, fp = [], [], 0, 0, 0, 0, []
    for p, y, v in zip(prob, labels, valid):
        pred, gt = (p >= 0.5) & v, y > 0
        inter, union = (pred & gt).sum(), (pred | gt).sum()
        dice.append(2 * inter / max(pred.sum() + gt.sum(), 1))
        iou.append(inter / max(union, 1))
        for k in range(1, int(y.max()) + 1):
            comp = y == k
            if comp.any():
                rec_n += 1
                rec_hits += int((pred & comp).sum() >= tau * comp.sum())
        lab, n = ndimage.label(pred, structure=connectivity(pred.ndim))
        prec_n += n
        hit = np.unique(lab[gt & (lab > 0)])
        prec_hits += hit.size
        fp.append(n - hit.size)
    return dict(dice=float(np.mean(dice)), iou=float(np.mean(iou)),
                lesion_recall=rec_hits / max(rec_n, 1), lesion_precision=prec_hits / max(prec_n, 1),
                fp_components_per_unit=float(np.mean(fp)), n_lesions=rec_n, n_units=len(prob))


def component_sizes(labels):
    out = []
    for y in labels:
        if y.max():
            out.append(np.bincount(y.ravel())[1:])
    sizes = np.concatenate(out) if out else np.zeros(0)
    return sizes[sizes > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arch", choices=ARCHS, default="unet-r34")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-units", type=int, default=100_000, help="smoke tests only")
    ap.add_argument("--tag", default="", help="suffix for smoke-test outputs")
    args = ap.parse_args()
    import segmentation_models_pytorch as smp
    import torch

    env = dict(torch=torch.__version__, smp=smp.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), python=platform.python_version())
    t0 = time.time()
    x, y, valid, ids, x_neg = prepare(args.dataset, args.size, negatives=True, max_units=args.max_units)
    n = len(x)
    rng = np.random.default_rng(5000 + args.seed)
    perm = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    tr_all, pool = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    n_val = max(1, int(round(VAL_FRACTION * len(tr_all))))
    vperm = rng.permutation(tr_all)
    val, tr = np.sort(vperm[:n_val]), np.sort(vperm[n_val:])
    # lesion-free FracAtlas images: split by the same fraction, training part only
    neg_tr = x_neg[rng.permutation(len(x_neg))[:int(round(TRAIN_FRACTION * len(x_neg)))]] if len(x_neg) else x_neg
    print(f"{args.dataset} {args.arch} seed {args.seed}: {n} units -> train {len(tr)} / val {len(val)} / "
          f"pool {len(pool)}; {len(neg_tr)} lesion-free training images; prep {time.time() - t0:.0f}s", flush=True)

    model, mean, std, curve, best_val = train(args.arch, x[tr], y[tr], x[val], y[val], valid[val], neg_tr,
                                              epochs=args.epochs, seed=args.seed, batch=args.batch)
    prob = predict(model, x[pool], mean, std, "cuda")
    metrics = lesion_metrics(prob, y[pool], valid[pool])
    sizes = component_sizes(y[tr_all])
    cuts = {f"q{q:g}": float(np.percentile(sizes, q)) for q in SMALL_QUANTILES}
    print(f"  pool metrics: {json.dumps(metrics)}\n  small cuts (D_train): {cuts}", flush=True)

    out_dir = OUT_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.arch}{args.tag}_seed{args.seed}"
    path = out_dir / f"{stem}.npz"
    np.savez_compressed(path, prob=prob.astype(np.float16), labels=y[pool], valid=valid[pool],
                        ids=np.asarray(ids)[pool])
    manifest = dict(
        protocol="P1 frozen model (pre-registration Amendment 3)", dataset=args.dataset, arch=args.arch,
        seed=args.seed, epochs=args.epochs, size=args.size, batch=args.batch,
        n_units=n, n_train=len(tr), n_val=len(val), n_pool=len(pool), n_negatives_train=len(neg_tr),
        train_ids=[str(ids[i]) for i in tr], val_ids=[str(ids[i]) for i in val],
        pool_ids=[str(ids[i]) for i in pool], best_val_dice=best_val, curve=curve,
        pool_metrics=metrics, small_cuts_from_train=cuts,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seconds=round(time.time() - t0), env=env,
    )
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"  saved {path}  ({manifest['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
