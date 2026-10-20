"""Frozen-model training for the revision runs, self-contained for Kaggle Notebooks.

Same protocol as train_frozen.py / train_frozen_msd.py (pre-registration
Amendment 3, P1). The dataset readers replicate app/datasets/loaders.py exactly
(Kvasir-SEG image/mask pairs; FracAtlas COCO polygons rasterised at native
resolution, lesion-free radiographs as training negatives; MSD Task08 tumours,
label 2, as 26-connected 3-D components), so the code runs without the rest of
the PUMS backend.

    python kaggle_train.py --dataset kvasir-seg --arch unet-r34 --seed 0 --data-root /tmp/pums_data

Output (identical format to train_frozen.py):
    <BACKEND_ROOT>/var/posteriors_frozen/<dataset>/<arch>_seed<m>.npz + .json      (2-D)
    <BACKEND_ROOT>/var/posteriors_frozen/<dataset>/<arch>_seed<m>/<case>.npz + .json (3-D)
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

from app.calibration.families import connectivity  # noqa: E402
from app.calibration.losses import gt_components  # noqa: E402
from app.core.config import BACKEND_ROOT  # noqa: E402

OUT_ROOT = BACKEND_ROOT / "var" / "posteriors_frozen"
TRAIN_FRACTION = 0.4
VAL_FRACTION = 0.1
SMALL_QUANTILES = (25.0, 33.0, 50.0)
MIN_COMPONENT_PX = 10          # 2-D, native resolution (as train_backbone.py)
MIN_COMPONENT_VOX = 10         # 3-D Task08 (as train_backbone_msd.py)
HU_LO, HU_HI = -100.0, 250.0   # liver CT window (as train_backbone_msd.py)
NEG_SLICE_FRACTION = 0.1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARCHS = ("unet-r34", "unet-mitb2", "fpn-effb3")


# ----------------------------------------------------------------- data (replicates app/datasets/loaders.py)
def _gray(path):
    from PIL import Image

    with Image.open(path) as h:
        return np.asarray(h.convert("L"), dtype=np.float64) / 255.0


def load_kvasir(root: Path):
    mask_dir = next(d for d in sorted(root.rglob("*")) if d.is_dir() and d.name.lower() in {"masks", "mask", "annotations"})
    image_dir = next(d for d in sorted(root.rglob("*")) if d.is_dir() and d.name.lower() in {"images", "image", "img"})
    from PIL import Image

    for mask_path in sorted(mask_dir.iterdir()):
        if mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        cands = [image_dir / mask_path.name, *(image_dir / f"{mask_path.stem}{e}" for e in IMAGE_EXTENSIONS)]
        ip = next((c for c in cands if c.exists()), None)
        if ip is None:
            continue
        image = _gray(ip)
        with Image.open(mask_path) as h:
            mask = np.asarray(h.convert("L")) > 127
        if mask.shape != image.shape or not mask.any():
            continue
        yield image, mask, mask_path.stem


def load_fracatlas(root: Path):
    from PIL import Image, ImageDraw

    coco_path = next(root.rglob("COCO_fracture_masks.json"))
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    image_dir = next(d for d in root.rglob("Fractured") if d.is_dir())
    for info in sorted(coco["images"], key=lambda r: r["file_name"]):
        path = image_dir / info["file_name"]
        anns = by_image.get(info["id"], [])
        if not path.exists() or not anns:
            continue
        image = _gray(path)
        canvas = Image.new("L", (image.shape[1], image.shape[0]), 0)
        draw = ImageDraw.Draw(canvas)
        for ann in anns:
            seg = ann["segmentation"]
            for poly in (json.loads(seg) if isinstance(seg, str) else seg):
                if len(poly) >= 6:
                    draw.polygon(list(map(float, poly)), outline=1, fill=1)
        mask = np.asarray(canvas) > 0
        if mask.any():
            yield image, mask, Path(info["file_name"]).stem


def fracatlas_negatives(root: Path):
    neg_dir = next(d for d in root.rglob("Non_fractured") if d.is_dir())
    skipped = 0
    for p in sorted(neg_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                yield _gray(p)
            except OSError as exc:  # a corrupt/truncated file must not stop the run; it is counted
                skipped += 1
                print(f"  skipped unreadable negative {p.name}: {exc}", flush=True)
    if skipped:
        print(f"  {skipped} lesion-free image(s) skipped", flush=True)


def letterbox(arr, size, *, nearest):
    import cv2

    h, w = arr.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=arr.dtype)
    canvas[:h, :w] = arr
    valid = np.zeros((side, side), dtype=np.uint8)
    valid[:h, :w] = 1
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    return out, cv2.resize(valid, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def prepare_2d(dataset, data_root: Path, size, max_units=None):
    # Streamed: each image is resized as soon as it is read (FracAtlas radiographs are
    # ~1760x2140; holding the 3366 lesion-free ones at native resolution would need ~100 GB).
    root = data_root / dataset
    pos = load_kvasir(root) if dataset == "kvasir-seg" else load_fracatlas(root)
    neg = fracatlas_negatives(root) if dataset == "fracatlas" else iter(())
    import itertools

    pos, neg = itertools.islice(pos, max_units), itertools.islice(neg, max_units)
    images, labels, valids, ids = [], [], [], []
    for image, mask, cid in pos:
        lab = gt_components(mask, MIN_COMPONENT_PX).astype(np.uint16)
        img, v = letterbox(image.astype(np.float32), size, nearest=False)
        lab_r, _ = letterbox(lab, size, nearest=True)
        images.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        labels.append(lab_r.astype(np.uint8))
        valids.append(v)
        ids.append(cid)
    x_neg = [(np.clip(letterbox(im.astype(np.float32), size, nearest=False)[0], 0, 1) * 255).astype(np.uint8)
             for im in neg]
    return (np.stack(images), np.stack(labels), np.stack(valids), ids,
            np.stack(x_neg) if x_neg else np.empty((0, size, size), np.uint8))


def load_volume(image_path, label_path, size):
    import cv2
    import nibabel as nib

    img = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(str(label_path)).dataobj)
    comps = gt_components(lab == 2, MIN_COMPONENT_VOX)
    assert comps.max() < 256, "more than 255 tumours in one volume"
    comps = comps.astype(np.uint8)
    img = np.clip((img - HU_LO) / (HU_HI - HU_LO), 0, 1)
    d = img.shape[2]
    x = np.empty((d, size, size), np.uint8)
    y = np.empty((d, size, size), np.uint8)
    for z in range(d):
        x[z] = (cv2.resize(img[:, :, z], (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        y[z] = cv2.resize(comps[:, :, z], (size, size), interpolation=cv2.INTER_NEAREST)
    return x, y


# ----------------------------------------------------------------- model (as train_frozen.py)
def build(arch):
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


def predict(model, x, mean, std, device="cuda", batch=8):
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


def train(arch, x_tr, y_tr, x_val, y_val, v_val, x_neg, *, epochs, seed, batch, device="cuda"):
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
            if len(x_neg):
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
        print(f"    epoch {ep + 1:3d}  loss {np.mean(losses):.4f}  val Dice {vd:.3f}{'  *' if vd == best else ''}",
              flush=True)
    model.load_state_dict(best_state)
    return model, mean, std, curve, best


def lesion_metrics(prob, labels, valid, tau=0.5):
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


def small_cuts(label_arrays):
    sizes = [np.bincount(y.ravel())[1:] for y in label_arrays if y.max()]
    sizes = np.concatenate(sizes) if sizes else np.zeros(0)
    sizes = sizes[sizes > 0]
    return {f"q{q:g}": float(np.percentile(sizes, q)) for q in SMALL_QUANTILES}


def split(n, seed):
    rng = np.random.default_rng(5000 + seed)
    perm = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    tr_all, pool = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    vperm = rng.permutation(tr_all)
    n_val = max(1, int(round(VAL_FRACTION * len(tr_all))))
    return rng, np.sort(vperm[n_val:]), np.sort(vperm[:n_val]), tr_all, pool


def env_info():
    import segmentation_models_pytorch as smp
    import torch

    return dict(torch=torch.__version__, smp=smp.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), python=platform.python_version(), platform="kaggle")


def run_2d(args):
    t0 = time.time()
    x, y, valid, ids, x_neg = prepare_2d(args.dataset, Path(args.data_root), args.size, args.max_units)
    rng, tr, val, tr_all, pool = split(len(x), args.seed)
    neg_tr = x_neg[rng.permutation(len(x_neg))[:int(round(TRAIN_FRACTION * len(x_neg)))]] if len(x_neg) else x_neg
    print(f"{args.dataset} {args.arch} seed {args.seed}: {len(x)} units -> train {len(tr)} / val {len(val)} / "
          f"pool {len(pool)}; {len(neg_tr)} lesion-free training images; prep {time.time() - t0:.0f}s", flush=True)
    model, mean, std, curve, best_val = train(args.arch, x[tr], y[tr], x[val], y[val], valid[val], neg_tr,
                                              epochs=args.epochs, seed=args.seed, batch=args.batch)
    prob = predict(model, x[pool], mean, std)
    metrics = lesion_metrics(prob, y[pool], valid[pool])
    cuts = small_cuts(y[tr_all])
    out_dir = args.out_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.arch}_seed{args.seed}.npz"
    np.savez_compressed(path, prob=prob.astype(np.float16), labels=y[pool], valid=valid[pool], ids=np.asarray(ids)[pool])
    manifest = dict(protocol="P1 frozen model (pre-registration Amendment 3)", dataset=args.dataset, arch=args.arch,
                    seed=args.seed, epochs=args.epochs, size=args.size, batch=args.batch, n_units=len(x),
                    n_train=len(tr), n_val=len(val), n_pool=len(pool), n_negatives_train=len(neg_tr),
                    train_ids=[str(ids[i]) for i in tr], val_ids=[str(ids[i]) for i in val],
                    pool_ids=[str(ids[i]) for i in pool], best_val_dice=best_val, curve=curve, pool_metrics=metrics,
                    small_cuts_from_train=cuts, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    seconds=round(time.time() - t0), env=env_info())
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"  pool metrics {json.dumps(metrics)}\n  small cuts {cuts}\n  saved {path} ({manifest['seconds']}s)", flush=True)


def run_task08(args):
    t0 = time.time()
    root = Path(args.data_root) / "msd-hepatic-vessel"
    images = sorted(p for p in root.rglob("imagesTr/*.nii.gz") if not p.name.startswith("._"))
    cases = [(p, p.parent.parent / "labelsTr" / p.name) for p in images]
    cases = [c for c in cases if c[1].exists()][: args.max_units]
    vols = [load_volume(i, lab, args.size) for i, lab in cases]
    ids = [c[0].name.split(".")[0] for c in cases]
    rng, tr, val, tr_all, pool = split(len(vols), args.seed)

    def slices(idx):
        xs, ys = [], []
        for v in idx:
            xv, yv = vols[v]
            pos = np.flatnonzero(yv.reshape(len(yv), -1).max(1) > 0)
            neg = np.setdiff1d(np.arange(len(yv)), pos)
            if len(neg):
                neg = rng.choice(neg, int(np.ceil(NEG_SLICE_FRACTION * len(neg))), replace=False)
            keep = np.concatenate([pos, neg]).astype(int)
            xs.append(xv[keep])
            ys.append(yv[keep])
        return np.concatenate(xs), np.concatenate(ys)

    x_tr, y_tr = slices(tr)
    x_val, y_val = slices(val)
    print(f"msd-hepatic-vessel {args.arch} seed {args.seed}: {len(vols)} volumes -> train {len(tr)} / val {len(val)} "
          f"/ pool {len(pool)} (patient-disjoint); {len(x_tr)} training slices; load {time.time() - t0:.0f}s", flush=True)
    model, mean, std, curve, best_val = train(args.arch, x_tr, y_tr, x_val, y_val, np.ones(y_val.shape, bool),
                                              np.empty((0,)), epochs=args.epochs, seed=args.seed, batch=args.batch)
    out_dir = args.out_root / "msd-hepatic-vessel" / f"{args.arch}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs, labs = [], []
    for v in pool:
        xv, yv = vols[v]
        p = predict(model, xv, mean, std, batch=32).astype(np.float16)
        np.savez_compressed(out_dir / f"{ids[v]}.npz", prob=p, labels=yv, valid=np.ones(yv.shape, bool))
        probs.append(p.astype(np.float32))
        labs.append(yv)
    metrics = lesion_metrics(probs, labs, [np.ones(y.shape, bool) for y in labs])
    cuts = small_cuts([vols[v][1] for v in tr_all])
    files = sorted(out_dir.glob("*.npz"))
    digest = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()
    manifest = dict(protocol="P1 frozen model, patient-disjoint (pre-registration Amendment 3)",
                    dataset="msd-hepatic-vessel", arch=args.arch, seed=args.seed, epochs=args.epochs, size=args.size,
                    n_units=len(vols), n_train=len(tr), n_val=len(val), n_pool=len(pool),
                    train_ids=[ids[i] for i in tr], val_ids=[ids[i] for i in val], pool_ids=[ids[i] for i in pool],
                    best_val_dice=best_val, curve=curve, pool_metrics=metrics, small_cuts_from_train=cuts,
                    sha256_of_file_hashes=digest, seconds=round(time.time() - t0), env=env_info())
    (out_dir.parent / f"{args.arch}_seed{args.seed}.json").write_text(json.dumps(manifest, indent=2))
    print(f"  pool metrics {json.dumps(metrics)}\n  small cuts {cuts}\n  saved {out_dir} ({manifest['seconds']}s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kvasir-seg", "fracatlas", "msd-hepatic-vessel"])
    ap.add_argument("--arch", choices=ARCHS, default="unet-r34")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data-root", default="/tmp/pums_data")
    ap.add_argument("--smoke", action="store_true", help="<=40 units, 1 epoch, throwaway output (Amendment 3)")
    args = ap.parse_args()
    args.max_units = 40 if args.smoke else None
    args.out_root = BACKEND_ROOT / "var" / "smoke" if args.smoke else OUT_ROOT
    task08 = args.dataset == "msd-hepatic-vessel"
    args.epochs = 1 if args.smoke else (args.epochs or (25 if task08 else 60))
    if task08 and args.smoke:
        args.max_units = 6
    args.size = args.size or (256 if task08 else 352)
    (run_task08 if task08 else run_2d)(args)


if __name__ == "__main__":
    main()
