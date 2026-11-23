# Author: bd78
r"""Train the bullet-hole network on training/labels/*.json and export it to ONNX.

Needs PyTorch, kept in its own environment so the application stays light:

    py -3.11 -m venv .venv-train
    .venv-train\Scripts\python -m pip install -r requirements-train.txt
    .venv-train\Scripts\python training\train_hole_model.py

Default run, two phases:
  1. validation: train on ~80 % of the labelled photos (split by group, so two crops of
     the same target never end up on both sides), score the held-out photos against
     their labels for a range of thresholds, and score the classic colour-rule detector
     on the same photos for comparison;
  2. final: train on every labelled photo, export target_analyzer/models/hole_net.onnx
     and hole_net.json (threshold chosen in phase 1, training summary), and check that
     OpenCV's DNN module gives the same heat map as PyTorch.

Options: --iters N (per phase, default 8000), --skip-val, --val-only, --seed S,
--device cpu|cuda, --base DIR (folder holding training/labels when the app's database
is not in the repository). The report of the run goes to training/report.json.

--cv K runs a K-fold cross-validation instead (photos split by group, every photo is
scored once by a model that never saw it, the threshold is chosen on the pooled
results); --update-meta then writes that threshold and those scores into the
existing hole_net.json. Results go to training/report_cv.json.

Model: a small U-Net (5 levels, ~0.6 M weights) that outputs a heat map at half the
input resolution; every hole centre is a Gaussian peak (CenterNet-style focal loss).
Input photos are resized so the longest side is NORM_SIDE (1024) px, then random
crops are taken with scale x0.55-1.8, rotation, flips, colour / light / blur / noise /
JPEG / shadow changes. Boxes listed under "ignore" in a label are left out of the loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from target_analyzer.detection import detect_impacts  # noqa: E402
from target_analyzer.hole_model import (MODEL_PATH, NORM_SIDE, STRIDE, decode_peaks,  # noqa: E402
                                        heat_to_image, prepare_input, work_scale)
from target_analyzer.images import load_image  # noqa: E402
from target_analyzer.labels import (LABELS_DIR, hole_radius, image_of, load_label,  # noqa: E402
                                    match_points, prf)

CROP = 320                 # training crop side (input pixels)
BATCH = 16
SCALE_RANGE = (0.55, 1.8)  # random zoom relative to the NORM_SIDE normalisation
HOLE_CROP_P = 0.65         # share of crops centred near a labelled hole
THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.80, 0.05)]


# ----------------------------------------------------------------------------- model
def conv_bn(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class Block(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(conv_bn(cin, cout), conv_bn(cout, cout))


class HoleNetTorch(nn.Module):
    """U-Net: input BGR 0-255 (N, 3, H, W), H and W multiples of 32; output logits (N, 1, H/2, W/2)."""

    def __init__(self, w=(16, 32, 64, 96, 128)):
        super().__init__()
        self.stem = conv_bn(3, w[0], stride=2)
        self.e1 = Block(w[0], w[0])
        self.e2 = Block(w[0], w[1])
        self.e3 = Block(w[1], w[2])
        self.e4 = Block(w[2], w[3])
        self.e5 = Block(w[3], w[4])
        self.d4 = Block(w[4] + w[3], w[3])
        self.d3 = Block(w[3] + w[2], w[2])
        self.d2 = Block(w[2] + w[1], w[1])
        self.d1 = Block(w[1] + w[0], w[0])
        self.head = nn.Sequential(conv_bn(w[0], w[0]), nn.Conv2d(w[0], 1, 1))
        nn.init.constant_(self.head[-1].bias, -2.19)  # prior p = 0.1 (CenterNet)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2.0, mode="nearest")

    def forward(self, x):
        x = (x - 128.0) / 64.0
        e1 = self.e1(self.stem(x))
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        e5 = self.e5(F.max_pool2d(e4, 2))
        d = self.d4(torch.cat([self._up(e5), e4], 1))
        d = self.d3(torch.cat([self._up(d), e3], 1))
        d = self.d2(torch.cat([self._up(d), e2], 1))
        d = self.d1(torch.cat([self._up(d), e1], 1))
        return self.head(d)


def focal_loss(logits: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CenterNet focal loss; peaks (gt == 1) are positives, the Gaussian tails damp the negatives."""
    logits = logits.float()
    pos = (gt >= 0.999).float()
    log_p, log_1p = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = torch.sigmoid(logits)
    pos_loss = log_p * (1 - p) ** 2 * pos * weight
    neg_loss = log_1p * p ** 2 * (1 - gt) ** 4 * (1 - pos) * weight
    return -(pos_loss.sum() + neg_loss.sum()) / (pos * weight).sum().clamp(min=1.0)


# ------------------------------------------------------------------------------ data
class Photo:
    """A labelled photo with a small pyramid of resized copies for fast crop warping."""

    def __init__(self, label_file: Path, base: Path = ROOT):
        self.label = load_label(label_file)
        self.name = Path(self.label["image"]).name
        self.group = self.label.get("group", self.name)
        img = load_image(image_of(self.label, base))
        if img is None:
            raise RuntimeError(f"cannot read {self.label['image']}")
        if (img.shape[1], img.shape[0]) != (self.label.get("width", img.shape[1]),
                                            self.label.get("height", img.shape[0])):
            raise RuntimeError(f"{self.name}: image size differs from the label")
        self.image = img
        self.s0 = NORM_SIDE / max(img.shape[:2])
        self.levels = []  # (factor, image), factors 0.5 s0, s0 and 2 s0 when not an upscale
        for f in (0.5 * self.s0, self.s0, 2 * self.s0):
            if f > 1.0 and f != self.s0:
                continue
            w, h = max(8, round(img.shape[1] * f)), max(8, round(img.shape[0] * f))
            interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
            self.levels.append((f, cv2.resize(img, (w, h), interpolation=interp)))
        holes = self.label["holes"]
        self.holes = np.array([[h["x"], h["y"], hole_radius(self.label, h)] for h in holes],
                              np.float32).reshape(-1, 3)
        self.ignore = [list(map(float, b)) for b in self.label.get("ignore", [])]


def _gauss_target(holes_crop: np.ndarray, out: int) -> np.ndarray:
    """Heat-map target (out x out) for hole centres / radii given in crop input pixels."""
    hm = np.zeros((out, out), np.float32)
    yy, xx = np.mgrid[0:out, 0:out].astype(np.float32)
    for x, y, r in holes_crop:
        cx, cy = (x - (STRIDE - 1) / 2) / STRIDE, (y - (STRIDE - 1) / 2) / STRIDE
        if not (-3 <= cx < out + 3 and -3 <= cy < out + 3):
            continue
        sigma = float(np.clip(0.3 * r, 1.5, 6.0)) / STRIDE
        x0, x1 = max(0, int(cx - 4 * sigma)), min(out, int(cx + 4 * sigma) + 2)
        y0, y1 = max(0, int(cy - 4 * sigma)), min(out, int(cy + 4 * sigma) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        g = np.exp(-((xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2) / (2 * sigma * sigma))
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
        px, py = int(round(cx)), int(round(cy))
        if 0 <= px < out and 0 <= py < out:
            hm[py, px] = 1.0
    return hm


def _photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:  # colour
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-15, 15)) % 180
        hsv[..., 1] *= rng.uniform(0.65, 1.35)
        hsv[..., 2] *= rng.uniform(0.7, 1.25)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.15:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if rng.random() < 0.3:  # gamma / contrast
        g = rng.uniform(0.7, 1.45)
        img = (255.0 * (img / 255.0) ** g).astype(np.uint8)
    if rng.random() < 0.3:  # hard or soft shadow over part of the crop
        h, w = img.shape[:2]
        a = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = (xx - rng.uniform(0, w)) * math.cos(a) + (yy - rng.uniform(0, h)) * math.sin(a)
        soft = rng.uniform(1.0, 40.0)
        k = 1.0 - (1.0 - rng.uniform(0.45, 0.85)) / (1.0 + np.exp(np.clip(-d / soft, -50, 50)))
        img = np.clip(img * k[..., None], 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.6))
    if rng.random() < 0.25:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 9), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 90)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def make_crop(photo: Photo, rng: random.Random):
    """One augmented training crop: (image CROP x CROP BGR, target, weight) at output resolution."""
    f = math.exp(rng.uniform(*map(math.log, SCALE_RANGE)))
    d = photo.s0 * f                      # original px -> crop px
    H, W = photo.image.shape[:2]
    if len(photo.holes) and rng.random() < HOLE_CROP_P:
        hx, hy, _ = photo.holes[rng.randrange(len(photo.holes))]
        span = 0.35 * CROP / d
        cx, cy = hx + rng.uniform(-span, span), hy + rng.uniform(-span, span)
    else:
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    # pyramid level: smallest factor >= d (the warp then only shrinks a little)
    above = [lv for lv in photo.levels if lv[0] >= d]
    b, src = min(above, key=lambda lv: lv[0]) if above else max(photo.levels, key=lambda lv: lv[0])
    theta = rng.uniform(0, 2 * math.pi)
    flip = rng.random() < 0.5
    k = d / b
    c, s = math.cos(theta) * k, math.sin(theta) * k
    A = np.array([[c, -s], [s, c]], np.float64)
    if flip:
        A = A @ np.array([[-1, 0], [0, 1]], np.float64)
    pc = np.array([(cx + 0.5) * b - 0.5, (cy + 0.5) * b - 0.5])
    t = np.array([CROP / 2 - 0.5, CROP / 2 - 0.5]) - A @ pc
    M = np.hstack([A, t[:, None]])
    border = tuple(int(v) for v in rng.choice([(0, 0, 0), (255, 255, 255)] +
                                              [tuple(rng.randrange(256) for _ in range(3))]))
    img = cv2.warpAffine(src, M, (CROP, CROP), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    img = _photometric(img, rng)

    def to_crop(pts):  # original px -> crop px
        lv = (np.asarray(pts, np.float64) + 0.5) * b - 0.5
        return lv @ A.T + t

    holes = np.zeros((0, 3), np.float32)
    if len(photo.holes):
        xy = to_crop(photo.holes[:, :2])
        holes = np.column_stack([xy, photo.holes[:, 2] * d]).astype(np.float32)
    out = CROP // STRIDE
    target = _gauss_target(holes, out)
    weight = np.ones((CROP, CROP), np.uint8)
    for x0, y0, x1, y1 in photo.ignore:
        poly = to_crop([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        cv2.fillConvexPoly(weight, np.round(poly).astype(np.int32), 0)
    weight = cv2.resize(weight, (out, out), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return img, target, weight


class Batches:
    """Background producer of training batches (OpenCV releases the GIL, threads are enough)."""

    def __init__(self, photos: list[Photo], seed: int, workers: int = 8):
        self.photos = photos
        self.weights = [math.sqrt(len(p.holes) + 4.0) for p in photos]
        self.pool = ThreadPoolExecutor(workers)
        self.seed = seed
        self.count = 0
        self.pending = [self._submit() for _ in range(4)]

    def _one(self, s: int):
        rng = random.Random(s)
        photo = rng.choices(self.photos, self.weights)[0]
        return make_crop(photo, rng)

    def _batch(self, s: int):
        items = [self._one(s * 1000 + i) for i in range(BATCH)]
        imgs = np.stack([it[0] for it in items]).transpose(0, 3, 1, 2).astype(np.float32)
        return imgs, np.stack([it[1] for it in items])[:, None], np.stack([it[2] for it in items])[:, None]

    def _submit(self):
        self.count += 1
        return self.pool.submit(self._batch, self.seed * 100003 + self.count)

    def next(self):
        fut = self.pending.pop(0)
        self.pending.append(self._submit())
        return fut.result()


# ------------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict(model: nn.Module, photo: Photo, device: str) -> np.ndarray:
    """Heat map of a whole photo at the default (no hint) scale, as the app computes it."""
    model.eval()
    blob, sx, sy, w, h = prepare_input(photo.image, work_scale(photo.image.shape))
    logits = model(torch.from_numpy(blob).to(device)).float().cpu().numpy()[0, 0]
    heat = 1.0 / (1.0 + np.exp(-logits))
    return heat[: -(-h // STRIDE), : -(-w // STRIDE)], sx, sy


def score_heat(heat, sx, sy, photo: Photo, thresholds) -> dict[float, dict]:
    res = {}
    peaks = decode_peaks(heat, min(thresholds))
    for t in thresholds:
        pts = [(heat_to_image(x, sx), heat_to_image(y, sy)) for x, y, s in peaks if s > t]
        m = match_points(pts, photo.label)
        res[t] = {k: m[k] for k in ("tp", "fp", "fn")}
    return res


def evaluate(model, photos, device, thresholds=THRESHOLDS) -> dict:
    per_photo = {p.name: score_heat(*predict(model, p, device), p, thresholds) for p in photos}
    totals = {}
    for t in thresholds:
        tp = sum(per_photo[n][t]["tp"] for n in per_photo)
        fp = sum(per_photo[n][t]["fp"] for n in per_photo)
        fn = sum(per_photo[n][t]["fn"] for n in per_photo)
        totals[t] = (tp, fp, fn, *prf(tp, fp, fn))
    best = max(thresholds, key=lambda t: totals[t][5])
    return {"per_photo": per_photo, "totals": totals, "best": best}


def classic_scores(photos) -> dict:
    out = {}
    for p in photos:
        imps = detect_impacts(p.image).impacts
        m = match_points([(i.x, i.y) for i in imps], p.label)
        out[p.name] = {k: m[k] for k in ("tp", "fp", "fn")}
    return out


def summary_line(tp, fp, fn) -> str:
    pr, rc, f1 = prf(tp, fp, fn)
    return f"TP {tp:4d}  FP {fp:4d}  FN {fn:4d}  precision {pr:.2f}  recall {rc:.2f}  F1 {f1:.2f}"


# -------------------------------------------------------------------------- training
def train(photos: list[Photo], iters: int, device: str, seed: int, val: list[Photo] | None = None,
          log_every: int = 250) -> nn.Module:
    torch.manual_seed(seed)
    model = HoleNetTorch().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warm = min(300, iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, iters - warm))))
    batches = Batches(photos, seed)
    amp = device == "cuda"
    t0, run = time.time(), 0.0
    for it in range(1, iters + 1):
        model.train()
        imgs, tgt, wgt = (torch.from_numpy(a).to(device, non_blocking=True) for a in batches.next())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(imgs)
        loss = focal_loss(logits, tgt, wgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % log_every == 0 or it == iters:
            msg = f"  iter {it:5d}/{iters}  loss {run:.3f}  lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:5.0f}s"
            if val and (it % (log_every * 4) == 0 or it == iters):
                ev = evaluate(model, val, device)
                tp, fp, fn, pr, rc, f1 = ev["totals"][ev["best"]]
                msg += f"  | val F1 {f1:.2f} (P {pr:.2f} R {rc:.2f} @ {ev['best']})"
            print(msg, flush=True)
    batches.pool.shutdown(wait=False, cancel_futures=True)
    return model.eval()


def split(photos: list[Photo], seed: int, frac: float = 0.2):
    groups: dict[str, list[Photo]] = {}
    for p in photos:
        groups.setdefault(p.group, []).append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = sum(len(p.holes) for p in photos)
    val, n = [], 0
    for k in keys:
        if n >= frac * total:
            break
        if any(len(p.holes) == 0 for p in groups[k]) and len(groups[k]) > 3:
            continue  # keep the large negative-only group (phone card photos) in training
        val += groups[k]
        n += sum(len(p.holes) for p in groups[k])
    names = {p.name for p in val}
    return [p for p in photos if p.name not in names], val


def model_out(path: Path) -> Path:
    """The --out model file, checked: an absolute path ending in .onnx."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".onnx":
        raise SystemExit(f"--out must be an .onnx file, got {path}")
    return path


def sidecar_of(model_path: Path) -> Path:
    """The metadata file written next to a model (hole_net.onnx -> hole_net.json)."""
    return model_out(model_path).with_suffix(".json")


def export(model: nn.Module, path: Path, meta: dict, check_photo: Photo, device: str) -> None:
    path = model_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.float().eval().cpu()
    dummy = torch.zeros(1, 3, 320, 320)
    kwargs = dict(input_names=["image"], output_names=["heat"], opset_version=17,
                  dynamic_axes={"image": {2: "h", 3: "w"}, "heat": {2: "h2", 3: "w2"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the dynamo switch
        torch.onnx.export(model, dummy, str(path), **kwargs)
    sidecar_of(path).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    # the app runs the network with OpenCV: check it gives the same heat map
    from target_analyzer.hole_model import HoleNet
    net = HoleNet(path)
    heat_cv, _, _ = net.heatmap(check_photo.image, work_scale(check_photo.image.shape))
    heat_pt, _, _ = predict(model, check_photo, "cpu")
    diff = float(np.abs(heat_cv - heat_pt).max())
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB); OpenCV vs PyTorch max |diff| = {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError("OpenCV DNN output differs from PyTorch")


def folds_by_group(photos: list[Photo], k: int) -> list[list[Photo]]:
    """K folds of whole groups, balanced by hole count (largest groups first)."""
    groups: dict[str, list[Photo]] = {}
    for p in photos:
        groups.setdefault(p.group, []).append(p)
    folds: list[list[Photo]] = [[] for _ in range(k)]
    load = [0] * k
    for g in sorted(groups, key=lambda g: (-sum(len(p.holes) + 2 for p in groups[g]), g)):
        i = min(range(k), key=lambda j: (load[j], j))
        folds[i] += groups[g]
        load[i] += sum(len(p.holes) + 2 for p in groups[g])
    return folds


def updated_meta(old: object, threshold: float, validation: dict) -> dict:
    """New model metadata with the cross-validation threshold and scores. It is rebuilt from
    known, typed fields only, so nothing read from the old file is written back as is."""
    if not isinstance(old, dict) or "norm_side" not in old:
        raise SystemExit("not a hole-model metadata file; not updated")
    meta = {"threshold": float(threshold), "norm_side": int(old["norm_side"]),
            "stride": int(old.get("stride", STRIDE)),
            "trained": date.fromisoformat(str(old.get("trained", date.today().isoformat()))).isoformat()}
    meta.update({key: int(old[key]) for key in ("photos", "holes", "iters") if key in old})
    meta["validation"] = validation
    return meta


def cross_validate(photos: list[Photo], k: int, iters: int, device: str, seed: int, meta_path: Path | None):
    folds = folds_by_group(photos, k)
    per_photo, fold_of = {}, {}
    for i, held in enumerate(folds):
        names = {p.name for p in held}
        train_set = [p for p in photos if p.name not in names]
        print(f"\n== fold {i + 1}/{k}: train {len(train_set)} photos, test {len(held)} photos "
              f"({sum(len(p.holes) for p in held)} holes)")
        model = train(train_set, iters, device, seed + 10 * i)
        per_photo.update(evaluate(model, held, device)["per_photo"])
        fold_of.update({p.name: i + 1 for p in held})
    totals = {}
    for t in THRESHOLDS:
        tp, fp, fn = (sum(per_photo[n][t][key] for n in per_photo) for key in ("tp", "fp", "fn"))
        totals[t] = (tp, fp, fn, *prf(tp, fp, fn))
    best = max(THRESHOLDS, key=lambda t: totals[t][5])
    classic = classic_scores(photos)
    print(f"\ncross-validation ({k} folds), network @ threshold {best}, every photo unseen by its model:")
    rows = []
    for p in sorted(photos, key=lambda p: p.name.lower()):
        m, c = per_photo[p.name][best], classic[p.name]
        rows.append({"photo": p.name, "group": p.group, "fold": fold_of[p.name], "holes": len(p.holes),
                     "network": m, "classic": c})
        print(f"  {p.name[:58]:58s} holes {len(p.holes):3d} | net TP {m['tp']:3d} FP {m['fp']:3d} FN {m['fn']:3d}"
              f" | classic TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d}")
    tot_n = [sum(r["network"][key] for r in rows) for key in ("tp", "fp", "fn")]
    tot_c = [sum(r["classic"][key] for r in rows) for key in ("tp", "fp", "fn")]
    print("  network : " + summary_line(*tot_n))
    print("  classic : " + summary_line(*tot_c))
    names = ("tp", "fp", "fn", "precision", "recall", "f1")
    report = {"folds": k, "iters": iters, "seed": seed, "threshold": best,
              "network": dict(zip(names, (*tot_n, *prf(*tot_n)))),
              "classic": dict(zip(names, (*tot_c, *prf(*tot_c)))),
              "by_threshold": {str(t): dict(zip(names, v)) for t, v in totals.items()},
              "photos": rows}
    (ROOT / "training" / "report_cv.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if meta_path is not None:
        mp = sidecar_of(meta_path)
        meta = updated_meta(json.loads(mp.read_text(encoding="utf-8")), best,
                            dict(report["network"], method=f"{k}-fold cross-validation, {len(photos)} photos"))
        mp.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        print(f"updated {mp}: threshold {best}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--cv", type=int, default=0, metavar="K")
    ap.add_argument("--base", type=Path, default=ROOT,
                    help="folder holding training/labels (the database folder of the app); default: the repository")
    ap.add_argument("--update-meta", action="store_true")
    args = ap.parse_args()
    args.out = model_out(args.out)

    files = sorted((args.base / LABELS_DIR).glob("*.json"))
    photos = []
    for f in files:
        lab = load_label(f)
        if lab.get("exclude"):
            print(f"skip {Path(lab['image']).name}: {lab['exclude']}")
            continue
        try:
            img_path = image_of(lab, args.base)
        except ValueError as exc:
            print(f"skip {f.name}: {exc}")
            continue
        if not img_path.is_file():
            print(f"skip {Path(lab['image']).name}: photo not found")
            continue
        photos.append(Photo(f, args.base))
    n_holes = sum(len(p.holes) for p in photos)
    print(f"{len(photos)} labelled photos, {n_holes} holes; device {args.device}")
    report = {"photos": len(photos), "holes": n_holes, "iters": args.iters, "seed": args.seed}
    threshold = None
    if args.cv:
        cross_validate(photos, args.cv, args.iters, args.device, args.seed, args.out if args.update_meta else None)
        return

    if not args.skip_val:
        tr, va = split(photos, args.seed)
        print(f"\n== phase 1: validation split: train {len(tr)} photos, validate {len(va)} photos "
              f"({sum(len(p.holes) for p in va)} holes):\n   " + "\n   ".join(p.name for p in va))
        model = train(tr, args.iters, args.device, args.seed, val=va)
        ev = evaluate(model, va, args.device)
        threshold = ev["best"]
        classic = classic_scores(va)
        print(f"\nvalidation, network @ threshold {threshold}:")
        rows = []
        for p in va:
            m, c = ev["per_photo"][p.name][threshold], classic[p.name]
            rows.append({"photo": p.name, "holes": len(p.holes), "network": m, "classic": c})
            print(f"  {p.name[:58]:58s} holes {len(p.holes):3d} | net TP {m['tp']:3d} FP {m['fp']:3d} FN {m['fn']:3d}"
                  f" | classic TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d}")
        tot_n = [sum(r["network"][k] for r in rows) for k in ("tp", "fp", "fn")]
        tot_c = [sum(r["classic"][k] for r in rows) for k in ("tp", "fp", "fn")]
        print("  network : " + summary_line(*tot_n))
        print("  classic : " + summary_line(*tot_c))
        report["validation"] = {
            "photos": rows, "threshold": threshold,
            "network": dict(zip(("tp", "fp", "fn", "precision", "recall", "f1"), (*tot_n, *prf(*tot_n)))),
            "classic": dict(zip(("tp", "fp", "fn", "precision", "recall", "f1"), (*tot_c, *prf(*tot_c)))),
            "by_threshold": {str(t): dict(zip(("tp", "fp", "fn", "precision", "recall", "f1"), v))
                             for t, v in ev["totals"].items()},
        }
    if not args.val_only:
        print(f"\n== phase 2: final model on all {len(photos)} photos")
        model = train(photos, args.iters, args.device, args.seed + 1)
        ev = evaluate(model, photos, args.device)
        if threshold is None:
            threshold = 0.35
        tp, fp, fn, pr, rc, f1 = ev["totals"][threshold]
        print(f"training set @ {threshold}: " + summary_line(tp, fp, fn) + "  (optimistic: seen in training)")
        report["final_training_set"] = {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": f1}
        meta = {"threshold": threshold, "norm_side": NORM_SIDE, "stride": STRIDE,
                "trained": time.strftime("%Y-%m-%d"), "photos": len(photos), "holes": n_holes,
                "iters": args.iters, "validation": report.get("validation", {}).get("network")}
        check = max(photos, key=lambda p: len(p.holes))
        export(model, args.out, meta, check, args.device)
    (ROOT / "training" / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
