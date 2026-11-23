# Author: bd78
"""Bullet-hole detector based on a small convolutional network (pattern recognition).

The network is trained by ``training/train_hole_model.py`` on the labelled photos in
``training/labels/`` and exported to ONNX; here it is run by OpenCV's DNN module, so
the application needs no deep-learning library. No Tk, no SQLite.

How it works: the photo is resized so its longest side is NORM_SIDE pixels (holes then
measure roughly 4-20 px in radius, the range seen in training), padded to a multiple of
32, and the network returns a *heat map* at half that resolution whose value is the
probability that a hole centre lies there. Local maxima above the threshold are the
holes. When the hole size is known (scale set in the UI), the resize factor is nudged
so the expected radius falls inside the trained range.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hole_net.onnx"   # metadata (threshold, training info) next to it as .json

NORM_SIDE = 1024        # longest side of the photo as seen by the network
STRIDE = 2              # heat-map pixels are STRIDE input pixels
MULTIPLE = 32           # the network downsamples 5 times: input sides must be multiples of 32
HINT_R_RANGE = (5.0, 18.0)  # radius (normalised px) a known hole size is brought into
MAX_INPUT_SIDE = 2560   # memory / time cap when a small hole forces an upscale
DEFAULT_THRESHOLD = 0.35


@dataclass
class ModelHit:
    x: float      # original image pixels
    y: float
    score: float  # peak value of the heat map, 0-1


def model_available(path: Path = MODEL_PATH) -> bool:
    return Path(path).is_file()


def work_scale(shape, hole_radius_px: float | None = None) -> float:
    """Resize factor original -> network input."""
    side = max(shape[:2])
    s = NORM_SIDE / side
    if hole_radius_px and hole_radius_px > 0:
        r = hole_radius_px * s
        lo, hi = HINT_R_RANGE
        if r < lo:
            s *= lo / r
        elif r > hi:
            s *= hi / r
    return min(s, MAX_INPUT_SIDE / side)


def prepare_input(image_bgr: np.ndarray, scale: float) -> tuple[np.ndarray, float, float, int, int]:
    """Resize and pad a BGR photo for the network.

    Returns (blob NCHW float32 BGR 0-255, sx, sy, w, h): sx, sy are the exact per-axis
    factors after rounding, (w, h) the size before padding.
    """
    H, W = image_bgr.shape[:2]
    w, h = max(1, round(W * scale)), max(1, round(H * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(image_bgr, (w, h), interpolation=interp)
    img = cv2.copyMakeBorder(img, 0, -h % MULTIPLE, 0, -w % MULTIPLE, cv2.BORDER_REPLICATE)
    blob = np.ascontiguousarray(img.astype(np.float32).transpose(2, 0, 1)[None])
    return blob, w / W, h / H, w, h


def decode_peaks(heat: np.ndarray, threshold: float) -> list[tuple[float, float, float]]:
    """Local maxima of a heat map above threshold, with sub-pixel refinement.

    Returns (x, y, score) in heat-map index coordinates (pixel j has its centre at j).
    """
    heat = heat.astype(np.float32)
    dil = cv2.dilate(heat, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((heat >= dil) & (heat > threshold))
    order = np.argsort(-heat[ys, xs])
    hh, ww = heat.shape
    out: list[tuple[float, float, float]] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if any(abs(px - x) <= 1.5 and abs(py - y) <= 1.5 for px, py, _ in out):
            continue  # plateau: two equal neighbours are both "maxima"
        y0, y1, x0, x1 = max(0, y - 1), min(hh, y + 2), max(0, x - 1), min(ww, x + 2)
        win = heat[y0:y1, x0:x1]
        wsum = float(win.sum())
        gy, gx = np.mgrid[y0:y1, x0:x1]
        out.append((float((gx * win).sum() / wsum), float((gy * win).sum() / wsum), float(heat[y, x])))
    return out


def heat_to_image(px: float, sx: float) -> float:
    """Heat-map index coordinate -> original image index coordinate (one axis)."""
    x_in = STRIDE * px + (STRIDE - 1) / 2.0   # centre of the STRIDE input pixels
    return (x_in + 0.5) / sx - 0.5            # undo cv2.resize (half-pixel convention)


class HoleNet:
    """The trained network, loaded once and reused for every photo."""

    def __init__(self, path: Path = MODEL_PATH):
        self.path = Path(path)
        self.net = cv2.dnn.readNet(str(self.path))
        meta_path = self.path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        self.threshold = float(self.meta.get("threshold", DEFAULT_THRESHOLD))

    def heatmap(self, image_bgr: np.ndarray, scale: float) -> tuple[np.ndarray, float, float]:
        """Heat map (probabilities, without the padding) and the per-axis factors."""
        blob, sx, sy, w, h = prepare_input(image_bgr, scale)
        self.net.setInput(blob)
        logits = np.asarray(self.net.forward(), np.float32).reshape(blob.shape[2] // STRIDE,
                                                                     blob.shape[3] // STRIDE)
        heat = 1.0 / (1.0 + np.exp(-logits))
        return heat[: -(-h // STRIDE), : -(-w // STRIDE)], sx, sy

    def detect(self, image_bgr: np.ndarray, hole_radius_px: float | None = None,
               threshold: float | None = None) -> list[ModelHit]:
        """Hole centres in original pixels, strongest first."""
        thr = self.threshold if threshold is None else threshold
        heat, sx, sy = self.heatmap(image_bgr, work_scale(image_bgr.shape, hole_radius_px))
        return [ModelHit(heat_to_image(x, sx), heat_to_image(y, sy), s)
                for x, y, s in decode_peaks(heat, thr)]

    def __call__(self, image_bgr, hole_radius_px=None):  # the interface detect_impacts expects
        return self.detect(image_bgr, hole_radius_px)
