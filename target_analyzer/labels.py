# Author: bd78
"""Training labels for the hole-detection network: file format, I/O and scoring.

One JSON file per photo in ``<base>/training/labels/`` (``<base>`` is the repository,
or the folder of the database for a packaged app)::

    {
      "image": "samples/1.jpg",          # relative to <base> (or absolute)
      "width": 1440, "height": 1920,     # size of the photo the coordinates refer to
      "holes": [{"x": 965.0, "y": 747.0, "r": 20.0}, ...],   # centres, original pixels;
                                         # r = hole radius in px, null = use r_default
      "r_default": 20.0,
      "ignore": [[x0, y0, x1, y1], ...], # boxes left out of training and scoring
                                         # (torn regions where single shots cannot be told apart)
      "exclude": "reason",               # optional: the whole photo is not used
      "group": "name",                   # optional: photos of the same target share a group
                                         # (kept on the same side of a train/validation split)
      "source": "..."                    # who / what made the label
    }

Pure Python (json), no Tk, no SQLite, no OpenCV.
"""
from __future__ import annotations

import json
from math import hypot
from pathlib import Path

LABELS_DIR = Path("training") / "labels"
IMAGES_DIR = Path("training") / "images"


def label_path(base: Path, image_name: str) -> Path:
    return Path(base) / LABELS_DIR / (Path(image_name).name + ".json")


def load_label(path: Path) -> dict:
    lab = json.loads(Path(path).read_text(encoding="utf-8"))
    lab.setdefault("holes", [])
    lab.setdefault("ignore", [])
    return lab


def dumps_label(label: dict) -> str:
    """JSON with one hole / one box per line, so label files stay readable and diff well."""
    lines = []
    for key, value in label.items():
        if key in ("holes", "ignore") and value:
            items = ",\n".join("  " + json.dumps(v, ensure_ascii=False) for v in value)
            lines.append(f' {json.dumps(key)}: [\n{items}\n ]')
        else:
            lines.append(f' {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}')
    return "{\n" + ",\n".join(lines) + "\n}\n"


def save_label(path: Path, label: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_label(label), encoding="utf-8")


def image_of(label: dict, base: Path) -> Path:
    p = Path(label["image"])
    return p if p.is_absolute() else Path(base) / p


def hole_radius(label: dict, hole: dict) -> float:
    r = hole.get("r")
    if r:
        return float(r)
    if label.get("r_default"):
        return float(label["r_default"])
    return max(label.get("width", 1000), label.get("height", 1000)) / 100.0


def in_boxes(x: float, y: float, boxes) -> bool:
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)


def match_points(pred: list[tuple[float, float]], label: dict, tol_factor: float = 1.0,
                 min_tol: float = 3.0) -> dict:
    """Score predicted centres against a label.

    A prediction matches an unmatched hole when it lies within max(min_tol,
    tol_factor * r) of its centre (greedy, closest pairs first). Predictions inside
    an ignore box are not counted. Returns tp, fp, fn and the matched pairs.
    """
    holes = label["holes"]
    preds = [(x, y) for x, y in pred if not in_boxes(x, y, label.get("ignore", []))]
    pairs = []
    for i, (x, y) in enumerate(preds):
        for j, h in enumerate(holes):
            d = hypot(x - h["x"], y - h["y"])
            if d <= max(min_tol, tol_factor * hole_radius(label, h)):
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_h, matched = set(), set(), []
    for d, i, j in pairs:
        if i in used_p or j in used_h:
            continue
        used_p.add(i)
        used_h.add(j)
        matched.append((i, j, d))
    tp = len(matched)
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(holes) - tp, "matched": matched}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 (1.0 when there is nothing to find and nothing found)."""
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
