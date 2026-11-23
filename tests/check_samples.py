# Author: bd78
r"""Run detection on the sample targets and write annotated images for review.

Usage: .venv\Scripts\python tests\check_samples.py [out_dir] [--net]

Prints, per sample, the number of blobs, impacts, detected target circles and
groups, then the group sizes (scale assumed from the largest circle = 200 mm, so
the cm values are only meaningful for the solid-disc targets). When the sample has a
label in training/labels/, the impacts are also scored against it (TP / FP / FN, a
hit within one hole radius of a labelled centre; boxes marked "ignore" are not
counted), with a total at the end. The annotated image shows the blobs (blue), the
impacts (red, numbered) and each target circle with its group label and shot count
(green).

--net also scores the trained network (target_analyzer/models/hole_net.onnx) and
writes <name>_net.jpg (network hits in magenta). Most samples were used to train it,
so its scores here are optimistic; training/report.json holds the scores on photos
it had not seen.
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from target_analyzer.detection import detect_impacts  # noqa: E402
from target_analyzer.images import load_image  # noqa: E402
from target_analyzer.labels import LABELS_DIR, label_path, load_label, match_points, prf  # noqa: E402
from target_analyzer.stats import group_stats  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
use_net = "--net" in sys.argv
out = Path(args[0]) if args else Path(__file__).parent / "out"
out.mkdir(parents=True, exist_ok=True)
net = None
if use_net:
    from target_analyzer.hole_model import HoleNet
    net = HoleNet()
EXTS = (".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".avif", ".heic", ".tif", ".tiff")
totals = {"rules": [0, 0, 0], "net": [0, 0, 0]}
for p in sorted(q for q in (ROOT / "samples").glob("*") if q.suffix.lower() in EXTS):
    img = load_image(p)
    if img is None:
        print(f"{p.name}: cannot read")
        continue
    res = detect_impacts(img)
    px_per_mm = 2 * res.disc.r / 200.0 if res.disc else 1.0  # 200 mm black disc
    st = group_stats([(i.x, i.y) for i in res.impacts], px_per_mm)
    counts = [sum(1 for i in res.impacts if i.group == g) for g in range(len(res.groups))]
    line = (f"{p.name}: disc={'yes' if res.disc else 'no'} targets={len(res.targets)} groups={len(res.groups)} "
            f"blobs={len(res.blobs)} impacts={len(res.impacts)} single_hole_area={res.single_hole_area:.0f}px2 | "
            + (f"W={st.width_mm/10:.1f}cm H={st.height_mm/10:.1f}cm D={st.diameter_mm/10:.1f}cm" if st else "no impact")
            + f" | shots/group={counts}")
    lp = label_path(ROOT, p.name)
    label = load_label(lp) if lp.is_file() else None
    if label is not None and label.get("exclude"):
        label = None
    hits = net.detect(img) if net else []
    if label is not None:
        m = match_points([(i.x, i.y) for i in res.impacts], label)
        line += f" | rules TP {m['tp']} FP {m['fp']} FN {m['fn']}"
        totals["rules"] = [a + m[k] for a, k in zip(totals["rules"], ("tp", "fp", "fn"))]
        if net:
            m = match_points([(h.x, h.y) for h in hits], label)
            line += f" | net TP {m['tp']} FP {m['fp']} FN {m['fn']}"
            totals["net"] = [a + m[k] for a, k in zip(totals["net"], ("tp", "fp", "fn"))]
    print(line)
    vis = img.copy()
    for g, grp in zip(res.groups, counts):
        if g.cx is not None:
            cv2.circle(vis, (int(g.cx), int(g.cy)), int(g.r), (0, 255, 0), 2)
            cv2.putText(vis, f"{g.label}:{grp}", (int(g.cx - g.r), int(g.cy - g.r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.drawContours(vis, res.blobs, -1, (255, 0, 0), 2)
    for k, imp in enumerate(res.impacts, 1):
        cv2.circle(vis, (int(imp.x), int(imp.y)), 14, (0, 0, 255), 3)
        cv2.putText(vis, str(k), (int(imp.x) + 16, int(imp.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(out / f"{p.stem}_detected.jpg"), vis)
    if net:
        vis = img.copy()
        rad = max(6, max(img.shape[:2]) // 120)
        for k, h in enumerate(hits, 1):
            cv2.circle(vis, (int(h.x), int(h.y)), rad, (255, 0, 255), 2)
            cv2.putText(vis, f"{k}", (int(h.x) + rad, int(h.y) - rad), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 + rad / 40, (255, 0, 255), 2)
        cv2.imwrite(str(out / f"{p.stem}_net.jpg"), vis)
for key, (tp, fp, fn) in totals.items():
    if tp + fp + fn:
        pr, rc, f1 = prf(tp, fp, fn)
        print(f"total vs {LABELS_DIR.as_posix()} ({key}): TP {tp} FP {fp} FN {fn} "
              f"precision {pr:.2f} recall {rc:.2f} F1 {f1:.2f}")
print("annotated images written to", out)
