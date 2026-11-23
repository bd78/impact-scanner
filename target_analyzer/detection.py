# Author: bd78
"""Bullet-hole detection on printed paper targets (OpenCV).

Two target styles are handled by one pipeline:

* Solid black disc (25 m precision pistol style): holes are torn paper that
  shows bright against the black. Inside the disc, "bright" pixels are holes.
* Ring targets (10 m air rifle / pistol cards): white centre, thin rings, wide
  black bands. Holes expose the backing board, which is grey or wood-coloured,
  i.e. neither paper-white nor ink-black, and often saturated in colour.

Pipeline
--------
1. Estimate the paper brightness globally (95th percentile) and locally (a
   grey-level closing that fills small dark objects but follows shadows), so
   uneven lighting does not matter much.
2. Ink mask = pixels far darker than paper. If the largest ink blob is a
   solid, circular disc, remember it (solid-disc style).
3. Candidate hole pixels:
   - mid-tone pixels: darker than the local paper but not ink (grey backing,
     holes on white paper), excluding the inside of a solid disc, where the
     print itself is grey under strong light;
   - saturated pixels (brown wood backing, anywhere);
   - inside a solid disc: pixels brighter than ~65 % of paper (torn paper).
4. Morphological opening removes the thin ring lines, digit strokes and the
   anti-aliased edges of the print; what remains are hole blobs.
5. Blobs much larger than a single hole are torn regions holding several
   shots; the shot count is estimated from the area (or distributed from the
   known number of shots fired) and the blob is split into that many centres
   with k-means.
6. Every circular black shape (the band of a ring target) is a *target*; when
   a photo shows several of them (a card of 10-25 diagrams) each impact is
   assigned to the nearest target and the impacts form one *group* per target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Disc:
    cx: float
    cy: float
    r: float
    solid: bool = True  # True: filled black disc; False: outer ring of a ring target


@dataclass
class Impact:
    x: float          # pixel coordinates in the original image
    y: float
    count: int = 1    # shots represented by this point (always 1 after splitting)
    area: float = 0.0
    manual: bool = False
    blob_id: int = -1  # id of the torn blob the point was split from (-1: isolated)
    group: int = 0     # index into the group list (one group per target diagram)


@dataclass
class Group:
    """A set of shots measured together: one diagram of a card, or one series."""
    label: str
    cx: float | None = None  # centre and radius (px) of the printed target the
    cy: float | None = None  # group belongs to, when it was found in the photo
    r: float | None = None


@dataclass
class DetectionResult:
    disc: Disc | None
    impacts: list[Impact]
    blobs: list[np.ndarray] = field(default_factory=list)  # contours, for drawing
    single_hole_area: float = 0.0
    debug: dict = field(default_factory=dict)  # intermediate masks (work resolution)
    targets: list[Disc] = field(default_factory=list)  # every circular black shape, largest first
    groups: list[Group] = field(default_factory=list)  # impacts[i].group indexes this list


WORK_SIZE = 1920  # longest image side used for detection; parameters are tuned for it

# --- backing colour picked by the user -----------------------------------------
# A hole shows whatever is behind the paper (cardboard, wood, foam, a black board).
# The user clicks inside a few holes; each click samples a DISC of pixels, so the
# texture of the backing (wood grain, cardboard fibres, JPEG noise) is captured as a
# colour distribution, not a single pixel value. A pixel matches when its Mahalanobis
# distance to one of the samples is below the tolerance (in standard deviations).
# Minimum spread (std-dev, OpenCV 8-bit Lab units) added to every sample. Lightness gets
# much more than the colour axes: the light changes across a photo (shadow, lit side of a
# hole) while the hue of the backing does not; it also keeps the covariance invertible for
# a flat colour (vector drawing). Tuned on 9 labelled photos where the colour rules fail:
# (L, a, b) = (4, 4, 4) -> F1 0.33, (16, 5, 5) -> 0.39, (20-30, 5-6, 5-6) -> 0.40-0.41,
# (16, 8, 8) -> 0.31 (with "use only this colour").
BACKING_MIN_STD = (20.0, 5.0, 5.0)
BACKING_KEEP = 0.8        # share of the disc pixels kept (the rest is torn edge / print)
BACKING_PRINT_SHARE = 0.5  # a sample matching more of a solid print area than this is the print
BACKING_PRINT_MIN = 0.005  # smallest print area (share of the image) checked for that, any shape;
                           # 13 photos, 'only this colour': F1 0.52 (disc only), 0.51 (2 %), 0.55 (0.5 %)
BACKING_DEFAULT_TOL = 3.0


@dataclass
class BackingSample:
    """Colour distribution (CIE Lab, OpenCV 8-bit scale) of a disc sampled inside a hole."""
    mean: tuple[float, float, float]
    cov: tuple[tuple[float, float, float], ...]  # 3 x 3, regularised
    n: int = 0                                    # pixels used

    @property
    def rgb_hex(self) -> str:
        lab = np.array([[self.mean]], np.float32).clip(0, 255).astype(np.uint8)
        b, g, r = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)[0, 0]
        return f"#{r:02x}{g:02x}{b:02x}"

    def to_dict(self) -> dict:
        return {"mean": [round(v, 3) for v in self.mean],
                "cov": [[round(v, 4) for v in row] for row in self.cov], "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> "BackingSample | None":
        try:
            mean = tuple(float(v) for v in d["mean"])
            cov = tuple(tuple(float(v) for v in row) for row in d["cov"])
            if len(mean) != 3 or len(cov) != 3 or any(len(r) != 3 for r in cov):
                return None
            np.linalg.inv(np.array(cov))  # must be invertible
            return cls(mean, cov, int(d.get("n", 0)))
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            return None


def _work_scale(shape) -> float:
    return min(1.0, WORK_SIZE / max(shape[:2]))


def sample_backing(image_bgr: np.ndarray, x: float, y: float, radius: float) -> BackingSample | None:
    """Colour distribution of the disc of `radius` px centred on (x, y) (original pixels).

    The pixels are processed exactly as in detect_impacts (work resolution, 5 px
    median) so the statistics match what the detector sees. The estimate is
    robust: the BACKING_KEEP closest pixels to the median are kept, so a disc
    that overlaps the torn edge or the print still describes the backing.
    """
    H, W = image_bgr.shape[:2]
    s = _work_scale(image_bgr.shape)
    m = int(np.ceil(radius + 12 / s))  # margin for the median filter
    x0, y0, x1, y1 = max(0, int(x - m)), max(0, int(y - m)), min(W, int(x + m) + 1), min(H, int(y + m) + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = image_bgr[y0:y1, x0:x1]
    if s < 1.0:
        crop = cv2.resize(crop, (max(3, round(crop.shape[1] * s)), max(3, round(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    crop = cv2.medianBlur(crop, 5)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float64)
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    cx, cy = (x - x0) * s, (y - y0) * s
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2 <= max(1.0, radius * s) ** 2).ravel()
    px = lab[inside]
    if len(px) < 3:
        return None
    centre = np.median(px, axis=0)
    for _ in range(2):  # trimmed estimate: drop the pixels farthest from the centre
        d = np.linalg.norm(px - centre, axis=1)
        keep = px[d <= np.quantile(d, BACKING_KEEP)] if len(px) > 10 else px
        centre = keep.mean(axis=0)
    cov = np.cov(keep.T) if len(keep) > 3 else np.zeros((3, 3))
    cov = cov + np.diag(np.square(BACKING_MIN_STD))
    return BackingSample(tuple(float(v) for v in centre), tuple(tuple(float(v) for v in r) for r in cov),
                         int(len(keep)))


def backing_match(lab: np.ndarray, samples: list[BackingSample], tol: float = BACKING_DEFAULT_TOL) -> np.ndarray:
    """Boolean mask of the pixels of a Lab image (H x W x 3) within `tol` std-devs of any sample."""
    out = np.zeros(lab.shape[:2], bool)
    if not samples:
        return out
    lab = lab.astype(np.float32)
    for smp in samples:
        inv = np.linalg.inv(np.array(smp.cov, np.float64)).astype(np.float32)
        diff = lab - np.array(smp.mean, np.float32)
        d2 = np.einsum("...i,ij,...j->...", diff, inv, diff, optimize=True)
        out |= d2 < tol * tol
    return out


def backing_match_image(image_bgr: np.ndarray, samples: list[BackingSample],
                        tol: float = BACKING_DEFAULT_TOL) -> np.ndarray:
    """backing_match on a whole photo, at work resolution (for a preview overlay)."""
    s = _work_scale(image_bgr.shape)
    if s < 1.0:
        H, W = image_bgr.shape[:2]
        image_bgr = cv2.resize(image_bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.medianBlur(image_bgr, 5), cv2.COLOR_BGR2Lab)
    return backing_match(lab, samples, tol)


def _drop_print_matches(backing_px: np.ndarray, ink: np.ndarray, disc_inside: np.ndarray | None,
                        h: int, w: int) -> None:
    """Remove (in place) the background-colour matches that are really print.

    When the picked colour is the colour of the print (black board behind black
    print), every printed area matches. Holes cannot be told from print there by
    colour, so a large solid print area mostly covered by the match is left to the
    other rules, whatever its shape: disc, band, silhouette, square. Print areas are
    the connected ink regions after an opening that erases lines, digits and the
    holes themselves, at least BACKING_PRINT_MIN of the image; the solid disc found
    by find_disc counts as one area.
    """
    if disc_inside is not None and backing_px[disc_inside].mean() > BACKING_PRINT_SHARE:
        backing_px &= ~disc_inside
    k = _odd(0.012 * max(h, w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    thick = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < BACKING_PRINT_MIN * h * w:
            continue
        x, y, bw, bh = stats[i, :4]
        if backing_px[y:y + bh, x:x + bw][labels[y:y + bh, x:x + bw] == i].mean() <= BACKING_PRINT_SHARE:
            continue
        # the opening shaved the outline of the area: widen it by the same kernel so
        # its anti-aliased edge and thin protrusions go too
        x0, y0, x1, y1 = max(0, x - k), max(0, y - k), min(w, x + bw + k), min(h, y + bh + k)
        comp = cv2.dilate((labels[y0:y1, x0:x1] == i).astype(np.uint8), kernel) > 0
        backing_px[y0:y1, x0:x1][comp] = False


def _odd(v: float) -> int:
    v = int(round(v))
    return max(3, v | 1)


def ink_mask(gray: np.ndarray, ink_ratio: float = 0.45) -> np.ndarray:
    """Pixels far darker than the paper (printed black)."""
    blur = cv2.medianBlur(gray, 5)
    paper = float(np.percentile(blur, 95))
    _, ink = cv2.threshold(blur, int(ink_ratio * paper), 255, cv2.THRESH_BINARY_INV)
    return ink


def _circular_ink_shapes(gray: np.ndarray, ink_ratio: float, min_r: float) -> list[Disc]:
    """Every closed ink blob that is roughly a circle of radius >= min_r, largest first."""
    ink = ink_mask(gray, ink_ratio)
    f = max(gray.shape) / 1920.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(15 * f), _odd(15 * f)))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Disc] = []
    for c in cnts:
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < min_r or cv2.contourArea(c) / (np.pi * r * r) < 0.7:
            continue
        circle = np.zeros_like(gray)
        cv2.circle(circle, (int(x), int(y)), int(r * 0.95), 255, -1)
        fill = cv2.countNonZero(cv2.bitwise_and(closed, circle)) / max(1, cv2.countNonZero(circle))
        out.append(Disc(float(x), float(y), float(r), solid=fill >= 0.75))
    out.sort(key=lambda d: -d.r)
    return out


def find_disc(gray: np.ndarray, ink_ratio: float = 0.45) -> Disc | None:
    """Largest circular black shape: a solid disc, or the outer band of a ring target."""
    shapes = _circular_ink_shapes(gray, ink_ratio, 0.05 * max(gray.shape))
    return shapes[0] if shapes else None


def find_targets(gray: np.ndarray, ink_ratio: float = 0.45, min_r_frac: float = 0.03) -> list[Disc]:
    """All target diagrams in the photo (circular black shapes), largest first.

    Keeps the shapes whose radius is at least min_r_frac of the longest image side
    and at least 60 % of the median radius, so the inner thin rings exposed by a
    torn band are not taken for extra targets; a circle whose centre lies inside
    a larger kept circle is dropped for the same reason. Reading order (rows, then
    left to right) is what `assign_groups` uses for the labels, not this order.
    """
    shapes = _circular_ink_shapes(gray, ink_ratio, min_r_frac * max(gray.shape))
    if not shapes:
        return []
    median_r = float(np.median([d.r for d in shapes]))
    kept: list[Disc] = []
    for d in shapes:  # largest first, so nesting is checked against bigger ones only
        if d.r < 0.6 * median_r:
            continue
        if any(np.hypot(d.cx - k.cx, d.cy - k.cy) < k.r for k in kept):
            continue
        kept.append(d)
    return kept


def _reading_order(targets: list[Disc]) -> list[Disc]:
    """Sort targets row by row (rows are bands of about one diameter), then left to right."""
    if not targets:
        return []
    band = 1.2 * float(np.median([d.r for d in targets]))
    return sorted(targets, key=lambda d: (round(d.cy / band), d.cx))


def assign_groups(impacts: list[Impact], targets: list[Disc]) -> list[Group]:
    """Give every impact the index of the nearest target; return the group list.

    With fewer than two targets everything goes into a single group "G1". Each
    target makes one group (even when no shot hit it) so the user can see the
    empty diagrams; labels are G1, G2, ... in reading order.
    """
    if len(targets) < 2:
        g = Group("G1", targets[0].cx, targets[0].cy, targets[0].r) if targets else Group("G1")
        for imp in impacts:
            imp.group = 0
        return [g]
    ordered = _reading_order(targets)
    for imp in impacts:
        imp.group = int(min(range(len(ordered)),
                            key=lambda i: (ordered[i].cx - imp.x) ** 2 + (ordered[i].cy - imp.y) ** 2))
    return [Group(f"G{i + 1}", d.cx, d.cy, d.r) for i, d in enumerate(ordered)]


def _split_blob(contour: np.ndarray, n: int) -> list[tuple[float, float]]:
    """Split a torn region into n centres with k-means on its pixel coordinates."""
    x, y, w, h = cv2.boundingRect(contour)
    sub = np.zeros((h, w), np.uint8)
    cv2.drawContours(sub, [contour - [x, y]], -1, 255, -1)
    ys, xs = np.nonzero(sub)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    n = max(1, min(n, len(pts)))
    if n == 1:
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return [(float(x + w / 2), float(y + h / 2))]
        return [(m["m10"] / m["m00"], m["m01"] / m["m00"])]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, _, centers = cv2.kmeans(pts, n, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return [(float(cx) + x, float(cy) + y) for cx, cy in centers]


def detect_impacts(
    image_bgr: np.ndarray,
    paper_ratio: float = 0.84,
    sat_thresh: int = 95,
    open_kernel: int = 11,
    min_area: float = 150.0,
    shots_fired: int | None = None,
    hole_area_hint: float | None = None,
    ink_ratio: float = 0.45,
    backing: list[BackingSample] | None = None,
    backing_tol: float = BACKING_DEFAULT_TOL,
    backing_only: bool = False,
    hole_model=None,
    use_rules: bool = True,
) -> DetectionResult:
    """Detect bullet holes in a target photo.

    paper_ratio: a pixel darker than this fraction of the local paper
        brightness (and not ink) is hole backing. Lower it if shadows are
        detected as holes, raise it if light-grey holes are missed.
    sat_thresh: HSV saturation (0-255) above which a darker-than-paper pixel
        counts as coloured backing (wood, ~120-180). Paper, grey backing and
        warm-lit paper in shadow (up to ~90) stay below it.
    open_kernel: size (px) of the opening that removes ring lines and digits;
        must stay smaller than a single hole.
    shots_fired: if known, the shots not accounted for by single holes are
        distributed over the large (torn) blobs proportionally to their area.
    hole_area_hint: expected area (px^2, original image) of the bright/grey
        core of a single hole (the UI derives it from the scale and a hole
        diameter); defaults to the median of the smaller half of the blobs.
    backing: colour samples taken inside holes (sample_backing). Pixels within
        backing_tol standard deviations of a sample are hole pixels, added to
        the rules above, or replacing them when backing_only is true.
    hole_model: optional trained network (hole_model.HoleNet, or any callable
        (image_bgr, hole_radius_px) -> hits with x, y, score in original pixels).
        With use_rules=False only the network finds the holes; otherwise its hits
        are merged with the rule-based impacts (a rule impact closer than one hole
        radius to a network hit is the same hole and is dropped).
    """
    original = image_bgr
    H, W = image_bgr.shape[:2]
    scale = 1.0
    if max(H, W) > WORK_SIZE:
        scale = WORK_SIZE / max(H, W)
        image_bgr = cv2.resize(image_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = image_bgr.shape[:2]
    f = max(h, w) / 1920.0
    blur_bgr = cv2.medianBlur(image_bgr, 5)
    gray = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

    paper_global = float(np.percentile(gray, 95))
    # printed black is grey (low chroma); the dark brown of a wooden backing in
    # shadow is just as dark but strongly coloured (orange hue, 5-25 on the
    # OpenCV 0-180 scale), so it must stay a candidate. The hue test keeps the
    # purple/green chromatic fringes of print edges in the ink mask.
    hue = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    chroma = blur_bgr.max(axis=2).astype(np.int16) - blur_bgr.min(axis=2).astype(np.int16)
    dark_wood = (sat > sat_thresh) & (chroma > 40) & ((hue <= 30) | (hue >= 170))
    ink = (gray < ink_ratio * paper_global) & ~dark_wood
    targets = find_targets(gray, ink_ratio)
    disc = find_disc(gray, ink_ratio)

    # local paper level: a grey-level closing fills every dark object smaller
    # than the kernel (holes, bands, print) with the surrounding paper value,
    # but leaves large-scale illumination steps (hard shadows) untouched
    kd = _odd(101 * f)
    local = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    local = cv2.GaussianBlur(local, (5, 5), 0).astype(np.float32)
    local = np.maximum(local, 1.0)
    rel = gray.astype(np.float32) / local

    midtone = (rel < paper_ratio) & ~ink
    # coloured backing (wood) is saturated AND darker than the paper overall;
    # the second condition rejects warm-lit paper. Large torn regions are not
    # filled by the closing, so this rule must not depend on the local level.
    coloured = (sat > sat_thresh) & (gray < 0.9 * paper_global) & ~ink
    cand = midtone | coloured

    inside = None
    if disc is not None and disc.solid:
        inside = np.zeros_like(gray)
        cv2.circle(inside, (int(disc.cx), int(disc.cy)), int(disc.r * 0.96), 255, -1)
        inside = inside > 0
        # the black print turns grey under strong light: drop mid-tones there,
        # and take bright (torn paper) pixels instead
        cand &= ~inside
        cand |= inside & ((gray > 0.65 * paper_global) | coloured)

    # backing colour picked by the user: not excluded by the ink mask, since a dark
    # backing behind white paper is exactly what the ink rule hides
    backing_px = None
    if backing:
        backing_px = backing_match(cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2Lab), backing, backing_tol)
        _drop_print_matches(backing_px, ink, inside, h, w)
        cand = backing_px if backing_only else (cand | backing_px)

    mask = cand.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    k = _odd(open_kernel)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0.15 * h * w  # anything bigger is background, not a hole

    def keep(c):
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            return False
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            return False  # touches the image border: paper edge, table
        hull = cv2.contourArea(cv2.convexHull(c))
        return hull <= 0 or area / hull >= 0.2  # thin arcs/strips (shadow edges) are not holes

    blobs = [c for c in cnts if keep(c)] if use_rules else []
    debug = {"ink": ink, "midtone": midtone, "coloured": coloured, "candidates": opened > 0}
    if backing_px is not None:
        debug["backing"] = backing_px
    if disc is not None:
        disc = Disc(disc.cx / scale, disc.cy / scale, disc.r / scale, disc.solid)
    targets = [Disc(t.cx / scale, t.cy / scale, t.r / scale, t.solid) for t in targets]
    impacts, single = _rule_impacts(blobs, scale, hole_area_hint, shots_fired)
    blobs = [(c / scale).astype(np.int32) for c in blobs]
    if hole_model is not None:
        r_hint = float(np.sqrt(hole_area_hint / np.pi)) if hole_area_hint else None
        hits = hole_model(original, r_hint)
        if shots_fired and not use_rules and len(hits) > shots_fired:
            # torn regions often give several peaks: keep the strongest, as many as shots
            hits = sorted(hits, key=lambda hit: -hit.score)[:shots_fired]
        r_same = float(np.sqrt(single / np.pi)) if single else (r_hint or max(H, W) / 150.0)
        net = [Impact(float(hit.x), float(hit.y), 1, np.pi * r_same ** 2, False, -1) for hit in hits]
        impacts = net + [i for i in impacts
                         if all(np.hypot(i.x - n.x, i.y - n.y) > r_same for n in net)]
        if not single and hole_area_hint:
            single = float(hole_area_hint)
    groups = assign_groups(impacts, targets)
    return DetectionResult(disc, impacts, blobs, single, debug, targets, groups)


def _rule_impacts(blobs, scale: float, hole_area_hint, shots_fired) -> tuple[list[Impact], float]:
    """Impacts from the hole blobs (work resolution): torn blobs are split into the
    estimated shot count. Returns the impacts and the single-hole area, both in
    original pixels."""
    if not blobs:
        return [], 0.0
    areas = np.array([cv2.contourArea(c) for c in blobs])
    if hole_area_hint:
        single = float(hole_area_hint) * scale ** 2
    else:
        # the smaller half of the blobs are (almost) all single holes; the
        # median ignores a stray tiny blob as well as the torn regions
        lower = np.sort(areas)[: max(1, len(areas) // 2)]
        single = float(np.median(lower))

    # blobs up to ~1.8x a single hole are counted as one shot
    counts = np.ones(len(blobs), dtype=int)
    multi = areas > 1.8 * single
    if multi.any():
        if shots_fired and shots_fired > int((~multi).sum()):
            remaining = shots_fired - int((~multi).sum())
            share = areas[multi] / areas[multi].sum() * remaining
            est = np.maximum(1, np.floor(share)).astype(int)
            leftover = remaining - int(est.sum())
            if leftover > 0:
                order = np.argsort(-(share - np.floor(share)))
                for i in order[:leftover]:
                    est[i] += 1
            counts[multi] = est
        else:
            # torn regions grow much faster than linearly with the number of
            # shots (paper falls out); sqrt(area ratio) matches the samples well.
            est = np.round(np.sqrt(areas[multi] / single))
            counts[multi] = np.clip(est, 2, 25).astype(int)

    impacts: list[Impact] = []
    for bid, (c, a, n) in enumerate(zip(blobs, areas, counts)):
        for (x, y) in _split_blob(c, int(n)):
            impacts.append(Impact(x / scale, y / scale, 1, float(a) / n / scale ** 2, False,
                                  bid if n > 1 else -1))
    return impacts, single / scale ** 2
