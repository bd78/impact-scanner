# Target Impact Analyser – Technical Reference

Audience: developers and AI coding agents working on this repository. This document
describes the structure, the data flow, the detection algorithm with its tuned
constants, the database schema, and the known limitations. Read it before changing
`target_analyzer/`.

## 1. Purpose and scope

A Windows desktop application (pure Python) that:

1. loads a photo of a paper shooting target;
2. lets the user define the pixel-to-millimetre scale;
3. detects bullet holes with classical computer vision (OpenCV colour rules), optionally
   helped by background colours the user picks inside holes (§4.10), or with a small
   convolutional network trained on labelled photos and run by OpenCV's DNN module
   (§10), or both;
4. splits the holes into *groups*: one per printed target diagram when the photo shows
   a card with several diagrams, otherwise one group; the user can create, delete and
   re-assign groups;
5. computes group-size statistics (width, height, extreme spread) per group, per photo
   (mean ± std-dev over its groups) and over several photos;
6. persists sessions, groups and impacts in SQLite and aggregates them.

There is no network access and no external service. One model file is shipped:
`target_analyzer/models/hole_net.onnx` (3.7 MB) with its metadata `hole_net.json`.
Training it needs PyTorch, in a separate environment (`.venv-train`,
`requirements-train.txt`); the application itself does not.

## 2. Repository layout

```
py-detect-impact/
├── setup.ps1               creates .venv and installs requirements.txt
├── run.ps1                 starts the app: .venv\Scripts\python -m target_analyzer [db]
├── requirements.txt        opencv-python-headless, numpy, pillow, pillow-heif
├── requirements-train.txt  + torch, onnx: training the hole network only (.venv-train, git-ignored)
├── README.md               short overview
├── docs/
│   ├── USER_GUIDE.md       end-user manual
│   └── TECHNICAL.md        this file
├── samples/                tuning photos (see §8)
├── training/
│   ├── train_hole_model.py trains, cross-validates and exports the hole network (§10)
│   ├── labels/             one JSON label per photo: hole centres, ignore boxes (§10.2)
│   ├── images/             photos copied by "Save as training label" (git-ignored: images/)
│   ├── report.json         scores of the last training run (validation split)
│   └── report_cv.json      scores of the last cross-validation
├── tests/
│   ├── check_samples.py    runs the detector on samples/, scores it against the labels,
│   │                       writes tests/out/*_detected.jpg (and *_net.jpg with --net)
│   └── out/                generated, git-ignored
└── target_analyzer/
    ├── __init__.py
    ├── __main__.py         entry point, chooses the database path
    ├── detection.py        hole detection (OpenCV), background colour samples – no Tk, no SQLite
    ├── hole_model.py       hole network inference (cv2.dnn), heat-map decoding – no Tk, no SQLite
    ├── labels.py           training-label format, I/O, scoring (match_points) – pure Python
    ├── models/             hole_net.onnx + hole_net.json (threshold, training summary)
    ├── stats.py            group statistics, unit conversion – pure Python
    ├── db.py               SQLite persistence, aggregation, settings
    ├── images.py           stored photo copies in <db folder>/images/ (resize, JPEG, unique names, path resolution)
    ├── export.py           CSV export of records for Excel (one row per group) – no Tk, no SQLite
    ├── recordfilter.py     filter patterns of the Records & averages list – pure Python
    ├── drawstyle.py        display style of impacts and groups (colours, line styles, widths) – pure Python
    └── ui.py               Tkinter application (the only module that imports the others)
```

Dependency direction: `ui.py` → `detection.py`, `hole_model.py`, `labels.py`,
`stats.py`, `db.py`, `images.py`, `export.py`, `recordfilter.py`, `drawstyle.py`. The lower modules do not
import each other except `db.py` → `stats.py` (for `GroupStats`, `summarize_groups`),
`db.py` → `images.py` (to delete orphaned copies and to derive source names during
migration) and `export.py` → `stats.py` (`to_unit`).
`detection.py` does not import `hole_model.py`: `detect_impacts` takes the network as a
plain callable. Keep it that way: detection, hole_model, labels, stats, db, images and
drawstyle must stay testable without a display. `training/train_hole_model.py` and
`tests/check_samples.py` import `detection`, `hole_model`, `images` and `labels`.

Runtime: Python 3.10+ (uses `X | None` annotations with `from __future__ import
annotations`), Tkinter from the standard library, SQLite from the standard library.
OpenCV is the *headless* build (no GUI functions such as `cv2.imshow`).

## 3. Data model

### 3.1 In-memory (`detection.py`)

```python
@dataclass
class Disc:            # largest circular black shape found
    cx, cy, r: float   # pixels, ORIGINAL image coordinates
    solid: bool        # True = filled disc (25 m pistol target), False = outer band of a ring target

@dataclass
class Impact:          # one shot
    x, y: float        # pixels, ORIGINAL image coordinates
    count: int = 1     # always 1 after splitting (kept for future use)
    area: float = 0.0  # blob area / count, original-pixel units
    manual: bool       # True if added by the user (drawn with a dashed outline)
    blob_id: int       # index of the torn blob it was split from, -1 if isolated
    group: int = 0     # index into the group list

@dataclass
class Group:           # a set of shots measured together
    label: str         # "G1", "G2", ... or user-given
    cx, cy, r: float | None   # printed target circle (pixels) when the group comes from one

@dataclass
class BackingSample:   # colour of the background seen through a hole, picked by the user (§4.10)
    mean: (L, a, b)    # OpenCV 8-bit Lab, robust mean of the sampled disc
    cov: 3x3           # covariance of the disc pixels + diag(BACKING_MIN_STD²)
    n: int             # pixels used
    # rgb_hex (swatch colour), to_dict() / from_dict() (settings JSON; None if invalid)

@dataclass
class DetectionResult:
    disc: Disc | None
    impacts: list[Impact]
    blobs: list[np.ndarray]   # OpenCV contours (original coordinates), for drawing
    single_hole_area: float   # reference area used to classify blobs
    debug: dict               # work-resolution boolean masks: ink, midtone, coloured, candidates (+ backing)
    targets: list[Disc]       # every circular black shape (target diagrams), largest first
    groups: list[Group]       # impacts[i].group indexes this list
```

The group index is an in-memory concept: deleting a group renumbers the indexes of the
following groups (`App.delete_group`). The `Group.label` is what the user sees and what
is stored.

### 3.2 Statistics (`stats.py`)

All physical quantities are stored in **millimetres**. `MM_PER_UNIT = {"mm": 1,
"cm": 10, "inch": 25.4}`; `to_unit()` / `from_unit()` convert for display and input.

```python
@dataclass
class GroupStats:
    n: int
    width_mm, height_mm: float       # extent of the centres along x / y
    diameter_mm: float               # extreme spread: max pairwise centre distance (O(n²), n ≤ ~100)
    center_x_px, center_y_px: float  # centroid, pixels
```

`group_stats(points_px, px_per_mm)` returns `None` for an empty list or a non-positive
scale. `mean_std(values)` returns the mean and the *sample* standard deviation
(ddof = 1; 0 when fewer than two values). `percentile(values, q)` is the q-th
percentile with linear interpolation between ranks (same convention as NumPy's
default); it is used for the P90 diameter.

`summarize_groups(list[GroupStats])` is the one aggregation used at every level
(photo, selection of records): `{metric: (mean, std)}` for `width_mm, height_mm,
diameter_mm, n_shots`, plus `count` (groups), `total_shots` and `diameter_p90_mm`.
Groups are pooled, never weighted by photo.

`cluster_points(points, max_dist)` is single-linkage clustering with a union-find
(O(n²) pairs, fine for a few hundred shots): two shots belong together when a chain
of shots at most `max_dist` apart joins them. Clusters are numbered in reading order
(row bands of `max_dist`, then x). It backs the "Auto-group: distance" button.

### 3.3 Database (`db.py`)

SQLite file, default `<repo>/targets.db` (see `__main__.py`). Schema is created with
`CREATE TABLE IF NOT EXISTS` on every start, then `_migrate()` upgrades older files;
`PRAGMA user_version` holds the schema version (`SCHEMA_VERSION = 3`).

A **session** is one photo. It owns one or more **groups**; every impact belongs to a
group.

```sql
CREATE TABLE sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,      -- ISO 8601, when saved
    shot_date       TEXT,               -- free text from the form (default: today ISO)
    image_path      TEXT,               -- "images/<stamp>_r<id>_<stem>.jpg", relative to the db folder
                                        -- (absolute path in records from older versions); see §3.4
    source_name     TEXT,               -- original file name of the photo ("IMG_5862.jpeg")
    weapon          TEXT,               -- free text, used for filtering (exact match)
    distance        TEXT,
    ammo            TEXT,
    notes           TEXT,
    unit            TEXT NOT NULL,      -- display unit chosen when saved ("cm" | "inch")
    px_per_mm       REAL NOT NULL,      -- scale of the photo
    n_groups        INTEGER NOT NULL DEFAULT 1,
    n_shots         INTEGER NOT NULL,   -- total over the groups
    width_mm        REAL NOT NULL,      -- MEAN over the groups (= the group value when n_groups = 1)
    height_mm       REAL NOT NULL,
    diameter_mm     REAL NOT NULL
);
CREATE TABLE groups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,       -- display order
    label       TEXT NOT NULL,
    cx_px       REAL, cy_px REAL, r_px REAL,   -- target circle in STORED-copy pixels, or NULL
    n_shots     INTEGER NOT NULL,
    width_mm    REAL NOT NULL,
    height_mm   REAL NOT NULL,
    diameter_mm REAL NOT NULL
);
CREATE TABLE impacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    group_id    INTEGER REFERENCES groups(id) ON DELETE CASCADE,
    x_px        REAL NOT NULL,          -- pixels of the STORED copy (see §3.4)
    y_px        REAL NOT NULL,
    manual      INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX idx_sessions_image_path ON sessions(image_path)
    WHERE image_path LIKE 'images/%';   -- two records never share a stored copy
CREATE TABLE settings (
    key         TEXT PRIMARY KEY,
    value       TEXT                    -- "draw_style": DrawStyle JSON (see §5)
                                        -- "backing_samples": JSON list of BackingSample.to_dict()
                                        -- "detector": "Colour rules" | "Neural network" | "Both"
);
```

`PRAGMA foreign_keys = ON` is set per connection so deleting a session deletes its
groups and impacts. Empty groups (no shot) are not stored: `App.save_record` skips them.
`get_setting(key, default)` / `set_setting(key, value)` (upsert) read and write the
settings table.

Migrations in `Database._migrate()`:

- drop the `mean_radius_mm` column of the first versions (`ALTER TABLE ... DROP
  COLUMN`, needs SQLite 3.35+; silently kept otherwise);
- `user_version < 2` → `_migrate_v2()`: adds `sessions.n_groups` and
  `impacts.group_id`, creates one group `G1` per session that has none, copying the
  session statistics, and attaches the session's impacts to it. It is idempotent (only
  sessions without groups are touched), so an interrupted upgrade finishes next start.
- `user_version < 3` → `_migrate_v3()`: adds `sessions.source_name` and fills it for
  existing rows with `images.source_name_of(image_path, all_paths)`: the file name of an
  absolute path, or the stem inside a stored copy name. A trailing `_<n>` is dropped only
  when the same path without it exists (then it was a clash suffix, otherwise it is part
  of the name, as in `IMG_5862`), and nested copy names (a record re-saved after *Load
  record* by an older version, `20261030_104309_20261029_220754_IMG_5862_2.jpg`) are
  unwrapped. It then creates the partial unique index; if an old database already
  shares a copy between records the index is skipped (the shared-copy logic of
  `delete_session` keeps working).

`Database.save_session(groups: list[GroupRecord], px_per_mm, unit, *, source_name,
store, **meta)` writes the session row (totals and means from `summarize_groups`), the
group rows and the impacts **in one transaction**. `store(sid) -> relative_path`, when
given, is called with the new session id before the commit and writes the photo copy;
on any exception the transaction is rolled back and a copy already written is deleted,
so a failed save leaves neither a row nor a file. `sessions_with_source(name)` lists
the ids saved from a photo of that name (the UI asks before saving another one).
`get_groups(sid)` / `get_impacts(sid)` read a record back; impacts carry `group_id` and
the UI maps it to the group index by position.

`Database.aggregate(ids)` pools the **groups** of the given sessions and returns the
`summarize_groups` dict plus `sessions` (number of sessions). A 25-diagram card thus
weighs 25 times a single-group photo; the P90 is over groups.

### 3.3.1 CSV export (`export.py`)

`App.export_records` (button *Export to Excel (CSV)…* in Records & averages) exports
the selected or all listed records. `export_rows(sessions, groups_of, db_path, unit,
decimal)` builds a header and one row per `groups` row: record id, date (`shot_date`,
else the date part of `created_at`), stored photo as an absolute path (relative
`image_path` taken from the database folder; the original photo's path is never
stored, only `source_name`), `source_name`, weapon, distance, ammo, group label,
`n_shots`, width and height converted to the display unit with 2 decimals. A record
without group rows gets one row with its session totals. `write_csv` writes UTF-8 with
a BOM (Excel then reads accents correctly). `excel_separators()` reads the Windows list
separator and decimal mark (`GetLocaleInfoEx`, `LOCALE_SLIST` / `LOCALE_SDECIMAL`)
because Excel opens a CSV with them; elsewhere `,` and `.`. `safe_text` prefixes text
starting with `= + - @ TAB CR` with `'` (CSV formula injection).

### 3.4 Stored photo copies (`images.py`)

`store_image(image_bgr, db_path, source_name, record_id) -> (relative_path, factor)`
writes a JPEG copy to `<db folder>/images/<YYYYmmdd_HHMMSS>_r<record_id>_<sanitised
stem>.jpg`, downscaled with `INTER_AREA` so the longest side is at most
`MAX_SIDE = 2500` px, quality `JPEG_QUALITY = 80` (≈ 0.5 MB for a 2500 px target photo;
a PNG was ≈ 4 MB).

Uniqueness: the record id (AUTOINCREMENT, never reused in a database) makes the name
unique even when the same photo, or two photos with the same name, are saved several
times in the same second. The file is opened with mode `"xb"` (exclusive create), so an
existing file is never overwritten and there is no check-then-write race between two
running instances; if the name is taken anyway (an old database restored next to a
newer `images/` folder) `_2`, `_3`, ... is appended. The JPEG is encoded with
`cv2.imencode` and written by Python's `open`, so non-ASCII paths work (`cv2.imwrite`
fails on them). `stored_factor(image)` gives the downscale factor without writing,
which the UI needs before the record id exists. Records saved by the short-lived PNG
version keep their `.png` path and load fine.

`Database.delete_session(sid)` deletes the row (impacts cascade) and then, if no
other session references the same `image_path`, calls `delete_stored_image`, which
only ever removes files inside `<db folder>/images/` (absolute paths from older
records are never touched). It returns True when a file was removed.

`factor` (≤ 1) is the ratio stored-copy pixels / original pixels. `App.save_record`
multiplies every impact coordinate, every group circle and `px_per_mm` by it before
calling `save_session`, so a record is self-consistent with its copy and `Load record`
(`resolve_image_path`) shows the impacts at the right place. Group statistics are
unaffected (they were computed in mm before saving). The original photo is never
referenced again.

## 4. Detection algorithm (`detection.py`)

Entry point:

```python
detect_impacts(image_bgr, paper_ratio=0.84, sat_thresh=95, open_kernel=11,
               min_area=150.0, shots_fired=None, hole_area_hint=None,
               ink_ratio=0.45, backing=None, backing_tol=3.0, backing_only=False,
               hole_model=None, use_rules=True) -> DetectionResult
```

All parameters except `hole_area_hint` and `ink_ratio` are exposed in the UI.
`backing*` are the user's background colours (§4.10); `hole_model` / `use_rules` select
the network and how its hits are merged with the rule-based impacts (§4.11, §10). With
the defaults the output is identical to the version without these parameters (checked
on every sample with a fixed `cv2.setRNGSeed`: `_split_blob` uses `cv2.kmeans` with
k-means++ seeding, whose global RNG makes the split points of torn regions vary
slightly from run to run).

### 4.1 Working resolution

If the longest side exceeds `WORK_SIZE = 1920`, the image is downscaled
(`INTER_AREA`) and every absolute constant below refers to that scale. Output
coordinates, contours and areas are converted back to original pixels
(`/ scale`, areas `/ scale²`). `hole_area_hint` is given in original pixels and is
converted *to* work units (`* scale²`). Kernels that must follow the print size use
`f = max(h, w) / 1920` as a multiplier (`_odd(101 * f)` etc.).

### 4.2 Pre-processing

```
blur_bgr = medianBlur(image, 5)          # removes JPEG noise and paper fibres
gray     = BGR2GRAY(blur_bgr)
hue, sat = BGR2HSV(blur_bgr)[..., 0:2]   # hue 0-180, saturation 0-255
chroma   = max(B,G,R) - min(B,G,R)
paper_global = percentile(gray, 95)      # the paper is the brightest large surface
dark_wood = (sat > sat_thresh) & (chroma > 40) & (hue <= 30 | hue >= 170)
ink      = (gray < ink_ratio * paper_global) & ~dark_wood   # printed black (≈ < 100 on a 225 paper)
```

`dark_wood` exists because a wooden backing seen through a hole in deep shadow is as
dark as print (gray 30–55 on the 22 Hunter card) but strongly orange (sat 170–220,
hue 9–16), whereas print is grey. Such pixels must not be ink, otherwise both hole rules
below exclude them. The chroma floor keeps warm-lit print (chroma ≤ 40 after the median)
in the ink mask: lowering it to 20–35 turned the thin ring lines of IMG_5861 and the
band interiors of IMG_5862 into candidates. The hue test rejects the purple/green
chromatic fringes along print edges. Very dark wood with chroma below 40 (gray ≈ 30,
cells 16 and 11 of the card) is still missed and needs a manual add.

### 4.3 Disc detection (`find_disc`)

Ink mask → closing with a 15 px ellipse (fills the thin white ring lines and digits)
→ largest external contour → `minEnclosingCircle`. Rejected if radius < 5 % of the
image or if `contourArea / (π r²) < 0.7` (not circular). `solid` is true when at
least 75 % of the pixels inside 0.95 r are ink: a filled disc. A ring target gives
`solid = False` (its white centre lowers the fill to ≈ 0.5). The UI uses the circle
for the "auto scale" tool in both cases.

### 4.4 Local paper level and the two pixel rules

```
local = GaussianBlur(morphologyEx(gray, MORPH_CLOSE, ellipse(101·f)), 5)
rel   = gray / local
```

A **grey-level closing** (dilate then erode) replaces every dark object smaller than
the kernel (holes, ring bands up to ~100 px, print) by the surrounding paper value,
but reproduces large-scale illumination steps exactly. This is what makes hard phone
shadows harmless; an earlier version used a plain dilation and produced a strip of
false positives along every shadow edge.

```
midtone  = (rel < paper_ratio) & ~ink                       # grey backing, holes on white paper
coloured = (sat > sat_thresh) & (gray < 0.9·paper_global) & ~ink   # wood backing
cand     = midtone | coloured
```

`coloured` deliberately uses the *global* paper level: a torn region wider than the
closing kernel is not filled by the closing, so `rel ≈ 1` inside it and the midtone
rule misses its interior; saturation does not have that problem. The
`gray < 0.9·paper_global` term rejects warm-lit bright paper; the `sat > 95` default
rejects shadowed warm paper (measured 75–90) while wood measures 120–180.

### 4.5 Solid-disc override

If a solid disc was found, inside 0.96 r the rules are replaced:

```
cand[inside] = (gray > 0.65·paper_global) | coloured
```

Reason: strongly lit black print reads as mid-grey (up to ≈ 125 on a 230 paper) and
would otherwise be flagged; inside the disc the holes are *bright* torn paper instead.

### 4.6 Morphology and blob filtering

```
mask   = close(cand, 5)          # joins the ragged edges of a hole
opened = open(mask, open_kernel) # erases ring lines, digit strokes, anti-aliased print edges
contours = findContours(opened, RETR_EXTERNAL)
```

A contour is kept when all of the following hold:

| Test | Rationale |
|---|---|
| `min_area ≤ area ≤ 0.15·h·w` | drops specks and background surfaces |
| bounding box does not touch the image border | paper edge, table, plastic sleeve |
| `area / convexHullArea ≥ 0.2` | thin arcs along shadow boundaries are not holes |

### 4.7 Single-hole reference and shot count

```
single = hole_area_hint·scale²  if given, else median(smallest half of the areas)
multi  = areas > 1.8·single
```

For multi blobs the count is:

- if `shots_fired` is known and exceeds the number of single blobs: the remaining shots
  are shared proportionally to area (floor, at least 1 each, remainder to the largest
  fractional parts);
- otherwise `clip(round(sqrt(area / single)), 2, 25)`. The square root was fitted on
  the samples: torn regions lose whole flaps of paper, so area grows much faster than
  linearly with the number of shots.

Each multi blob is then split into `count` centres by `cv2.kmeans` on its filled
pixel coordinates (`_split_blob`). Single blobs use the contour centroid.

The UI passes `hole_area_hint = π·(Ø/2 · px_per_mm)²` with Ø from the "Hole core Ø
(mm)" field (default 4.0) whenever the scale is set. Without a scale the fallback
median is fragile when many blobs are noise, which is why the status line asks the
user to set the scale first.

### 4.8 Targets and groups

`find_targets(gray, ink_ratio, min_r_frac=0.03)` reuses the disc search for *every*
closed ink contour: circularity ≥ 0.7, radius ≥ 3 % of the longest side. It then drops
shapes smaller than 60 % of the median radius and shapes whose centre lies inside a
larger kept circle, so the inner rings exposed when a band is torn open are not extra
targets. `find_disc` is the same search restricted to radius ≥ 5 % and returns the
largest shape. `detect_impacts` runs `find_targets` at work resolution and converts the
circles back to original pixels.

`assign_groups(impacts, targets)` gives every impact the index of the nearest target
centre and returns one `Group` per target (empty ones included, so the user sees the
diagrams without shots), labelled `G1…Gn` in reading order (`_reading_order`: rows are
bands of 1.2 × median radius, then x). With fewer than two targets everything goes into
a single `G1` that carries the disc circle when there is one. Shots far from every
diagram (on the blank paper between cells) still go to the nearest one; the UI's
*Move impact to selected group* tool corrects that.

On the 22 Hunter card (`samples/fe341a73-….jpg`, 1440×1920, 25 numbered diagrams plus
small ones in the margins) 27 of the 30 circles are found; the three missed ones have a
band broken by the grid line or pencil marks. The circles found are correct and the
assignment makes no mistake on the 89 default-parameter impacts.

### 4.9 Debug output

`DetectionResult.debug` holds the boolean masks `ink`, `midtone`, `coloured`,
`candidates` at work resolution. `tests/check_samples.py` shows how to visualise a
run; to inspect the masks, stack them with `np.hstack` after converting to `uint8`.

### 4.10 Hole background colour (user samples)

Purpose: the rules of §4.4 know only "darker than paper" and "saturated"; a black
backing behind white paper is ink to them, a white backing behind black print is the
solid-disc case only. The user can instead show what a hole looks like by clicking
inside one.

`sample_backing(image_bgr, x, y, radius)` (original pixels) crops the disc with a
margin, applies exactly the detector's pre-processing (resize to the work resolution,
5 px median), converts to Lab (OpenCV 8-bit: L 0-255, a and b offset by 128), and
estimates a **robust colour distribution**: start from the per-channel median, keep the
`BACKING_KEEP = 80 %` pixels closest to the current centre, recompute the mean, twice.
So a disc that overlaps the torn edge or the print still describes the backing. The
covariance of the kept pixels is the texture (wood grain, fibres, JPEG noise); a
minimum spread `BACKING_MIN_STD = (20, 5, 5)` (L, a, b) is added on the diagonal:

- lightness gets four times more than the colour axes because the light changes
  across a photo (shadow, lit side of a hole) while the hue of the backing does not;
- it keeps the covariance invertible for a flat colour (vector drawings).

The values come from nine labelled photos on which the colour rules fail, one sample
each, "use only this colour" mode (TP / FP / FN against the labels):

| min std (L, a, b) | 4, 4, 4 | 10, 4, 4 | 16, 5, 5 | 22, 6, 6 | 30, 5, 5 | 16, 8, 8 |
|---|---|---|---|---|---|---|
| F1 | 0.33 | 0.36 | 0.39 | 0.41 | 0.40 | 0.31 |

(colour rules alone on the same photos: F1 0.15). These figures were measured before
the print rule below; with it, (22, 6, 6) gives 0.43 on the same photos.

`backing_match(lab, samples, tol)` marks the pixels whose squared Mahalanobis distance
to any sample is below `tol²` (`tol` in standard deviations, UI default 3).
`backing_match_image` does it on a whole photo at work resolution, for the UI overlay.
In `detect_impacts` the match is computed on the median-filtered work image, after the
solid-disc override, and is **not** masked by `ink` (a dark backing behind white paper
is exactly what the ink rule hides):

```
cand = backing_px                      if backing_only
cand = midtone | coloured | backing_px  otherwise
```

**Print of the same colour** (`_drop_print_matches`). A backing of the colour of the
print (black board behind black print) matches the whole print. Holes cannot be told
from print there by colour, so print areas that the match covers for more than
`BACKING_PRINT_SHARE = 50 %` are removed from `backing_px` and left to the other rules.
Print areas are found **whatever their shape** (targets are not always circles: discs,
bands, silhouettes, squares): the connected components of `ink` after an opening with
an ellipse of 1.2 % of the longest side (erases lines, digits and the holes themselves),
at least `BACKING_PRINT_MIN = 0.5 %` of the image, widened back by the same kernel to
take their anti-aliased edge. The solid disc of §4.5, when found, counts as one area
(its lit parts are not `ink`). On 13 labelled photos, "use only this colour": F1 0.52
with the disc alone, 0.51 with a 2 % minimum (the inner bands of a silhouette, cut
apart by white outlines, are smaller), 0.55 with 0.5 %; the splatter sheet drops from 31
false hits to 4. A torn region that shows a black backing and covers more than 0.5 % of
the image would be removed too; none of the labelled photos has one.

Then the usual closing, opening and blob filters apply. Limits: holes inside print of
the background's colour stay invisible to colour (the network, §10, sees them by
shape); a background as bright as the paper matches the paper, which then forms one
huge region rejected by `max_area`, taking the holes on the paper with it. The UI warns
when one sample matches more than `BACKING_WARN_SHARE = 20 %` of the photo. One sample
covers one lighting; uneven light needs a sample in the lit and in the shadowed part.
Samples are stored in the `settings` table (`backing_samples`) and kept from photo to
photo, since a range usually uses the same backer.

### 4.11 Merging network hits

With `hole_model` given, `detect_impacts` calls `hole_model(original_image,
r_hint)` (`r_hint = sqrt(hole_area_hint / π)`, the expected hole radius in original
pixels, or None) and gets hits with `x, y, score` in original pixels (§10.5). With
`use_rules=False` the rule blobs are skipped and the impacts are the hits; when
`shots_fired` is given and smaller than the number of hits, only that many hits are
kept, strongest first (inside torn regions the network often puts several peaks: 140
hits for about 90 shots on the Hunter card). Otherwise
the impacts are all the hits plus every rule impact farther than one hole radius
(`sqrt(single_hole_area / π)`, else `r_hint`, else max side / 150) from every hit. Hits
get `blob_id = -1` and `area = π r²` of that radius. Targets, disc and groups are
computed as before.

## 5. User interface (`ui.py`)

`App(tk.Tk)` with a `ttk.Notebook`: tabs *Analyse*, *Records & averages* and
*Parameters*. Each tab name has its own background colour (`TAB_COLOURS`: blue, green,
orange). ttk cannot colour tabs one by one and the Windows theme ignores tab colours,
so the notebook uses the style `Coloured.TNotebook` whose tab layout is empty (no tab
row), and `_build` draws a bar of `tk.Label`s instead. A click calls `nb.select`;
`_update_tab_bar` (on `<<NotebookTabChanged>>`, so `nb.select()` from code works too)
shows the selected tab in its full colour with white bold text and the others in a
light tint (`tint(colour, TAB_TINT = 0.7)`), and colours the 4 px strip under the bar.

State held on the `App` instance:

| Attribute | Meaning |
|---|---|
| `image_path`, `image_bgr`, `pil_image` | the loaded photo (BGR for OpenCV, PIL RGB for display) |
| `source_name` | original file name of the photo; *Load record* restores it from the record, so a re-save keeps it instead of the stored copy's name |
| `style: DrawStyle` | display style, loaded from the `draw_style` setting at start |
| `view_scale`, `view_off` | fit-to-canvas transform; `_to_view()` / `_to_image()` convert coordinates both ways |
| `mode` | `None`, `"scale"`, `"add"`, `"remove"`, `"assign"` or `"backing"`; `on_click()` dispatches on it |
| `scale_points`, `px_per_mm` | the 2-point tool and the resulting scale |
| `impacts: list[Impact]`, `disc` | detection result and user edits; `impact.group` is the group index |
| `groups: list[Group]`, `cur_group` | the groups and the index selected in the group table |
| `group_stats: dict[int, GroupStats]`, `summary` | per-group statistics (only groups with shots, only with a scale) and the `summarize_groups` dict for the photo |
| `var_*` | Tk variables bound to the parameter entries |
| `backing: list[BackingSample]` | background colours picked by the user, loaded from / saved to the `backing_samples` setting |
| `backing_mask`, `backing_mask_k` | "show matching pixels" overlay: PIL "L" image at work resolution (0 or 150) and its pixels per photo pixel |
| `single_hole_area` | from the last detection (original px), used as `r_default` of an exported training label |
| `_hole_net` | `HoleNet`, loaded on first use (`_get_hole_net`) |

Group handling: `_members()` (group index → impact indexes), `_ensure_group()` (creates
`G1` when the user adds a shot with no group), `new_group` / `rename_group` /
`delete_group` (deleting removes the group's impacts after confirmation and renumbers
the indexes above it), `auto_group_circles` (`find_targets` + `assign_groups` on the
current impacts), `auto_group_distance` (`cluster_points`; threshold asked in the
display unit, default 5 cm, or in pixels without a scale), the *assign* click mode
(moves the nearest impact into `cur_group`). Labels are kept unique
(`_unique_label`). The *add* mode adds to `cur_group`.

Drawing (`redraw` → `_draw_group_overlay`, one call per group, the selected group
last so it is on top): target circle, box around the impacts with the centroid cross,
the impacts numbered within the group, and the group name above the top-left corner of
the box, outside it, `GROUP_LABEL_GAP = 10` screen px up (the box of a single shot is
its circle; a group without shots is labelled at the box of its target circle). Each element takes its
colour, dash and width from `self.style` through `_line(key, group_index)`; the colour
`"group"` resolves to `GROUP_COLOURS[i % n]`, a palette without red or yellow so the
defaults stay distinct (impacts yellow `#ffff00` solid 2 px, hand-added impacts yellow
dashed, groups in palette colours dashed 1 px, selected group red `#ff0000` circle
solid 4 px and box solid 2 px). The impact radius is `style.impact_radius(zoom)`:
`impact_size` screen pixels at zoom *fit*, multiplied by the zoom when
`scale_with_zoom` is true (the default), so a circle keeps the same size relative to the
photo; the box margin follows it. Line widths and fonts stay in screen pixels. The
preview always uses `impact_size`. The green disc from the auto-scale tool is drawn only while no group carries a
circle.

**Display style** (`drawstyle.py`): `DrawStyle(lines: {key: LineStyle(colour, style,
width)}, impact_size)` for the keys of `ELEMENTS` (`impact`, `impact_manual`,
`group_circle`, `group_box`, `selected_circle`, `selected_box`). Line styles map to the
Tk dash strings `"" / "-" / "." / "-."`, the portable form that scales with the width.
`to_json` / `from_json`: the parser accepts only `#rrggbb` or `"group"`, known styles and
widths in `WIDTH_RANGE` (1–12), radii in `IMPACT_SIZE_RANGE` (3–40); anything else falls
back to the default, so a damaged or newer setting never breaks the display.

*Parameters* tab (`_build_params`): one row per element with a colour swatch
(`colorchooser.askcolor`), a *Group colour* tick box, a line style combobox and a width
spinbox, plus the impact radius, the *grow with the zoom* switch and *Reset to
defaults*. Every change goes through
`_style_from_widgets`, which reads all widgets (a half-typed number is ignored), saves
the JSON with `set_setting`, redraws the photo and the preview canvas
(`_draw_preview`: two sample groups drawn with the same `_draw_group_overlay`).

Other key methods: `open_image` / `_load_image` (`images.load_image()`: Pillow with EXIF
transpose and the optional `pillow_heif` opener, AVIF and WebP included; returns BGR; also
used by `tests/check_samples.py` and the training script), `analyze` (builds the
hint, calls `detect_impacts`, takes `impacts` and `groups` from the result),
`update_stats` (recomputes every `GroupStats`, the photo summary, fills the group table
and the labels), `rotate_photo(clockwise)` (buttons *Rotate left / right*:
`images.rotate_image` turns `image_bgr` with `cv2.rotate`, and `images.rotate_xy` moves
every image coordinate the same way: impacts, group circle centres, `scale_points`,
`disc`; pixel centres are integers, so clockwise is `(h-1-y, x)` and four turns give back
the exact coordinates; radii, `px_per_mm` and the hole-colour samples are unchanged;
`update_stats` then swaps width and height; `rotation` counts the degrees since the file
was opened and is reset by `_load_image`), `save_record` (asks for confirmation when `sessions_with_source`
finds earlier records from the same file name; one `GroupRecord` per non-empty group,
coordinates and circles scaled by `stored_factor`; passes a `store` callback so the copy
is named with the new record id inside the save transaction), `_refresh_records` (the
table has a *photo* column with `source_name`; it lists the sessions for which
`recordfilter.row_matches` accepts every non-empty pattern of the filter row
`RECORD_FILTERS`, refreshed 250 ms after the last key press, and keeps the
selection of the rows still listed; patterns: wildcards `* ?` through `fnmatchcase`
on case-folded text, `> >= < <= = <> !=` and `a..b` compared as numbers when both
sides start with a number that is not part of a date or time, else as text, a lone
number compared numerically, `!` negates), `compute_average`
(pooled group statistics: mean ± std and P90 in both cm and inch, independent of the
display unit), `load_record` → `_load_record_id(sid)` (restores groups and impacts, mapping
`group_id` to the group position; `open_image` also calls it, through
`_offer_saved_record`, when `sessions_with_source` finds records for the chosen file
name and the user accepts the latest one), `delete_records` (rows and orphaned stored copies, see
`Database.delete_session`).

**Hole background colour and detector** (`_build_backing`, below the detection
parameters). *Pick in a hole* sets the mode `"backing"`; `_on_motion` (canvas
`<Motion>`) draws the sampling disc (tag `brush`, magenta) of `_brush_radius()` screen
pixels under the cursor, and a click calls `_add_backing_sample`, which samples a disc
of `radius / view_scale` photo pixels, so the user zooms in to sample small holes.
Swatches (`_update_swatches`, at most `MAX_SWATCHES = 7` plus "+n") remove their sample
on click. `_backing_changed` refreshes the swatches and, when *Show matching pixels* is
ticked, recomputes `backing_mask`; `redraw` composites `BACKING_OVERLAY` (magenta) over
the visible region through that mask, resized with the same box as the photo.
*Tolerance (σ)* is read by `_backing_tol()` (0.5-10, default 3); *Use only this colour*
maps to `backing_only`. The **Detector** combobox (`DETECTORS`: Colour rules, Neural
network, Both) is saved in the `detector` setting; a network choice without a loadable
model falls back to Colour rules. `analyze` passes `hole_model=self._get_hole_net()`
and `use_rules = detector != "Neural network"`. `lbl_model` shows the training summary
from `hole_net.json` (`_model_info`).

**Training labels** (`save_training_label`, button *Save as training label*): writes
the current impacts to `<db folder>/training/labels/<source name>.json` (format §10.2,
`r: null`, `r_default` from *Hole core Ø* × scale, else the last single-hole area, else
1 % of the longest side). The image path is relative to the db folder; a photo outside
it is copied to `<db folder>/training/images/` (a different file with the same name
gets `_2`, `_3`, ...). A turned photo (`rotation` ≠ 0) is not the file on disk: it is
written as shown to `training/images/<photo file name>_rot<degrees>.png` (e.g. `a.jpg_rot90.png`, so
`a.jpg` and `a.png` do not share one; written to a `.tmp` file first, then swapped in) (`_label_photo`). With the default database the db folder is the repository, so
the labels land in `training/labels/` next to the bundled ones; otherwise pass
`--base <db folder>` to the training script.

**Zoom and pan.** `zoom` is relative to "fit to canvas" (1 = whole photo, clamped to
`[1, ZOOM_MAX = 40]`); `view_scale = fit × zoom` screen px per image px and `view_off`
is the screen position of image pixel (0, 0). `_update_view()` recomputes both for the
canvas size (after a resize it keeps the image point at the canvas centre);
`_clamp_view()` centres an axis on which the photo is smaller than the canvas and
otherwise forbids panning past the image edge, so near the edges at low zoom the
point under the cursor can shift. Entry points: `zoom_at(factor, vx, vy)` keeps the
image point under (vx, vy) fixed (wheel, Ctrl+click = `ZOOM_STEP²`,
Ctrl+right-click / Ctrl+Shift+click = 1/`ZOOM_STEP²`, `+`/`-` keys and buttons at
the centre); the slider (`ttk.Scale` on `log(zoom)`, so each step is the same ratio)
calls `_on_zoom_slider`, and `redraw` sets its variable back, which does not fire the
command; `zoom_fit` (button, `0`/`Home`); `zoom_to_box` / `zoom_to_group` (button and
double-click in the group table: the group's circle × 1.1, or the box of its shots plus
a margin). Pan: right or middle drag always, left drag when no tool is active
(`_pan_start/_move/_end`). `<Control-Button-1>` is more specific than `<Button-1>`, so
Tk does not run `on_click` for a Ctrl+click and no tool fires. Keys are bound on the
canvas, which takes the focus on click, so typing in the entries never zooms.

`redraw` resamples only the visible part of the photo (`Image.resize(..., box=...)`),
so its cost is bounded by the canvas size: about 25 ms per zoom step on a 12 Mpx photo.
`BILINEAR` below 4 screen px per image px, `NEAREST` above (crisp pixels for placing a
shot). Impact circles scale with the zoom (see *Drawing*); click tolerance for *Remove*
and *Move* is the larger of `CLICK_TOLERANCE_PX` = 25 screen pixels and the impact
radius, so a click anywhere inside an enlarged circle hits it.
The scale dialog is a `simpledialog.Dialog` subclass (`ScaleDialog`) returning
`(value, unit)`.

`__main__.py`: `python -m target_analyzer [db_path]`; default database is
`<repo>/targets.db`.

## 6. Testing

There is no unit-test suite yet. Two checks exist:

1. `tests/check_samples.py` – runs `detect_impacts` with defaults on every image in
   `samples/` (all formats, read with `images.load_image`), prints blob/impact/target/
   group counts, the group sizes (scale assumed from the detected circle = 200 mm, so
   the cm values are only meaningful for the solid-disc targets) and the shots per
   group, scores the impacts against `training/labels/` when the photo has a label
   (TP / FP / FN, `labels.match_points`, total at the end), and writes annotated images
   to `tests/out/` (target circles with `label:count` in green). `--net` also scores
   the network and writes `*_net.jpg`; most samples are training photos, so those
   scores are optimistic (see §10.4 for unseen-photo scores). Run it after any change
   to `detection.py` and look at the pictures; counts alone can hide regressions.
2. A scripted Tk smoke test was used during development (load the card → scale →
   analyse → add / move / remove a shot → delete an empty and a full group → new and
   rename → save → records table → average → load record and compare → distance
   clustering on `3.jpg` → load a migrated v1 record). It is not committed; `App`
   methods can be driven directly as long as a display is available
   (`app.after(..., run); app.mainloop()`), with `simpledialog.askfloat/askstring`,
   `messagebox.askyesno/showinfo` and `ScaleDialog` patched via `unittest.mock`. Test
   the database layer on a *copy* of `targets.db` (the file in the repository holds
   real records).

When printing the aggregate text on a Windows console set `PYTHONIOENCODING=utf-8`
(it contains `±` and `σ`).

## 7. Expected behaviour on the target styles

| Style | Example | Hole appearance | Rules that fire |
|---|---|---|---|
| Solid black disc, white ring lines and digits inside (25 m precision pistol, ISSF, black Ø 200 mm) | `samples/1-3.jpg` | bright torn paper on black; holes outside the disc are grey on white | inside: bright > 0.65 paper; outside: midtone |
| Ring target, white centre, thin black rings, wide black bands, several diagrams per card (10 m air rifle/pistol) | `samples/IMG_58xx.jpeg` | grey backing (gray ≈ 0.65 paper) or wood backing (orange, sat 120–180); also holes on the blank paper between diagrams | midtone and coloured |
| Card of 25 small ring diagrams (.22 "Hunter" card, 5 × 5 numbered cells plus margin diagrams), one group per cell | `samples/fe341a73-….jpg` | wood backing, from bright orange to nearly black in the phone shadow; black backing in a few cells | coloured, midtone, `dark_wood` exclusion; `find_targets` for the groups |

Measured levels on the samples (after the 5 px median): paper 209–235, ink 15–60
(up to 125 in a lit corner of the disc), grey backing ≈ 150, wood 108–150 gray with
hue ≈ 16/180, shadowed warm paper sat 75–90.

## 8. Sample set and tuning history

| File | Content | Notes |
|---|---|---|
| `samples/1.jpg` | .45, 15 m, solid disc | large torn region (≈ 6 shots), one hole on the paper below the disc |
| `samples/2.jpg` | .45, 10 m, solid disc | two big torn regions, few isolated holes |
| `samples/3.jpg` | SP01, 10 m, solid disc | ≈ 15 clean holes, two small clusters |
| `samples/IMG_5857–5863.jpeg` | 10 m air targets, iPhone 3024×4032 | grey or wood backing, hard phone shadows, crumpled plastic sleeve, coloured pasters on IMG_5860 |
| `samples/fe341a73-….jpg` | .22 Hunter card, 25 cells, 1440×1920 | the multi-group sample: ≈ 90 shots in 23 cells, 2–8 per cell; half the card in a hard shadow; two cells with a black backing (undetectable by colour) |
| 38 other files (stock photos, product shots, drawings; JPEG, JFIF, WebP, AVIF, 192–1820 px) | splatter / reactive targets (yellow halo, several per sheet), silhouettes (black or white print), coloured rings (archery, red, orange), vector drawings, a target on a tree trunk | added for the network (§10). The colour rules were never tuned on them and mostly fail (F1 0.14 on them, §10.4). `images (1).jfif` is a half-size copy of `51idwtDcGnL…jpg`, `images (4).jfif` a crop of `shooting-target-with-bullet-holes-conceptual…jpg` (same `group` in their labels). Three are not labelled (`exclude`): `depositphotos_…` (≈ 150 overlapping holes), `paper-target-with-gun-shot-holes-AA632W` (≈ 80), `images (2).jfif` (192×128 thumbnail) |

Failure modes fixed during tuning, in order: (1) ring lines and digits detected →
opening; (2) lit black print detected as midtone → disc override; (3) 4032 px photos
→ work resolution 1920; (4) strips along hard shadow edges → closing-based local
level; (5) chromatic fringes and warm-lit paper as "coloured" → `gray < 0.9 paper`
and `sat > 95`; (6) a stray tiny blob shrinking the reference area so every hole was
split → median of the smaller half plus UI hint; (7) thin shadow arcs → solidity.

(8) dark wood in shadow classified as ink and therefore invisible → `dark_wood`
exclusion (§4.2); on IMG_5860–5863 this also joined the previously fragmented torn
regions into whole blobs (blob count 21–27 → 5–11 with the same or fewer impacts).

Known leftovers: a thin arc at a shadow boundary on IMG_5861; brightly lit wood
(> 0.9 paper) partly missed; nearly black wood (chroma < 40) and black backings missed;
without a scale, holes may still be over-split; coloured pasters are detected as
impacts by design (they usually cover real holes); three of the 30 circles of the
Hunter card are not found (band broken by a grid line or pencil marks).

## 9. Extension points

- **Non-circular targets** (silhouettes, squares, IPSC-style cardboard): the network
  (§10) and the background colour with its print rule (§4.10) do not depend on the
  target shape. Three older parts do: `find_disc` and the auto-scale button (use the
  2-point tool), the solid-disc override of §4.5 (holes in black print of another shape
  are not found by the colour rules: use the network), and `find_targets` /
  `assign_groups` (no circle gives one group G1: use *Auto-group: distance* or manual
  groups). Generalising `_circular_ink_shapes` to any closed print outline (contours
  of the thick ink areas, assignment with `cv2.pointPolygonTest`) would lift the last
  two.
- **Per-target-type presets**: store parameter sets (paper ratio, saturation, kernel,
  hole Ø, disc diameter) and expose a combobox in `_build_analyze`.
- **Perspective correction**: detect the outer circle as an ellipse (`cv2.fitEllipse`
  on the disc contour) and warp to a circle before detection; the scale tool would
  then be exact in every direction.
- **Scoring**: ring radii are known for ISSF targets; with each `Group`'s circle and
  the scale, each impact's ring value is `floor` of its distance to the group centre.
- **More display settings**: add a key to `drawstyle.ELEMENTS` and `default_lines()`
  and use it through `App._line`; the Parameters tab builds its rows from `ELEMENTS`.
- **Schema changes**: bump `SCHEMA_VERSION`, add a `_migrate_vN()` step in
  `Database._migrate()` guarded by `PRAGMA user_version`; existing users have
  `targets.db` files. Keep every step idempotent.
- **Group labels from the print**: the cell numbers of a card could be read (template
  matching of the digits) to label the groups 1–25 instead of G1–G27 in reading order.
- **Tests**: `detection.py` and `stats.py` are display-free; pytest cases can assert
  impact counts within a tolerance on `samples/` and exact values on synthetic images.

## 10. Hole detection network (`hole_model.py`, `training/`)

### 10.1 Purpose

The colour rules of §4 encode what a hole looks like on the targets they were tuned
on. On other targets they fail: on the 45 labelled photos they score precision 0.14,
recall 0.22 (`tests/check_samples.py`). A convolutional network learns the *look* of a
hole from examples (round outline, torn edge, contrast with whatever surrounds it)
whatever the colours, so splatter targets, silhouettes, coloured rings, black holes on
white or white holes on black are all the same problem to it. It is offered as a
second detector, not a replacement: the colour rules remain the default and still
split large torn regions better (§10.4).

### 10.2 Labels (`labels.py`, `training/labels/`)

One JSON file per photo, format in the docstring of `labels.py`: hole centres in
original pixels with an optional radius (`r`, else `r_default`), `ignore` boxes, an
optional `exclude` reason and `group`. `match_points(pred, label)` scores a detection:
greedy one-to-one matching, closest pairs first, a prediction matches a hole within
`max(3 px, r)` of its centre, predictions inside an ignore box are not counted;
`prf(tp, fp, fn)` gives precision, recall, F1. Both are used by the training script,
`tests/check_samples.py` and the evaluation numbers below.

How the bundled labels were made (November 2026, by the AI assistant, from zoomed crops
of each photo with a coordinate grid; positions are good to a few pixels at the crop
scale, not perfect):

- 36 stock / product / drawing photos and `1.jpg`, `2.jpg`, `3.jpg`: every visible hole
  placed by hand; in torn clusters one point per shot that can be told apart (a guess
  where they overlap);
- `IMG_5857`, `IMG_5858`: the colour-rule output, pruned of false hits (plastic sleeve,
  fragments). These two labels therefore favour the colour rules;
- `IMG_5859`–`IMG_5863`: no hole labels, the torn regions in `ignore` boxes. They teach
  the network what is *not* a hole: print, ring lines, hard phone shadows, the plastic
  sleeve, pasters;
- the Hunter card: 38 isolated holes, the 30 torn cells in `ignore` boxes;
- `group`: `images (1).jfif` is `51idwtDcGnL…` at half size, `images (4).jfif` a crop
  of `shooting-target-with-bullet-holes-conceptual…`, and the seven iPhone photos show
  the same card; a group stays on one side of every train / validation split;
- `exclude`: `depositphotos_…` (≈ 150 overlapping holes), `paper-target-with-gun-shot-
  holes-AA632W` (≈ 80), `images (2).jfif` (192 × 128 thumbnail).

Total: 45 photos, 605 holes. The stock / product / drawing photos are **not in the
repository** (third-party copyright; `samples/*` is git-ignored except the original
tuning photos): their labels are kept, and training skips a label whose photo is missing. *Save as training label* in the app writes the same
format (§5); `ignore` boxes and `group` are added by editing the JSON.

### 10.3 Model and training (`training/train_hole_model.py`)

- **Input**: the photo in BGR 0–255, resized so its longest side is `NORM_SIDE = 1024`
  (holes then measure about 4–20 px in radius), padded to a multiple of 32. The
  normalisation `(x − 128) / 64` is inside the network, so the ONNX file takes raw
  pixels.
- **Network** (`HoleNetTorch`): a U-Net. Stem 3 × 3 conv stride 2 (16 channels), five
  encoder blocks of two 3 × 3 conv + batch-norm + ReLU (16, 32, 64, 96, 128 channels)
  with 2 × 2 max-pooling between them, a decoder of nearest-neighbour upsampling,
  concatenation with the encoder block and the same double conv, then a 3 × 3 conv and
  a 1 × 1 conv to one channel of logits at half the input resolution (`STRIDE = 2`).
  0.93 M weights, receptive field large enough for a hole and its surroundings. ONNX
  file 3.7 MB, operators Conv, MaxPool, Resize, Concat, Relu, Sub, Div only.
- **Target**: a Gaussian per hole on the half-resolution grid, σ = clip(0.3 r, 1.5, 6)
  input pixels / 2, the pixel holding the centre set to 1. **Loss**: the CenterNet
  focal loss (positives = pixels at 1, negatives damped by (1 − target)⁴), normalised
  by the number of positives; pixels inside `ignore` boxes have weight 0. Output bias
  initialised to −2.19 (prior 0.1).
- **Data**: random 320 × 320 crops, batch 16, built by 8 threads (OpenCV releases the
  GIL). A photo is drawn with weight √(holes + 4); 65 % of the crops are centred near a
  labelled hole. Geometry in one `warpAffine` from a small pyramid of the photo: zoom
  × 0.55–1.8 (log-uniform) around the 1024 normalisation, rotation 0–360°, mirror,
  border filled with black, white or a random colour. Photometric: HSV jitter (hue ±15,
  saturation × 0.65–1.35, value × 0.7–1.25, p 0.5), grey (p 0.15), gamma 0.7–1.45
  (p 0.3), a straight soft or hard shadow darkening one side to 45–85 % (p 0.3), blur
  (p 0.25), noise (p 0.25), JPEG quality 35–90 (p 0.3).
- **Optimiser**: AdamW, lr 2e-3, weight decay 1e-4, 300 warm-up iterations then
  cosine decay, 8000 iterations, bfloat16 autocast on the GPU. About 7 minutes per
  model on an RTX 5060 (Python 3.11, PyTorch 2.14 + CUDA 13); hours on a CPU.
- **Runs**: by default phase 1 trains on a split of the photos (by group, ≈ 20 % of the
  holes held out), scores the held-out photos for thresholds 0.10–0.75 and the colour
  rules on the same photos, then phase 2 trains on all photos and exports. `--cv K`
  runs a K-fold cross-validation instead (folds of whole groups balanced by hole
  count); `--update-meta` writes its threshold and scores into `hole_net.json`.
- **Export**: `torch.onnx.export` (TorchScript exporter, opset 17, dynamic height and
  width), then the heat map of one photo is computed by OpenCV DNN and compared with
  PyTorch (max difference 4 × 10⁻⁷ with OpenCV 5.1; 5.0, used by the app, gives the same
  output as 5.1 to the last bit).

### 10.4 Results

Scores against the labels (§10.2), a hit within one hole radius of a labelled centre. **Cross-validation** (4 folds of whole groups, 8000 iterations per model): every photo is scored by a model that never saw it, so this is the best estimate of the behaviour on new photos. The colour rules run with their defaults and no scale. Threshold chosen on the pooled results: **0.3** (stored in `hole_net.json`).

| Photos | n | holes | network TP / FP / FN | precision / recall / F1 | colour rules TP / FP / FN | precision / recall / F1 |
|---|---|---|---|---|---|---|
| all | 45 | 605 | 423 / 45 / 182 | 0.90 / 0.70 / 0.79 | 132 / 807 / 473 | 0.14 / 0.22 / 0.17 |
| original tuning photos (§8) | 11 | 129 | 33 / 16 / 96 | 0.67 / 0.26 / 0.37 | 85 / 76 / 44 | 0.53 / 0.66 / 0.59 |
| stock, product and drawing photos | 34 | 476 | 390 / 29 / 86 | 0.93 / 0.82 / 0.87 | 47 / 731 / 429 | 0.06 / 0.10 / 0.07 |

Per photo where the two detectors differ most (network vs colour rules, TP / FP / FN):

| Photo | holes | network | colour rules |
|---|---|---|---|
| `IMG_5858.jpeg` | 31 | 2 / 3 / 29 | 31 / 8 / 0 |
| `IMG_5857.jpeg` | 19 | 1 / 3 / 18 | 19 / 20 / 0 |
| `2.jpg` | 11 | 2 / 2 / 9 | 5 / 12 / 6 |
| `1.jpg` | 15 | 8 / 1 / 7 | 12 / 11 / 3 |
| `fe341a73-bda5-48af-9d48-90d44ce664cc.jpg` | 38 | 10 / 4 / 28 | 8 / 14 / 30 |
| `3.jpg` | 15 | 10 / 0 / 5 | 10 / 7 / 5 |
| `stock-photo-image-of-shooting-target-with-bullet` | 16 | 12 / 3 / 4 | 5 / 43 / 11 |
| `1000_F_555000618_3RfWOZ8t3ZqLdEhGbxgijkEPN7CZh3N` | 8 | 4 / 0 / 4 | 0 / 1 / 8 |
| `bullet-holes-in-shooting-target-JOHF05345.jpg` | 13 | 10 / 4 / 3 | 1 / 21 / 12 |
| `shooting-target-on-tree-trunk-with-bullet-holes-` | 12 | 7 / 1 / 5 | 0 / 105 / 12 |
| `stock-photo--target-with-bullet-holes-1491283967` | 7 | 6 / 0 / 1 | 7 / 58 / 0 |
| `images (3).jfif` | 9 | 8 / 4 / 1 | 0 / 0 / 9 |
| `shot-target-human-silhouette-with-bullet-circula` | 25 | 23 / 0 / 2 | 6 / 31 / 19 |
| `images (4).jfif` | 12 | 8 / 0 / 4 | 0 / 0 / 12 |
| `images.jfif` | 4 | 4 / 0 / 0 | 2 / 14 / 2 |
| `target-27267845.webp` | 9 | 6 / 0 / 3 | 0 / 6 / 9 |
| `51idwtDcGnL._AC_UF894,1000_QL80_.jpg` | 35 | 35 / 1 / 0 | 5 / 20 / 30 |
| `gettyimages-174091422-612x612.jpg` | 10 | 8 / 1 / 2 | 0 / 1 / 10 |
| `human-target-bullet-holes-gun-firing-range-shape` | 8 | 6 / 0 / 2 | 0 / 0 / 8 |
| `Target-Gun-with-Bullet-Holes-Vector-Graphics-238` | 17 | 14 / 0 / 3 | 1 / 35 / 16 |
| `shooting-range-target-bullet-holes-260nw-2268053` | 9 | 8 / 1 / 1 | 0 / 0 / 9 |
| `51JP5apBD1L._AC_SY300_SX300_QL70_ML2_.avif` | 21 | 21 / 1 / 0 | 1 / 16 / 20 |
| `images (1).jfif` | 35 | 35 / 0 / 0 | 1 / 6 / 34 |
| `practice-target-bullet-holes-14605844.webp` | 11 | 11 / 1 / 0 | 0 / 5 / 11 |
| `shooting-range-gun-target-bullet-260nw-573036892` | 31 | 29 / 0 / 2 | 0 / 0 / 31 |
| `81biofuEVVL._AC_SL1500_.avif` | 21 | 20 / 0 / 1 | 0 / 142 / 21 |
| `81LsvBJLCSL._AC_SL1500_.avif` | 44 | 44 / 0 / 0 | 0 / 7 / 44 |
| `81Tt0sK4dvL._AC_SL1500_.avif` | 25 | 25 / 0 / 0 | 0 / 27 / 25 |
| `gun-target-bullet-holes-vector-illustration-succ` | 11 | 11 / 0 / 0 | 0 / 0 / 11 |
| `istockphoto-163650165-612x612.jpg` | 10 | 10 / 0 / 0 | 0 / 0 / 10 |

Threshold trade-off (cross-validation, all photos): 0.15: P 0.67 R 0.78, 0.25: P 0.85 R 0.73, 0.35: P 0.93 R 0.67, 0.45: P 0.96 R 0.61, 0.55: P 0.98 R 0.54.

Earlier single split of the shipped training run (`training/report.json`, 9 held-out photos, 121 holes): network TP 98 FP 8 FN 23 (F1 0.86), colour rules TP 26 FP 230 FN 95 (F1 0.14). The shipped model, trained on all photos, scores F1 0.91 on them (`tests/check_samples.py --net`), which only shows it learned the labels: use the cross-validation figures.

Reading: the network is far better on targets the colour rules were not tuned for
(splatter, silhouettes, drawings, coloured rings, black holes on white: F1 0.87 against
0.07). On the original tuning photos the **colour rules remain better** (0.59 against
0.37): those photos are dominated by large torn regions, which the rules split by size
while the network finds only the isolated holes; when the seven phone photos of the 10 m
card (one group) were held out, the network had seen no photo of that card at all; and
the IMG_5857/5858 labels come from the colour rules. Hence *Colour rules* stays the
default; labelling more of these photos (§10.6) is what would close the gap.

### 10.5 Inference (`hole_model.py`)

`HoleNet(path)` loads the ONNX file with `cv2.dnn.readNet` and the threshold from the
JSON next to it. `detect(image_bgr, hole_radius_px=None, threshold=None)`:

1. `work_scale`: `NORM_SIDE / longest side`; when the hole radius is known (scale and
   *Hole core Ø* set in the UI) the factor is changed so that radius falls in
   `HINT_R_RANGE = (5, 18)` normalised px, the range of the training; the input side is
   capped at `MAX_INPUT_SIDE = 2560`.
2. `prepare_input`: resize (`INTER_AREA` down, `INTER_LINEAR` up), pad right and
   bottom by replication to a multiple of 32, NCHW float32.
3. Forward pass, sigmoid, crop the padding.
4. `decode_peaks`: pixels equal to their 3 × 3 maximum and above the threshold,
   strongest first, a peak within 1.5 heat-map pixels of a stronger one dropped
   (plateaus), position refined by the weighted centroid of the 3 × 3 window.
5. `heat_to_image`: heat-map index → input index (`2 j + 0.5`) → original index,
   undoing the half-pixel convention of `cv2.resize`.

`HoleNet.__call__(image, hole_radius_px)` is the interface `detect_impacts` expects
(§4.11). About 90–180 ms per photo on a desktop CPU, whatever the photo size.

### 10.6 Retraining

```powershell
py -3.11 -m venv .venv-train
.venv-train\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU; omit for CPU
.venv-train\Scripts\python -m pip install -r requirements-train.txt
.venv-train\Scripts\python training\train_hole_model.py                    # validation split + final model (≈ 15 min on GPU)
.venv-train\Scripts\python training\train_hole_model.py --cv 4 --update-meta   # honest scores + threshold (≈ 30 min)
.venv\Scripts\python tests\check_samples.py --net                          # look at tests\out\*_net.jpg
```

Then commit `target_analyzer/models/hole_net.onnx`, `hole_net.json`,
`training/report.json`, `training/report_cv.json` and the new labels (not the photos in
`training/images/`, which are git-ignored). Labels from the app have no `ignore`
boxes: for a torn region where shots cannot be placed, add a box by hand rather than
guessing points, and give photos of the same target a common `group`. The training
script reads every label in `training/labels/` and skips `exclude`d ones; a label whose
width and height differ from the photo stops it (the photo was replaced or rotated).

### 10.7 Limitations

- **Few labels, mostly stock photos**: 45 photos, of which 11 are the original tuning
  photos of §8 (3 solid-disc photos, 7 phone photos of one 10 m card, the Hunter card)
  and only 6 of those have hole labels. More labelled photos of the targets actually shot is the
  single most effective improvement (§10.6).
- **Torn regions**: the network sees holes, not shots. A large tear gives fewer peaks
  than shots, or several peaks on one tear (140 hits for ≈ 90 shots on the Hunter
  card). *Shots fired* keeps the strongest hits in network mode (§4.11); otherwise use
  the colour rules for that photo or correct by hand.
- **Labels are approximate** (§10.2), and those of `IMG_5857`/`IMG_5858` come from the
  colour rules.
- Coloured pasters over holes are not in the training data: the network does not
  report them (the colour rules do, by design).
- The background colours (§4.10) are not used by the network.
- No radius per hole: impacts from the network get the single-hole area of the photo.
