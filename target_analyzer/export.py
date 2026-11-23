# Author: bd78
"""Export saved records to a CSV file that Excel opens directly.

One row per group. Excel reads a CSV with the list separator and decimal mark of the
Windows regional settings (';' and ',' in France, ',' and '.' in the US), so both are
taken from the system; the file is UTF-8 with a BOM so accents survive.

No Tk, no SQLite: the caller passes plain mappings (sqlite3.Row works).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Mapping

from .stats import to_unit

FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def excel_separators() -> tuple[str, str]:
    """(list separator, decimal mark) Excel uses on this machine."""
    sep, dec = ",", "."
    if sys.platform == "win32":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(8)
            get = ctypes.windll.kernel32.GetLocaleInfoEx
            if get(None, 0x0C, buf, len(buf)):   # LOCALE_SLIST
                sep = buf.value or sep
            if get(None, 0x0E, buf, len(buf)):   # LOCALE_SDECIMAL
                dec = buf.value or dec
        except (AttributeError, OSError):
            pass
    if sep == dec:  # unusable combination: fall back to what Excel accepts anywhere
        sep = ";" if dec == "," else ","
    return sep, dec


def safe_text(value) -> str:
    """Text cell that Excel shows as typed: a leading = + - @ would start a formula."""
    s = "" if value is None else str(value)
    return "'" + s if s.startswith(FORMULA_START) else s


def number(value: float, decimal: str, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", decimal)


def photo_full_path(stored: str, db_path: str | Path) -> str:
    """Absolute path of a record's stored photo (relative paths are from the database folder)."""
    if not stored:
        return ""
    p = Path(stored)
    if not p.is_absolute():
        p = Path(db_path).resolve().parent / p
    return str(p)


HEADER = ("Record", "Date", "Photo (full path)", "Photo name", "Weapon", "Distance", "Ammo",
          "Group", "Holes", "Width ({u})", "Height ({u})")


def export_rows(sessions: Iterable[Mapping], groups_of: Mapping[int, list[Mapping]],
                db_path: str | Path, unit: str, decimal: str = ".") -> list[list[str]]:
    """Header + one row per group of each session. A record without group rows gets
    one row with its own totals."""
    rows = [[h.format(u=unit) for h in HEADER]]
    for s in sessions:
        date = s["shot_date"] or (s["created_at"] or "")[:10]
        head = [str(s["id"]), safe_text(date), safe_text(photo_full_path(s["image_path"] or "", db_path)),
                safe_text(s["source_name"]), safe_text(s["weapon"]), safe_text(s["distance"]),
                safe_text(s["ammo"])]
        groups = groups_of.get(s["id"]) or [{"label": "", "n_shots": s["n_shots"],
                                             "width_mm": s["width_mm"], "height_mm": s["height_mm"]}]
        for g in groups:
            rows.append(head + [safe_text(g["label"]), str(g["n_shots"]),
                                number(to_unit(g["width_mm"], unit), decimal),
                                number(to_unit(g["height_mm"], unit), decimal)])
    return rows


def write_csv(path: str | Path, rows: list[list[str]], sep: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, delimiter=sep, quoting=csv.QUOTE_MINIMAL).writerows(rows)
