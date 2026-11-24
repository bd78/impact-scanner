# Author: bd78
"""Filter patterns for the Records & averages list. Pure Python, no Tk, no SQLite.

A pattern is matched against the text of one column, ignoring case and spaces at both ends:

    (empty)        everything
    9mm            exactly "9mm"
    25             a number alone: the same number (25, 25.0, "25 m")
    *luger*        contains "luger";  9mm*  starts with;  *.jpg  ends with
    ?              any single character (IMG_58?7.jpg)
    >5  >=5  <10  <=10  =5  <>5      compare: as numbers when both sides are numbers
                                     (25, 25 m, 2,5), else as text (dates: >=2026-11-01)
    5..10                            between, inclusive (numbers or text, e.g. dates)
    !pattern                         NOT: rows that do not match the pattern
"""
from __future__ import annotations

import re
from fnmatch import fnmatchcase

OPERATORS = (">=", "<=", "<>", "!=", ">", "<", "=")
_NUMBER = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)(?![\d.,/:-])")  # not a date or a time
_PURE_NUMBER = re.compile(r"^[-+]?\d+(?:[.,]\d+)?$")


def _number(text: str) -> float | None:
    """Leading number of a text ('25 m' -> 25.0, '2,5' -> 2.5), or None."""
    m = _NUMBER.match(text)
    return float(m.group(1).replace(",", ".")) if m else None


def _key(a: str, b: str):
    """Compare as numbers when both texts start with one, else as case-folded text."""
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return na, nb
    return a.casefold(), b.casefold()


def _compare(value: str, op: str, target: str) -> bool:
    a, b = _key(value, target)
    return {">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b, "=": a == b,
            "<>": a != b, "!=": a != b}[op]


def matches(value, pattern: str) -> bool:
    """True if the column text `value` matches `pattern` (see the module doc)."""
    p = (pattern or "").strip()
    if not p:
        return True
    if p.startswith("!") and not p.startswith("!="):
        return not matches(value, p[1:])
    v = "" if value is None else str(value).strip()
    for op in OPERATORS:
        if p.startswith(op):
            return _compare(v, op, p[len(op):].strip())
    if ".." in p:
        lo, hi = (s.strip() for s in p.split("..", 1))
        return (not lo or _compare(v, ">=", lo)) and (not hi or _compare(v, "<=", hi))
    if "*" in p or "?" in p:
        # '[' would start a character set in fnmatch: take it literally
        return fnmatchcase(v.casefold(), p.casefold().replace("[", "[[]"))
    if _PURE_NUMBER.match(p):  # 25 matches 25, 25.0 and "25 m"
        return _number(v) == _number(p)
    return v.casefold() == p.casefold()


def row_matches(row: dict, patterns: dict) -> bool:
    """All non-empty patterns match their column (AND)."""
    return all(matches(row.get(col), pat) for col, pat in patterns.items())
