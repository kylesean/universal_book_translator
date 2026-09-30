"""Structural-first asset verification (Phase 1, level 1 of 2).

A reconstructed asset is only trustworthy if it survives a check. This module
implements the *cheap* level first -- structure, not pixels:

- **formula:** the delivered math is non-empty, brace-balanced and carries no
  engine error marker. (The expensive level already exists as
  :mod:`ubt.adapters.pdf.formula_witness`: a structural raster comparison against
  the source crop. Its verdicts are honoured in
  :func:`ubt.core.content.adapt.graph_from_blocks`.)
- **table:** the reconstruction did not shatter into single-character cells or
  inconsistent column counts -- the exact failure the rigid engine exists to
  prevent.

A FAIL is corruption (Axiom A), never a silent pass. ``reconcile`` records it;
the opaque source-crop fallback for tables lands in Phase 1b.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ubt.core.content.nodes import AssetKind

# A cell this short is a fragment, not content.
_SHATTER_CELL_MAX = 1
# Shatter needs *both* a clear majority of one-character cells and enough cells
# to be meaningful: a two-column table with single-char headers (H, H) is not
# shattered, while a five-column table of single characters is.
_SHATTER_CELL_SHARE = 0.5
_SHATTER_MIN_CELLS = 4
_ENGINE_ERROR_MARKERS = ("data-mjx-error", "mathjax error", "unsupported")
_MATH_ERROR_RE = re.compile(r"\b(?:error|failed)\b", re.IGNORECASE)


class StructuralVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # nothing structural to judge (no recognized table/math)


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Outcome of one structural asset check."""

    verdict: StructuralVerdict
    detail: str = ""

    @property
    def verified(self) -> bool:
        return self.verdict is StructuralVerdict.PASS

    @property
    def corrupt(self) -> bool:
        return self.verdict is StructuralVerdict.FAIL


def _balanced(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack:
                return False
            if stack.pop() != pairs[ch]:
                return False
    return not stack


def verify_formula_structure(text: str) -> VerifyResult:
    """Structural sanity of a reconstructed formula's delivered text."""
    body = (text or "").strip()
    if not body:
        return VerifyResult(StructuralVerdict.FAIL, "empty formula")
    lowered = body.lower()
    if any(marker in lowered for marker in _ENGINE_ERROR_MARKERS):
        return VerifyResult(StructuralVerdict.FAIL, "engine error marker in emitted math")
    if not _balanced(body):
        return VerifyResult(StructuralVerdict.FAIL, "unbalanced delimiters")
    return VerifyResult(StructuralVerdict.PASS, "structural")


def _cells_from_markdown(text: str) -> list[str]:
    cells: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if set(stripped) <= set("|-: "):
            continue  # separator row
        cells.extend(part.strip() for part in stripped.strip("|").split("|"))
    return cells


def _cells_from_html(text: str) -> list[str]:
    return [
        re.sub(r"<[^>]+>", "", cell).strip()
        for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", text, flags=re.IGNORECASE | re.DOTALL)
    ]


def verify_table_structure(text: str) -> VerifyResult:
    """Detect a shattered table reconstruction (single-char cells / ragged rows)."""
    body = (text or "").strip()
    if not body:
        return VerifyResult(StructuralVerdict.FAIL, "empty table")
    cells = _cells_from_markdown(body) or _cells_from_html(body)
    if not cells:
        # Not a recognized table grammar: cannot judge structurally. Not a
        # corruption finding -- the pixel level (Phase 1b) covers it.
        return VerifyResult(StructuralVerdict.SKIP, "unrecognized table grammar")
    shattered = sum(1 for c in cells if len(c) <= _SHATTER_CELL_MAX)
    if shattered >= _SHATTER_MIN_CELLS and shattered / len(cells) > _SHATTER_CELL_SHARE:
        return VerifyResult(
            StructuralVerdict.FAIL,
            f"{shattered}/{len(cells)} cells are single-character (shattered)",
        )
    return VerifyResult(StructuralVerdict.PASS, "structural")


def verify_asset_structure(asset_kind: AssetKind, text: str) -> VerifyResult:
    """Dispatch the structural check for a reconstructed asset."""
    if asset_kind is AssetKind.FORMULA:
        return verify_formula_structure(text)
    if asset_kind is AssetKind.TABLE:
        return verify_table_structure(text)
    return VerifyResult(StructuralVerdict.SKIP, "no structural policy")


__all__ = [
    "StructuralVerdict",
    "VerifyResult",
    "verify_asset_structure",
    "verify_formula_structure",
    "verify_table_structure",
]
