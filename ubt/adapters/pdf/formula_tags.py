"""Recover original equation numbers from the source PDF text layer.

Docling's VLM formula items usually carry only the transcribed formula: the
printed equation number sits in the margin as a separate text object and gets
dropped with the rest of the page furniture. The chapter counter then prints a
position-based number (3.40) exactly where the book prints the author's tag
(A.1). The text layer still carries the tag, so the renderer reads it back
from the formula's row band — deterministic, no model.

The matcher is a pure function over ``(text, boxes)`` so it can be tested
without a PDF; ``recover_formula_tag`` is the thin pdfium adapter.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized

# Letter-prefixed tags (A.12a) and numeric chapter tags (3.11) only; the
# formula body's own parentheses ("( 2 r - 1 )") never match.
_TAG_RE = re.compile(
    r"\(\s*([A-Za-z]\s*\.\s*\d{1,2}\s*[a-z]?|\d{1,2}\s*\.\s*\d{1,2}\s*[a-z]?)\s*\)"
)
_MAX_TAG_HEIGHT_PT = 14.0
_BAND_SLACK_PT = 2.0


@lru_cache(maxsize=64)
@pdfium_serialized
def _page_text_boxes(
    pdf_path: str, page_no: int, mtime_ns: int
) -> tuple[str, tuple[tuple[float, float, float, float], ...]]:
    """Per-character text and boxes for one page (1-based page number).

    ``mtime_ns`` is part of the cache key (not otherwise used): without it a
    long-lived process would serve stale tags after the PDF is overwritten
    in place — the page profiler's cache key already covers size + mtime.
    """
    import contextlib

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        page = doc[page_no - 1]
        try:
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_range()
                boxes = tuple(
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    for box in (textpage.get_charbox(i) for i in range(textpage.count_chars()))
                )
            finally:
                with contextlib.suppress(Exception):
                    textpage.close()
        finally:
            with contextlib.suppress(Exception):
                page.close()
    finally:
        with contextlib.suppress(Exception):
            doc.close()
    return str(text), boxes


def match_tag_in_page(
    text: str,
    boxes: tuple[tuple[float, float, float, float], ...] | list[tuple[float, float, float, float]],
    y0: float,
    y1: float,
    x1: float,
) -> str | None:
    """Tag-like ``(…)`` nearest the formula's right edge, inside its row band.

    Candidates must vertically overlap ``[y0, y1]`` (the formula bbox) and be
    no taller than a printed tag; among them the one nearest the formula's
    right edge wins, which keeps a neighbouring column's tag out.
    """
    best: tuple[float, str] | None = None
    for m in _TAG_RE.finditer(text):
        start, end = m.start(), m.end() - 1
        if end >= len(boxes):
            continue
        left = min(boxes[i][0] for i in range(start, end + 1))
        bottom = min(boxes[i][1] for i in range(start, end + 1))
        top = max(boxes[i][3] for i in range(start, end + 1))
        center_y = (bottom + top) / 2.0
        if center_y < y0 - _BAND_SLACK_PT or center_y > y1 + _BAND_SLACK_PT:
            continue
        if (top - bottom) > _MAX_TAG_HEIGHT_PT:
            continue
        distance = abs(left - x1)
        tag = re.sub(r"\s+", "", m.group(1))
        if best is None or distance < best[0]:
            best = (distance, tag)
    return best[1] if best else None


def recover_formula_tag(pdf_path: Path | str, page_no: int, bbox: Any) -> str | None:
    """Original printed equation tag near a formula bbox, or None.

    Never raises: a missing PDF, an unreadable page or a layout where no tag
    can be located all return None so callers keep their previous numbering.
    """
    if bbox is None:
        return None
    try:
        mtime_ns = Path(pdf_path).stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    try:
        text, boxes = _page_text_boxes(str(pdf_path), int(page_no), mtime_ns)
    except Exception:  # recovery must never break rendering
        return None
    return match_tag_in_page(text, boxes, float(bbox.y0), float(bbox.y1), float(bbox.x1))
