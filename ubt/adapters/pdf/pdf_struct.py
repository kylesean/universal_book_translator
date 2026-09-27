"""pikepdf-backed structural reads: page count, geometry, content-stream ops, resources.

Deliberately pikepdf rather than pdf_oxide, even though pdf_oxide is now the
render/text engine: pikepdf resolves *inherited* page attributes
(``/MediaBox``/``/Rotate`` on the Pages tree node) exactly like pypdf did,
while pdf_oxide 0.3.78 silently defaults an inherited ``/MediaBox`` to
US-Letter (verified against a hand-built inherited-geometry PDF). Geometry
feeds correctness gates (artifact parity, out-of-bounds), so it stays on the
engine with correct inheritance semantics. Raw content-stream operator counts
(`:func:`count_ops``) reproduce pypdf's ``get_contents().operations`` counts
exactly (147/147 pages on the test corpus), which keeps the calibrated
``PROFILE_VECTOR_PATH_OPS`` knob valid across this migration.

This module replaces every runtime ``pypdf`` use; pypdf survives only as a
dev-extra fixture writer for tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import pikepdf
from pikepdf import Dictionary, Name, Page, Pdf

logger = logging.getLogger(__name__)

__all__ = [
    "content_stream_bytes",
    "count_ops",
    "open_pdf",
    "page_box",
    "page_boxes",
    "page_count",
    "page_size",
    "page_sizes",
    "resource_font_count",
    "resource_image_count",
]


def open_pdf(path: Path | str) -> Pdf:
    """Open a PDF for structural reads. Raises pikepdf errors on bad input."""
    return pikepdf.open(str(path))


def _box_size(box: Any) -> tuple[float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return abs(x1 - x0), abs(y1 - y0)


def page_count(path: Path | str) -> int:
    """Number of pages; raises on unreadable input (callers keep their own fallback)."""
    with pikepdf.open(str(path)) as pdf:
        return len(pdf.pages)


def page_size(page: Page) -> tuple[float, float]:
    """(width_pt, height_pt) of one page, inheritance-resolved."""
    return _box_size(page.MediaBox)


def page_sizes(path: Path | str) -> dict[int, tuple[float, float]]:
    """1-indexed ``{page_no: (width_pt, height_pt)}`` for every page."""
    with pikepdf.open(str(path)) as pdf:
        return {idx: page_size(page) for idx, page in enumerate(pdf.pages, start=1)}


def page_box(page: Page) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) MediaBox of one page, inheritance-resolved.

    Unlike :func:`page_size` this keeps the origin, which a bounds check needs:
    a MediaBox of ``[10 10 610 810]`` is the same 600x800 size as ``[0 0 600 800]``
    but its visible right edge is at x=610, not x=600.
    """
    x0, y0, x1, y1 = (float(v) for v in page.MediaBox)
    return x0, y0, x1, y1


def page_boxes(path: Path | str) -> dict[int, tuple[float, float, float, float]]:
    """1-indexed ``{page_no: (x0, y0, x1, y1)}`` MediaBoxes for every page."""
    with pikepdf.open(str(path)) as pdf:
        return {idx: page_box(page) for idx, page in enumerate(pdf.pages, start=1)}


def count_ops(page: Page, opnames: Iterable[str]) -> int:
    """Count content-stream operators whose name is in ``opnames``.

    Semantics equal pypdf's per-page raw operator count (no Form-XObject
    recursion), so thresholds calibrated against pypdf carry over unchanged.
    """
    wanted = frozenset(opnames)
    n = 0
    instructions = cast("list[tuple[Sequence[Any], Any]]", list(pikepdf.parse_content_stream(page)))
    for _operands, operator in instructions:
        if str(operator) in wanted:
            n += 1
    return n


def content_stream_bytes(page: Page) -> bytes:
    """Decoded (Flate-expanded) page content stream(s), concatenated."""
    raw = page.get("/Contents")
    if raw is None:
        return b""
    # pikepdf returns a ``pikepdf.Array`` for an array ``/Contents`` — NOT a
    # Python ``list`` — so an ``isinstance(raw, list)`` test fell through to the
    # single-object branch, where ``read_bytes()`` raised on the Array and the
    # broad ``except`` below turned every such page into ``b""``.
    objs = list(raw) if isinstance(raw, (pikepdf.Array, list)) else [raw]
    chunks: list[bytes] = []
    for obj in objs:
        try:
            resolved: Any = obj
            if hasattr(resolved, "get_object"):
                resolved = resolved.get_object()
            chunks.append(bytes(resolved.read_bytes()))
        except Exception:  # one broken stream must not lose the page
            continue
    return b"".join(chunks)


def _resources(page: Page) -> Dictionary:
    res = page.get("/Resources")
    return res if isinstance(res, Dictionary) else Dictionary()


def resource_image_count(page: Page) -> int:
    """Images declared in the page resources (``/XObject`` subtype ``/Image``)."""
    xobjects = _resources(page).get("/XObject")
    if not isinstance(xobjects, Dictionary):
        return 0
    n = 0
    # pikepdf's stubs do not declare Dictionary.values() (dynamic proxy); it
    # exists at runtime, hence the cast.
    for ref in cast("dict[Any, Any]", xobjects).values():
        try:
            obj: Any = ref
            if hasattr(obj, "get_object"):
                obj = obj.get_object()
            if isinstance(obj, Dictionary) and obj.get(Name("/Subtype")) == Name("/Image"):
                n += 1
        except Exception:  # unresolvable entry is not an image
            continue
    return n


def resource_font_count(page: Page) -> int:
    """Fonts declared in the page resources (``/Font`` dictionary size)."""
    fonts = _resources(page).get("/Font")
    return len(fonts) if isinstance(fonts, Dictionary) else 0
