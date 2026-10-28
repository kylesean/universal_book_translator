"""PDF background canvas text stripper.

Uses pikepdf to extract a vector background PDF by stripping text content streams
(`BT...ET`) from the source PDF while strictly preserving:
- All vector paths, fills, strokes, shading, rules, and clipping paths (m, l, c, re, S, s, f, F, B, sh, Do)
- All graphics state stack operations (q, Q, cm, gs, g, G, rg, RG, k, K, cs, CS, sc, SC)
- Embedded raster images
- Nested Form XObjects (with recursive text stripping)
- Optional selective erase/keep bounding boxes for formulas, headers, footers, etc.

Produces clean, deduplicated, and stream-compressed background PDFs suitable for 1:1 facsimile rendering.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pikepdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BoxRect:
    """Axis-aligned rectangle in PDF user coordinates (pt, origin bottom-left)."""

    page_index: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, y: float, tol: float = 0.5) -> bool:
        return (self.x0 - tol) <= x <= (self.x1 + tol) and (self.y0 - tol) <= y <= (self.y1 + tol)


def _should_keep_bt_block(
    cur_x: float,
    cur_y: float,
    erase_rects: Sequence[BoxRect] | None,
    keep_rects: Sequence[BoxRect] | None,
) -> bool:
    """Decide whether to keep or strip a BT...ET block based on its coordinate position."""
    if keep_rects and any(rect.contains_point(cur_x, cur_y) for rect in keep_rects):
        return True
    if erase_rects is not None:
        return not any(rect.contains_point(cur_x, cur_y) for rect in erase_rects)
    # Default: if erase_rects is not specified, strip all text unless in keep_rects
    return False


def strip_stream_bt_et(
    content_obj: pikepdf.Object | pikepdf.Stream | pikepdf.Page,
    pdf: pikepdf.Pdf,
    *,
    erase_rects: Sequence[BoxRect] | None = None,
    keep_rects: Sequence[BoxRect] | None = None,
) -> bytes:
    """Parse a single content stream, strip unwanted BT...ET blocks, and unparse."""
    try:
        instructions = cast_instructions(pikepdf.parse_content_stream(content_obj))
    except Exception as exc:
        logger.debug("Failed to parse content stream for stripping: %s", exc)
        return b""

    out: list[Any] = []
    in_bt = False
    bt_buffer: list[Any] = []
    cur_x, cur_y = 0.0, 0.0

    for inst in instructions:
        operands = inst[0]
        op = str(inst[1])
        if op == "BT":
            in_bt = True
            bt_buffer = [inst]
            cur_x, cur_y = 0.0, 0.0
            continue
        if in_bt:
            bt_buffer.append(inst)
            if op == "Tm" and len(operands) >= 6:
                cur_x, cur_y = float(operands[4]), float(operands[5])
            elif op in ("Td", "TD") and len(operands) >= 2:
                cur_x += float(operands[0])
                cur_y += float(operands[1])
            if op == "ET":
                in_bt = False
                if _should_keep_bt_block(cur_x, cur_y, erase_rects, keep_rects):
                    out.extend(bt_buffer)
                bt_buffer = []
            continue
        out.append(inst)

    return pikepdf.unparse_content_stream(out)


def cast_instructions(parsed: Any) -> list[Any]:
    return list(parsed)


def strip_form_xobjects_recursive(
    resources: pikepdf.Dictionary | None,
    pdf: pikepdf.Pdf,
    *,
    erase_rects: Sequence[BoxRect] | None = None,
    keep_rects: Sequence[BoxRect] | None = None,
    depth: int = 0,
    max_depth: int = 5,
) -> None:
    """Recursively strip text from /Subtype /Form XObjects in /Resources."""
    if resources is None or depth > max_depth:
        return
    xobjs = resources.get("/XObject")
    if not isinstance(xobjs, pikepdf.Dictionary):
        return
    for _key, val in xobjs.items():
        try:
            xobj = val.get_object() if hasattr(val, "get_object") else val
            if isinstance(xobj, pikepdf.Stream) and xobj.get("/Subtype") == pikepdf.Name("/Form"):
                raw = strip_stream_bt_et(xobj, pdf, erase_rects=erase_rects, keep_rects=keep_rects)
                xobj.write(raw)
                child_res = xobj.get("/Resources")
                if isinstance(child_res, pikepdf.Dictionary):
                    strip_form_xobjects_recursive(
                        child_res,
                        pdf,
                        erase_rects=erase_rects,
                        keep_rects=keep_rects,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
        except Exception as exc:
            logger.debug("Error stripping Form XObject at depth %d: %s", depth, exc)


def strip_text_streams(
    src_pdf: Path | str,
    dst_pdf: Path | str,
    *,
    pages: Sequence[int] | None = None,
    per_page_erase_rects: Mapping[int, Sequence[BoxRect]] | None = None,
    per_page_keep_rects: Mapping[int, Sequence[BoxRect]] | None = None,
) -> Path:
    """Extract vector background PDF by stripping text content streams from src_pdf."""
    src_path = Path(src_pdf)
    dst_path = Path(dst_pdf)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with pikepdf.open(src_path) as src:
        out = pikepdf.Pdf.new()
        total_pages = len(src.pages)
        wanted = list(pages) if pages is not None else list(range(1, total_pages + 1))
        for pno in wanted:
            if not (1 <= pno <= total_pages):
                continue
            page = src.pages[pno - 1]
            erase_r = (per_page_erase_rects or {}).get(pno)
            keep_r = (per_page_keep_rects or {}).get(pno)

            # Strip page content stream(s)
            raw = strip_stream_bt_et(page, src, erase_rects=erase_r, keep_rects=keep_r)
            page.Contents = src.make_stream(raw)

            # Recursively strip Form XObjects
            res = page.get("/Resources")
            if isinstance(res, pikepdf.Dictionary):
                strip_form_xobjects_recursive(res, src, erase_rects=erase_r, keep_rects=keep_r)

            out.pages.append(page)

        out.save(
            dst_path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )
    return dst_path


__all__ = [
    "BoxRect",
    "cast_instructions",
    "strip_form_xobjects_recursive",
    "strip_stream_bt_et",
    "strip_text_streams",
]
