"""PDF geometry extraction for the rigid typesetter (the only I/O here).

Wraps pdfium: reading-order text rows plus image-object bounds used as hard
zone guards. Textless (scanned) pages fall back to authored line rows
(VLM-measured member lines, else bbox slices) and get a sampled background
color so painted zones can cover the raster glyphs. Pages are read once and
the result is pure data (:class:`~ubt.adapters.pdf.rigid.zones.PageFacts`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ubt.adapters.pdf.pdf_struct import page_sizes
from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.adapters.pdf.rigid.zones import PageFacts, Rect
from ubt.adapters.pdf.textgeom import LineBox, extract_lines, synthetic_vlm_lines
from ubt.core.ir.models import IRBlock
from ubt.core.policy.layout_policy import BG_SAMPLE_SCALE, STROKE_BLACK_CUTOFF

logger = logging.getLogger(__name__)

# Guard inflation: figure images get a small breathing band so a zone never
# touches their frame.
IMAGE_GUARD_PAD_PT = 3.0


def _page_rotation(pdf: object, page_no: int) -> int:
    """Return the page's effective /Rotate in degrees (0 when upright).

    pdfium returns unrotated text rects while ``get_width/height`` (and the
    Docling bboxes zones are matched against) use the rotated display frame,
    and ``pikepdf.add_overlay`` authors in content space. Populating zones from
    disagreeing frames silently misplaces or clips the translation, so the rigid
    engine cannot typeset a rotated page correctly.
    """
    page = pdf[page_no - 1]  # type: ignore[index]
    try:
        return int(page.get_rotation()) % 360
    finally:
        page.close()


@pdfium_serialized
def extract_pages(
    pdf_path: Path | str,
    page_numbers: list[int],
    blocks: Sequence[IRBlock] | None = None,
) -> dict[int, PageFacts]:
    """Read rows and image guards for the given 1-based pages."""
    import pypdfium2 as pdfium

    path = Path(pdf_path)
    by_page: dict[int, list[IRBlock]] = {}
    for block in blocks or ():
        if block.bbox is not None:
            by_page.setdefault(block.bbox.page, []).append(block)

    # pdfium's ``get_mediabox()`` does not inherit ``/MediaBox`` from the page
    # tree (it silently falls back to US-Letter), while every text rect this
    # module returns is in the page's true user space. pikepdf resolves
    # inheritance exactly (same reason ``pdf_struct`` exists), so the frame used
    # for zones and the overlay must come from here, not pdfium. Resolve all
    # pages once; failure degrades to the pdfium frame rather than aborting.
    try:
        resolved_sizes = page_sizes(path)
    except Exception:  # noqa: BLE001 - geometry fallback, never fatal
        logger.debug("pikepdf page-size resolution failed for %s", path, exc_info=True)
        resolved_sizes = {}

    facts: dict[int, PageFacts] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        total = len(pdf)
        for page_no in page_numbers:
            if page_no < 1 or page_no > total:
                continue
            rotation = _page_rotation(pdf, page_no)
            if rotation:
                # A rotated page's coordinate frames disagree with pdfium
                # text rects, so rigid cannot place it. Demote just this page to
                # source-visible (its blocks get no zone → a no_zone skip) rather
                # than aborting the whole render; the rest of the book still ships
                # rigid. An all-rotated document trips _assert_paintable upstream.
                logger.warning(
                    "rigid: skipping page %d (/Rotate %d° unsupported); its text "
                    "stays untranslated. Use render_engine='publication' to cover "
                    "rotated pages.",
                    page_no,
                    rotation,
                )
                continue
            lines, size = extract_lines(path, page_no)
            size = resolved_sizes.get(page_no, size)
            bg: str | None = None
            if not lines and by_page.get(page_no):
                lines = _textless_rows(page_no, size, by_page[page_no])
                if lines:
                    bg = _sample_page_bg(pdf, page_no)
            images: list[Rect] = []
            try:
                from pypdfium2 import PdfImage

                guard_page = pdf[page_no - 1]
                try:
                    for obj in guard_page.get_objects():
                        if isinstance(obj, PdfImage):
                            x0, y0, x1, y1 = (float(v) for v in obj.get_bounds())
                            images.append(
                                (
                                    x0 - IMAGE_GUARD_PAD_PT,
                                    y0 - IMAGE_GUARD_PAD_PT,
                                    x1 + IMAGE_GUARD_PAD_PT,
                                    y1 + IMAGE_GUARD_PAD_PT,
                                )
                            )
                finally:
                    guard_page.close()
            except Exception:  # guards are best-effort, never fatal
                logger.debug("image guard extraction failed on page %d", page_no, exc_info=True)
            facts[page_no] = PageFacts(
                page=page_no,
                width=size[0],
                height=size[1],
                lines=tuple(lines),
                images=tuple(images),
                bg=bg,
            )
    finally:
        pdf.close()
    return facts


def _textless_rows(
    page_no: int,
    size: tuple[float, float],
    blocks: Sequence[IRBlock],
) -> list[LineBox]:
    """Row evidence for a page without an extractable text layer.

    VLM-measured member lines win; otherwise the block bboxes are sliced
    proportionally to the source line count (the historical P9-A path).
    """
    rows = synthetic_vlm_lines(blocks)
    if rows:
        return rows
    from ubt.adapters.pdf.coordinate_resolver import synthesize_line_boxes_for_blocks

    logger.debug(
        "page %d has no text rows; using bbox slices for %d block(s)", page_no, len(blocks)
    )
    return synthesize_line_boxes_for_blocks(blocks, size)


def _sample_page_bg(pdf: object, page_no: int) -> str | None:
    """Median page-raster color, dark glyph pixels excluded (scan covers)."""
    try:
        page = pdf[page_no - 1]  # type: ignore[index]
        try:
            img = page.render(scale=BG_SAMPLE_SCALE).to_pil().convert("RGB")
        finally:
            page.close()
        small = img.resize((max(1, img.width // 4), max(1, img.height // 4)))
        rs: list[int] = []
        gs: list[int] = []
        bs: list[int] = []
        for pr, pg, pb in small.get_flattened_data():
            if max(pr, pg, pb) < STROKE_BLACK_CUTOFF:
                continue
            rs.append(pr)
            gs.append(pg)
            bs.append(pb)
        if not rs:
            return None
        rs.sort()
        gs.sort()
        bs.sort()
        mid = len(rs) // 2
        return f"#{rs[mid]:02x}{gs[mid]:02x}{bs[mid]:02x}"
    except Exception:  # sampling is best-effort, never fatal
        logger.debug("background sampling failed on page %d", page_no, exc_info=True)
        return None
