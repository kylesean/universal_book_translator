"""Page transcription: rendered image -> anchored IRBlocks.

Fail-closed seams throughout: unknown driver (KeyError), unmeasured
geometry on a textless page (ValueError), and empty transcripts (no blocks)
all surface loudly instead of injecting guess blocks into the ledger.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from pathlib import Path

from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
from ubt.adapters.pdf.vlm.anchor import AnchorStats, anchor_transcript
from ubt.adapters.pdf.vlm.registry import get_driver
from ubt.adapters.pdf.vlm.types import VlmDriver
from ubt.core.ir.models import (
    BlockProvenance,
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)

logger = logging.getLogger(__name__)

#: Four-tier fallback tier vocabulary (``UBTConfig.ocr_fallback``). The env
#: read below is the direct-caller fallback (tests, library use) for a caller
#: that arrives without a config handle — the resolved config always wins
#: through ``AdapterRuntimeConfig.ocr_fallback``.
FALLBACK_ENV_VAR = "UBT_OCR_FALLBACK"


class VlmFallbackMode(StrEnum):
    """Fallback tiers for VLM ingestion."""

    OFF = "off"
    MISSING = "missing"
    WEAK = "weak"
    ALL = "all"


def get_fallback_mode(fallback: str | None = None) -> VlmFallbackMode:
    """Resolve the fallback tier: config value first, env for direct callers.

    ``off`` stays the default on every path: a scanned page the OCR channel
    cannot read must reject the book, not silently ship a partial read.
    """
    raw = (fallback or os.environ.get(FALLBACK_ENV_VAR, "")).strip().lower()
    if raw in ("1", "true", "missing"):
        return VlmFallbackMode.MISSING
    if raw in ("weak", "degraded"):
        return VlmFallbackMode.WEAK
    if raw in ("all", "full"):
        return VlmFallbackMode.ALL
    return VlmFallbackMode.OFF


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 10.0


def order_lines_by_column(
    boxes: list[tuple[float, float, float, float]],
    page_width: float,
) -> list[int]:
    """Reading-order permutation of line indices for a column-aware page.

    A VLM driver recognizes text in its own order — typically top-down across
    the whole page — which interleaves the columns of a multi-column scan
    (line 1 of column 1, line 1 of column 2, line 2 of column 1, …). The pdfium
    path already owns a gutter detector, so reuse it here rather than keeping a
    second column heuristic. A single-column page (or an unreadable width) keeps
    its order.
    """
    if len(boxes) <= 1 or page_width <= 0:
        return list(range(len(boxes)))
    from ubt.adapters.pdf.textgeom import LineBox, column_order

    line_boxes = [LineBox("", box) for box in boxes]
    ordered = column_order(line_boxes, page_width)
    position = {id(lb): i for i, lb in enumerate(line_boxes)}
    return [position[id(lb)] for lb in ordered]


def group_lines_to_paragraphs(
    lines: list[tuple[str, tuple[float, float, float, float]]],
    gap_mult: float = 1.5,
) -> list[list[int]]:
    """Group line indices into paragraphs by vertical rhythm, in the given order.

    ``lines`` must already be in reading order (see :func:`order_lines_by_column`):
    grouping only decides where one paragraph ends and the next begins. Without
    it, a dense scan page yields hundreds of line-blocks and the LLM stages grind
    for hours. Same-column overlap + gap ≤ mult*median line height joins;
    anything else breaks. Pure function.
    """
    if not lines:
        return []
    heights = [b[3] - b[1] for _, b in lines]
    med = _median(heights)
    groups: list[list[int]] = []
    current: list[int] = []
    prev_box: tuple[float, float, float, float] | None = None
    for i, (_text, box) in enumerate(lines):
        if prev_box is not None:
            gap = prev_box[1] - box[3]
            x_overlap = min(prev_box[2], box[2]) - max(prev_box[0], box[0])
            same_col = x_overlap > 0.3 * min(prev_box[2] - prev_box[0], box[2] - box[0])
            if not (same_col and 0 <= gap <= gap_mult * med):
                groups.append(current)
                current = []
        current.append(i)
        prev_box = box
    if current:
        groups.append(current)
    return groups


def transcribe_page_to_blocks(
    pdf_path: Path,
    page_no: int,
    id_prefix: str = "pdf_main",
    start_index: int = 1,
    driver_name: str | None = None,
    scale: float = 2.0,
    driver: VlmDriver | None = None,
) -> tuple[list[IRBlock], AnchorStats]:
    """Transcribe one PDF page into IRBlocks (proofread or recognition mode).

    ``start_index`` seeds block numbering; callers own spine order.
    """
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            if page_no < 1 or page_no > len(pdf):
                raise IndexError(f"Page number {page_no} out of bounds (1..{len(pdf)})")
            page = pdf[page_no - 1]
            try:
                # ``get_width``/``get_height`` are the DISPLAYED size (the
                # page's /Rotate already applied), matching what
                # ``render`` produces: the driver gets a bitmap that looks like
                # the page as a reader sees it. The text layer, by contrast,
                # reports unrotated user-space rects -- pdfium's text rects
                # ignore /Rotate, as do the render compositor's boxes. The
                # driver is told the rotation so it can bring its boxes into
                # that frame instead of handing back a sideways page.
                width, height = float(page.get_width()), float(page.get_height())
                rotation = int(page.get_rotation())
                try:
                    cropbox = page.get_cropbox()
                    origin_x = float(cropbox[0])
                    origin_y = float(cropbox[1])
                except Exception:
                    origin_x = 0.0
                    origin_y = 0.0
                image = page.render(scale=scale).to_pil().convert("RGB")
                textpage = page.get_textpage()
                try:
                    pdfium_lines = _harvest_text_lines(textpage)
                finally:
                    textpage.close()
            finally:
                page.close()
        finally:
            pdf.close()

    if driver is None:
        driver = get_driver(driver_name)
    try:
        transcript = driver.recognize(
            image, (width, height), scale, rotation, origin_pt=(origin_x, origin_y)
        )
    except TypeError:
        transcript = driver.recognize(image, (width, height), scale, rotation)
    anchored, stats = anchor_transcript(
        pdfium_lines, transcript, (width, height), rotation, origin_pt=(origin_x, origin_y)
    )
    logger.info(
        "vlm p%d via %s: %d lines matched=%d vlm_only=%d pdfium_only=%d",
        page_no,
        transcript.engine,
        len(anchored),
        stats.matched,
        stats.vlm_only,
        stats.pdfium_only,
    )
    blocks: list[IRBlock] = []
    # Column-aware reading order first: a two-column scan would otherwise be
    # read line-by-line across the gutter. Then paragraph grouping: hundreds of
    # line-blocks would grind the LLM stages for hours, and rhythm-joined
    # paragraphs translate AND align better.
    anchored = [anchored[i] for i in order_lines_by_column([a.box for a in anchored], width)]
    groups = group_lines_to_paragraphs([(a.text, a.box) for a in anchored])
    for g, member_ids in enumerate(groups):
        members = [anchored[i] for i in member_ids]
        text = "\n".join(m.text for m in members)
        if not text.strip():
            continue
        x0 = min(m.box[0] for m in members)
        y0 = min(m.box[1] for m in members)
        x1 = max(m.box[2] for m in members)
        y1 = max(m.box[3] for m in members)
        provenances = sorted({m.provenance for m in members})
        blocks.append(
            IRBlock(
                element=make_element(
                    id=f"{id_prefix}#v{page_no:03d}{g:04d}",
                    flow_id=FlowID.MAIN_STORY,
                    spine_index=start_index + g,
                    block_type=BlockType.NARRATIVE,
                    bbox=BoundingBox(page=page_no, x0=x0, y0=y0, x1=x1, y1=y1),
                    source_text=text,
                ),
                provenance=BlockProvenance(
                    parser=f"vlm:{transcript.engine}",
                    anchor_provenance="+".join(provenances),
                    needs_review=any(m.needs_review for m in members),
                    # A truncated driver transcript loses the page tail; carry
                    # the flag so the ledger/report can show it rather than
                    # trusting a half-page as complete.
                    ocr_truncated=transcript.truncated,
                    anchor_stats={
                        "matched": stats.matched,
                        "vlm_only": stats.vlm_only,
                        "pdfium_only": stats.pdfium_only,
                    },
                    vlm_lines=[{"text": m.text, "box": list(m.box)} for m in members],
                ),
            )
        )
    return blocks, stats


def _harvest_text_lines(textpage: object) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Best-effort pdfium line harvest (rect rows); empty when textless."""
    try:
        n = textpage.count_rects(0, -1)  # type: ignore[attr-defined]
    except Exception:
        return []
    lines: list[tuple[str, tuple[float, float, float, float]]] = []
    for i in range(n):
        try:
            rect = textpage.get_rect(i)  # type: ignore[attr-defined]
            text = (textpage.get_text_bounded(*rect) or "").strip()  # type: ignore[attr-defined]
        except Exception:
            continue
        if text:
            lines.append((text, (rect[0], rect[1], rect[2], rect[3])))
    return lines
