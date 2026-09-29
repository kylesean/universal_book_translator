"""Page transcription: rendered image -> anchored IRBlocks.

Fail-closed seams throughout: unknown driver (KeyError), unmeasured
geometry on a textless page (ValueError), and empty transcripts (no blocks)
all surface loudly instead of injecting guess blocks into the ledger.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
from ubt.adapters.pdf.vlm.anchor import AnchorStats, anchor_transcript
from ubt.adapters.pdf.vlm.registry import get_driver
from ubt.adapters.pdf.vlm.types import VlmDriver
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock

logger = logging.getLogger(__name__)

#: Four-tier fallback switch: "off" (default), "missing", "weak", "all".
#: Accepts truthy strings ("1", "true") as "missing" for environment compatibility.
FALLBACK_ENV_VAR = "UBT_VLM_SCAN_FALLBACK"


class VlmFallbackMode(StrEnum):
    """Fallback tiers for VLM ingestion."""

    OFF = "off"
    MISSING = "missing"
    WEAK = "weak"
    ALL = "all"


def get_fallback_mode() -> VlmFallbackMode:
    raw = os.environ.get(FALLBACK_ENV_VAR, "").strip().lower()
    if raw in ("1", "true", "missing"):
        return VlmFallbackMode.MISSING
    if raw in ("weak", "degraded"):
        return VlmFallbackMode.WEAK
    if raw in ("all", "full"):
        return VlmFallbackMode.ALL
    return VlmFallbackMode.OFF


def fallback_enabled() -> bool:
    return get_fallback_mode() != VlmFallbackMode.OFF


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 10.0


def group_lines_to_paragraphs(
    lines: list[tuple[str, tuple[float, float, float, float]]],
    gap_mult: float = 1.5,
) -> list[list[int]]:
    """Group line indices into paragraphs by vertical rhythm.

    Without this, a dense scan page yields hundreds of line-blocks and the
    LLM stages grind for hours. Same-column overlap + gap ≤ mult*median
    line height joins; anything else breaks. Pure function.
    """
    if not lines:
        return []
    heights = [b[3] - b[1] for _, b in lines]
    med = _median(heights)
    order = sorted(range(len(lines)), key=lambda i: (-lines[i][1][3], lines[i][1][0]))
    groups: list[list[int]] = []
    current: list[int] = []
    prev_box: tuple[float, float, float, float] | None = None
    for i in order:
        box = lines[i][1]
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


@dataclass(frozen=True)
class VlmEvidence:
    """One measured member line re-entering render pairing."""

    text: str
    box: tuple[float, float, float, float]


def synthetic_vlm_lines(blocks: Sequence[IRBlock] | None) -> list[VlmEvidence]:
    """VLM-measured member lines for render pairing on textless pages.

    Textless scan pages yield zero pdfium lines, so VLM paragraph blocks
    could never pair and never render. Member lines stored in provenance
    (``vlm_lines``) re-enter here as pairing evidence: same text the block
    was built from, same measured boxes. Ordinary blocks (no ``vlm_lines``)
    contribute nothing — zero behavior change elsewhere. Lives here (not in
    the render engine) to avoid an import cycle; the engine wraps these in
    LineBox.
    """
    out: list[VlmEvidence] = []
    for block in blocks or []:
        members = (getattr(block, "provenance", None) or {}).get("vlm_lines") or []
        for member in members:
            try:
                text = str(member.get("text", "")).strip()
                x0, y0, x1, y1 = (float(v) for v in member.get("box", ()))
            except (TypeError, ValueError, AttributeError):
                continue
            if text and x1 > x0 and y1 > y0:
                out.append(VlmEvidence(text, (x0, y0, x1, y1)))
    return out


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
                width, height = float(page.get_width()), float(page.get_height())
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
    transcript = driver.recognize(image, (width, height), scale)
    anchored, stats = anchor_transcript(pdfium_lines, transcript, (width, height))
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
    # Paragraph grouping first: hundreds of line-blocks would grind the LLM
    # stages for hours; rhythm-joined paragraphs translate AND align better.
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
                id=f"{id_prefix}#v{page_no:03d}{g:04d}",
                flow_id=FlowID.MAIN_STORY,
                spine_index=start_index + g,
                block_type=BlockType.NARRATIVE,
                bbox=BoundingBox(page=page_no, x0=x0, y0=y0, x1=x1, y1=y1),
                source_text=text,
                provenance={
                    "parser": f"vlm:{transcript.engine}",
                    "anchor_provenance": "+".join(provenances),
                    "needs_review": any(m.needs_review for m in members),
                    "anchor_stats": {
                        "matched": stats.matched,
                        "vlm_only": stats.vlm_only,
                        "pdfium_only": stats.pdfium_only,
                    },
                    "vlm_lines": [{"text": m.text, "box": list(m.box)} for m in members],
                },
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
