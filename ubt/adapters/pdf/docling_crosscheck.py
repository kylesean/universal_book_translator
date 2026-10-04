"""PDFium line geometry vs Docling layout IoU cross-check (Defense 1).

In the analyze stage, born-digital PDF pages can suffer from deep-learning
layout model hallucinations (e.g. cross-column spans, phantom bounding boxes,
or misclassified raster regions). This module performs lightweight (~2ms)
cross-checking by extracting physical text lines via PDFium (the ground-truth
glyph stream) and computing the Intersection-over-Union (IoU) with Docling's
bounding boxes.

When a text block's IoU with physical text lines falls below the threshold
(default 0.60), or when a block with text content has no physical lines under
it, the block is gracefully demoted to PRESERVED_OPAQUE:
- Its element confidence is set to Confidence.UNKNOWN.
- Its skip_translate and policy_translate are set to False/True (skip translation).
- Its provenance records the mismatch details.
- LayerCompositor leaves Layer 0 (the original canvas) intact without masking.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path

from ubt.adapters.pdf.textgeom import LineBox, extract_lines
from ubt.core.ir.models import BlockType, IRBlock
from ubt.model.ast import Confidence
from ubt.model.span import CompositeSpan, PhysicalBox

logger = logging.getLogger(__name__)

#: Text block types subject to physical glyph grounding verification.
_VERIFIABLE_TYPES = frozenset(
    {BlockType.NARRATIVE, BlockType.DIALOGUE, BlockType.HEADING, BlockType.LIST_ITEM}
)

#: Default IoU threshold below which a Docling box is considered ungrounded or displaced.
DEFAULT_MIN_IOU = 0.60


def _box_area(x0: float, y0: float, x1: float, y1: float) -> float:
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = _box_area(ix0, iy0, ix1, iy1)
    if inter <= 0:
        return 0.0
    union = _box_area(*a) + _box_area(*b) - inter
    return inter / union if union > 0 else 0.0


def _boxes_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def cross_check_blocks_with_pdfium(
    blocks: Sequence[IRBlock],
    pdf_path: Path,
    *,
    min_iou: float = DEFAULT_MIN_IOU,
) -> list[IRBlock]:
    """Verify Docling text block bounding boxes against PDFium glyph lines.

    For each prose text block on a born-digital page:
    1. Extracts cached PDFium lines for the page.
    2. If the page is textless (scanned image), skips verification (left to OCR/VLM).
    3. Finds PDFium lines intersecting the block's bounding box.
    4. Computes the IoU between the block's bbox and the union of intersecting lines.
    5. If IoU < min_iou (or zero intersecting lines for substantial text), marks the
       block with Confidence.UNKNOWN and skip_translate=True so it sinks to
       PRESERVED_OPAQUE on Layer 0.
    """
    path = Path(pdf_path)
    output: list[IRBlock] = []
    lines_by_page: dict[int, list[LineBox]] = {}

    for block in blocks:
        if block.block_type not in _VERIFIABLE_TYPES or block.skip_translate:
            output.append(block)
            continue
        bbox = block.bbox
        if bbox is None or bbox.page <= 0:
            output.append(block)
            continue

        span = block.element.span
        eval_boxes = (
            span.boxes
            if isinstance(span, CompositeSpan)
            else (PhysicalBox.of(bbox.page, (bbox.x0, bbox.y0, bbox.x1, bbox.y1)),)
        )

        failed_reason: str | None = None
        text_len = len(block.source_text.strip())
        all_inter_lines: list[LineBox] = []

        for pbox in eval_boxes:
            p_page = pbox.page
            if p_page not in lines_by_page:
                try:
                    page_lines, _ = extract_lines(path, p_page)
                    lines_by_page[p_page] = page_lines
                except Exception as exc:
                    logger.debug(
                        "PDFium line extraction failed for cross-check page %d: %s",
                        p_page,
                        exc,
                    )
                    lines_by_page[p_page] = []

            page_lines = lines_by_page[p_page]
            if not page_lines:
                # Scanned / raster-only page: no born-digital lines to cross-check against.
                continue

            doc_box = pbox.bbox
            if _box_area(*doc_box) <= 0:
                continue

            inter_lines = [line for line in page_lines if _boxes_intersect(doc_box, line.rect)]
            if not inter_lines:
                if text_len > 10:
                    failed_reason = f"no_physical_lines(p{p_page})"
                    break
                continue

            all_inter_lines.extend(inter_lines)
            ux0 = min(line.rect[0] for line in inter_lines)
            uy0 = min(line.rect[1] for line in inter_lines)
            ux1 = max(line.rect[2] for line in inter_lines)
            uy1 = max(line.rect[3] for line in inter_lines)
            iou = _box_iou(doc_box, (ux0, uy0, ux1, uy1))

            effective_min_iou = 0.40 if text_len <= 5 else min_iou
            if iou < effective_min_iou:
                failed_reason = f"iou_mismatch(p{p_page}: iou={iou:.2f} < {effective_min_iou:.2f})"
                break

        if failed_reason is not None:
            logger.info(
                "Block %s on page %d failed cross-check (%s); sinking to PRESERVED_OPAQUE",
                block.id,
                bbox.page,
                failed_reason,
            )
            fused_elem = dataclasses.replace(
                block.element, confidence=Confidence.UNKNOWN, skip_translate=True
            )
            demoted = IRBlock(element=fused_elem)
            demoted.skip_translate = True
            demoted.policy_translate = False
            demoted.provenance = {
                **block.provenance,
                "iou_crosscheck": f"opaque: {failed_reason}",
            }
            demoted.error_flags = list(block.error_flags) + [
                f"skip:preserved_opaque({failed_reason})"
            ]
            output.append(demoted)
            continue

        if all_inter_lines:
            try:
                from ubt.adapters.pdf.textgeom import _aggregate_line_styles
                from ubt.core.ir.models import StyleMeta

                fsz, is_bold, is_italic = _aggregate_line_styles(all_inter_lines)
                if fsz >= 4.5:
                    block.style = StyleMeta(font_size=fsz)
                    block.provenance["font_size"] = fsz
                if is_bold:
                    block.provenance["is_bold"] = True
                if is_italic:
                    block.provenance["is_italic"] = True
            except Exception as exc:
                logger.debug("Failed to extract line styles for block %s: %s", block.id, exc)

        output.append(block)

    return output


__all__ = ["DEFAULT_MIN_IOU", "cross_check_blocks_with_pdfium"]
