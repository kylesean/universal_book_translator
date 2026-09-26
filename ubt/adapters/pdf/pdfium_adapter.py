"""3 born-digital PDF fast path: pypdfium2 text extraction.

pypdfium2 is Apache-2.0 (the bundled PDFium binary is BSD-3-Clause), which
keeps the repository's Zero-AGPL delivery guarantee intact. The adapter
reuses the Docling mainline delivery pipeline (Typst reconstruction,
facing-page interleaving, rigid typesetting) so only the extraction leg
differs.

Routing: ``UBT_PDF_ENGINE=auto`` picks this engine for clean single-column
born-digital PDFs (see :mod:`ubt.adapters.pdf.engine_selector`); scans and
multi-column layouts stay on Docling. ``UBT_PDF_ENGINE=pdfium`` forces the
fast path explicitly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.adapters.pdf.plain_text_extractor import classify_plain_text_block
from ubt.core.ir.models import BookManifest, BoundingBox, IRBlock

logger = logging.getLogger(__name__)

# A vertical gap larger than this multiple of the median line height starts
# a new paragraph (in-paragraph leading is ~0.3-0.5x line height; paragraph spacing is >=0.7x).
_PARAGRAPH_GAP_RATIO = 0.75


def _paragraph_bbox(
    page_num: int, rects: Sequence[tuple[float, float, float, float]]
) -> BoundingBox | None:
    """Union of a paragraph's line rects in PDF bottom-left points.

    PDFium reports rects as ``(left, bottom, right, top)`` in page points with
    the origin at the bottom-left, which is exactly the native PDF convention
    the rest of the PDF stack expects. Discarding this geometry (the previous
    behavior) left every fast-path block with a zero-area bbox, silently
    disabling visual-scalpel repair, formula witness and zone building.
    """
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    if x1 <= x0 or y1 <= y0:
        return None
    return BoundingBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1)


@pdfium_serialized
def extract_blocks_with_pdfium(path: Path) -> list[IRBlock]:
    """Column-aware geometric line harvesting: reading order preserved across columns."""
    import pypdfium2 as pdfium

    from ubt.adapters.pdf.textgeom import extract_lines

    blocks: list[IRBlock] = []
    block_idx = 1
    pdf = pdfium.PdfDocument(str(path))
    try:
        total_pages = len(pdf)
    finally:
        pdf.close()

    for page_num in range(1, total_pages + 1):
        try:
            lines, size = extract_lines(path, page_num)
        except Exception as exc:
            logger.debug("extract_lines failed on page %d of %s: %s", page_num, path.name, exc)
            continue
        if not lines:
            continue

        heights = sorted(max(1.0, ln.rect[3] - ln.rect[1]) for ln in lines)
        median_h = heights[len(heights) // 2]

        paragraph_lines: list[list[str]] = [[]]
        paragraph_rects: list[list[tuple[float, float, float, float]]] = [[]]
        prev_rect: tuple[float, float, float, float] | None = None

        for ln in lines:
            start_new = False
            if prev_rect is not None:
                # If x position shifted significantly (new column), or current line is higher than previous (top of next col),
                # or vertical gap exceeds paragraph threshold
                horizontal_jump = abs(ln.rect[0] - prev_rect[0]) > 0.25 * size[0]
                vertical_gap = prev_rect[1] - ln.rect[3]
                is_above = ln.rect[1] > prev_rect[3]
                if horizontal_jump or is_above or vertical_gap > _PARAGRAPH_GAP_RATIO * median_h:
                    start_new = True

            if start_new:
                paragraph_lines.append([])
                paragraph_rects.append([])

            if ln.text.strip():
                paragraph_lines[-1].append(ln.text.strip())
                paragraph_rects[-1].append(ln.rect)
            prev_rect = ln.rect

        for p_lines, p_rects in zip(paragraph_lines, paragraph_rects, strict=True):
            if not p_lines:
                continue
            normalized = " ".join(p_lines).strip()
            bbox = _paragraph_bbox(page_num, p_rects)
            block = classify_plain_text_block(
                normalized,
                block_idx,
                page_num,
                bbox=bbox,
            )
            if block is not None:
                blocks.append(block)
                block_idx += 1

    return blocks


class PDFiumAdapter(DoclingPDFAdapter):
    """Born-digital PDF fast path: CPU-cheap pypdfium2 extraction, zero model downloads.

    Inherits the full Docling-mainline render stack (publication Typst,
    rigid typesetting, alternating bilingual interleaving) — only extraction
    is replaced with geometric text harvesting.
    """

    @property
    def engine_name(self) -> str:
        return "pdfium"

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Manifest identical to the Docling mainline except the engine tag."""
        manifest = await super().extract_manifest(input_path)
        return manifest.model_copy(
            update={"metadata": {**manifest.metadata, "pdf_parser_engine": "pdfium"}}
        )

    def _extract_blocks_sync(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        """pypdfium2 geometric extraction; degrade to plain pdf_oxide on failure."""
        try:
            blocks = self._extract_with_pdfium(path)
        except Exception as exc:
            logger.warning(
                "pypdfium2 extraction failed on '%s' (%s); falling back to the "
                "plain pdf_oxide text extractor",
                path.name,
                exc,
            )
            blocks = self._extract_with_oxide(path)
        if page_range is not None:
            first, last = page_range
            blocks = [b for b in blocks if b.bbox is None or first <= b.bbox.page <= last]
        return blocks

    @pdfium_serialized
    def _extract_with_pdfium(self, path: Path) -> list[IRBlock]:
        """Geometric line harvesting with column-aware reading order."""
        return extract_blocks_with_pdfium(path)
