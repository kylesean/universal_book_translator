"""Shared plain-text extraction helpers for lightweight PDF text paths.

Consumers are the pdf_oxide fallback inside the Docling family
(``ubt.adapters.pdf.docling_parser`` imports :func:`pages_to_blocks`) and the
core port ``ubt.core.ports.sample_pdf_pages`` (this module's
:func:`sample_pdf_pages`), so core needs neither to name this adapter module
nor to import a PDF library. The pdfium reader does not come through here: it
types its own blocks from real geometry in ``ubt.adapters.pdf.reader_pdf``, and
the shared rules it must agree with live in :mod:`ubt.analyze.structure`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ubt.analyze.structure import (
    is_list_prefix,
    looks_like_debris,
    looks_like_heading,
    looks_like_listing,
)
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element


def sample_page_indices(page_count: int, max_samples: int = 10) -> list[int]:
    """0-based page indices to probe, deduplicated and at most ``max_samples`` long.

    Avoids front-matter sampling bias (where pages 0..9 are almost always cover,
    blank, TOC, and preface) by sampling page 0 plus evenly distributed fractions
    across 10% to 90% of the document.
    """
    if page_count <= 0:
        return []
    if page_count <= max_samples:
        return list(range(page_count))
    fractions = [i / max_samples for i in range(1, max_samples)]
    indices = [0] + [min(int(f * page_count), page_count - 1) for f in fractions]
    seen: set[int] = set()
    result: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def sample_pdf_pages(path: Path) -> tuple[int, bool, str]:

    try:
        from ubt.adapters.pdf.pdfium_gate import open_document

        with open_document(path) as doc:
            page_count = len(doc)
            sample_indices = sample_page_indices(page_count)
            text_samples: list[str] = []
            for i in sample_indices:
                page = doc[i]
                try:
                    tp = page.get_textpage()
                    try:
                        text_samples.append(tp.get_text_range() or "")
                    finally:
                        tp.close()
                finally:
                    page.close()
        full_sample = "\n".join(text_samples)
        is_scanned = len(full_sample.strip()) < 100 * max(1, min(5, page_count))
        return page_count, is_scanned, full_sample
    except Exception:  # pdfium probe failed; the oxide extractor below is the fallback
        pass

    from ubt.adapters.pdf import oxide_render

    texts = oxide_render.extract_page_texts(path)
    if not texts:
        return 1, False, ""
    page_count = len(texts)
    sample_indices = sample_page_indices(page_count)
    full_sample = "\n".join(texts[i] for i in sample_indices)
    is_scanned = len(full_sample.strip()) < 100 * max(1, min(5, page_count))
    return page_count, is_scanned, full_sample


def classify_plain_text_block(
    normalized: str,
    block_idx: int,
    page_num: int,
    bbox: BoundingBox | None = None,
) -> IRBlock | None:
    """Classify one normalized paragraph into its typed IRBlock.

    ``bbox`` carries real geometry when the extractor has it (pdfium); the
    oxide fallback has none and leaves a zero-area placeholder. Returns
    ``None`` for empty input (the block counter must not advance).
    """
    if not normalized:
        return None
    resolved_bbox = bbox or BoundingBox(page=page_num, x0=0.0, y0=0.0, x1=0.0, y1=0.0)
    # Gate 1 (math closed loop): shattered-equation debris must never reach
    # the LLM as prose. Structural detection (Docling FORMULA) stays primary;
    # this is the safety net for plain-text paths (pdfium/oxide) that cannot
    # emit FORMULA. skip_translate ships it verbatim via is_static_skip.
    # The debris/listing/title/list rules all live in the shared analyzer
    # (single source of truth: one owner), so a plain-text engine cannot type a line differently
    # from the reader -- the analyzer produces the types directly instead of a
    # later flow-repair pass re-typing them.
    if looks_like_debris(normalized):
        return _plain_block(normalized, block_idx, resolved_bbox, BlockType.FORMULA, skip=True)
    if looks_like_listing(normalized):
        return _plain_block(normalized, block_idx, resolved_bbox, BlockType.CODE, skip=True)
    is_list = is_list_prefix(normalized)
    block_type = (
        BlockType.HEADING
        if looks_like_heading(normalized)
        else (BlockType.LIST_ITEM if is_list else BlockType.NARRATIVE)
    )
    return _plain_block(normalized, block_idx, resolved_bbox, block_type, skip=False)


def _plain_block(
    normalized: str,
    block_idx: int,
    bbox: BoundingBox,
    block_type: BlockType,
    *,
    skip: bool,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=f"pdf_main#b{block_idx:04d}",
            spine_index=block_idx,
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            source_text=normalized,
            bbox=bbox,
            skip_translate=skip,
        )
    )


def pages_to_blocks(pages: Iterable[tuple[int, str]]) -> list[IRBlock]:
    """Blank-line paragraph segmentation over ``(page_number, page_text)`` pairs.

    This is the canonical oxide fallback segmentation: paragraphs split on
    blank lines, hard-wrapped lines rejoined with single spaces. CRLF output
    (as produced by pdfium) is normalized before splitting.
    """
    blocks: list[IRBlock] = []
    block_idx = 1
    for page_num, page_text in pages:
        unified = page_text.replace("\r\n", "\n").replace("\r", "\n")
        for paragraph in unified.split("\n\n"):
            lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
            normalized = " ".join(lines)
            block = classify_plain_text_block(normalized, block_idx, page_num)
            if block is not None:
                blocks.append(block)
                block_idx += 1
    return blocks
