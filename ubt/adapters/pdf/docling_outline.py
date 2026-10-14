"""The PDF-outline/TOC extraction leg of the Docling parser.

Split out of :mod:`ubt.adapters.pdf.docling_parser` (see that module's
docstring): the bookmark-tree reader, the document-index block builder and the
formula-region cropper are one seam -- they read the source PDF's *own*
structure (outline, page furniture, drawn regions) rather than Docling's
converted model.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

from ubt.core.cleaners.lnds_pruner import normalize_academic_pdf_math
from ubt.core.ir.models import (
    BlockProvenance,
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)

logger = logging.getLogger(__name__)


def read_pdf_outline_entries(
    pdf_path: Path | None,
) -> list[tuple[int, str, int]]:
    """Level-0/1 entries from the PDF's own bookmark tree: (level, title, page).

    Docling's ``DOCUMENT_INDEX`` extraction is lossy for a specific, common
    shape: a TOC page that sets chapter titles ("1 Introduction") in a
    different typeface or column from their sub-sections. On the arXiv
    2101.03961 reference paper Docling returned all 21 sub-section rows but
    **none** of the 15 top-level chapters, because the extractor reads each
    line through ``parse_toc_entry_line`` and those lines carry no terminal
    page number in the geometry it inspects.

    The PDF's bookmark tree already holds exactly those entries, with their
    hierarchy, and costs one library call. Using it is strictly better than a
    "heading calibrator" that guesses levels from font size and regex: the
    level here is *declared* by the document, not inferred.

    Returns an empty list when the PDF has no outline (most reflowed or
    print-to-PDF sources) — the caller then keeps Docling's own extraction.
    """
    if pdf_path is None or not pdf_path.exists():
        return []
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:  # pragma: no cover - library/loader failure
        logger.debug("PDF outline unavailable: %s", exc)
        return []
    try:
        entries: list[tuple[int, str, int]] = []
        for bookmark in doc.get_toc():
            title = (bookmark.get_title() or "").strip()
            if not title:
                continue
            page = 0
            try:
                dest = bookmark.get_dest()
                page = int(dest.get_index()) + 1 if dest is not None else 0
            except Exception:
                page = 0
            entries.append((int(bookmark.level), title, page))
        return entries
    finally:
        doc.close()


def _extract_document_index_blocks(
    item: Any,
    pdf_path: Path | None,
    page_lines_cache: dict[int, list[Any]],
    start_index: int,
) -> list[IRBlock]:
    """Extract per-line TOC IRBlocks from a Docling DOCUMENT_INDEX item."""
    from ubt.adapters.pdf.docling_blocks import parse_toc_entry_line
    from ubt.adapters.pdf.textgeom import extract_lines

    out: list[IRBlock] = []
    for prov in getattr(item, "prov", []) or []:
        page_no = int(getattr(prov, "page_no", 0))
        prov_bbox = getattr(prov, "bbox", None)
        if page_no < 1 or prov_bbox is None or pdf_path is None or not pdf_path.exists():
            continue
        if page_no not in page_lines_cache:
            try:
                page_lines_cache[page_no] = extract_lines(pdf_path, page_no)[0]
            except Exception as exc:
                logger.debug("TOC line extraction failed on page %d: %s", page_no, exc)
                page_lines_cache[page_no] = []

        p_left = float(getattr(prov_bbox, "l", 0.0))
        p_bottom = float(getattr(prov_bbox, "b", 0.0))
        p_right = float(getattr(prov_bbox, "r", 0.0))
        p_top = float(getattr(prov_bbox, "t", 0.0))

        matched_lines = [
            ln
            for ln in page_lines_cache[page_no]
            if (p_bottom - 6.0) <= (ln.rect[1] + ln.rect[3]) / 2.0 <= (p_top + 6.0)
            and ln.rect[0] >= p_left - 12.0
            and ln.rect[2] <= p_right + 12.0
        ]
        matched_lines.sort(key=lambda ln: (-ln.rect[3], ln.rect[0]))
        for ln in matched_lines:
            parsed = parse_toc_entry_line(ln.text)
            if parsed is None:
                continue
            title, toc_page, has_leaders = parsed
            idx = start_index + len(out)
            out.append(
                IRBlock(
                    element=make_element(
                        id=f"pdf_main#b{idx:04d}",
                        spine_index=idx,
                        block_type=BlockType.HEADING if ln.bold else BlockType.NARRATIVE,
                        flow_id=FlowID.MAIN_STORY,
                        source_text=normalize_academic_pdf_math(title),
                        bbox=BoundingBox(
                            page=page_no,
                            x0=float(ln.rect[0]),
                            y0=float(ln.rect[1]),
                            x1=float(ln.rect[2]),
                            y1=float(ln.rect[3]),
                        ),
                    ),
                    provenance=BlockProvenance(
                        source_page=page_no,
                        toc_entry=True,
                        toc_page=toc_page,
                        toc_leaders=has_leaders,
                        is_bold=ln.bold,
                    ),
                )
            )
    return out


#: A heading's leading label: ``4``, ``4.1``, ``A``, ``C`` (appendix letters).
#: Section rows of one paper carry these consistently; prose lines do not.
_TOC_LABEL_RE = re.compile(r"^(\d+(?:\.\d+)*|[A-Z])(?:\.?\s+|\.)")


def _toc_label(title: str) -> tuple[int, tuple[int, ...], str]:
    """Parse a TOC row's leading label into a sort-friendly shape.

    ``("4 Downstream Results")`` -> ``(0, (4,), "")``; ``4.1`` -> ``(0, (4, 1), "")``;
    appendix ``C`` -> ``(1, (), "C")``. Unlabelled rows sort by title, after
    nothing in particular — they are the rare case.
    """
    m = _TOC_LABEL_RE.match(title.strip())
    if not m:
        return (2, (), "")
    label = m.group(1)
    if label[0].isdigit():
        parts = tuple(int(p) for p in label.split("."))
        return (0, parts, "")
    return (1, (), label)


def _toc_sort_key(block: IRBlock) -> tuple[int, int, tuple[int, ...], str, str]:
    """Reading-order key for a synthesized TOC row: ``(page, depth, label, stable)``.

    The page is the row's *destination*, not the page it was read from: both
    sources store the destination in ``toc_page`` (Docling parses the printed
    number off the entry line, the outline stringifies the bookmark's target
    page). Sorting on it restores reading order when the two sources are merged.

    Rows whose destination is not a number sort last, in insertion order — a
    malformed entry must not be reordered to the front on a guess.

    Rows sharing a destination page are ordered by their leading label instead
    of insertion order: ``4 Downstream Results`` and ``4.1 Fine-Tuning`` both
    point at page 14 of the arXiv 2101.03961 reference paper, and Docling reads
    ``4.1`` first, so the stable sort delivered a contents page where the
    section opened *above* the chapter that contains it. The label puts the
    shorter (parent) number first; on a tie (two rows with the same label —
    should not happen) the source order still decides.
    """
    raw = str(block.provenance.toc_page or "").strip()
    if not raw.isdigit():
        return (sys.maxsize, 0, (), "", "")
    depth, parts, letter = _toc_label(block.source_text or "")
    title = (block.source_text or "").strip().lower()
    return (int(raw), depth, parts, letter, title)


def _extract_outline_only_blocks(
    pdf_path: Path | None,
    already: list[IRBlock],
    start_index: int,
) -> list[IRBlock]:
    """TOC rows for outline entries Docling's DOCUMENT_INDEX extraction missed.

    Merges the two sources rather than replacing either: Docling supplies the
    sub-section rows with their real page geometry, and the PDF outline supplies
    the top-level chapters it drops.
    An entry already present verbatim in ``already`` is skipped, so the two
    sources never produce a duplicate row.

    The synthesized block has no bbox: the outline states a *destination page*,
    not a rectangle on the TOC page. The publication renderer reflows, so the
    entry is typeset as ordinary text in reading order rather than drawn at a
    guessed position on a fixed page.
    """
    entries = read_pdf_outline_entries(pdf_path)
    if not entries:
        return []
    seen = {(b.source_text or "").strip().lower() for b in already}
    out: list[IRBlock] = []
    for level, title, dest_page in entries:
        key = title.strip().lower()
        if key in seen:
            continue
        # The outline title and the Docling row often differ only by the
        # numeric prefix ('1 Introduction' vs 'Introduction'); match on the
        # stripped form too before adding a row.
        stripped = re.sub(r"^[0-9A-F]+\.?\s+", "", title).strip().lower()
        if stripped and any(stripped == s for s in seen):
            continue
        seen.add(key)
        idx = start_index + len(out)
        out.append(
            IRBlock(
                element=make_element(
                    id=f"pdf_main#b{idx:04d}",
                    spine_index=idx,
                    block_type=BlockType.HEADING,
                    flow_id=FlowID.MAIN_STORY,
                    source_text=normalize_academic_pdf_math(title),
                    # The outline declares the true depth; Docling's Markdown
                    # pass emits every heading as level 1, which flattens the
                    # document (40 headings all rendered as <h1>). Bookmark
                    # levels are 0-based, heading levels 1-based.
                    level=max(1, level + 1),
                ),
                provenance=BlockProvenance(
                    source_page=dest_page,
                    toc_entry=True,
                    toc_page=str(dest_page),
                    toc_leaders=False,
                    is_bold=level == 0,
                ),
            )
        )
    return out


def _crop_formula_asset(
    item: Any,
    assets_dir: Path,
    pdf_path: Path | None,
    ordinal: int,
) -> IRBlock | None:
    """Recover a formula Docling could not read by cropping its source region.

    A display equation the extractor left as a bare ``$$`` is content, not
    markup: the renderer rebuilds the page from the IR and has
    no source canvas to preserve, so without this the equation simply vanishes.
    The crop becomes an IMAGE block, which the renderer already places and the
    quality report already counts.

    Returns ``None`` when the region cannot be cropped (no provenance, no bbox,
    no source PDF, or a region too small to be an equation), so the caller keeps
    the ``$$`` placeholder and the block stays visible as a known gap rather
    than a silent one.
    """
    if pdf_path is None or not pdf_path.exists():
        return None
    prov = item.prov[0] if getattr(item, "prov", None) else None
    if prov is None or getattr(prov, "bbox", None) is None:
        return None
    page_no = int(getattr(prov, "page_no", 0) or 0)
    if page_no < 1:
        return None
    b = prov.bbox
    bbox = BoundingBox(
        page=page_no,
        x0=float(getattr(b, "l", 0.0)),
        y0=float(getattr(b, "b", 0.0)),
        x1=float(getattr(b, "r", 0.0)),
        y1=float(getattr(b, "t", 0.0)),
    )
    if bbox.x1 - bbox.x0 < 8 or bbox.y1 - bbox.y0 < 6:
        # A degenerate box is a stray glyph, not an equation; cropping it would
        # add noise to the page.
        return None
    asset_path = assets_dir / f"formula_p{page_no}_{ordinal:04d}.png"
    try:
        from ubt.adapters.pdf.visual_scalpel import crop_block_pil

        assets_dir.mkdir(parents=True, exist_ok=True)
        cropped = crop_block_pil(pdf_path, page_no, bbox)
        if getattr(cropped, "width", 0) < 8 or getattr(cropped, "height", 0) < 6:
            return None
        cropped.save(asset_path)
    except Exception as exc:
        logger.debug("Formula crop failed for page %d: %s", page_no, exc)
        return None
    return IRBlock(
        element=make_element(
            id=f"pdf_main#fml_{ordinal:04d}",
            spine_index=ordinal,
            block_type=BlockType.IMAGE,
            flow_id=FlowID.MAIN_STORY,
            source_text=str(asset_path),
            bbox=bbox,
            skip_translate=True,
        ),
        target_text=str(asset_path),
    )


__all__ = [
    "_crop_formula_asset",
    "_extract_document_index_blocks",
    "_extract_outline_only_blocks",
    "_toc_sort_key",
    "read_pdf_outline_entries",
]
