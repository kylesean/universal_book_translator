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


def _toc_rows_from_lines(
    lines: list[Any],
    width: float,
) -> list[tuple[str, str, bool, Any, tuple[float, float, float, float]]]:
    """``(title, page, has_leaders, line, box)`` for the TOC rows on one page.

    Two shapes reach this function, and the page's own geometry decides which:

    1. **Title and page number on one line** (``1.2.3 Title ......... 42``).
       :func:`parse_toc_entry_line` reads it directly.
    2. **Title and page number on separate lines**, which is what ``textgeom``
       produces for a leader-dotted row: the dot-leader gutter splits the row
       into a title line and a bare right-margin numeral. Per-line parsing finds
       nothing here — measured on arXiv 2608.25512, where it returned 0 rows for
       all 62 lines of a two-page contents section. The native reader already
       solves this shape by pairing on the baseline
       (:func:`ubt.adapters.pdf.reader_pdf._find_toc_pairs`); this reuses it
       rather than growing a second pairing rule.

    The fallback only runs when per-line parsing produced nothing, so a
    single-line TOC keeps its existing (exact) reading and this cannot change it.
    """
    from ubt.adapters.pdf.docling_blocks import parse_toc_entry_line

    rows: list[tuple[str, str, bool, Any, tuple[float, float, float, float]]] = []
    for ln in lines:
        parsed = parse_toc_entry_line(ln.text)
        if parsed is None:
            continue
        title, toc_page, has_leaders = parsed
        box = (float(ln.rect[0]), float(ln.rect[1]), float(ln.rect[2]), float(ln.rect[3]))
        rows.append((title, toc_page, has_leaders, ln, box))
    if rows:
        return rows

    from ubt.adapters.pdf.reader_pdf import _find_toc_pairs

    for title_line, number_line, page in _find_toc_pairs(lines, width):
        # The delivered row spans the whole line -- title through page number --
        # so the renderer can redraw leaders and number in one box.
        box = (
            min(title_line.rect[0], number_line.rect[0]),
            min(title_line.rect[1], number_line.rect[1]),
            max(title_line.rect[2], number_line.rect[2]),
            max(title_line.rect[3], number_line.rect[3]),
        )
        rows.append((title_line.text, page, True, title_line, box))
    return rows


def _extract_document_index_blocks(
    item: Any,
    pdf_path: Path | None,
    page_lines_cache: dict[int, tuple[list[Any], float]],
    start_index: int,
) -> list[IRBlock]:
    """Extract per-line TOC IRBlocks from a Docling DOCUMENT_INDEX item.

    ``page_lines_cache`` holds ``(lines, page width)`` per page: the width is
    what :func:`_toc_rows_from_lines`' baseline fallback needs to tell a
    right-margin page number from a title, and the caller extracts the pair once
    per page anyway.
    """
    from ubt.adapters.pdf.textgeom import extract_lines

    out: list[IRBlock] = []
    for prov in getattr(item, "prov", []) or []:
        page_no = int(getattr(prov, "page_no", 0))
        prov_bbox = getattr(prov, "bbox", None)
        if page_no < 1 or prov_bbox is None or pdf_path is None or not pdf_path.exists():
            continue
        if page_no not in page_lines_cache:
            try:
                lines, (width, _height) = extract_lines(pdf_path, page_no)
                page_lines_cache[page_no] = (lines, width)
            except Exception as exc:
                logger.debug("TOC line extraction failed on page %d: %s", page_no, exc)
                page_lines_cache[page_no] = ([], 0.0)

        page_lines, width = page_lines_cache[page_no]
        p_left = float(getattr(prov_bbox, "l", 0.0))
        p_bottom = float(getattr(prov_bbox, "b", 0.0))
        p_right = float(getattr(prov_bbox, "r", 0.0))
        p_top = float(getattr(prov_bbox, "t", 0.0))

        region_lines = [
            ln
            for ln in page_lines
            if (p_bottom - 6.0) <= (ln.rect[1] + ln.rect[3]) / 2.0 <= (p_top + 6.0)
            and ln.rect[0] >= p_left - 12.0
            and ln.rect[2] <= p_right + 12.0
        ]
        region_lines.sort(key=lambda ln: (-ln.rect[3], ln.rect[0]))
        for title, toc_page, has_leaders, ln, box in _toc_rows_from_lines(region_lines, width):
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
                            x0=box[0],
                            y0=box[1],
                            x1=box[2],
                            y1=box[3],
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

    # Keyed on title *and* destination page. Title alone is not unique in a
    # real contents: arXiv 2608.25512 lists "4.3.2 Temporal Composability"
    # (p. 44) and "7.3 Temporal Composability" (p. 79), and folding them
    # together would silently drop a real section from the delivered contents.
    # The destination is what tells two same-named rows apart, and both sides
    # of this merge already carry it (``toc_page``).
    def _key(title: str, page: str) -> tuple[str, str]:
        return title.strip().lower(), page.strip()

    seen = {_key(b.source_text or "", str(b.provenance.toc_page or "")) for b in already}
    out: list[IRBlock] = []
    for level, title, dest_page in entries:
        key = _key(title, str(dest_page))
        if key in seen:
            continue
        # The outline title and the Docling row often differ only by the
        # numeric prefix ('1 Introduction' vs 'Introduction'); match on the
        # stripped form too before adding a row.
        stripped = re.sub(r"^[0-9A-F]+\.?\s+", "", title).strip().lower()
        if stripped and any(stripped == t and str(dest_page) == p for t, p in seen):
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


def _figure_asset_dpi_tag() -> str:
    """The delivered-crop DPI tag, read lazily (the module's docling-import pattern).

    Asset filenames carry the resolution they were cut at, so raising the DPI
    cuts new files instead of replaying old ones from the sha-keyed asset dir.
    """
    from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI_TAG

    return FIGURE_ASSET_DPI_TAG


def assign_heading_levels_from_outline(
    pdf_path: Path | None,
    blocks: list[IRBlock],
) -> int:
    """Restore heading depth from the source PDF's own bookmark tree.

    Docling labels every heading ``section_header`` but reports them all at
    ``level 1`` (verified on four cached conversions: 26, 53 and 73 headings,
    every one at level 1), so a rebuilt book typesets a chapter title and its
    4.2.2 sub-subsection identically and the PDF outline comes out flat -- a
    114-entry sidebar with nothing collapsible. The source PDF *does* declare the
    hierarchy in its bookmarks (the reference paper carries 109 entries at levels
    0/1/2), and that declaration is what this reads.

    Matching is by label when both sides carry one (``4.2`` / ``4.2.2`` at the
    front of the title), then by normalized title. A heading with no outline
    counterpart keeps ``level 1``: guessing from font size is exactly the
    "heading calibrator" the outline reader exists to avoid.

    Returns the number of headings whose level changed.
    """
    entries = read_pdf_outline_entries(pdf_path)
    if not entries:
        return 0
    by_label: dict[str, int] = {}
    by_title: dict[str, int] = {}
    for level, title, _page in entries:
        # Bookmark levels are 0-based, heading levels 1-based; clamp because a
        # deeply nested bookmark (level 5+) has no Typst heading equivalent.
        depth = min(6, max(1, level + 1))
        by_title.setdefault(_normalize_heading_key(title), depth)
        label_match = _TOC_LABEL_RE.match(title.strip())
        if label_match:
            by_label.setdefault(label_match.group(1), depth)

    changed = 0
    for block in blocks:
        if block.block_type is not BlockType.HEADING:
            continue
        text = (block.source_text or "").strip()
        if not text:
            continue
        block_depth: int | None = None
        label_match = _TOC_LABEL_RE.match(text)
        if label_match:
            block_depth = by_label.get(label_match.group(1))
        if block_depth is None:
            block_depth = by_title.get(_normalize_heading_key(text))
        if block_depth is None:
            continue
        if getattr(block.element, "level", 1) != block_depth:
            block.set_heading_level(block_depth)
            changed += 1
    return changed


def _normalize_heading_key(title: str) -> str:
    """A heading title reduced to its comparable core.

    Strips a leading section label, collapses whitespace, folds case, and drops
    punctuation differences Docling introduces (the outline says
    ``2 What Is a Harness?`` while the block may read ``2. What Is a Harness``).
    """
    stripped = _TOC_LABEL_RE.sub("", title.strip())
    folded = re.sub(r"[\s\u00a0]+", " ", stripped).strip().lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", folded)


def _crop_formula_asset(
    item: Any,
    assets_dir: Path,
    pdf_path: Path | None,
    ordinal: int,
) -> IRBlock | None:
    """Recover a formula Docling could not read by cropping its source region.

    A display equation the extractor left as a bare ``$$`` is content, not
    markup: the renderer rebuilds the page from the IR and has no source canvas,
    so without this the equation simply vanishes. The crop becomes an IMAGE
    block, which the renderer places and the quality report counts.

    Delegates to :func:`ubt.adapters.pdf.docling_blocks.crop_region_asset` — the
    single crop implementation, shared with the lossy-code recovery. The two were
    65-line near-copies (normalized similarity 0.9956) differing only in the size
    floor (8 × 6pt here, 60 × 24pt for code), the asset name and the id prefix.
    """
    from ubt.adapters.pdf.docling_blocks import crop_region_asset

    return crop_region_asset(
        item,
        assets_dir,
        pdf_path,
        ordinal,
        kind="formula",
        min_width_pt=8,
        min_height_pt=6,
    )


__all__ = [
    "_crop_formula_asset",
    "_extract_document_index_blocks",
    "_extract_outline_only_blocks",
    "_toc_sort_key",
    "read_pdf_outline_entries",
]
