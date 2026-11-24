"""Docling structural recovery, table cell parsing, and layout metadata helpers."""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.docling_blocks import (
    merge_table_continuation_fragments,
    resolve_overlapping_formula_blocks,
)
from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
from ubt.analyze.structure import pdf_list_marker
from ubt.core.ir.models import (
    BlockProvenance,
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    StyleMeta,
    make_element,
)
from ubt.model.ast import RegionKind
from ubt.model.span import CompositeSpan, PhysicalBox, Span, boxes_to_provenance

logger = logging.getLogger(__name__)


#: An ordered-marker run must be at least this long to be re-typed as a list:
#: a lone ``(2024) ...`` citation opens like a marker but is not a list item.
_ORDERED_MARKER_MIN_RUN = 2

#: How many times the *same* text must appear in a page-header/footer position
#: before it is treated as running chrome rather than one-off content.
#:
#: The comparison is ``>=``, not ``>``: a running head recurs once per page it
#: spans, and three occurrences is already repetition — a title-page credit line
#: appears once, a real section heading is never re-emitted by the *layout*
#: model in the header band at all. With the old strict ``> 3`` a head that
#: spanned exactly three pages survived as a ``HEADING`` block in the main
#: story, so "3.2 Unified FinFET compact model" and "Appendix Explicit surface
#: potential model" were typeset as chapter-level headings mid-chapter, in the
#: source language, three times each.
_CHROME_REPEAT_MIN = 3
#: Ordered-marker shapes (numbered ``(1)`` / ``1.`` / ``1)``, lettered ``a.`` /
#: ``(a)``, CJK ``一、``). Bullets are deliberately excluded: Docling already
#: types real bullet lists, so a bullet match here is more likely a symbol.
#: Floor for a one-off page footer to count as a footnote paragraph rather than
#: chrome (see ``_is_prose_footer``). A page number or running title is far below.
_FOOTER_PROSE_MIN_CHARS = 80
_FOOTER_PROSE_MIN_WORDS = 12

#: How much of a reference entry's text to match against the page's own text
#: layer when recovering the ``[n]`` Docling dropped. 40 characters is well past
#: the point where two entries in one bibliography can share a prefix (they start
#: with different author names) and short enough to tolerate a re-wrapped line.
_REFERENCE_PREFIX_CHARS = 40

_ORDERED_MARKER_RE = re.compile(
    r"^(?:"
    r"[（(]?[0-9０-９]{1,3}(?:[.．][0-9０-９]{1,3})*[.．、)）]"
    r"|[（(]?[a-zA-ZＡ-Ｚａ-ｚ][.．)）]"
    r"|[一二三四五六七八九十百]+[、.．)）]"
    r")$"
)
#: A first-line indent is this many em beyond the body margin (matches
#: the retired native reader's ``INDENT_FACTOR``); the two readers must agree on what an indent is.
_INDENT_FACTOR = 1.0
#: A single-line block carries no body margin to compare against, so an indent
#: is only measurable on a block of at least this many source lines.
_INDENT_MIN_LINES = 2


def _ordered_marker(block: IRBlock) -> tuple[str, str] | None:
    """``(marker, text)`` when a body paragraph opens with an ordered marker.

    Docling's own list typing misses parenthesized enumerations like ``(1)`` that
    it reads as one paragraph, so the marker survives only if the LLM reproduces
    it -- which it does inconsistently. Re-typing the run here lets the
    compositor's marker restorer put the number back deterministically.
    """
    if block.skip_translate or block.block_type is not BlockType.NARRATIVE:
        return None
    if block.region is not RegionKind.BODY:
        return None
    parsed = pdf_list_marker(block.source_text or "")
    if parsed is None:
        return None
    marker, text = parsed
    if not _ORDERED_MARKER_RE.match(marker):
        return None
    return marker, text


def _as_list_item(block: IRBlock, marker: str, text: str) -> IRBlock:
    element = make_element(
        id=block.id,
        spine_index=block.spine_index,
        block_type=BlockType.LIST_ITEM,
        flow_id=block.flow_id,
        region=block.region,
        source_text=text,
        bbox=block.bbox,
        span=block.element.span,
        skip_translate=block.skip_translate,
        confidence=block.element.confidence,
        decorative=block.element.decorative,
        marker=marker,
    )
    return block.model_copy(update={"element": element})


def restore_ordered_markers(blocks: list[IRBlock]) -> list[IRBlock]:
    """Re-type runs of marker-led body paragraphs as list items.

    Only a *run* (``_ORDERED_MARKER_MIN_RUN`` or more consecutive marker-led
    paragraphs) is converted, so a paragraph that merely starts with a bracketed
    year or a reference stays prose.
    """
    out = list(blocks)
    index = 0
    while index < len(out):
        if _ordered_marker(out[index]) is None:
            index += 1
            continue
        run: list[tuple[int, str, str]] = []
        cursor = index
        while cursor < len(out):
            parsed = _ordered_marker(out[cursor])
            if parsed is None:
                break
            run.append((cursor, parsed[0], parsed[1]))
            cursor += 1
        if len(run) >= _ORDERED_MARKER_MIN_RUN:
            for position, marker, text in run:
                out[position] = _as_list_item(out[position], marker, text)
        index = cursor
    return out


#: An appendix heading opens with a lone capital letter (``A``/``B``/``C`` ...),
#: then a capitalized word. The letter must be a *marker*, not an opening
#: article, so the word after it must be capitalized too -- prose reads
#: ``A new approach``, an appendix heading reads ``A Another Example ...``.
#: Only the opening is anchored: the title may run on ("A Another Example for
#: OSTD APIs"), which is why the document-consistency corroboration below, not
#: the shape alone, is what keeps prose out.
_APPENDIX_HEADING_RE = re.compile(r"^([A-Z])\s+[A-Z]")
#: An appendix heading is a title: short enough to be one, and it never ends in
#: sentence punctuation (a terminal period would make ``A B.`` read as prose).
_APPENDIX_HEADING_MAX_CHARS = 80
#: ...and it fits on one source line. A wrapped paragraph opening on
#: ``A Capitalized`` word is prose, not a heading.
_APPENDIX_HEADING_MAX_LINE_HEIGHT_PT = 26.0


def _appendix_heading_marker(block: IRBlock) -> str | None:
    """The appendix letter when ``block`` has the appendix-heading shape.

    Shape only: block type plays no part, so an already-typed heading
    corroborates the document's appendix-marking style and an untyped
    paragraph can be re-typed against it.
    """
    if block.skip_translate or block.flow_id is not FlowID.MAIN_STORY:
        return None
    text = (block.source_text or "").strip()
    if not text or len(text) > _APPENDIX_HEADING_MAX_CHARS:
        return None
    if text[-1] in ".,;:!?" or "\n" in text:
        return None
    match = _APPENDIX_HEADING_RE.match(text)
    if match is None:
        return None
    # One printed line: a heading wraps at titles, not at paragraph length.
    bbox = block.bbox
    if bbox is not None and (bbox.y1 - bbox.y0) > _APPENDIX_HEADING_MAX_LINE_HEIGHT_PT:
        return None
    return match.group(1)


def restore_appendix_headings(blocks: list[IRBlock]) -> list[IRBlock]:
    """Re-type letter-marked paragraphs as appendix headings.

    Docling's layout model reads most appendix headings as section headers but
    misses some -- measured on the ATC'25 paper, "B Complete results ..." and
    "C Complete results ..." came out as headings while "A Another Example for
    OSTD APIs" came out as a body paragraph and shipped set as body text. The
    shape is unambiguous (a lone capital letter, then a capitalized word, one
    line, no terminal punctuation), and what makes re-typing safe is document
    consistency, the same corroboration ``restore_ordered_markers`` demands of
    a list: this document must already carry a letter-marked heading, and the
    candidate's letter must sit next to one of theirs in the appendix sequence
    -- the missed heading is always the sequence's neighbour the model skipped
    (``A`` beside typed ``B``/``C``). Prose that merely opens with a capitalized
    word stays prose in a paper with no letter-marked headings.
    """
    heading_markers = {
        marker
        for block in blocks
        if block.block_type is BlockType.HEADING
        and (marker := _appendix_heading_marker(block)) is not None
    }
    # Appendix letters run consecutively: a typed "B" vouches for a missed "A"
    # (or "C"), not for an arbitrary narrative that happens to open "A CPU ...".
    corroborated = set(heading_markers)
    for marker in heading_markers:
        corroborated.add(chr(ord(marker) - 1))
        corroborated.add(chr(ord(marker) + 1))
    if not heading_markers:
        return blocks
    out = list(blocks)
    for index, block in enumerate(out):
        if block.block_type is not BlockType.NARRATIVE:
            continue
        marker = _appendix_heading_marker(block)
        if marker is None or marker not in corroborated:
            continue
        element = make_element(
            id=block.id,
            spine_index=block.spine_index,
            block_type=BlockType.HEADING,
            flow_id=block.flow_id,
            region=block.region,
            source_text=block.source_text,
            bbox=block.bbox,
            span=block.element.span,
            skip_translate=block.skip_translate,
            confidence=block.element.confidence,
            decorative=block.element.decorative,
            level=1,
        )
        out[index] = block.model_copy(update={"element": element})
    return out


def _is_prose_footer(text: str) -> bool:
    """Whether a one-off page footer is a footnote paragraph, not chrome.

    A footer is prose the author wrote when it reads like a sentence: several
    words AND sentence punctuation/length. A page number, a running title, a
    copyright strip or an arXiv stamp is short and carries none of that, so it
    stays skipped. Kept precision-first — a false positive would translate a
    watermark into the wrong language.
    """
    body = (text or "").strip()
    if len(body) < _FOOTER_PROSE_MIN_CHARS:
        return False
    if len(body.split()) < _FOOTER_PROSE_MIN_WORDS:
        return False
    # A footnote reads as one or more sentences: it ends in sentence punctuation
    # OR contains an internal sentence boundary (". " / "。"). A bare label, DOI
    # or copyright strip carries neither.
    if body.endswith((".", "。", "：", ":")):
        return True
    return ". " in body or "。" in body


def _table_reader_enabled() -> bool:
    """Whether the opt-in vision table reader is configured."""
    from ubt.adapters.pdf import table_reader

    return table_reader.enabled()


def _read_table_cells(item: Any, pdf_path: Path | None, docling_markdown: str) -> str | None:
    """Docling's table with its data cells re-read from the page image, or ``None``.

    The crop is Docling's own table bbox — the reader is a region reader, and
    feeding it a page collapses several tables into one (measured). The header
    row and the column count stay Docling's: the reader misreads header text,
    and the header is what translation and the leak metric key on.
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
    if bbox.x1 - bbox.x0 < 24 or bbox.y1 - bbox.y0 < 16:
        # Too small to be a table; cropping it would send noise to the model.
        return None
    try:
        from ubt.adapters.pdf import table_reader
        from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI, crop_block_pil

        image = crop_block_pil(pdf_path, page_no, bbox, dpi=FIGURE_ASSET_DPI)
    except Exception as exc:  # noqa: BLE001 - no crop means no upgrade
        logger.debug("table crop failed on page %s: %s", page_no, exc)
        return None
    read = table_reader.read_table_image_sync(image)
    if read is None:
        return None
    docling_rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in docling_markdown.splitlines()
        if line.strip().startswith("|") and "---" not in line
    ]
    merged = table_reader.merge_with_docling(docling_rows, read)
    if merged is None:
        return None
    header, _, _ = docling_markdown.partition("\n|---")
    separator = "|" + "|".join("---" for _ in merged[0]) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in merged)
    if header:
        return f"{header}\n{separator}\n{body}"
    return f"{separator}\n{body}"


def _recover_reference_marker(text: str, page_text: str) -> str | None:
    """The bracketed number Docling dropped from a reference entry, or ``None``.

    Docling types a reference as a ``list_item`` but leaves ``marker`` empty and
    strips the number out of ``text``: on arXiv 2609.20519 the source carries 108
    citation markers, every ``[n]`` prefix survives in the page's own text layer,
    and the extraction kept none of them — the delivered bibliography lost its
    numbering.

    Recovered by *text prefix*, not by coordinates: the page text holds
    ``[1] Mark Chen, Jerry Tworek, …`` while Docling's item holds
    ``Mark Chen, Jerry Tworek, …``. A prefix match cannot confuse two entries
    (unlike a bbox match, whose coordinate systems did not line up in testing and
    would have silently numbered the wrong paragraphs).
    """
    head = " ".join((text or "").split())[:_REFERENCE_PREFIX_CHARS]
    if not head or not page_text:
        return None
    # Compare on a whitespace-collapsed haystack so a line break *inside a word*
    # ("Da\nvid") still matches: the page's text layer wraps mid-word where
    # Docling's text has no break at all, and a pure regex over the raw text
    # misses those entries (measured: 52/54 recovered with this, vs 1/57 for a
    # raw-text regex).
    collapsed = " ".join(page_text.split())
    pattern = r"(\[\d{1,3}\])\s*" + re.escape(head)
    match = re.search(pattern, collapsed)
    if match:
        return match.group(1)
    # Second failure mode: Docling drops a space *inside* an organisation name —
    # "[38] Axiom Math. AxiomProver…" arrives as "AxiomMath. AxiomProver…" — so
    # retry with both sides space-stripped. This recovers the two organisation
    # entries the whitespace pass missed (measured: 54/54 on the reference list).
    compact_head = head.replace(" ", "")
    compact_page = collapsed.replace(" ", "")
    match = re.search(
        r"(\[\d{1,3}\])\s*" + re.escape(compact_head),
        compact_page,
    )
    if match:
        return match.group(1)
    # Third: the page layer uses typographic punctuation the extractor flattens
    # ("Anthropic’s" arrives as "Anthropic's"), so compare with apostrophes and
    # dashes folded to their ASCII forms.
    folded_head = re.sub(r"[\u2018\u2019\u02bc]", "'", compact_head)
    folded_page = re.sub(r"[\u2018\u2019\u02bc]", "'", compact_page)
    match = re.search(r"(\[\d{1,3}\])\s*" + re.escape(folded_head), folded_page)
    return match.group(1) if match else None


def _iter_pdf_page_texts(pdf_path: Path, first: int, last: int) -> list[str]:
    """Text of pages ``first..last`` (1-based, clamped), unreadable pages as ``""``."""
    out: list[str] = []
    try:
        import pypdfium2 as pdfium

        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(pdf_path))
            try:
                total = len(pdf)
                for page_no in range(max(1, first), min(total, last) + 1):
                    try:
                        out.append(pdf[page_no - 1].get_textpage().get_text_range())
                    except Exception:  # noqa: BLE001 - one bad page must not lose the rest
                        out.append("")
            finally:
                pdf.close()
    except Exception as exc:  # noqa: BLE001 - recovery is best-effort
        logger.debug("reference-marker recovery: %s unreadable (%s)", pdf_path, exc)
    return out


def _line_in_box(rect: tuple[float, float, float, float], box: BoundingBox) -> bool:
    center_y = (rect[1] + rect[3]) / 2.0
    if not (box.y0 - 1.0 <= center_y <= box.y1 + 1.0):
        return False
    return min(rect[2], box.x1) - max(rect[0], box.x0) > 1.0


def _first_line_indent(box: BoundingBox, lines: list[Any]) -> float | None:
    """Indent (pt) of the block's first line past its own body margin, or ``None``."""
    inside = [line for line in lines if line.text.strip() and _line_in_box(line.rect, box)]
    if len(inside) < _INDENT_MIN_LINES:
        return None
    inside.sort(key=lambda line: -line.rect[3])
    body_x0 = min(float(line.rect[0]) for line in inside[1:])
    first = inside[0]
    indent = float(first.rect[0]) - body_x0
    font = float(first.font_size) if first.font_size and first.font_size > 0 else (box.y1 - box.y0)
    if indent > _INDENT_FACTOR * max(font, 1.0) and indent < (box.x1 - box.x0) * 0.5:
        return indent
    return None


def _is_centered(box: BoundingBox, lines: list[Any]) -> bool:
    """Whether any line of the block hangs symmetrically inside its own box.

    A centered title's short lines sit evenly between the margins; a
    left-aligned heading's lines start at the box edge (zero left gap). One
    qualifying line is enough — a multi-line centered title usually has its
    widest line touching both margins, so the evidence lives in the others.
    """
    inside = [line for line in lines if line.text.strip() and _line_in_box(line.rect, box)]
    width = box.x1 - box.x0
    if width <= 0:
        return False
    gap_min = max(6.0, 0.02 * width)
    tolerance = max(6.0, 0.05 * width)
    for line in inside:
        left = float(line.rect[0]) - box.x0
        right = box.x1 - float(line.rect[2])
        if left > gap_min and right > gap_min and abs(left - right) <= tolerance:
            return True
    return False


def _line_boxes(box: BoundingBox, lines: list[Any]) -> tuple[PhysicalBox, ...]:
    """The block's own lines as a reading-order box chain (top to bottom)."""
    inside = [line for line in lines if line.text.strip() and _line_in_box(line.rect, box)]
    inside.sort(key=lambda line: -line.rect[3])
    return tuple(
        PhysicalBox.of(
            int(box.page),
            (float(line.rect[0]), float(line.rect[1]), float(line.rect[2]), float(line.rect[3])),
        )
        for line in inside
    )


def annotate_layout_metadata(blocks: list[IRBlock], pdf_path: Path | None) -> list[IRBlock]:
    """Record per-block layout facts the block box alone cannot carry.

    Docling gives one bounding box per block (the margin), so three facts are
    invisible in its output and must come from the page's own line rects
    (textgeom):

    - a body paragraph's first-line indent (``first_line_indent_pt``);
    - a list item's marker-column indent, recorded in the same field — the
      block box starts at the *wrapped* lines' margin, so the indent is what
      separates the marker column from the body margin;
    - a heading's centering (``alignment="center"``) — a centered title's
      short lines hang symmetrically inside the box.

    **These are measurements, and the Typst renderer does not read them.**
    That is deliberate, not an oversight: the target's typography follows the
    target language's conventions (see the ``first_line_indent_pt`` docstring
    in :mod:`ubt.core.ir.models`), and the renderer takes its indent amount
    from :data:`ubt.layout.theme._TYPOGRAPHIC_CONVENTIONS`. They were live
    inputs for the overlay renderer, which drew each fragment back onto its
    source page and therefore had to copy the source's own geometry; that
    engine was removed when PDF delivery moved to Typst. The records stay
    because they are cheap, they persist in the ledger, and they are what a
    future opt-in "match the source layout" mode would consume.

    A page whose lines cannot be read (a scan with no text layer) simply gets
    no metadata.
    """
    if pdf_path is None:
        return blocks
    from ubt.adapters.pdf.textgeom import extract_lines

    lines_by_page: dict[int, list[Any]] = {}

    def _page_lines(page: int) -> list[Any]:
        if page not in lines_by_page:
            try:
                lines_by_page[page] = list(extract_lines(pdf_path, page)[0])
            except Exception:
                lines_by_page[page] = []
        return lines_by_page[page]

    for block in blocks:
        if block.skip_translate:
            continue
        box = block.bbox
        if box is None or box.page <= 0:
            continue
        if block.block_type in (BlockType.NARRATIVE, BlockType.LIST_ITEM):
            if block.region is not RegionKind.BODY:
                continue
            indent = _first_line_indent(box, _page_lines(int(box.page)))
            if indent is not None:
                block.style = (block.style or StyleMeta()).model_copy(
                    update={"first_line_indent_pt": indent}
                )
        elif block.block_type is BlockType.HEADING:
            if block.region not in (RegionKind.BODY, RegionKind.TITLE):
                continue
            page_lines = _page_lines(int(box.page))
            if _is_centered(box, page_lines):
                block.style = (block.style or StyleMeta()).model_copy(
                    update={"alignment": "center"}
                )
            # A multi-line heading also records its own line boxes: the
            # compositor flows the translation across the source's line
            # structure, so the title breaks where the source title breaks
            # (usually at the subtitle colon) instead of at an arbitrary
            # width-fill point, which can strand a particle ("的") at a line
            # head and mangle the phrasing.
            line_boxes = _line_boxes(box, page_lines)
            if len(line_boxes) >= 2:
                block.element = dataclasses.replace(
                    block.element, span=CompositeSpan(boxes=line_boxes)
                )
                # Mirror the chain into the one provenance key the ledger can
                # rebuild a CompositeSpan from (``ledger_base._row_to_block``).
                # Without it the export stage -- which always reloads its blocks
                # from the ledger -- reads the heading back as a single span
                # holding the *first* line's box: only that line is masked and
                # the whole translation is squeezed into it, while the rest of
                # the source heading stays in the source language on the page.
                block.provenance.physical_boxes = boxes_to_provenance(line_boxes)
    return blocks


def type_docling_blocks(blocks: list[IRBlock]) -> list[IRBlock]:
    """The analyzer's own typing stage: raw Docling blocks -> final typed blocks.

    This is the Docling analyzer producing its own types (native analyzer AST type production). It
    trusts Docling's labels end to end -- reading order, chrome (PAGE_HEADER /
    PAGE_FOOTER), headings (TITLE / SECTION_HEADER), captions (CAPTION and the
    ``FIG. N`` title shape) and segmentation -- and the emission loop types
    debris/listing directly from the shared rules. There is no caption fuse /
    decouple / latch pass re-guessing boundaries afterwards.

    Rendering-correctness merges:
    1. Vertically overlapping FORMULA boxes are unioned so the shared region is cropped once;
    2. Loose code/formula fragments stranded immediately below a table are merged into the table.
    """
    blocks = resolve_overlapping_formula_blocks(blocks)
    blocks = merge_table_continuation_fragments(blocks)
    blocks = restore_ordered_markers(blocks)
    blocks = restore_appendix_headings(blocks)
    blocks = reanchor_figure_captions(blocks)
    return blocks


def reanchor_figure_captions(blocks: list[IRBlock]) -> list[IRBlock]:
    """Re-anchor figure and table captions rigidly to their mother figure/table BBoxes.

    Identifies figure captions (FlowID.CAPTION, RegionKind.CAPTION, or matching
    `Fig(ure)? N` patterns) and finds the corresponding BlockType.IMAGE on the
    same page located immediately above the caption.
    Locks the caption vertical position to +4pt below the image BBox (y1 = img.bbox.y0 - 4.0pt).

    Identifies table captions (matching `Table N` patterns) and finds the corresponding
    BlockType.TABLE on the same page located immediately below the caption (standard academic layout).
    Locks the caption vertical position to +4pt above the table BBox (y0 = tbl.bbox.y1 + 4.0pt).

    Synchronizes both `element.span` and `provenance.physical_boxes` to guarantee
    spatial consistency across all downstream layout solvers.
    """
    images_by_page: dict[int, list[IRBlock]] = {}
    tables_by_page: dict[int, list[IRBlock]] = {}
    for b in blocks:
        if b.bbox and b.bbox.page:
            if b.block_type == BlockType.IMAGE:
                images_by_page.setdefault(b.bbox.page, []).append(b)
            elif b.block_type == BlockType.TABLE:
                tables_by_page.setdefault(b.bbox.page, []).append(b)

    if not images_by_page and not tables_by_page:
        return blocks

    caption_pattern = re.compile(r"^(?:fig(?:ure)?|图)\.?\s*[\d\w]+", re.IGNORECASE)
    table_pattern = re.compile(r"^(?:table|表)\.?\s*[\d\w]+", re.IGNORECASE)

    updated_blocks: list[IRBlock] = []
    for b in blocks:
        if not b.bbox or not b.bbox.page:
            updated_blocks.append(b)
            continue

        raw_txt = b.source_text.strip()
        is_table_caption = bool(table_pattern.match(raw_txt))
        # A picture carries the figure's own BBox. Docling also labels a picture
        # with captions as CAPTION-role, and treating it as a caption candidate
        # re-anchored the exhibit itself: measured on the LoopCD paper, p4's
        # figure box was dragged to the equation crop 105pt above it, so the
        # picture box ended up covering the preceding paragraph and the CELS
        # source-overlap check condemned the page to a reflow.
        is_caption_cand = b.block_type is not BlockType.IMAGE and (
            b.flow_id == FlowID.CAPTION
            or b.region == RegionKind.CAPTION
            or bool(caption_pattern.match(raw_txt))
        )

        if (
            is_caption_cand
            and not is_table_caption
            and (caption_bbox := b.bbox) is not None
            and caption_bbox.page in images_by_page
        ):
            p_images = images_by_page[caption_bbox.page]
            cap_mid_x = (caption_bbox.x0 + caption_bbox.x1) / 2.0
            cap_y1 = caption_bbox.y1

            # Captions sit below the artwork in most layouts, but a panel-style
            # figure carries its caption at the top of the card. Docling already
            # measures that position; when the caption is flush (<=24pt) with a
            # horizontally overlapping picture *below* it, keep the measured
            # BBox and only record the parent. Without this, the caption was
            # dragged to an unrelated equation crop further up the page.
            over_cap: list[tuple[float, IRBlock]] = []
            for img in p_images:
                img_bbox = img.bbox
                if img_bbox is None:
                    continue
                overlap = min(img_bbox.x1, caption_bbox.x1) - max(img_bbox.x0, caption_bbox.x0)
                gap = caption_bbox.y0 - img_bbox.y1
                if overlap > 0.0 and -15.0 <= gap <= 24.0:
                    over_cap.append((gap, img))
            if over_cap:
                _, nearest = min(over_cap, key=lambda item: abs(item[0]))
                prov_dict = b.provenance.model_dump()
                prov_dict["caption_parent_id"] = nearest.id
                prov_dict["is_anchored_caption"] = True
                b = b.model_copy(update={"provenance": BlockProvenance(**prov_dict)})
                updated_blocks.append(b)
                continue

            best_img: IRBlock | None = None
            best_cost = float("inf")
            for img in p_images:
                if not img.bbox:
                    continue
                dist = img.bbox.y0 - cap_y1
                horiz_overlap = max(
                    0.0,
                    min(img.bbox.x1, caption_bbox.x1) - max(img.bbox.x0, caption_bbox.x0),
                )
                img_w = max(1.0, img.bbox.x1 - img.bbox.x0)
                img_mid_x = (img.bbox.x0 + img.bbox.x1) / 2.0
                horiz_dist = abs(cap_mid_x - img_mid_x)
                horiz_close = horiz_dist <= max(100.0, img_w)
                if -15.0 <= dist <= 150.0 and (horiz_overlap > 0.0 or horiz_close):
                    v_penalty = dist if dist >= 0 else abs(dist) * 2.0
                    cost = v_penalty + horiz_dist * 0.15
                    if cost < best_cost:
                        best_cost = cost
                        best_img = img

            if best_img is not None and best_img.bbox is not None:
                anchored_y1 = round(best_img.bbox.y0 - 4.0, 2)
                cap_h = max(12.0, caption_bbox.y1 - caption_bbox.y0)
                anchored_y0 = round(anchored_y1 - cap_h, 2)
                new_span = Span(
                    page=caption_bbox.page,
                    bbox=(caption_bbox.x0, anchored_y0, caption_bbox.x1, anchored_y1),
                )
                prov_dict = b.provenance.model_dump()
                prov_dict["caption_parent_id"] = best_img.id
                prov_dict["is_anchored_caption"] = True
                prov_dict["physical_boxes"] = [
                    {
                        "page": caption_bbox.page,
                        "bbox": [
                            caption_bbox.x0,
                            anchored_y0,
                            caption_bbox.x1,
                            anchored_y1,
                        ],
                    }
                ]

                new_element = dataclasses.replace(b.element, span=new_span)
                b = b.model_copy(
                    update={
                        "element": new_element,
                        "provenance": BlockProvenance(**prov_dict),
                    }
                )

        elif (
            is_table_caption
            and (table_caption_bbox := b.bbox) is not None
            and table_caption_bbox.page in tables_by_page
        ):
            # ``b`` is reassigned with ``model_copy`` in the branch above, which
            # makes mypy drop the loop-top ``b.bbox`` narrowing for the rest of
            # the chain; the local keeps every read below tied to the checked
            # value.
            p_tables = tables_by_page[table_caption_bbox.page]
            cap_mid_x = (table_caption_bbox.x0 + table_caption_bbox.x1) / 2.0
            cap_y0 = table_caption_bbox.y0

            best_tbl: IRBlock | None = None
            best_cost = float("inf")
            for tbl in p_tables:
                if not tbl.bbox:
                    continue
                # Academic tables place caption above the table (dist: caption bottom to table top)
                dist = cap_y0 - tbl.bbox.y1
                horiz_overlap = max(
                    0.0,
                    min(tbl.bbox.x1, table_caption_bbox.x1)
                    - max(tbl.bbox.x0, table_caption_bbox.x0),
                )
                tbl_w = max(1.0, tbl.bbox.x1 - tbl.bbox.x0)
                tbl_mid_x = (tbl.bbox.x0 + tbl.bbox.x1) / 2.0
                horiz_dist = abs(cap_mid_x - tbl_mid_x)
                horiz_close = horiz_dist <= max(100.0, tbl_w)
                if -15.0 <= dist <= 150.0 and (horiz_overlap > 0.0 or horiz_close):
                    v_penalty = dist if dist >= 0 else abs(dist) * 2.0
                    cost = v_penalty + horiz_dist * 0.15
                    if cost < best_cost:
                        best_cost = cost
                        best_tbl = tbl

            if best_tbl is not None and best_tbl.bbox is not None:
                anchored_y0 = round(best_tbl.bbox.y1 + 4.0, 2)
                cap_h = max(12.0, table_caption_bbox.y1 - table_caption_bbox.y0)
                anchored_y1 = round(anchored_y0 + cap_h, 2)
                new_span = Span(
                    page=table_caption_bbox.page,
                    bbox=(table_caption_bbox.x0, anchored_y0, table_caption_bbox.x1, anchored_y1),
                )
                prov_dict = b.provenance.model_dump()
                prov_dict["caption_parent_id"] = best_tbl.id
                prov_dict["is_anchored_caption"] = True
                prov_dict["physical_boxes"] = [
                    {
                        "page": table_caption_bbox.page,
                        "bbox": [
                            table_caption_bbox.x0,
                            anchored_y0,
                            table_caption_bbox.x1,
                            anchored_y1,
                        ],
                    }
                ]

                new_element = dataclasses.replace(b.element, span=new_span)
                b = b.model_copy(
                    update={
                        "element": new_element,
                        "provenance": BlockProvenance(**prov_dict),
                    }
                )

        updated_blocks.append(b)

    return updated_blocks
