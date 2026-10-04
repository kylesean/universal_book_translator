"""Native PDF reader: a PDF -> the typed Document AST (native AST reader and verification seam).

This is the "difficulty concentrated in one place" architectural principle. ``adapters``
supply *raw geometry* (text lines, their boxes and font metrics via
:mod:`ubt.adapters.pdf.textgeom`); this module owns *structure* -- reading order,
paragraph grouping, heading detection, and page furniture (chrome, listings) --
and emits a :class:`~ubt.model.ast.Document` directly, with no ``IRBlock`` detour.

One rule set, stated once and documented, replaces the competing heuristics the
old flow lived with:

1. a vertical gap wider than :data:`PARAGRAPH_GAP_FACTOR` of the font starts a
   new paragraph;
2. a font-size change over :data:`FONT_CHANGE_RATIO` starts a new paragraph;
3. a first-line indent -- ``x0`` moving right by more than
   :data:`INDENT_FACTOR` font sizes *from the body margin* -- starts a new
   paragraph; a hanging indent (the previous line is already off the margin,
   e.g. a bullet's wrapped continuation) does not, so a list item stays whole;
4. a line beginning with a bullet/number starts a list item;
5. a bare number in the top or bottom margin band is a page number -- page
   furniture, kept verbatim in a ``RegionKind.PAGE_NUMBER`` region;
6. a line carrying unambiguous program syntax is a :class:`CodeBlock` (kept
   verbatim), never prose.

A heading is decided by *typography* (font size relative to the page's body
size, or bold + short + no terminal punctuation), never by length alone -- the
rule that made the old plain-text path mislabel every fragment.

The reader never drops text for being unsure: an element it cannot classify is
emitted as prose with :attr:`Confidence.INFERRED`. Text loss would only come from
a page whose geometry cannot be read at all, and that is reported by omission.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.analyze._identity import file_digest
from ubt.analyze.normalize import normalize_text
from ubt.analyze.structure import (
    is_bare_page_number,
    looks_like_debris,
    looks_like_listing,
    pdf_list_marker,
)
from ubt.core.policy.layout_policy import FOOTER_BAND_PT, HEADER_BAND_PT
from ubt.model.ast import (
    CodeBlock,
    Confidence,
    Document,
    ElementT,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Region,
    RegionKind,
)
from ubt.model.span import CanonicalSource, CompositeSpan, PageGeometry, Span

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ubt.adapters.pdf.textgeom import LineBox

#: Extra leading (as a fraction of the font size) that starts a new paragraph.
PARAGRAPH_GAP_FACTOR = 0.55
#: Relative font-size change that starts a new paragraph.
FONT_CHANGE_RATIO = 0.15
#: A first-line indent larger than this many font sizes starts a new paragraph.
INDENT_FACTOR = 1.0
#: Font size, relative to the page body size, at which a line reads as a heading.
HEADING_RATIOS = ((1.5, 1), (1.3, 2), (1.15, 3))
_HEADING_MAX_CHARS = 100
_SENTENCE_END = (".", "!", "?", "\u3002", ":", ";")
_WS_RE = re.compile(r"\s+")

# --- Table of contents ----------------------------------------------------- #
#: A TOC title opens with a section number: ``1.``, ``1.1.``, ``4.3.``.
_SECTION_NUM_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3})*\.?\s+\S")
#: A TOC page number is a bare arabic/roman numeral.
_TOC_PAGE_RE = re.compile(r"^[0-9ivxlcdmIVXLCDM]+$")
#: Baselines within this many points are the same TOC row.
_TOC_BASELINE_TOL_PT = 2.5
#: A TOC page number sits in the right margin; a title starts in the left half.
_TOC_PAGE_MIN_X_RATIO = 0.6
_TOC_TITLE_MAX_X_RATIO = 0.6
#: Fewer pairs than this and the page is not a TOC (a stray numbered line is not).
_TOC_MIN_PAIRS = 3


def _find_toc_pairs(lines: list[LineBox], width: float) -> list[tuple[LineBox, LineBox, str]]:
    """Pair TOC title lines with their right-margin page numbers by baseline.

    ``textgeom`` reads a title and its page number as two lines (the dot-leader
    gutter separates them), so the reader must pair them back or a table of
    contents scatters into translated titles plus a run of bare page-number
    blocks. A page only counts as a TOC when several numbered rows pair up
    (``_TOC_MIN_PAIRS``); once it does, unnumbered rows on the same page
    (``Abstract``, ``References``, ``Acknowledgments``) are paired too. The
    numbered-pair floor is what keeps an equation page -- which also has
    right-margin numerals -- from being mistaken for a TOC.
    """
    if width <= 0:
        return []
    right_numbers = [
        line
        for line in lines
        if _TOC_PAGE_RE.match(line.text.strip()) and line.rect[0] > _TOC_PAGE_MIN_X_RATIO * width
    ]

    def _left(line: LineBox) -> bool:
        return line.rect[0] <= _TOC_TITLE_MAX_X_RATIO * width

    def _baseline(line: LineBox) -> float:
        return (line.rect[1] + line.rect[3]) / 2.0

    def _pair(titles: list[LineBox], used: set[int]) -> list[tuple[LineBox, LineBox, str]]:
        out: list[tuple[LineBox, LineBox, str]] = []
        for line in titles:
            baseline = _baseline(line)
            for candidate in right_numbers:
                if id(candidate) in used:
                    continue
                if abs(_baseline(candidate) - baseline) <= _TOC_BASELINE_TOL_PT:
                    out.append((line, candidate, candidate.text.strip()))
                    used.add(id(candidate))
                    break
        return out

    used: set[int] = set()
    numbered = [ln for ln in lines if _left(ln) and _SECTION_NUM_RE.match(ln.text.strip())]
    pairs = _pair(numbered, used)
    if len(pairs) < _TOC_MIN_PAIRS:
        return []
    paired_ids = {id(title) for title, _number, _page in pairs}
    unnumbered = [
        ln
        for ln in lines
        if _left(ln)
        and id(ln) not in paired_ids
        and not _SECTION_NUM_RE.match(ln.text.strip())
        and any(ch.isalpha() for ch in ln.text.strip())
    ]
    pairs.extend(_pair(unnumbered, used))
    return pairs


@dataclasses.dataclass(frozen=True, slots=True)
class _Group:
    """One provisional element before the canonical text and spans are assigned."""

    text: str
    page: int
    bbox: tuple[float, float, float, float]
    max_size: float
    all_bold: bool
    #: Set for a table-of-contents row: the page number its leaders point at.
    toc_page: str | None = None


def _body_size(lines: list[LineBox]) -> float:
    """Char-weighted modal font size of a page -- its body text size."""
    weights: dict[float, int] = {}
    for line in lines:
        if line.font_size <= 0:
            continue
        key = round(line.font_size, 1)
        weights[key] = weights.get(key, 0) + max(1, len(line.text.strip()))
    if not weights:
        return 10.0
    return max(weights, key=lambda key: weights[key])


def _join_lines(lines: list[LineBox]) -> str:
    """Join a paragraph's lines, healing end-of-line hyphenation."""

    def _clean(text: str) -> str:
        return _WS_RE.sub(" ", text).strip()

    out = ""
    for line in lines:
        piece = _clean(line.text)
        if not piece:
            continue
        if not out:
            out = piece
        elif out.endswith("-") and not out.endswith("--") and piece[:1].islower():
            out = out[:-1] + piece
        else:
            out = f"{out} {piece}"
    # Canonicalize presentation characters before any span is assigned, so the
    # element text, the source slice and Span.chars all see the same stream.
    return normalize_text(out)


def _body_margin(lines: list[LineBox]) -> float:
    """The page's body left margin: the modal ``x0`` of its non-empty lines."""
    weights: dict[float, int] = {}
    for line in lines:
        if not line.text.strip():
            continue
        key = round(line.rect[0], 0)
        weights[key] = weights.get(key, 0) + 1
    if not weights:
        return 0.0
    return max(weights, key=lambda key: weights[key])


def _starts_new_paragraph(previous: LineBox, current: LineBox, body: float, body_x0: float) -> bool:
    font = max(previous.font_size, current.font_size, body, 1.0)
    indent = INDENT_FACTOR * font
    # A first-line indent -- measured from the *body margin* -- starts a
    # paragraph. A *hanging* indent does not: when the previous line is already
    # off the margin (a bullet's wrapped continuation, a block quote's second
    # line), the current line continues the same unit.
    if current.rect[0] - body_x0 > indent and previous.rect[0] - body_x0 <= indent:
        return True
    # Font-size change (a heading or a different text block).
    if abs(current.font_size - previous.font_size) > FONT_CHANGE_RATIO * font:
        return True
    # Vertical gap: previous line's bottom minus this line's top.
    gap = previous.rect[1] - current.rect[3]
    return gap > PARAGRAPH_GAP_FACTOR * font


def _group_lines(lines: list[LineBox], body: float) -> list[list[LineBox]]:
    groups: list[list[LineBox]] = []
    current: list[LineBox] = []
    body_x0 = _body_margin(lines)
    for line in lines:
        if not line.text.strip():
            continue
        # Rule 4: a line opening a list item starts a new group, so consecutive
        # bullets/numbers are separate items rather than one merged paragraph.
        # A wrapped continuation carries no marker, so it stays with its item.
        starts_list = pdf_list_marker(line.text) is not None
        if current and (starts_list or _starts_new_paragraph(current[-1], line, body, body_x0)):
            groups.append(current)
            current = []
        current.append(line)
    if current:
        groups.append(current)
    return groups


def _group_from_lines(
    lines: list[LineBox],
    page: int,
    *,
    toc_page: str | None = None,
    toc_bbox: tuple[float, float, float, float] | None = None,
) -> _Group | None:
    text = _join_lines(lines)
    if not text:
        return None
    sizes = [line.font_size for line in lines if line.font_size > 0]
    # A TOC row's box spans the whole row (title through page number), so the
    # compositor can redraw title + leaders + number across the row width.
    bbox = toc_bbox or (
        min(line.rect[0] for line in lines),
        min(line.rect[1] for line in lines),
        max(line.rect[2] for line in lines),
        max(line.rect[3] for line in lines),
    )
    return _Group(
        text=text,
        page=page,
        bbox=bbox,
        max_size=max(sizes) if sizes else 0.0,
        all_bold=all(line.bold for line in lines),
        toc_page=toc_page,
    )


def _heading_level(group: _Group, body: float) -> int | None:
    if body > 0:
        ratio = group.max_size / body
        for threshold, level in HEADING_RATIOS:
            if ratio >= threshold:
                return level
    if (
        group.all_bold
        and len(group.text) < _HEADING_MAX_CHARS
        and not group.text.endswith(_SENTENCE_END)
    ):
        return 3
    return None


def _is_page_number(group: _Group, page_height: float) -> bool:
    """Whether a group is a bare page number sitting in a margin band.

    A page number is page furniture only where furniture lives: the top or
    bottom margin. A bare number *inside* the body is content -- a table cell, a
    TOC leader, an index entry -- and is left to translate. This is the reader's
    own geometry talking, not a guess.
    """
    if page_height <= 0 or not is_bare_page_number(group.text):
        return False
    _x0, y0, _x1, y1 = group.bbox
    return y0 < FOOTER_BAND_PT or y1 > page_height - HEADER_BAND_PT


def _classify(
    group: _Group, body: float, element_id: str, spine_index: int, *, page_height: float
) -> tuple[ElementT, RegionKind]:
    """Turn a provisional group into its typed element and page region.

    Never drops a group: an element it cannot classify is prose at
    :attr:`Confidence.INFERRED`. The region kind is the reader's own layout
    judgement (body vs page furniture), which is exactly what the AST's
    :class:`RegionKind` models -- the pipeline no longer re-derives it.
    """
    confidence = Confidence.INFERRED
    span = Span(page=group.page, bbox=group.bbox)
    if group.toc_page is not None:
        # A TOC row translates like prose, but its dot leaders and page number
        # are source layout the compositor must redraw (the reader drops the
        # leader/​number lines, so a plain text overlay would leave a gap). Keep
        # the title as the translatable text and carry the page number for the
        # TOC-aware renderer; the box spans the whole row.
        return (
            Paragraph(
                id=element_id,
                spine_index=spine_index,
                text=group.text,
                toc_page=group.toc_page,
                span=span,
                confidence=confidence,
            ),
            RegionKind.BODY,
        )
    if _is_page_number(group, page_height):
        return (
            Paragraph(
                id=element_id,
                spine_index=spine_index,
                text=group.text,
                skip_translate=True,
                span=span,
                confidence=confidence,
            ),
            RegionKind.PAGE_NUMBER,
        )
    # Heading before list: a numbered section title (``1.1. Background``) shares
    # the ordered-marker shape with a list item, and only the typography tells
    # them apart -- a heading is larger/bold, a list item is body text.
    level = _heading_level(group, body)
    if level is not None:
        return (
            Heading(
                id=element_id,
                spine_index=spine_index,
                text=group.text,
                level=level,
                span=span,
                confidence=confidence,
            ),
            RegionKind.BODY,
        )
    list_marker = pdf_list_marker(group.text)
    if list_marker is not None:
        marker, item_text = list_marker
        return (
            ListItem(
                id=element_id,
                spine_index=spine_index,
                text=item_text,
                marker=marker,
                span=span,
                confidence=confidence,
            ),
            RegionKind.BODY,
        )
    if looks_like_listing(group.text):
        return (
            CodeBlock(
                id=element_id,
                spine_index=spine_index,
                text=group.text,
                skip_translate=True,
                span=span,
                confidence=confidence,
            ),
            RegionKind.BODY,
        )
    if looks_like_debris(group.text):
        return (
            Formula(
                id=element_id,
                spine_index=spine_index,
                source=group.text,
                skip_translate=True,
                span=span,
                confidence=confidence,
            ),
            RegionKind.BODY,
        )
    return (
        Paragraph(
            id=element_id,
            spine_index=spine_index,
            text=group.text,
            span=span,
            confidence=confidence,
        ),
        RegionKind.BODY,
    )


def _with_chars(element: ElementT, chars: tuple[int, int]) -> ElementT:
    """Return a copy of ``element`` whose span carries a character range."""
    span: Span | CompositeSpan
    if isinstance(element.span, CompositeSpan):
        span = CompositeSpan(boxes=element.span.boxes, chars=chars)
    else:
        span = Span(page=element.span.page, bbox=element.span.bbox, chars=chars)
    return dataclasses.replace(element, span=span)


def _page_count(pdf_path: Path) -> int:
    from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK, open_document

    with PDFIUM_LOCK, open_document(pdf_path) as document:
        return len(document)


def read_pdf(
    path: str | Path,
    *,
    doc_id: str | None = None,
    pages: Iterable[int] | None = None,
) -> Document:
    """Read a born-digital PDF into a typed :class:`Document`.

    ``pages`` is a 1-based selection; ``None`` reads every page. A page whose
    geometry cannot be extracted is skipped (its text is genuinely unavailable),
    never silently merged.
    """
    from ubt.adapters.pdf import textgeom

    pdf_path = Path(path)
    wanted = list(range(1, _page_count(pdf_path) + 1)) if pages is None else sorted(set(pages))

    geometry: list[PageGeometry] = []
    provisional: list[tuple[ElementT, RegionKind]] = []
    texts: list[str] = []
    sequence = 0

    for page_no in wanted:
        try:
            lines, (width, height) = textgeom.extract_lines(pdf_path, page_no)
        except Exception:  # an unreadable page is reported by absence, not lost silently
            continue
        geometry.append(PageGeometry(index=page_no, width_pt=width, height_pt=height))
        # extract_lines already glues row fragments and applies column reading
        # order (textgeom.py), so it returns ready-to-group lines.
        ordered = list(lines)
        # Pair each TOC title with its right-margin page number, drop the
        # standalone number lines, and widen each row's box to span the whole
        # row so the compositor can redraw title + leaders + number.
        toc_pairs = _find_toc_pairs(ordered, width)
        toc_page_by_id = {id(title): page for title, _number, page in toc_pairs}
        toc_bbox_by_id = {
            id(title): (
                min(title.rect[0], number.rect[0]),
                min(title.rect[1], number.rect[1]),
                max(title.rect[2], number.rect[2]),
                max(title.rect[3], number.rect[3]),
            )
            for title, number, _page in toc_pairs
        }
        drop_ids = {id(number) for _title, number, _page in toc_pairs}
        if drop_ids:
            ordered = [line for line in ordered if id(line) not in drop_ids]
        body = _body_size(ordered)
        for line_group in _group_lines(ordered, body):
            toc_page = next(
                (toc_page_by_id[id(line)] for line in line_group if id(line) in toc_page_by_id),
                None,
            )
            toc_bbox = next(
                (toc_bbox_by_id[id(line)] for line in line_group if id(line) in toc_bbox_by_id),
                None,
            )
            group = _group_from_lines(line_group, page_no, toc_page=toc_page, toc_bbox=toc_bbox)
            if group is None:
                continue
            element, region_kind = _classify(
                group, body, f"pdf#{sequence:05d}", sequence, page_height=height
            )
            texts.append(group.text)
            provisional.append((element, region_kind))
            sequence += 1

    # Regions are the reader's own layout judgement: consecutive elements of the
    # same page-furniture kind (body vs page number) form one region.
    regions: list[Region] = []
    current_kind = RegionKind.BODY
    current: list[ElementT] = []
    cursor = 0
    for (element, region_kind), text in zip(provisional, texts, strict=True):
        element = _with_chars(element, (cursor, cursor + len(text)))
        cursor += len(text) + 1
        if current and region_kind is not current_kind:
            regions.append(
                Region(id=f"r{len(regions)}", kind=current_kind, elements=tuple(current))
            )
            current = []
        current_kind = region_kind
        current.append(element)
    if current:
        regions.append(Region(id=f"r{len(regions)}", kind=current_kind, elements=tuple(current)))

    source = CanonicalSource(
        doc_id=doc_id or file_digest(pdf_path),
        path=str(pdf_path),
        text="\n".join(texts),
        pages=tuple(geometry),
    )
    return Document(source=source, regions=tuple(regions))


__all__ = ["read_pdf"]
