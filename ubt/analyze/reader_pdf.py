"""Native PDF reader: a PDF -> the typed Document AST (ADR-0001 Phase 1).

This is the "difficulty concentrated in one place" the ADR calls for. ``adapters``
supply *raw geometry* (text lines, their boxes and font metrics via
:mod:`ubt.adapters.pdf.textgeom`); this module owns *structure* -- reading order,
paragraph grouping, heading detection -- and emits a
:class:`~ubt.model.ast.Document` directly, with no ``IRBlock`` detour.

One rule set, stated once and documented, replaces the competing heuristics the
old flow lived with:

1. a vertical gap wider than :data:`PARAGRAPH_GAP_FACTOR` of the font starts a
   new paragraph;
2. a font-size change over :data:`FONT_CHANGE_RATIO` starts a new paragraph;
3. a first-line indent (``x0`` moves right by more than :data:`INDENT_FACTOR`
   font sizes) starts a new paragraph;
4. a line beginning with a bullet/number starts a list item.

A heading is decided by *typography* (font size relative to the page's body
size, or bold + short + no terminal punctuation), never by length alone -- the
rule that made the old plain-text path mislabel every fragment.

The reader never drops text for being unsure: an element it cannot classify is
emitted as prose with :attr:`Confidence.INFERRED`. Text loss would only come from
a page whose geometry cannot be read at all, and that is reported by omission.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.model.ast import (
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
from ubt.model.span import CanonicalSource, PageGeometry, Span

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
_LIST_PREFIXES = ("\u2022", "\u25e6", "\u2023", "-", "*", "\u00b7", "\u2013")
_SENTENCE_END = (".", "!", "?", "\u3002", ":", ";")
_WS_RE = re.compile(r"\s+")


@dataclasses.dataclass(frozen=True, slots=True)
class _Group:
    """One provisional element before the canonical text and spans are assigned."""

    text: str
    page: int
    bbox: tuple[float, float, float, float]
    max_size: float
    all_bold: bool


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
    return out


def _starts_new_paragraph(previous: LineBox, current: LineBox, body: float) -> bool:
    font = max(previous.font_size, current.font_size, body, 1.0)
    # Column/margin jump or a first-line indent.
    if current.rect[0] - previous.rect[0] > INDENT_FACTOR * font:
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
    for line in lines:
        if not line.text.strip():
            continue
        if current and _starts_new_paragraph(current[-1], line, body):
            groups.append(current)
            current = []
        current.append(line)
    if current:
        groups.append(current)
    return groups


def _group_from_lines(lines: list[LineBox], page: int) -> _Group | None:
    text = _join_lines(lines)
    if not text:
        return None
    sizes = [line.font_size for line in lines if line.font_size > 0]
    return _Group(
        text=text,
        page=page,
        bbox=(
            min(line.rect[0] for line in lines),
            min(line.rect[1] for line in lines),
            max(line.rect[2] for line in lines),
            max(line.rect[3] for line in lines),
        ),
        max_size=max(sizes) if sizes else 0.0,
        all_bold=all(line.bold for line in lines),
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


def _classify(group: _Group, body: float, element_id: str, spine_index: int) -> ElementT:
    """Turn a provisional group into its typed element (never drops it)."""
    confidence = Confidence.INFERRED
    if group.text.startswith(_LIST_PREFIXES):
        marker = group.text[0]
        return ListItem(
            id=element_id,
            spine_index=spine_index,
            text=group.text[len(marker) :].strip(),
            marker=marker,
            span=Span(page=group.page, bbox=group.bbox),
            confidence=confidence,
        )
    level = _heading_level(group, body)
    if level is not None:
        return Heading(
            id=element_id,
            spine_index=spine_index,
            text=group.text,
            level=level,
            span=Span(page=group.page, bbox=group.bbox),
            confidence=confidence,
        )
    from ubt.core.validators.math_guard import looks_like_math_debris

    if looks_like_math_debris(group.text):
        return Formula(
            id=element_id,
            spine_index=spine_index,
            source=group.text,
            skip_translate=True,
            span=Span(page=group.page, bbox=group.bbox),
            confidence=confidence,
        )
    return Paragraph(
        id=element_id,
        spine_index=spine_index,
        text=group.text,
        span=Span(page=group.page, bbox=group.bbox),
        confidence=confidence,
    )


def _with_chars(element: ElementT, chars: tuple[int, int]) -> ElementT:
    """Return a copy of ``element`` whose span carries a character range."""
    span = Span(page=element.span.page, bbox=element.span.bbox, chars=chars)
    return dataclasses.replace(element, span=span)


def _page_count(pdf_path: Path) -> int:
    from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK, open_document

    with PDFIUM_LOCK, open_document(pdf_path) as document:
        return len(document)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


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
    provisional: list[ElementT] = []
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
        body = _body_size(ordered)
        for line_group in _group_lines(ordered, body):
            group = _group_from_lines(line_group, page_no)
            if group is None:
                continue
            texts.append(group.text)
            provisional.append(_classify(group, body, f"pdf#{sequence:05d}", sequence))
            sequence += 1

    elements: list[ElementT] = []
    cursor = 0
    for element, text in zip(provisional, texts, strict=True):
        elements.append(_with_chars(element, (cursor, cursor + len(text))))
        cursor += len(text) + 1

    source = CanonicalSource(
        doc_id=doc_id or _file_digest(pdf_path),
        path=str(pdf_path),
        text="\n".join(texts),
        pages=tuple(geometry),
    )
    region = Region(id="r0", kind=RegionKind.BODY, elements=tuple(elements))
    return Document(source=source, regions=(region,) if elements else ())


__all__ = ["read_pdf"]
