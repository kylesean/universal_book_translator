"""Assemble a reader's elements into a ``Document`` (ADR-0001 Phase 1).

Every native reader decides its own structure; this is the shared tail. Each
element's text is normalized (ADR §12 Q3), then stamped with a character range
into the canonical stream -- which *is* the concatenation of those texts. One
owner for the "``Span.chars`` slices the element's own text out of
``CanonicalSource.text``" promise, so a reader cannot stamp offsets a different
way than its sibling.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ubt.analyze.normalize import normalize_text
from ubt.model.ast import (
    Document,
    ElementT,
    Formula,
    Region,
    RegionKind,
    Table,
    TextElement,
)
from ubt.model.span import CanonicalSource, PageGeometry, Span


def element_text(element: ElementT) -> str:
    """The text a ``Span.chars`` range indexes for this element class."""
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    return ""


def normalized(element: ElementT) -> ElementT:
    """The element with its canonical text normalized (ADR §12 Q3)."""
    if isinstance(element, TextElement):
        return dataclasses.replace(element, text=normalize_text(element.text))
    if isinstance(element, Formula):
        return dataclasses.replace(element, source=normalize_text(element.source))
    if isinstance(element, Table):
        return dataclasses.replace(element, markup=normalize_text(element.markup))
    return element


def number(elements: Sequence[ElementT], prefix: str) -> list[ElementT]:
    """Give each element its stable id and reading-order index.

    Ids are ``<prefix>#<n>`` in reading order -- the same id space the PDF reader
    uses (``pdf#…``), so a reader's elements are addressable without an
    ``IRBlock`` detour.
    """
    return [
        dataclasses.replace(element, id=f"{prefix}#{index:05d}", spine_index=index)
        for index, element in enumerate(elements)
    ]


def assemble(
    elements: Sequence[ElementT],
    *,
    doc_id: str,
    path: str = "",
    pages: tuple[PageGeometry, ...] = (),
) -> Document:
    """Stamp reading-order char ranges and build the canonical stream they index.

    Each element keeps the ``span.page`` its reader gave it (``0`` for a format
    without pagination); the reader's structural choice (element class, region)
    is preserved, only the character range is added here.
    """
    texts: list[str] = []
    stamped: list[ElementT] = []
    cursor = 0
    for element in elements:
        element = normalized(element)
        text = element_text(element)
        stamped.append(
            dataclasses.replace(
                element, span=Span(page=element.span.page, chars=(cursor, cursor + len(text)))
            )
        )
        texts.append(text)
        cursor += len(text) + 1
    source = CanonicalSource(doc_id=doc_id, path=path, text="\n".join(texts), pages=pages)
    if not stamped:
        return Document(source=source, regions=())
    region = Region(id="r0", kind=RegionKind.BODY, elements=tuple(stamped))
    return Document(source=source, regions=(region,))


__all__ = ["assemble", "element_text", "normalized", "number"]
