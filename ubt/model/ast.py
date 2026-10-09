"""The typed Document AST (core document model).

This is the one place that answers "what is this piece of the document?". An
element's *type* is its structure; the region it sits in is its layout. ``IRBlock``
spread that answer across ``block_type`` plus role enums that could disagree;
here a single class is the source of truth and the bridge re-derives the rest.

The element set is closed and deliberately small -- one class per kind the
extractors can actually distinguish today. Rich inline structure (emphasis,
links, inline math as nodes) is *not* modelled yet: the current extractors
carry flat text, and inventing an inline tree they cannot fill would be
speculative. When a reader can fill it, it gets added here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from ubt.model.span import CanonicalSource, CompositeSpan, Span


class Confidence(StrEnum):
    """How sure the analyzer is about an element's type and span.

    A three-valued enum, not a float: it exists to trigger a *fallback* ("low
    confidence -> preserve the source rather than guess"), which needs a
    decision, not a number.
    """

    VERIFIED = "verified"  # ground-truth geometry/typography agreed
    INFERRED = "inferred"  # a rule decided it; usually right, not proven
    UNKNOWN = "unknown"  # placement failed; treat as preserve-only


class FlowKind(StrEnum):
    """Context-window partition a text element belongs to (mirrors ``FlowID``)."""

    MAIN = "main_story"
    SIDEBAR = "sidebar_aside"
    FOOTNOTE = "footnote"
    TABLE_GRID = "table_grid"
    CAPTION = "caption"


class RegionKind(StrEnum):
    """Page-furniture role of a region; the one layout vocabulary (single source of truth)."""

    BODY = "body"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    CAPTION = "caption"
    FOOTNOTE = "footnote"


class ElementKind(StrEnum):
    """Flat tag for an element class, for serialization and switching."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    DIALOGUE = "dialogue"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    CODE_BLOCK = "code_block"
    FORMULA = "formula"
    TABLE = "table"
    FIGURE = "figure"


@dataclass(frozen=True, slots=True)
class Element:
    """Base of the AST union: identity, provenance span, and coarse flags.

    Never instantiated directly -- use one of the concrete classes below. Each
    concrete class narrows :attr:`kind` to a ``Literal`` so the closed union can
    be serialized and deserialized without guessing the class back from shape.
    """

    id: str
    spine_index: int
    span: Span | CompositeSpan
    confidence: Confidence = Confidence.INFERRED
    #: A decorative element (banner, ornament) may be dropped; anything else
    #: must resolve to a translated or preserved realization.
    decorative: bool = False
    flow: FlowKind = FlowKind.MAIN
    skip_translate: bool = False
    #: The page-furniture role this element sits in (the extractor's finding,
    #: single layout vocabulary). ``Region`` groups elements by it.
    region: RegionKind = RegionKind.BODY
    #: Discriminator for the closed union; each concrete class narrows it.
    kind: ElementKind = ElementKind.PARAGRAPH
    font_size: float = 0.0

    @property
    def is_text(self) -> bool:
        return isinstance(self, TextElement)

    @property
    def is_asset(self) -> bool:
        return isinstance(self, (Formula, Table, Figure))


@dataclass(frozen=True, slots=True)
class TextElement(Element):
    """An element that carries translatable prose (or a deliberate keep)."""

    text: str = ""


@dataclass(frozen=True, slots=True)
class Heading(TextElement):
    level: int = 1
    kind: Literal[ElementKind.HEADING] = ElementKind.HEADING


@dataclass(frozen=True, slots=True)
class Paragraph(TextElement):
    #: For a table-of-contents row, the page number its dot leaders point at.
    #: The reader pairs the row's title with the right-margin number and keeps
    #: only the title as translatable text; the compositor redraws the leaders
    #: and the number so a translated entry keeps its TOC layout.
    toc_page: str = ""
    kind: Literal[ElementKind.PARAGRAPH] = ElementKind.PARAGRAPH


@dataclass(frozen=True, slots=True)
class Dialogue(TextElement):
    kind: Literal[ElementKind.DIALOGUE] = ElementKind.DIALOGUE


@dataclass(frozen=True, slots=True)
class ListItem(TextElement):
    marker: str = ""
    kind: Literal[ElementKind.LIST_ITEM] = ElementKind.LIST_ITEM


@dataclass(frozen=True, slots=True)
class Caption(TextElement):
    region: RegionKind = RegionKind.CAPTION
    kind: Literal[ElementKind.CAPTION] = ElementKind.CAPTION


@dataclass(frozen=True, slots=True)
class CodeBlock(TextElement):
    kind: Literal[ElementKind.CODE_BLOCK] = ElementKind.CODE_BLOCK


@dataclass(frozen=True, slots=True)
class Formula(Element):
    """A display equation, carried as its source markup (never retyped by an LLM)."""

    source: str = ""
    kind: Literal[ElementKind.FORMULA] = ElementKind.FORMULA


@dataclass(frozen=True, slots=True)
class Table(Element):
    """A table, carried as its structured markup."""

    markup: str = ""
    kind: Literal[ElementKind.TABLE] = ElementKind.TABLE


@dataclass(frozen=True, slots=True)
class Figure(Element):
    """An image/vector asset, referenced by id (never re-drawn from text)."""

    asset_id: str = ""
    kind: Literal[ElementKind.FIGURE] = ElementKind.FIGURE


#: The closed union of concrete element classes.
ElementT = (
    Heading | Paragraph | Dialogue | ListItem | Caption | CodeBlock | Formula | Table | Figure
)

#: Element classes that carry translatable text.
TEXT_ELEMENTS: tuple[type[Element], ...] = (
    Heading,
    Paragraph,
    Dialogue,
    ListItem,
    Caption,
    CodeBlock,
)

#: Element classes that are immutable assets.
ASSET_ELEMENTS: tuple[type[Element], ...] = (Formula, Table, Figure)

#: The closed set of concrete element classes -- the runtime form of
#: :data:`ElementT`. Capability declarations and exhaustive dispatches are built
#: from it, so adding a class is one edit here.
ELEMENT_CLASSES: tuple[type[Element], ...] = (
    Heading,
    Paragraph,
    Dialogue,
    ListItem,
    Caption,
    CodeBlock,
    Formula,
    Table,
    Figure,
)


def source_slice(element: Element, source: CanonicalSource) -> str:
    """The exact source text one element carries, from its character span.

    Falls back to the element's own carried source when the document has no
    canonical stream or the span does not index it -- the slice is then coarser,
    but still the source bytes rather than a reconstruction.
    """
    chars = element.span.chars
    if chars is not None:
        start, end = chars
        if 0 <= start <= end <= len(source.text):
            return source.text[start:end]
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    if isinstance(element, Figure):
        return element.asset_id
    return ""


@dataclass(frozen=True, slots=True)
class Region:
    """A contiguous run of elements sharing one page-furniture role."""

    id: str
    kind: RegionKind
    elements: tuple[ElementT, ...] = ()

    @property
    def text_elements(self) -> tuple[TextElement, ...]:
        return tuple(el for el in self.elements if isinstance(el, TextElement))


@dataclass(frozen=True, slots=True)
class Document:
    """A whole understood document: its source and its regions, in reading order."""

    source: CanonicalSource
    regions: tuple[Region, ...] = ()

    @property
    def elements(self) -> tuple[ElementT, ...]:
        return tuple(el for region in self.regions for el in region.elements)


__all__ = [
    "ASSET_ELEMENTS",
    "ELEMENT_CLASSES",
    "TEXT_ELEMENTS",
    "Caption",
    "CodeBlock",
    "Confidence",
    "Dialogue",
    "Document",
    "Element",
    "ElementKind",
    "ElementT",
    "Figure",
    "FlowKind",
    "Formula",
    "Heading",
    "ListItem",
    "Paragraph",
    "Region",
    "RegionKind",
    "Table",
    "TextElement",
    "CanonicalSource",
    "Span",
    "source_slice",
]
