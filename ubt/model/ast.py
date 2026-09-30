"""The typed Document AST (ADR-0001 model layer).

This is the one place that answers "what is this piece of the document?". An
element's *type* is its structure; the region it sits in is its layout; an
optional :class:`SemanticKind` carries the semantic axes that are neither
(element type cannot say "this paragraph is an abstract"). ``IRBlock`` spread
that answer across ``block_type`` plus three role enums that could disagree;
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

from ubt.model.span import CanonicalSource, Span


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
    """Page-furniture role of a region (mirrors ``LayoutRole``)."""

    BODY = "body"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    CAPTION = "caption"
    FOOTNOTE = "footnote"


class SemanticKind(StrEnum):
    """Semantic axis the element type cannot express (mirrors ``SemanticRole``)."""

    MAIN_TEXT = "main_text"
    ABSTRACT = "abstract"
    REFERENCE = "reference"
    METADATA = "metadata"
    AFFILIATION = "affiliation"
    UNKNOWN = "unknown"


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

    Never instantiated directly -- use one of the concrete classes below.
    """

    id: str
    spine_index: int
    span: Span
    confidence: Confidence = Confidence.INFERRED
    #: A decorative element (banner, ornament) may be dropped; anything else
    #: must resolve to a translated or preserved realization.
    decorative: bool = False
    flow: FlowKind = FlowKind.MAIN
    semantic: SemanticKind = SemanticKind.MAIN_TEXT
    skip_translate: bool = False

    @property
    def kind(self) -> ElementKind:
        return _ELEMENT_KIND[type(self)]

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


@dataclass(frozen=True, slots=True)
class Paragraph(TextElement):
    pass


@dataclass(frozen=True, slots=True)
class Dialogue(TextElement):
    pass


@dataclass(frozen=True, slots=True)
class ListItem(TextElement):
    marker: str = ""


@dataclass(frozen=True, slots=True)
class Caption(TextElement):
    pass


@dataclass(frozen=True, slots=True)
class CodeBlock(TextElement):
    pass


@dataclass(frozen=True, slots=True)
class Formula(Element):
    """A display equation, carried as its source markup (never retyped by an LLM)."""

    source: str = ""


@dataclass(frozen=True, slots=True)
class Table(Element):
    """A table, carried as its structured markup."""

    markup: str = ""


@dataclass(frozen=True, slots=True)
class Figure(Element):
    """An image/vector asset, referenced by id (never re-drawn from text)."""

    asset_id: str = ""


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

_ELEMENT_KIND: dict[type[Element], ElementKind] = {
    Heading: ElementKind.HEADING,
    Paragraph: ElementKind.PARAGRAPH,
    Dialogue: ElementKind.DIALOGUE,
    ListItem: ElementKind.LIST_ITEM,
    Caption: ElementKind.CAPTION,
    CodeBlock: ElementKind.CODE_BLOCK,
    Formula: ElementKind.FORMULA,
    Table: ElementKind.TABLE,
    Figure: ElementKind.FIGURE,
}


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

    def element(self, element_id: str) -> ElementT | None:
        for element in self.elements:
            if element.id == element_id:
                return element
        return None


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
    "SemanticKind",
    "Table",
    "TextElement",
    "CanonicalSource",
    "Span",
]
