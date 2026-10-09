"""``ubt.model`` -- the pure type layer of the document-compiler kernel.

Zero dependencies on the pipeline, pydantic, or any IO: the model can be
imported, reasoned about, and serialized without a document in hand.

- :mod:`ubt.model.span` -- :class:`Span`, :class:`PageGeometry`,
  :class:`CanonicalSource`: where an element came from.
- :mod:`ubt.model.ast` -- the typed Document AST (:class:`Document`,
  :class:`Region`, and the closed element union).
- :mod:`ubt.model.segment` -- the translation-unit types (:class:`Segment`,
  :class:`Placeholder`).

The AST is the one place that answers "what is this piece of the document?".
An element's *type* is its structure; the region it sits in is its layout.
"""

from __future__ import annotations

from ubt.model.ast import (
    ASSET_ELEMENTS,
    TEXT_ELEMENTS,
    Caption,
    CodeBlock,
    Confidence,
    Dialogue,
    Document,
    Element,
    ElementKind,
    ElementT,
    Figure,
    FlowKind,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Region,
    RegionKind,
    Table,
    TextElement,
    source_slice,
)
from ubt.model.segment import QA, Placeholder, Provenance, Segment, SegmentState
from ubt.model.span import BBox, CanonicalSource, PageGeometry, Span

__all__ = [
    "ASSET_ELEMENTS",
    "TEXT_ELEMENTS",
    "BBox",
    "CanonicalSource",
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
    "PageGeometry",
    "Paragraph",
    "Placeholder",
    "Provenance",
    "QA",
    "Region",
    "RegionKind",
    "Segment",
    "SegmentState",
    "Span",
    "Table",
    "TextElement",
    "source_slice",
]
