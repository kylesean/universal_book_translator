"""``ubt.model`` -- the pure type layer of the document-compiler kernel.

Zero dependencies on the pipeline, pydantic, or any IO: the model can be
imported, reasoned about, and serialized without a document in hand.

- :mod:`ubt.model.span` -- :class:`Span`, :class:`PageGeometry`,
  :class:`CanonicalSource`: where an element came from.
- :mod:`ubt.model.ast` -- the typed Document AST (:class:`Document`,
  :class:`Region`, and the closed element union).
- :mod:`ubt.model.fidelity` -- the :class:`Fidelity` lattice, :class:`Proof`,
  and :class:`Attestation`.

This layer is the kernel's centre of gravity: the pipeline's ``IRBlock`` is
bridged into a :class:`Document` (see :mod:`ubt.analyze.bridge`) so the rest of
the kernel can reason over typed structure instead of a bag of optional role
enums.
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
)
from ubt.model.fidelity import (
    FIDELITY_DESCENT,
    Attestation,
    Fidelity,
    Proof,
    ProofKind,
    ProofOutcome,
)
from ubt.model.segment import QA, Placeholder, Provenance, Segment, SegmentState
from ubt.model.span import BBox, CanonicalSource, PageGeometry, Span

__all__ = [
    "ASSET_ELEMENTS",
    "FIDELITY_DESCENT",
    "TEXT_ELEMENTS",
    "Attestation",
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
    "Fidelity",
    "Figure",
    "FlowKind",
    "Formula",
    "Heading",
    "ListItem",
    "PageGeometry",
    "Paragraph",
    "Placeholder",
    "Proof",
    "ProofKind",
    "ProofOutcome",
    "Provenance",
    "QA",
    "Region",
    "RegionKind",
    "Segment",
    "SegmentState",
    "Span",
    "Table",
    "TextElement",
]
