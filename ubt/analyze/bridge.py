"""Bridge: ``IRBlock`` <-> the typed Document AST.

``IRBlock`` carries its structure as the typed :class:`~ubt.model.ast.Element`
it holds (single source of truth: one attribute, one origin); the pipeline's mutable fields are
execution state plus typography. This module projects a flat block list into a
:class:`~ubt.model.ast.Document` and back -- the shape the HTML/EPUB views and
the content graph consume.

The round-trip preserves everything that defines the document: the element
(class, flow, region, span, text, ``skip_translate``) and the block id /
reading order. It deliberately does **not** carry pipeline state (status,
``target_text``, scores, flags): those belong to execution, not understanding.

The bridge round-trip is pinned losslessly by
`tests/unit/analyze/test_bridge_roundtrip.py`.
"""

from __future__ import annotations

import dataclasses

from ubt.core.ir.models import IRBlock
from ubt.model.ast import Document, ElementT, Region, RegionKind
from ubt.model.span import CanonicalSource, CompositeSpan, PageGeometry


def element_from_block(block: IRBlock) -> ElementT:
    """The typed element a block is made of (the block's structure)."""
    return block.element


def document_from_blocks(
    blocks: list[IRBlock],
    *,
    doc_id: str,
    path: str = "",
    pages: tuple[PageGeometry, ...] = (),
) -> Document:
    """Build a :class:`Document` from a flat, unordered list of ``IRBlock``s."""
    ordered = sorted(blocks, key=lambda block: block.spine_index)
    regions: list[Region] = []
    current_kind = RegionKind.BODY
    current: list[ElementT] = []
    for block in ordered:
        element = element_from_block(block)
        kind = element.region
        if current and kind is not current_kind:
            regions.append(
                Region(id=f"r{len(regions)}", kind=current_kind, elements=tuple(current))
            )
            current = []
        current_kind = kind
        current.append(element)
    if current:
        regions.append(Region(id=f"r{len(regions)}", kind=current_kind, elements=tuple(current)))
    canonical_text = "\n".join(b.source_text for b in ordered)
    source = CanonicalSource(doc_id=doc_id, path=path, text=canonical_text, pages=tuple(pages))
    return Document(source=source, regions=tuple(regions))


def block_from_element(element: ElementT, *, region_kind: RegionKind) -> IRBlock:
    """Rebuild one ``IRBlock`` from its element (structure only; no pipeline state)."""
    block = IRBlock(element=dataclasses.replace(element, region=region_kind))
    span = element.span
    if isinstance(span, CompositeSpan) and len(span.boxes) > 1:
        # Mirror the element's box chain into the one provenance key the ledger
        # can rebuild a CompositeSpan from (ledger_base._row_to_block); without
        # it the chain collapses to the first box across the round-trip.
        block.provenance = {
            "physical_boxes": [{"page": box.page, "bbox": list(box.bbox)} for box in span.boxes]
        }
    return block


def blocks_from_document(document: Document) -> list[IRBlock]:
    """Rebuild the ``IRBlock`` list from a :class:`Document`, in reading order."""
    return [
        block_from_element(element, region_kind=region.kind)
        for region in document.regions
        for element in region.elements
    ]


__all__ = [
    "block_from_element",
    "blocks_from_document",
    "document_from_blocks",
    "element_from_block",
]
