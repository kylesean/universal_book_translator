"""Bootstrap bridge: ``IRBlock`` <-> the typed Document AST.

``IRBlock`` is the pipeline's working form: content plus a dozen pipeline
fields (status, scores, translation). The AST models only the *document* -- what
an element is, where it came from, how it reads. The bridge projects the one
into the other so the kernel can migrate to the typed model piecemeal.

The round-trip preserves everything that defines the document:

    id, spine_index, block_type, flow_id, layout_role,
    structure_role, source_text, skip_translate, bbox

It deliberately does **not** carry pipeline state (status, ``target_text``,
scores, flags, provenance): those belong to execution, not understanding, and
are re-attached by the pipeline. ``structure_role`` is re-derived from the
element type and ``layout_role`` from the region kind -- the element class and
the region are the single source of those axes, which is exactly the
"one attribute, one origin" rule ``IRBlock``'s three role enums broke.

The `scripts/shadow_ast.py` acceptance proves the round-trip is lossless over
the corpus.
"""

from __future__ import annotations

from typing import Any

from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
)
from ubt.model.ast import (
    Caption,
    CodeBlock,
    Dialogue,
    Document,
    Element,
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
from ubt.model.span import CanonicalSource, PageGeometry, Span

_FLOW_TO_KIND: dict[FlowID, FlowKind] = {
    FlowID.MAIN_STORY: FlowKind.MAIN,
    FlowID.SIDEBAR_ASIDE: FlowKind.SIDEBAR,
    FlowID.FOOTNOTE: FlowKind.FOOTNOTE,
    FlowID.TABLE_GRID: FlowKind.TABLE_GRID,
    FlowID.CAPTION: FlowKind.CAPTION,
}
_KIND_TO_FLOW: dict[FlowKind, FlowID] = {kind: flow for flow, kind in _FLOW_TO_KIND.items()}

# --------------------------------------------------------------------------- #
# IRBlock -> Document
# --------------------------------------------------------------------------- #


def _span(block: IRBlock) -> Span:
    bbox = block.bbox
    if bbox is None:
        return Span()
    return Span(page=bbox.page, bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1))


def _region_kind(block: IRBlock) -> RegionKind:
    """The page-furniture role that groups this block (layout axis, single source)."""
    if block.layout_role is not None:
        return block.layout_role
    if block.flow_id is FlowID.CAPTION:
        return RegionKind.CAPTION
    if block.flow_id is FlowID.FOOTNOTE:
        return RegionKind.FOOTNOTE
    return RegionKind.BODY


def element_from_block(block: IRBlock) -> ElementT:
    """Project one ``IRBlock`` into its typed element (document structure only)."""
    common: dict[str, Any] = {
        "id": block.id,
        "spine_index": block.spine_index,
        "span": _span(block),
        "flow": _FLOW_TO_KIND.get(block.flow_id, FlowKind.MAIN),
        "skip_translate": block.skip_translate,
    }
    text = block.source_text or ""
    block_type = block.block_type
    if block_type is BlockType.NARRATIVE:
        if block.layout_role is RegionKind.CAPTION or block.flow_id is FlowID.CAPTION:
            return Caption(text=text, **common)
        return Paragraph(text=text, **common)
    if block_type is BlockType.DIALOGUE:
        return Dialogue(text=text, **common)
    if block_type is BlockType.HEADING:
        return Heading(text=text, level=1, **common)
    if block_type is BlockType.LIST_ITEM:
        return ListItem(text=text, marker="", **common)
    if block_type is BlockType.CODE:
        return CodeBlock(text=text, **common)
    if block_type is BlockType.FORMULA:
        return Formula(source=text, **common)
    if block_type is BlockType.TABLE:
        return Table(markup=text, **common)
    # BlockType.IMAGE is the only remaining kind.
    return Figure(asset_id=text, **common)


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
        kind = _region_kind(block)
        if current and kind is not current_kind:
            regions.append(
                Region(id=f"r{len(regions)}", kind=current_kind, elements=tuple(current))
            )
            current = []
        current_kind = kind
        current.append(element)
    if current:
        regions.append(Region(id=f"r{len(regions)}", kind=current_kind, elements=tuple(current)))
    source = CanonicalSource(doc_id=doc_id, path=path, pages=tuple(pages))
    return Document(source=source, regions=tuple(regions))


# --------------------------------------------------------------------------- #
# Document -> IRBlock
# --------------------------------------------------------------------------- #


def _element_block_type(element: Element) -> BlockType:
    if isinstance(element, Heading):
        return BlockType.HEADING
    if isinstance(element, Dialogue):
        return BlockType.DIALOGUE
    if isinstance(element, ListItem):
        return BlockType.LIST_ITEM
    if isinstance(element, CodeBlock):
        return BlockType.CODE
    if isinstance(element, Formula):
        return BlockType.FORMULA
    if isinstance(element, Table):
        return BlockType.TABLE
    if isinstance(element, Figure):
        return BlockType.IMAGE
    # Paragraph and Caption both live in the NARRATIVE block type.
    return BlockType.NARRATIVE


def _element_source(element: Element) -> str:
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    if isinstance(element, Figure):
        return element.asset_id
    return ""


def _bbox(span: Span) -> BoundingBox | None:
    if span.bbox is None:
        return None
    x0, y0, x1, y1 = span.bbox
    return BoundingBox(page=span.page, x0=x0, y0=y0, x1=x1, y1=y1)


def block_from_element(element: ElementT, *, region_kind: RegionKind) -> IRBlock:
    """Rebuild one ``IRBlock`` from its element (structure only; no pipeline state)."""
    block = IRBlock(
        id=element.id,
        flow_id=_KIND_TO_FLOW[element.flow],
        spine_index=element.spine_index,
        block_type=_element_block_type(element),
        bbox=_bbox(element.span),
        source_text=_element_source(element),
        skip_translate=element.skip_translate,
        layout_role=region_kind,
    )
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
