"""The typed AST round-trips blocks without loss.

The ADR's Phase-1 gate, on synthetic fixtures: bridge a flat ``IRBlock`` list
into the typed :class:`~ubt.model.ast.Document` and back, and the
document-defining fields must be identical -- id, spine_index, block_type,
flow_id, region, source_text, skip_translate, bbox. Pipeline state (status,
``target_text``, scores) is deliberately *not* carried: execution belongs to
the pipeline, not to understanding.
"""

from __future__ import annotations

import pytest

from ubt.analyze.bridge import blocks_from_document, document_from_blocks
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.model.ast import (
    Caption,
    CodeBlock,
    ElementT,
    FlowKind,
    Formula,
    Heading,
    Paragraph,
    RegionKind,
)
from ubt.model.span import Span

pytestmark = pytest.mark.fast


def _element(element: ElementT) -> IRBlock:
    return IRBlock(element=element)


def _blocks() -> list[IRBlock]:
    # Deliberately out of reading order: the document must restore spine order.
    return [
        _element(
            Paragraph(
                id="p1",
                spine_index=1,
                span=Span(page=1, chars=(12, 52)),
                text="Body text between the headings.",
            )
        ),
        _element(
            Heading(id="h1", spine_index=0, span=Span(page=1, chars=(0, 7)), text="Chapter One")
        ),
        _element(
            CodeBlock(
                id="c1",
                spine_index=2,
                span=Span(page=1, chars=(52, 66)),
                text="x = y + 1",
                skip_translate=True,
            )
        ),
        _element(
            Caption(
                id="cap1",
                spine_index=3,
                span=Span(page=1, chars=(66, 84)),
                text="Figure 1: the result.",
                region=RegionKind.CAPTION,
            )
        ),
        _element(
            Paragraph(
                id="fn1",
                spine_index=4,
                span=Span(page=1, chars=(84, 100)),
                text="A footnote worth keeping.",
                flow=FlowKind.FOOTNOTE,
                region=RegionKind.FOOTNOTE,
            )
        ),
        _element(
            Formula(
                id="f1",
                spine_index=5,
                span=Span(page=2, bbox=(10.0, 20.0, 210.0, 45.0)),
                source="E = mc^2",
            )
        ),
    ]


def _projection(block: IRBlock) -> dict[str, object]:
    return {
        "spine_index": block.spine_index,
        "block_type": block.block_type.value,
        "flow_id": block.flow_id.value,
        "region": block.region.value,
        "source_text": block.source_text or "",
        "skip_translate": block.skip_translate,
        "bbox": None
        if block.bbox is None
        else (block.bbox.page, block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1),
    }


def test_every_document_defining_field_survives_the_round_trip() -> None:
    blocks = _blocks()
    document = document_from_blocks(blocks, doc_id="doc", path="book.md")
    rebuilt = blocks_from_document(document)

    assert {b.id for b in rebuilt} == {b.id for b in blocks}
    by_id = {b.id: b for b in rebuilt}
    for before in blocks:
        after = by_id[before.id]
        assert _projection(after) == _projection(before), before.id


def test_reading_order_is_restored_from_spine_index() -> None:
    document = document_from_blocks(_blocks(), doc_id="doc", path="book.md")
    rebuilt = blocks_from_document(document)
    assert [b.id for b in rebuilt] == ["h1", "p1", "c1", "cap1", "fn1", "f1"]


def test_consecutive_same_region_blocks_group_and_boundaries_split_regions() -> None:
    document = document_from_blocks(_blocks(), doc_id="doc", path="book.md")
    assert [(r.kind, len(r.elements)) for r in document.regions] == [
        (RegionKind.BODY, 3),
        (RegionKind.CAPTION, 1),
        (RegionKind.FOOTNOTE, 1),
        (RegionKind.BODY, 1),
    ]


def test_execution_state_is_deliberately_not_carried() -> None:
    # The bridge understands the document; the pipeline owns its state. A block
    # rebuilt from a document is structure only: pending, unscored, untranslated.
    blocks = _blocks()
    for block in blocks:
        block.status = BlockStatus.MTQE_PASSED
        block.target_text = "translated elsewhere"
    document = document_from_blocks(blocks, doc_id="doc", path="book.md")
    rebuilt = {b.id: b for b in blocks_from_document(document)}
    for block_id, block in rebuilt.items():
        assert block.status is BlockStatus.PENDING, block_id
        assert block.target_text is None, block_id


def test_block_type_and_flow_projection_matches_the_ir_vocabulary() -> None:
    by_id = {b.id: b for b in _blocks()}
    assert by_id["h1"].block_type is BlockType.HEADING
    assert by_id["c1"].block_type is BlockType.CODE
    assert by_id["c1"].skip_translate is True
    assert by_id["cap1"].flow_id is FlowID.MAIN_STORY  # region, not flow, carries it
    assert by_id["fn1"].flow_id is FlowID.FOOTNOTE
