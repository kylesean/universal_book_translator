from __future__ import annotations

import pytest

from ubt.adapters.pdf.docling_parser import drop_phantom_continuation_boxes
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.model.span import CompositeSpan, PhysicalBox

pytestmark = pytest.mark.fast


def _block(block_id: str, page: int, y0: float, y1: float, *, chain: bool = False) -> IRBlock:
    boxes = (
        (
            PhysicalBox.of(page, (50.0, y0, 550.0, y1)),
            PhysicalBox.of(page + 1, (50.0, 398.9, 550.0, 414.4)),
        )
        if chain
        else ()
    )
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="text",
            bbox=BoundingBox(page=page, x0=50.0, y0=y0, x1=550.0, y1=y1),
            span=CompositeSpan(boxes=boxes) if boxes else None,
        )
    )
    if chain:
        block.provenance.physical_boxes = [{"page": b.page, "bbox": list(b.bbox)} for b in boxes]
    return block


def test_a_chain_box_owned_by_another_block_is_dropped() -> None:
    para = _block("p1", 5, 94.0, 124.0, chain=True)
    caption = _block("c1", 6, 398.9, 414.4)

    out = drop_phantom_continuation_boxes([para, caption])
    by_id = {b.id: b for b in out}

    assert by_id["p1"].provenance.physical_boxes == []
    assert not isinstance(by_id["p1"].element.span, CompositeSpan)
    assert by_id["p1"].bbox is not None
    assert by_id["p1"].bbox.page == 5
    assert by_id["c1"].bbox is not None
    assert by_id["c1"].bbox.page == 6


def test_a_genuine_cross_page_continuation_is_kept() -> None:
    para = _block("p1", 5, 94.0, 124.0, chain=True)

    out = drop_phantom_continuation_boxes([para])
    (only,) = out

    assert len(only.provenance.physical_boxes) == 2
    assert isinstance(only.element.span, CompositeSpan)
    assert len(only.element.span.boxes) == 2


def test_a_single_box_block_is_untouched() -> None:
    plain = _block("p1", 5, 94.0, 124.0)
    out = drop_phantom_continuation_boxes([plain])
    assert out[0].provenance.physical_boxes == []
