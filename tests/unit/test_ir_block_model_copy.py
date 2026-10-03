"""Unit tests for IRBlock.model_copy override supporting structural and state updates."""

from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, FlowID, IRBlock, make_element
from ubt.model.ast import RegionKind

pytestmark = pytest.mark.fast


def test_ir_block_model_copy_source_text() -> None:
    el = make_element(
        id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="Original Text"
    )
    block = IRBlock(element=el)
    assert block.source_text == "Original Text"

    copied = block.model_copy(update={"source_text": "Updated Text"})
    assert copied.source_text == "Updated Text"
    assert block.source_text == "Original Text"  # original untouched
    # Ensure with_source_text produces identical outcome
    assert block.with_source_text("Updated Text").source_text == "Updated Text"


def test_ir_block_model_copy_structural_fields() -> None:
    el = make_element(
        id="b1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Hello",
        bbox=BoundingBox(page=1, x0=10, y0=20, x1=100, y1=200),
        skip_translate=False,
    )
    block = IRBlock(element=el)

    new_bbox = BoundingBox(page=2, x0=5, y0=10, x1=50, y1=100)
    copied = block.model_copy(
        update={
            "id": "b2",
            "spine_index": 2,
            "skip_translate": True,
            "bbox": new_bbox,
            "region": RegionKind.FOOTNOTE,
            "flow_id": FlowID.FOOTNOTE,
        }
    )

    assert copied.id == "b2"
    assert copied.spine_index == 2
    assert copied.skip_translate is True
    assert copied.bbox == new_bbox
    assert copied.region == RegionKind.FOOTNOTE
    assert copied.flow_id == FlowID.FOOTNOTE


def test_ir_block_model_copy_mixed_structural_and_execution_state() -> None:
    el = make_element(id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="Source")
    block = IRBlock(element=el, status=BlockStatus.PENDING)

    copied = block.model_copy(
        update={
            "source_text": "New Source",
            "target_text": "Target Translation",
            "status": BlockStatus.MTQE_PASSED,
            "mtqe_score": 0.98,
        }
    )

    assert copied.source_text == "New Source"
    assert copied.target_text == "Target Translation"
    assert copied.status == BlockStatus.MTQE_PASSED
    assert copied.mtqe_score == 0.98
