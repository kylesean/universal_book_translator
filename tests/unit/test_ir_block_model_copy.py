"""Unit tests for IRBlock.model_copy override supporting structural and state updates."""

from __future__ import annotations

import pytest

from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BoundingBox,
    FlowID,
    InlineRun,
    IRBlock,
    StyleMeta,
    make_element,
)
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


def test_ir_block_model_copy_isolates_mutable_state() -> None:
    # A shallow copy shares error_flags/provenance, so a later append on the
    # copy mutated the original block. The default is deep now.
    el = make_element(id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="Hi")
    block = IRBlock(element=el)
    block.error_flags.append("orig")
    block.provenance.parser = "v"

    copied = block.model_copy(update={"target_text": "你好"})
    assert copied.error_flags is not block.error_flags
    assert copied.provenance is not block.provenance
    copied.error_flags.append("added_on_copy")
    copied.provenance.parser = "v2"
    assert block.error_flags == ["orig"]
    assert block.provenance.parser == "v"
    assert copied.provenance.parser == "v2"


def test_ir_block_from_element_factories() -> None:
    el1 = make_element(id="e1", spine_index=0, block_type=BlockType.NARRATIVE, source_text="One")
    el2 = make_element(id="e2", spine_index=1, block_type=BlockType.NARRATIVE, source_text="Two")

    block1 = IRBlock.from_element(el1)
    assert block1.element is el1
    assert block1.id == "e1"
    assert block1.source_text == "One"

    blocks = IRBlock.from_elements([el1, el2])
    assert len(blocks) == 2
    assert blocks[0].id == "e1"
    assert blocks[1].id == "e2"


def test_style_meta_round_trips_target_runs() -> None:
    style = StyleMeta(
        inline_runs=(InlineRun(text="50.0%", bold=True),),
        target_runs=(InlineRun(text="重点", bold=True),),
    )
    restored = StyleMeta.model_validate_json(style.model_dump_json())
    assert restored.inline_runs == style.inline_runs
    assert restored.target_runs == style.target_runs


def test_style_meta_defaults_target_runs_to_empty() -> None:
    assert StyleMeta().target_runs == ()
