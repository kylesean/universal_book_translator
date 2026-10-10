"""Unit tests for table overlay generation in LayerCompositor."""

from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.render.outputs import _overlayable, overlays_from_blocks

pytestmark = pytest.mark.fast


def test_simple_table_is_overlayable() -> None:
    block = IRBlock(
        element=make_element(
            id="tbl_1",
            spine_index=1,
            block_type=BlockType.TABLE,
            source_text="| A | B |\n| --- | --- |\n| 1 | 2 |",
            bbox=BoundingBox(x0=50.0, y0=500.0, x1=350.0, y1=600.0, page=1),
        )
    )
    block.target_text = "| 甲 | 乙 |\n| --- | --- |\n| 一 | 二 |"
    block.status = BlockStatus.MTQE_PASSED

    assert _overlayable(block) is True

    overlays = overlays_from_blocks([block])
    assert len(overlays) == 1
    assert overlays[0].kind == "table"
    assert overlays[0].page == 1
    assert "甲" in overlays[0].text


def test_complex_table_is_not_overlayable() -> None:
    # 7 columns -> complex table
    block = IRBlock(
        element=make_element(
            id="tbl_complex",
            spine_index=1,
            block_type=BlockType.TABLE,
            source_text="| C1 | C2 | C3 | C4 | C5 | C6 | C7 |\n| --- | --- | --- | --- | --- | --- | --- |\n| 1 | 2 | 3 | 4 | 5 | 6 | 7 |",
            bbox=BoundingBox(x0=50.0, y0=500.0, x1=400.0, y1=600.0, page=1),
        )
    )
    block.target_text = "| 栏1 | 栏2 | 栏3 | 栏4 | 栏5 | 栏6 | 栏7 |\n| --- | --- | --- | --- | --- | --- | --- |\n| 1 | 2 | 3 | 4 | 5 | 6 | 7 |"
    block.status = BlockStatus.MTQE_PASSED

    assert _overlayable(block) is False

    overlays = overlays_from_blocks([block])
    assert len(overlays) == 0


def test_skipped_table_is_not_overlayable() -> None:
    block = IRBlock(
        element=make_element(
            id="tbl_skipped",
            spine_index=1,
            block_type=BlockType.TABLE,
            source_text="| A | B |\n| --- | --- |\n| 1 | 2 |",
            bbox=BoundingBox(x0=50.0, y0=500.0, x1=350.0, y1=600.0, page=1),
        )
    )
    block.target_text = "| 甲 | 乙 |\n| --- | --- |\n| 一 | 二 |"
    block.skip_translate = True

    assert _overlayable(block) is False
