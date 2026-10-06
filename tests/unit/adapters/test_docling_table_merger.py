"""Table continuation fragments stranded below a Docling table are merged into it."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.docling_blocks import merge_table_continuation_fragments
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _make_block(
    block_id: str,
    block_type: BlockType,
    page: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str,
) -> IRBlock:
    elem = make_element(
        id=block_id,
        spine_index=1,
        block_type=block_type,
        source_text=text,
        bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
    )
    return IRBlock(element=elem)


def test_merge_table_continuation_fragments_absorbs_code_and_formulas() -> None:
    # Table block
    tbl = _make_block("b01", BlockType.TABLE, 1, 70.0, 200.0, 540.0, 400.0, "| Col1 | Col2 |")
    # Immediate continuation fragments
    frag1 = _make_block("b02", BlockType.NARRATIVE, 1, 75.0, 185.0, 150.0, 195.0, "ScheduleSource")
    frag2 = _make_block("b03", BlockType.FORMULA, 1, 155.0, 185.0, 170.0, 195.0, "←→")
    frag3 = _make_block(
        "b04", BlockType.NARRATIVE, 1, 75.0, 170.0, 140.0, 180.0, "next_fire(ctx) ->"
    )
    frag4 = _make_block(
        "b05",
        BlockType.NARRATIVE,
        1,
        145.0,
        170.0,
        530.0,
        180.0,
        "Fire | None Returns next candidate fire",
    )
    # Following separate section heading
    heading = _make_block("b06", BlockType.HEADING, 1, 70.0, 100.0, 200.0, 120.0, "5. Conclusion")

    blocks = [tbl, frag1, frag2, frag3, frag4, heading]
    merged = merge_table_continuation_fragments(blocks)

    assert len(merged) == 2
    assert merged[0].id == "b01"
    assert merged[0].block_type == BlockType.TABLE
    # Bounding box y0 extended from 200.0 down to lowest fragment 170.0
    assert merged[0].bbox is not None
    assert merged[0].bbox.y0 == 170.0
    assert merged[0].bbox.y1 == 400.0
    assert "ScheduleSource" in merged[0].source_text
    assert "next_fire(ctx) ->" in merged[0].source_text
    # Heading preserved as second block
    assert merged[1].id == "b06"
    assert merged[1].block_type == BlockType.HEADING


def test_merge_table_continuation_fragments_does_not_swallow_distant_prose() -> None:
    tbl = _make_block("b01", BlockType.TABLE, 1, 70.0, 300.0, 540.0, 400.0, "| A | B |")
    # Gap is 300.0 - 250.0 = 50.0 > 25.0 max gap
    prose = _make_block(
        "b02",
        BlockType.NARRATIVE,
        1,
        70.0,
        150.0,
        540.0,
        250.0,
        "This is an ordinary narrative paragraph discussing the table results.",
    )

    blocks = [tbl, prose]
    merged = merge_table_continuation_fragments(blocks)

    assert len(merged) == 2
    assert merged[0].bbox is not None
    assert merged[0].bbox.y0 == 300.0
    assert merged[1].id == "b02"


def test_merge_table_continuation_fragments_does_not_swallow_near_prose() -> None:
    # A short body sentence sits within the gap/containment window; the old
    # gate absorbed anything under 120 chars or containing ":", so it was
    # swallowed into the table as grid markup instead of translated as prose.
    tbl = _make_block("b01", BlockType.TABLE, 1, 70.0, 200.0, 540.0, 400.0, "| A | B |")
    prose = _make_block(
        "b02",
        BlockType.NARRATIVE,
        1,
        70.0,
        188.0,
        540.0,
        198.0,
        "The results are shown below:",
    )

    blocks = [tbl, prose]
    merged = merge_table_continuation_fragments(blocks)

    assert len(merged) == 2
    assert merged[1].id == "b02"
    assert merged[0].bbox is not None
    assert merged[0].bbox.y0 == 200.0
