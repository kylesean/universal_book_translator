from __future__ import annotations

import pytest

from ubt.adapters.pdf.docling_blocks import merge_table_continuation_fragments
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element

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
    flow_id: FlowID = FlowID.MAIN_STORY,
) -> IRBlock:
    elem = make_element(
        id=block_id,
        spine_index=1,
        block_type=block_type,
        flow_id=flow_id,
        source_text=text,
        bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
    )
    return IRBlock(element=elem)


def test_merge_table_continuation_fragments_absorbs_code_and_formulas() -> None:
    tbl = _make_block("b01", BlockType.TABLE, 1, 70.0, 200.0, 540.0, 400.0, "| Col1 | Col2 |")
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
    heading = _make_block("b06", BlockType.HEADING, 1, 70.0, 100.0, 200.0, 120.0, "5. Conclusion")

    blocks = [tbl, frag1, frag2, frag3, frag4, heading]
    merged = merge_table_continuation_fragments(blocks)

    assert len(merged) == 2
    assert merged[0].id == "b01"
    assert merged[0].block_type == BlockType.TABLE
    assert merged[0].bbox is not None
    assert merged[0].bbox.y0 == 170.0
    assert merged[0].bbox.y1 == 400.0
    assert "ScheduleSource" in merged[0].source_text
    assert "next_fire(ctx) ->" in merged[0].source_text
    assert merged[1].id == "b06"
    assert merged[1].block_type == BlockType.HEADING


def test_merge_table_continuation_fragments_does_not_swallow_distant_prose() -> None:
    tbl = _make_block("b01", BlockType.TABLE, 1, 70.0, 300.0, 540.0, 400.0, "| A | B |")
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


def test_a_footnote_directly_below_a_table_is_not_absorbed() -> None:
    tbl = _make_block("b01", BlockType.TABLE, 1, 70.0, 200.0, 540.0, 400.0, "| A | B |")
    note = _make_block(
        "b02",
        BlockType.NARRATIVE,
        1,
        70.0,
        188.0,
        300.0,
        198.0,
        "1) See Table 2 for the full comparison.",
        flow_id=FlowID.FOOTNOTE,
    )

    merged = merge_table_continuation_fragments([tbl, note])

    assert [b.id for b in merged] == ["b01", "b02"]
    assert merged[0].source_text == "| A | B |"
