from __future__ import annotations

import pytest

from ubt.adapters.pdf.facsimile_plan import build_page_fit_models, obstacle_top
from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    block_type: BlockType,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            source_text=text,
            bbox=BoundingBox(page=1, x0=x0, y0=y0, x1=x1, y1=y1),
        )
    )


def test_obstacle_top_is_the_blocks_top_edge() -> None:
    block = _block("h1", BlockType.HEADING, x0=50.0, y0=470.0, x1=300.0, y1=490.0, text="Title")

    assert obstacle_top(block.bbox) == 490.0


def test_heading_and_table_obstacles_use_the_edge_the_planner_uses() -> None:
    prose = _block(
        "p1",
        BlockType.NARRATIVE,
        x0=50.0,
        y0=500.0,
        x1=300.0,
        y1=520.0,
        text="A paragraph above the exhibit.",
    )
    heading = _block("h1", BlockType.HEADING, x0=50.0, y0=470.0, x1=300.0, y1=490.0, text="Title")

    model = build_page_fit_models(
        {1: [prose, heading]},
        [(400.0, 600.0)],
        {1: (10.0, 1.2)},
        total_pages=1,
    )[1]

    record = model.columns[0][0]
    # The heading's top edge is y1=490; using y0=470 would let the flow expand
    # 20pt into the heading's own band.
    assert record["obstacle_top"] == 490.0
