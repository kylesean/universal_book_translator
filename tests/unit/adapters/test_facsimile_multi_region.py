"""Regression test: multi-region column classification and grouping.

Pinned invariants:
1. Centered multi-line blocks straddling the center line or having overlapping
   horizontal projections on the same page must NOT be fragmented into distinct
   columns (which causes solve_flow starvation and downstream text collision).
2. Genuine multi-column blocks (e.g. left column flowing to right column) must
   be correctly grouped into disjoint region boxes.
3. Blocks spanning across page boundaries must be grouped across pages.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.facsimile_plan import _physical_box_region, group_multi_region_boxes
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.model.span import PhysicalBox

pytestmark = pytest.mark.fast


def _block_with_pboxes(
    block_id: str,
    boxes: list[tuple[int, tuple[float, float, float, float]]],
    block_type: BlockType = BlockType.HEADING,
) -> IRBlock:
    page = boxes[0][0]
    x0 = min(b[1][0] for b in boxes)
    y0 = min(b[1][1] for b in boxes)
    x1 = max(b[1][2] for b in boxes)
    y1 = max(b[1][3] for b in boxes)
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=block_type,
            source_text="source text",
            bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
        )
    )
    block.target_text = "target text"
    block.provenance.physical_boxes = [{"page": p, "bbox": list(bb)} for p, bb in boxes]
    return block


def test_centered_multiline_block_does_not_split_into_columns() -> None:
    """A 2-line centered header straddling the center line must remain unified."""
    page_width = 612.0
    # b0005 case: line 1 midpoint = 308.93 pt, line 2 midpoint = 306.29 pt
    boxes = [
        (1, (102.98, 243.72, 514.88, 261.79)),
        (1, (123.19, 226.78, 489.40, 240.79)),
    ]
    block = _block_with_pboxes("b0005", boxes)
    grouped = group_multi_region_boxes(block, page_width)
    assert grouped is None, "Centered multi-line header must not be split into multi-column flow"


def test_straddling_physical_box_is_classified_as_column_zero() -> None:
    """Boxes straddling the page center line are column 0 (spanning/centered)."""
    page_width = 612.0
    # Box midpoint is slightly right of center (306.29 > 306.0), but left edge is at 123.19
    pbox = PhysicalBox.of(1, (123.19, 226.78, 489.40, 240.79))
    reg = _physical_box_region(pbox, page_width)
    assert reg == (1, 0), "Straddling box must be classified as column 0"


def test_genuine_multicolumn_block_is_grouped_into_two_regions() -> None:
    """A paragraph flowing from column 0 to column 1 must yield 2 grouped boxes."""
    page_width = 612.0
    boxes = [
        # 2 lines in left column
        (2, (54.0, 100.0, 290.0, 120.0)),
        (2, (54.0, 120.0, 290.0, 140.0)),
        # 2 lines in right column
        (2, (322.0, 700.0, 558.0, 720.0)),
        (2, (322.0, 720.0, 558.0, 740.0)),
    ]
    block = _block_with_pboxes("p_split", boxes, block_type=BlockType.NARRATIVE)
    grouped = group_multi_region_boxes(block, page_width)
    assert grouped is not None
    assert len(grouped) == 2
    assert grouped[0].page == 2
    assert grouped[0].bbox == (54.0, 100.0, 290.0, 140.0)
    assert grouped[1].page == 2
    assert grouped[1].bbox == (322.0, 700.0, 558.0, 740.0)


def test_cross_page_block_is_grouped_across_pages() -> None:
    """A block spanning across page boundaries must yield boxes on each page."""
    page_width = 612.0
    boxes = [
        (1, (54.0, 60.0, 290.0, 80.0)),
        (2, (54.0, 720.0, 290.0, 740.0)),
    ]
    block = _block_with_pboxes("p_cross", boxes, block_type=BlockType.NARRATIVE)
    grouped = group_multi_region_boxes(block, page_width)
    assert grouped is not None
    assert len(grouped) == 2
    assert grouped[0].page == 1
    assert grouped[1].page == 2
