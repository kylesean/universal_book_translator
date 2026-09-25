"""Figure weaving must not destroy the parser's multi-column reading order."""

from ubt.adapters.pdf.docling_render import _weave_images_preserving_order
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock


def _blk(i: int, page: int, x0: float, y0: float, x1: float, y1: float) -> IRBlock:
    return IRBlock(
        id=f"b{i}",
        spine_index=i,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text=f"block {i}",
        bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
    )


def _img(i: int, page: int, x0: float, y0: float, x1: float, y1: float) -> IRBlock:
    blk = _blk(i, page, x0, y0, x1, y1)
    blk.block_type = BlockType.IMAGE
    return blk


def test_left_column_image_inserted_inside_left_flow() -> None:
    # Two-column page in docling reading order: all left column, then right.
    blocks = [
        _blk(1, 1, 40, 600, 250, 660),  # L1 top
        _blk(2, 1, 40, 400, 250, 460),  # L2 middle
        _blk(3, 1, 40, 100, 250, 200),  # L3 bottom
        _blk(4, 1, 290, 600, 500, 660),  # R1 top
        _blk(5, 1, 290, 100, 500, 460),  # R2 rest
    ]
    img = _img(9, 1, 40, 300, 250, 380)  # left column, between L2 and L3
    out = _weave_images_preserving_order(blocks, [img])
    ids = [b.id for b in out]
    assert ids == ["b1", "b2", "b9", "b3", "b4", "b5"]
    # The regression this guards: a pure y sort interleaved columns
    # (b1, b4, b2, b9, b3, b5) and shattered the reading order.


def test_full_width_image_lands_above_both_columns_content() -> None:
    blocks = [
        _blk(1, 1, 40, 100, 250, 500),  # left lower half
        _blk(2, 1, 290, 100, 500, 500),  # right lower half
    ]
    img = _img(9, 1, 40, 520, 500, 650)  # spans both columns, above both
    out = _weave_images_preserving_order(blocks, [img])
    assert [b.id for b in out] == ["b9", "b1", "b2"]


def test_images_on_multiple_pages_group_by_page() -> None:
    blocks = [
        _blk(1, 1, 40, 100, 500, 200),
        _blk(2, 2, 40, 100, 500, 200),
    ]
    imgs = [
        _img(9, 2, 40, 300, 500, 400),
        _img(8, 1, 40, 300, 500, 400),
    ]
    out = _weave_images_preserving_order(blocks, imgs)
    assert [b.id for b in out] == ["b8", "b1", "b9", "b2"]  # image above text reads first
