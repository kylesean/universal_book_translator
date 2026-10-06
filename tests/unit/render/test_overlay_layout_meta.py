"""The overlay carries the source layout facts the fragment needs.

Two geometry facts live outside the block's bounding box and are recorded by
the analyzer from the page's own line rects (``annotate_layout_metadata``):

- a list item's marker-column indent — the block box starts at the wrapped
  lines' margin, so the compositor would otherwise draw "(1)" flush with the
  body margin where the source hangs it to the right;
- a centered heading — without the flag the title fragment hugs the left
  margin where the source centered it.

These tests pin the overlay lowering for both facts and the Typst source the
typesetter emits for them.
"""

from __future__ import annotations

import pytest

from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    StyleMeta,
    make_element,
)
from ubt.render.outputs import overlays_from_blocks

pytestmark = pytest.mark.fast


def _block(block_type: BlockType, *, style: StyleMeta | None = None) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            source_text="source",
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=354.0, y1=712.0, page=1),
        )
    )
    block.style = style
    block.target_text = "target"
    return block


def test_a_list_items_marker_indent_reaches_the_overlay() -> None:
    style = StyleMeta(first_line_indent_pt=22.2)
    block = _block(BlockType.LIST_ITEM, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.indent_pt == pytest.approx(22.2)
    assert overlay.align_center is False


def test_a_narrative_paragraph_keeps_its_own_indent_rule() -> None:
    style = StyleMeta(first_line_indent_pt=17.4)
    block = _block(BlockType.NARRATIVE, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.indent_pt == pytest.approx(17.4)


def test_a_centered_heading_reaches_the_overlay() -> None:
    style = StyleMeta(alignment="center")
    block = _block(BlockType.HEADING, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.align_center is True
    assert overlay.indent_pt is None


def test_a_left_aligned_heading_is_not_marked_centered() -> None:
    block = _block(BlockType.HEADING, style=StyleMeta())
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.align_center is False


def test_centering_is_heading_only() -> None:
    # A narrative block with a stray alignment value stays left-aligned: the
    # flag is a heading fact, not a general indent substitute.
    style = StyleMeta(alignment="center")
    block = _block(BlockType.NARRATIVE, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.align_center is False


def test_the_typesetter_emits_the_centering_rule() -> None:
    from ubt.render.outputs import TypstFragmentTypesetter

    ts = TypstFragmentTypesetter(cache_dir=":temp:")
    centered = ts._text_source("标题", 200.0, 40.0, 12.0, align_center=True)
    flush = ts._text_source("标题", 200.0, 40.0, 12.0)
    assert "#set align(center)" in centered
    assert "#set align(center)" not in flush
