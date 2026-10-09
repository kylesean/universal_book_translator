"""The list marker survives the overlay lowering.

The reader stores a ``ListItem``'s bullet on the element and strips it from the
text; the compositor draws only the text and masks the source glyph away with the
rest of the region, so the overlay builder must put the marker back. This is the
regression the retired rigid typesetter's ``_with_list_marker`` used to cover and
the unified compositor dropped.
"""

from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.render.outputs import overlays_from_blocks

pytestmark = pytest.mark.fast


def _block(block_type: BlockType, *, source: str, target: str, marker: str = "") -> IRBlock:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=block_type,
            source_text=source,
            marker=marker,
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=354.0, y1=712.0, page=1),
        )
    )
    block.target_text = target
    return block


def test_a_markerless_list_item_gets_its_bullet_back() -> None:
    block = _block(BlockType.LIST_ITEM, source="first item", target="第一项", marker="•")
    (overlay,) = overlays_from_blocks([block])
    assert overlay.text == "• 第一项"


def test_an_item_that_reproduced_its_marker_is_not_doubled() -> None:
    block = _block(BlockType.LIST_ITEM, source="first item", target="• 第一项", marker="•")
    (overlay,) = overlays_from_blocks([block])
    assert overlay.text == "• 第一项"


@pytest.mark.parametrize("marker", ["", None])
def test_an_item_with_no_extracted_marker_falls_back_to_a_bullet(marker: str | None) -> None:
    block = _block(BlockType.LIST_ITEM, source="first item", target="第一项", marker=marker or "")
    (overlay,) = overlays_from_blocks([block])
    assert overlay.text == "• 第一项"


@pytest.mark.parametrize(
    "target", ["1. 引言", "（1）引言", "１. 引言", "（a）项", "一、项", "3、项"]
)
def test_a_full_width_ordered_marker_is_not_doubled(target: str) -> None:
    # A model that switches to CJK punctuation must not get a second marker.
    block = _block(BlockType.LIST_ITEM, source="first item", target=target, marker="1.")
    (overlay,) = overlays_from_blocks([block])
    assert overlay.text == target


def test_a_non_list_block_is_left_untouched() -> None:
    block = _block(BlockType.NARRATIVE, source="A paragraph.", target="一段。")
    (overlay,) = overlays_from_blocks([block])
    assert overlay.text == "一段。"


def test_a_toc_row_becomes_a_toc_overlay_carrying_its_page_number() -> None:
    # The reader stores the page number in provenance (toc_entry/toc_page); the
    # overlay must keep the TOC kind and the number so the compositor can redraw
    # title + leaders + number instead of a plain text overlay.
    block = _block(BlockType.NARRATIVE, source="3. Effects", target="3. 效应")
    block.provenance = {"toc_entry": True, "toc_page": "9"}
    (overlay,) = overlays_from_blocks([block])
    assert overlay.kind == "toc"
    assert overlay.toc_page == "9"
    assert overlay.text == "3. 效应"
