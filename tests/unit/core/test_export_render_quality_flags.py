"""``low_legibility_font`` is a defect flag, never a skip flag.

The compositor draws a monolingual target below the readable floor (6pt) rather
than keep the source when the box is too short for the target's line at 6pt.
That draw must reach the quality report *without* reaching the delivery
contract's ``source_kept`` count -- a ``render_skip:`` flag would do the latter,
which is the very outcome drawing it avoided. These tests pin the two halves:
the flag lands on the block, and a stale one from an earlier render is cleared.
"""

from __future__ import annotations

import pytest

from ubt.core.content.adapt import kept_in_source
from ubt.core.engine.stages.export import (
    RENDER_QUALITY_FLAGS,
    apply_render_quality_flags,
)
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(block_id: str, *, flags: list[str] | None = None) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="effect.",
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=81.6, y1=708.3, page=1),
        )
    )
    block.target_text = "[模拟翻译] effect."
    block.status = BlockStatus.MTQE_PASSED
    block.error_flags = list(flags or [])
    return block


def test_the_flag_lands_on_the_block_without_keeping_the_source() -> None:
    blocks = [_block("b1")]
    changed = apply_render_quality_flags(blocks, [("b1", "low_legibility_font")])
    assert changed == {"b1"}
    assert blocks[0].error_flags == ["low_legibility_font"]
    # The block delivered a translation; a quality flag must not read as a keep.
    assert kept_in_source(blocks[0]) is False


def test_every_quality_flag_is_a_plain_flag_not_a_skip() -> None:
    for flag in RENDER_QUALITY_FLAGS:
        assert not flag.startswith(("render_skip:", "inplace_skip:"))


def test_a_stale_flag_from_an_earlier_render_is_cleared() -> None:
    blocks = [_block("b1", flags=["low_legibility_font", "some_other_defect"])]
    changed = apply_render_quality_flags(blocks, [])
    assert changed == {"b1"}
    assert blocks[0].error_flags == ["some_other_defect"]


def test_an_unchanged_block_is_not_reported_as_changed() -> None:
    blocks = [_block("b1", flags=["low_legibility_font"])]
    assert apply_render_quality_flags(blocks, [("b1", "low_legibility_font")]) == set()
    assert blocks[0].error_flags == ["low_legibility_font"]


def test_unknown_block_ids_and_unknown_flags_are_ignored() -> None:
    blocks = [_block("b1")]
    changed = apply_render_quality_flags(
        blocks, [("gone", "low_legibility_font"), ("b1", "not_a_quality_flag")]
    )
    assert changed == set()
    assert blocks[0].error_flags == []
