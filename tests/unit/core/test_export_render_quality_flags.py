"""Render-quality flags are defects, never skip flags.

The compositor draws a monolingual target below the readable floor (6pt) rather
than keep the source when the box is too short for the target's line at 6pt; the
renderer delivers a formula as readable Unicode when it cannot typeset it as
math; and a block whose text still carries a pipeline sentinel is escaped as
prose rather than mangled. Each of those draws must reach the quality report
*without* reaching the delivery contract's ``source_kept`` count -- a
``render_skip:`` flag would do the latter, which is the very outcome drawing it
avoided. These tests pin the two halves: the flag lands on the block, and a
stale one from an earlier render is cleared.
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


def test_the_new_render_quality_flags_land_and_never_read_as_kept() -> None:
    """The formula-degradation and sentinel-leak flags behave like the font one.

    Both are drawn-and-delivered defects, not source keeps, so each must land on
    its block and leave ``kept_in_source`` false.
    """
    blocks = [_block("f1", flags=None), _block("s1", flags=None)]
    changed = apply_render_quality_flags(
        blocks, [("f1", "formula_degraded"), ("s1", "sentinel_leaked")]
    )
    assert changed == {"f1", "s1"}
    assert blocks[0].error_flags == ["formula_degraded"]
    assert blocks[1].error_flags == ["sentinel_leaked"]
    assert all(kept_in_source(b) is False for b in blocks)


def _delivery_block(
    block_id: str, target: str, *, status: BlockStatus = BlockStatus.FAILED
) -> IRBlock:
    from ubt.core.ir.models import make_element

    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Hello world.",
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=81.6, y1=708.3, page=1),
        )
    )
    block.target_text = target
    block.status = status
    return block


def test_the_delivery_map_strips_a_failed_draft_wrapper() -> None:
    """The map the HTML/EPUB companions receive must carry placed *text*.

    ``_delivery_translations`` fed ``target_text`` through verbatim, and the
    companion views treat a map entry as an override that beats their own
    ``delivered_text`` fallback. So the PDF showed the clean sentence while both
    companions shipped ``<mark class="ubt-failed-draft" title="Repair error: …">``
    — wrapper class and provider exception text — escaped onto the page.
    """
    from ubt.core.engine.stages.export import _delivery_translations

    wrapped = (
        '<mark class="ubt-failed-draft" title="Repair error: Model API error (500)">'
        "你好世界。</mark>"
    )
    placed = _delivery_translations([_delivery_block("b1", wrapped)])
    assert placed == {"b1": "你好世界。"}


def test_the_delivery_map_strips_a_structure_mismatch_wrapper() -> None:
    """Export wraps the draft itself when the HTML delta check fails."""
    from ubt.core.engine.stages.export import _delivery_translations

    wrapped = (
        '<mark class="ubt-failed-draft" title="html_structure_mismatch: expected &lt;p&gt;">'
        "你好世界。</mark>"
    )
    placed = _delivery_translations([_delivery_block("b1", wrapped)])
    assert placed["b1"] == "你好世界。"
    assert "ubt-failed-draft" not in placed["b1"]


def test_the_delivery_map_agrees_with_the_pdf_renderer() -> None:
    """One rule for every deliverable: the map value equals ``delivered_text``."""
    from ubt.core.engine.stages.export import _delivery_translations
    from ubt.render.delivered_text import delivered_text

    wrapped = (
        '<mark class="ubt-failed-draft" title="Repair error: Model API error (500)">'
        "你好世界。</mark>"
    )
    block = _delivery_block("b1", wrapped)
    assert _delivery_translations([block])["b1"] == delivered_text(block)


def test_the_delivery_map_omits_assets_and_source_kept_blocks() -> None:
    from ubt.core.engine.stages.export import _delivery_translations

    plain = _delivery_block("b1", "你好世界。", status=BlockStatus.MTQE_PASSED)
    empty = _delivery_block("b2", "   ", status=BlockStatus.MTQE_PASSED)
    assert _delivery_translations([plain, empty]) == {"b1": "你好世界。"}
