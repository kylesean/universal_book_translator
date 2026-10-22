"""Continuation fusion: joining one paragraph that a PDF split across boxes.

A born-digital PDF has no notion of "paragraph", so a paragraph crossing a page
or column break arrives as two blocks. Fusion merges them into one element so the
translation flows across the chain. These tests pin the two halves of that
contract: the run is detected from reading order and geometry, and the fused
block never carries a *partial* translation.
"""

from __future__ import annotations

import pytest

from ubt.core.ir.continuation import fuse_continuation_blocks, join_continuous_text
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element

pytestmark = pytest.mark.fast


def _prose(
    block_id: str,
    text: str,
    *,
    page: int,
    y0: float,
    target: str | None = None,
    x0: float = 10.0,
    x1: float = 200.0,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text=text,
            bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y0 + 10.0),
        ),
        target_text=target,
    )


def test_a_page_break_run_fuses_into_one_block() -> None:
    first = _prose(
        "pdf_main#b0001", "This paragraph continues", page=1, y0=700.0, target="这一段继续"
    )
    second = _prose("pdf_main#b0002", "onto the next page.", page=2, y0=700.0, target="到下一页。")
    fused = fuse_continuation_blocks([first, second])

    assert len(fused) == 1
    assert fused[0].source_text == "This paragraph continues onto the next page."
    assert fused[0].target_text == "这一段继续到下一页。"


def test_a_half_translated_run_does_not_ship_a_partial_target() -> None:
    """The fused target is all-or-nothing.

    Concatenating only the translated halves deleted the untranslated text: the
    two blocks are now one, so a re-run never drafts the missing part and the
    reader gets a paragraph with a hole. An empty target sends the whole fused
    paragraph back through the draft stage instead.
    """
    first = _prose(
        "pdf_main#b0001", "This paragraph continues", page=1, y0=700.0, target="这一段继续"
    )
    second = _prose("pdf_main#b0002", "onto the next page.", page=2, y0=700.0, target=None)

    fused = fuse_continuation_blocks([first, second])

    assert len(fused) == 1
    assert not fused[0].target_text, "a partial run must not carry a partial translation"


def test_an_emptied_target_also_blocks_the_fusion_target() -> None:
    """A whitespace-only target counts as untranslated, not as translated."""
    first = _prose("pdf_main#b0001", "First half continues", page=1, y0=700.0, target="前半")
    second = _prose("pdf_main#b0002", "second half here.", page=2, y0=700.0, target="   ")

    fused = fuse_continuation_blocks([first, second])

    assert len(fused) == 1
    assert not fused[0].target_text


def test_the_fused_block_keeps_the_whole_box_chain() -> None:
    """A resumed run rebuilds the span from ``physical_boxes`` alone."""
    first = _prose("pdf_main#b0001", "This paragraph continues", page=1, y0=700.0)
    second = _prose("pdf_main#b0002", "onto the next page.", page=2, y0=700.0)

    fused = fuse_continuation_blocks([first, second])

    assert fused[0].provenance.fused_block_ids == ["pdf_main#b0001", "pdf_main#b0002"]
    assert len(fused[0].provenance.physical_boxes) == 2


def test_blocks_that_do_not_form_a_run_are_untouched() -> None:
    first = _prose("pdf_main#b0001", "A complete sentence.", page=1, y0=700.0)
    second = _prose("pdf_main#b0002", "Another complete one.", page=2, y0=700.0)

    assert fuse_continuation_blocks([first, second]) == [first, second]


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["one", "two"], "one two"),
        (["中文", "续接"], "中文续接"),
        (["atten-", "tion"], "atten-tion"),
        (["  ", "only"], "only"),
    ],
)
def test_join_continuous_text_handles_script_and_hyphen_boundaries(
    pieces: list[str], expected: str
) -> None:
    assert join_continuous_text(pieces) == expected
