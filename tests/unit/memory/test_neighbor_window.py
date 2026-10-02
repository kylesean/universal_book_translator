"""Contract tests for the read-only neighbor sliding-window context builder."""

from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockType, FlowID, IRBlock, make_element
from ubt.core.memory.neighbor_window import DEFAULT_NEIGHBOR_CHARS, NeighborContextBuilder

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    spine_index: int,
    source: str,
    *,
    flow: FlowID = FlowID.MAIN_STORY,
    target: str | None = None,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=spine_index,
        block_type=BlockType.NARRATIVE,
        flow_id=flow,
        source_text=source,
        skip_translate=False,
    )
    return IRBlock(element=element, target_text=target)


def test_default_window_is_three_hundred_chars() -> None:
    assert DEFAULT_NEIGHBOR_CHARS == 300


# --------------------------------------------------------------------------- #
# extract_excerpts
# --------------------------------------------------------------------------- #


def test_extracts_tail_of_prev_and_head_of_next() -> None:
    excerpts = NeighborContextBuilder(5).extract_excerpts("abcdefghij", "klmnopqrst")
    assert excerpts == {"prev_excerpt": "fghij", "next_excerpt": "klmno"}


def test_short_texts_are_kept_whole() -> None:
    assert NeighborContextBuilder(5).extract_excerpts("ab", "cd") == {
        "prev_excerpt": "ab",
        "next_excerpt": "cd",
    }


def test_absent_and_blank_sides_are_empty() -> None:
    builder = NeighborContextBuilder(5)
    assert builder.extract_excerpts(None, None) == {"prev_excerpt": "", "next_excerpt": ""}
    assert builder.extract_excerpts("   ", "  ") == {"prev_excerpt": "", "next_excerpt": ""}


# --------------------------------------------------------------------------- #
# format_prompt_block
# --------------------------------------------------------------------------- #


def test_no_context_renders_nothing() -> None:
    assert NeighborContextBuilder().format_prompt_block(None, None) == ""


def test_prev_only_renders_the_preceding_label() -> None:
    rendered = NeighborContextBuilder().format_prompt_block("prev", None)
    assert rendered.startswith("### Reference Context (read-only, NOT part of the task)\n\n")
    assert "[READ-ONLY PRECEDING CONTEXT: DO NOT TRANSLATE OR ECHO]\nprev" in rendered
    assert "SUBSEQUENT" not in rendered


def test_both_sides_render_both_labels_in_order() -> None:
    rendered = NeighborContextBuilder().format_prompt_block("prev", "next")
    assert rendered.index("PRECEDING") < rendered.index("SUBSEQUENT")
    assert "prev\n\n[READ-ONLY SUBSEQUENT CONTEXT" in rendered
    assert rendered.endswith("next")


# --------------------------------------------------------------------------- #
# extract_from_blocks
# --------------------------------------------------------------------------- #


def test_neighbors_are_filtered_to_the_same_flow_and_sorted_by_spine() -> None:
    target = _block("c", 2, "T")
    surrounding = [
        _block("a", 1, "A", target="Atgt"),
        _block("b", 0, "B", target="Btgt"),
        _block("c", 2, "T"),
        _block("d", 3, "NextSrcText"),
        _block("x", 1, "FootnoteSrc", flow=FlowID.FOOTNOTE),
    ]
    rendered = NeighborContextBuilder().extract_from_blocks(target, surrounding)
    assert "Atgt" in rendered  # previous finished translation, spine-ordered
    assert "Btgt" not in rendered
    assert "NextSrcText" in rendered  # next uses the source while untranslated
    assert "FootnoteSrc" not in rendered  # footnote flow is isolated


def test_previous_prefers_translation_then_falls_back_to_source() -> None:
    target = _block("c", 2, "T")
    # The preceding block has no target yet -> its source is used.
    rendered = NeighborContextBuilder().extract_from_blocks(
        target, [_block("b", 1, "Bsrc"), target]
    )
    assert "Bsrc" in rendered


def test_fallbacks_fill_the_gaps_for_an_isolated_block() -> None:
    target = _block("c", 2, "T")
    rendered = NeighborContextBuilder().extract_from_blocks(
        target,
        [target],
        fallback_prev_text="PB",
        fallback_next_text="NB",
    )
    assert "PB" in rendered
    assert "NB" in rendered


def test_fallbacks_apply_when_the_target_is_not_in_the_window() -> None:
    target = _block("c", 2, "T")
    unknown = _block("zz", 9, "Z")
    rendered = NeighborContextBuilder().extract_from_blocks(
        unknown,
        [target],
        fallback_prev_text="PB",
        fallback_next_text="NB",
    )
    assert "PB" in rendered
    assert "NB" in rendered


def test_an_isolated_block_without_fallbacks_renders_nothing() -> None:
    target = _block("c", 2, "T")
    assert NeighborContextBuilder().extract_from_blocks(target, [target]) == ""
