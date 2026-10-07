"""Contract tests for the summary primitives the hierarchical manager uses."""

from __future__ import annotations

import pytest

from ubt.core.memory.rolling_summary import (
    build_summary_prompt,
    collect_chapter_text,
    deterministic_summary,
    extract_chapter_id,
)

pytestmark = pytest.mark.fast


class _Block:
    def __init__(
        self,
        source_text: str,
        target_text: str | None = None,
        draft_text: str | None = None,
    ) -> None:
        self.source_text = source_text
        self.target_text = target_text
        self.draft_text = draft_text


# --------------------------------------------------------------------------- #
# extract_chapter_id
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("block_id", "expected"),
    [
        ("ch_001#b0002", "ch_001"),
        ("ch_001:b0002", "ch_001"),
        ("ch#a:b", "ch"),  # first delimiter wins
        ("b0002", ""),  # no delimiter = single-chapter document
        ("", ""),
    ],
)
def test_extract_chapter_id(block_id: str, expected: str) -> None:
    assert extract_chapter_id(block_id) == expected


# --------------------------------------------------------------------------- #
# collect_chapter_text
# --------------------------------------------------------------------------- #


def test_collect_prefers_target_then_draft_then_source() -> None:
    blocks = [
        _Block("src1", "tgt1"),
        _Block("src2"),
        _Block("src4", draft_text="draft4"),
    ]
    assert collect_chapter_text(blocks) == "tgt1\nsrc2\ndraft4"


def test_collect_skips_blank_blocks() -> None:
    assert collect_chapter_text([_Block("  "), _Block("real")]) == "real"


def test_collect_caps_at_max_chars() -> None:
    blocks = [_Block("a" * 10), _Block("b" * 10), _Block("c" * 10)]
    assert collect_chapter_text(blocks, max_chars=25) == "a" * 10 + "\n" + "b" * 10 + "\nccc"


def test_collect_of_no_blocks_is_empty() -> None:
    assert collect_chapter_text([]) == ""


# --------------------------------------------------------------------------- #
# deterministic_summary
# --------------------------------------------------------------------------- #


def test_deterministic_summary_returns_short_text_unchanged() -> None:
    assert deterministic_summary("hello") == "hello"


def test_deterministic_summary_cuts_at_a_sentence_boundary() -> None:
    assert deterministic_summary("One. Two. Three. Four.", max_chars=10) == "One. Two."


def test_deterministic_summary_ignores_an_abbreviation_boundary() -> None:
    # The only period in the window is "Fig." — a 4-char boundary is too short
    # to trust, so the hard cut wins.
    assert deterministic_summary("Fig. 3 shows", max_chars=6) == "Fig. 3"


def test_deterministic_summary_honours_a_five_char_boundary() -> None:
    assert deterministic_summary("abcd.efghij", max_chars=6) == "abcd."


def test_deterministic_summary_honours_a_cjk_boundary() -> None:
    assert deterministic_summary("第一句。第二句。第三句。", max_chars=8) == "第一句。第二句。"


# --------------------------------------------------------------------------- #
# build_summary_prompt
# --------------------------------------------------------------------------- #


def test_build_summary_prompt_embeds_language_and_content() -> None:
    system_prompt, user_prompt = build_summary_prompt("content here", "zh")
    assert "zh" in system_prompt
    assert "Output ONLY the summary text." in system_prompt
    assert "content here" in user_prompt
    assert "zh" in user_prompt
