"""Contract tests for rolling cross-chapter continuity summaries."""

from __future__ import annotations

import asyncio

import pytest

from ubt.core.memory.rolling_summary import (
    build_summary_prompt,
    collect_chapter_text,
    deterministic_summary,
    extract_chapter_id,
    summarize_chapter,
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


# --------------------------------------------------------------------------- #
# summarize_chapter
# --------------------------------------------------------------------------- #


def _blocks() -> list[_Block]:
    return [_Block("x", "translated chapter text here")]


def test_summarize_returns_a_valid_llm_summary() -> None:
    async def complete(_system: str, _user: str) -> str:
        return "A fine summary of the chapter."

    assert asyncio.run(summarize_chapter(_blocks(), complete)) == "A fine summary of the chapter."


def test_summarize_strips_wrapping_quotes_and_whitespace() -> None:
    async def complete(_system: str, _user: str) -> str:
        return '  "Hello summary."  '

    assert asyncio.run(summarize_chapter(_blocks(), complete)) == "Hello summary."


def test_summarize_accepts_a_summary_at_the_minimum_length() -> None:
    async def complete(_system: str, _user: str) -> str:
        return "abcdefghij"  # exactly 10 chars == _MIN_SUMMARY_CHARS

    assert asyncio.run(summarize_chapter(_blocks(), complete)) == "abcdefghij"


def test_summarize_normalizes_internal_whitespace() -> None:
    async def complete(_system: str, _user: str) -> str:
        return "aa   bb   cc   dd"

    assert asyncio.run(summarize_chapter(_blocks(), complete)) == "aa bb cc dd"


def test_summarize_degrades_a_too_short_summary_to_the_excerpt() -> None:
    async def complete(_system: str, _user: str) -> str:
        return "hi"

    assert asyncio.run(summarize_chapter(_blocks(), complete)) == "translated chapter text here"


def test_summarize_truncates_the_fallback_excerpt_of_a_long_chapter() -> None:
    chapter = "Chapter body sentence. " * 30  # 690 chars > the 400-char cap

    async def complete(_system: str, _user: str) -> str:
        return "hi"  # too short -> fall back to the chapter excerpt

    result = asyncio.run(summarize_chapter([_Block("x", chapter)], complete))
    assert result.endswith(".")
    assert len(result) <= 400  # the deterministic excerpt cap
    assert len(result) < len(chapter)


def test_summarize_degrades_on_a_provider_error() -> None:
    async def complete(_system: str, _user: str) -> str:
        raise RuntimeError("boom")

    assert asyncio.run(summarize_chapter(_blocks(), complete)) == "translated chapter text here"


def test_summarize_truncates_an_over_long_summary_at_a_boundary() -> None:
    long = ("Sentence one is here. " * 60).strip()
    assert len(long) > 800

    async def complete(_system: str, _user: str) -> str:
        return long

    result = asyncio.run(summarize_chapter(_blocks(), complete))
    assert 400 < len(result) <= 800
    assert result.endswith(".")
    assert long.startswith(result)


def test_summarize_of_no_text_is_empty() -> None:
    async def complete(_system: str, _user: str) -> str:
        return "A valid summary here."

    assert asyncio.run(summarize_chapter([], complete)) == ""
