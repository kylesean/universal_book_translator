"""Tests for rolling cross-chapter continuity summaries."""

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.memory.rolling_summary import (
    collect_chapter_text,
    deterministic_summary,
    extract_chapter_id,
    summarize_chapter,
)
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def test_extract_chapter_id() -> None:
    assert extract_chapter_id("ch_001#b0002") == "ch_001"
    assert extract_chapter_id("pdf_main#b0017") == "pdf_main"
    assert extract_chapter_id("no_separator_id") == ""  # single-chapter doc


class _FakeBlock:
    def __init__(self, source: str, target: str | None = None, draft: str | None = None) -> None:
        self.source_text = source
        self.target_text = target
        self.draft_text = draft


def test_collect_chapter_text_prefers_translation() -> None:
    blocks = [
        _FakeBlock("Source one.", "译文一。"),
        _FakeBlock("Source two.", None, "草稿二。"),
        _FakeBlock("Source three."),
    ]
    text = collect_chapter_text(blocks)
    assert "译文一。" in text
    assert "草稿二。" in text
    assert "Source three." in text
    assert "Source one." not in text  # translated version wins


def test_collect_chapter_text_caps_input() -> None:
    blocks = [_FakeBlock("x" * 100) for _ in range(50)]
    assert len(collect_chapter_text(blocks, max_chars=500)) <= 500


def test_deterministic_summary_truncates_at_sentence_boundary() -> None:
    text = "第一句。第二句。" + "y" * 500 + "最后一句。"
    summary = deterministic_summary(text, max_chars=100)
    assert len(summary) <= 100
    assert summary.endswith("。")


def test_deterministic_summary_ignores_an_abbreviation_boundary() -> None:
    """A lone early period must not collapse the summary to a three-char fragment.

    The cut is taken at the last sentence boundary in the window, but when the
    only period belongs to an abbreviation ("Fig.", "e.g.", "v1.2") that
    boundary sits at index 3 and the summary degrades to "Fig." — starving the
    next chapter's continuity prompt. A boundary shorter than
    ``_MIN_BOUNDARY_CHARS`` is ignored in favour of the hard cut.
    """
    text = "Fig. " + "x" * 600
    summary = deterministic_summary(text, max_chars=100)
    assert summary == text[:100]
    assert len(summary) == 100


def test_deterministic_summary_still_honours_a_late_boundary() -> None:
    """A boundary past the minimum length is still preferred over the hard cut."""
    text = "A sentence. " + "y" * 600
    assert deterministic_summary(text, max_chars=100) == "A sentence."


@pytest.mark.asyncio
async def test_summarize_chapter_uses_llm_output() -> None:
    blocks = [_FakeBlock("Chapter content here.", "本章内容译文。")]

    async def fake_complete(system: str, user: str) -> str:
        return "主人公进入森林，遇到；术语已确立。"

    summary = await summarize_chapter(blocks, fake_complete)
    assert summary == "主人公进入森林，遇到；术语已确立。"


@pytest.mark.asyncio
async def test_summarize_chapter_truncates_overlong_llm_summary() -> None:
    """A >cap summary is truncated, not thrown away for a short head excerpt.

    The tail is the part the next chapter's continuity needs, and the bulk call
    was already paid for.
    """
    blocks = [_FakeBlock("SOURCE-EXCERPT " + "s" * 200)]
    long_summary = "MODEL-SUMMARY " + ("情节推进。" * 200)

    async def fake_complete(system: str, user: str) -> str:
        return long_summary

    summary = await summarize_chapter(blocks, fake_complete)
    assert summary.startswith("MODEL-SUMMARY")
    assert len(summary) <= 800
    assert "SOURCE-EXCERPT" not in summary


@pytest.mark.asyncio
async def test_summarize_chapter_degrades_on_garbage() -> None:
    blocks = [_FakeBlock("A first sentence. A second one follows here.")]

    async def fake_complete(system: str, user: str) -> str:
        raise RuntimeError("provider down")

    summary = await summarize_chapter(blocks, fake_complete)
    assert "A first sentence." in summary


@pytest.mark.asyncio
async def test_summarize_chapter_empty_blocks() -> None:
    async def fake_complete(system: str, user: str) -> str:  # pragma: no cover
        raise AssertionError("should not be called")

    assert await summarize_chapter([], fake_complete) == ""


@pytest.mark.asyncio
async def test_pipeline_feeds_chapter_summary_into_next_chapter(tmp_path: Path) -> None:
    """Chapter 2 draft prompts must carry chapter 1's summary; exactly one summary call."""
    md = tmp_path / "two_chapters.md"
    md.write_text(
        "# Chapter 1: The Forest\n\nAlice walked into the dark forest and found a door.\n"
        "The door was locked and glowed faintly in the mist.\n\n"
        "# Chapter 2: The Door\n\nBehind the door a staircase descended into silence.\n"
        "She counted the steps as she went down into the deep.\n",
        encoding="utf-8",
    )
    provider = MockModelProvider(default_response="这是默认翻译。")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db", rate_limit_rpm=600),
        router=router,
        qe_runner=MockQERunner(default_score=0.9),
    )

    async for _ in orchestrator.run(
        input_path=md,
        output_path=tmp_path / "out.md",
        target_lang="zh",
        job_id="job_rolling_e2e",
    ):
        pass

    summary_calls = [c for c in provider.call_history if "continuation summary" in c["prompt"]]
    assert len(summary_calls) == 1, "expected exactly one rolling-summary call (chapter 1)"
    # Summary happens after chapter 1 drafts and before chapter 2 drafts
    # (draft prompts only — repair-stage prompts also embed source text)
    summary_idx = provider.call_history.index(summary_calls[0])
    ch1_last = min(
        i
        for i, c in enumerate(provider.call_history)
        if "Alice walked into the dark forest" in c["prompt"]
        and "Source Paragraph to Translate" in c["prompt"]
    )
    ch2_first = min(
        i
        for i, c in enumerate(provider.call_history)
        if "staircase" in c["prompt"] and "Source Paragraph to Translate" in c["prompt"]
    )
    assert ch1_last < summary_idx < ch2_first

    # The mock's summary response becomes the rolling summary in chapter 2 prompts
    assert any(
        "这是默认翻译。" in c["prompt"] and "staircase" in c["prompt"]
        for c in provider.call_history
    )


@pytest.mark.asyncio
async def test_pipeline_rolling_summary_disabled(tmp_path: Path) -> None:
    md = tmp_path / "two_chapters.md"
    md.write_text(
        "# Chapter 1: One\n\nFirst chapter narrative text goes here in bulk.\n\n"
        "# Chapter 2: Two\n\nSecond chapter narrative text goes here in bulk.\n",
        encoding="utf-8",
    )
    provider = MockModelProvider(default_response="这是默认翻译。")
    router = ModelRouter(provider=provider)
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db", rate_limit_rpm=600, enable_rolling_summary=False),
        router=router,
        qe_runner=MockQERunner(default_score=0.9),
    )

    async for _ in orchestrator.run(
        input_path=md,
        output_path=tmp_path / "out.md",
        target_lang="zh",
        job_id="job_rolling_off",
    ):
        pass

    assert not any("continuation summary" in c["prompt"] for c in provider.call_history)
