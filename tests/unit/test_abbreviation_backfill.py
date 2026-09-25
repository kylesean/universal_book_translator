"""Tests for the bulk abbreviation-translation backfill channel."""

from pathlib import Path
from typing import Any

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.memory.abbreviation_backfill import (
    backfill_abbreviation_translations,
    build_backfill_prompt,
)
from ubt.core.memory.bible import BibleEntry
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def _pair(acronym: str, expansion: str) -> dict[str, Any]:
    return {"source": expansion, "translation": "", "aliases": [acronym], "kind": "term"}


@pytest.mark.asyncio
async def test_backfill_fills_wellformed_response() -> None:
    entries = [_pair("WM", "Working Memory"), _pair("LSTM", "Long Short-Term Memory")]

    async def fake_complete(system: str, user: str) -> str:
        return "WM = 工作记忆（WM）\nLSTM = 长短期记忆网络（LSTM）"

    result, filled = await backfill_abbreviation_translations(entries, fake_complete)
    assert filled == 2
    assert result[0]["translation"] == "工作记忆（WM）"
    assert result[1]["translation"] == "长短期记忆网络（LSTM）"


@pytest.mark.asyncio
async def test_backfill_tolerates_numbering_and_noise() -> None:
    entries = [_pair("SDPA", "scaled dot-product attention")]

    async def fake_complete(system: str, user: str) -> str:
        return (
            "Sure! Here are the translations:\n"
            "1. SDPA = 缩放点积注意力（SDPA）\n"
            "- WM = 工作记忆\n"
            "This should be ignored."
        )

    result, filled = await backfill_abbreviation_translations(entries, fake_complete)
    assert filled == 1
    assert result[0]["translation"] == "缩放点积注意力（SDPA）"


@pytest.mark.asyncio
async def test_backfill_garbage_degrades_gracefully() -> None:
    entries = [_pair("WM", "Working Memory")]

    async def fake_complete(system: str, user: str) -> str:
        return "工作记忆是一个认知系统，无法解析成对账格式。"

    result, filled = await backfill_abbreviation_translations(entries, fake_complete)
    assert filled == 0
    assert result[0]["translation"] == ""


@pytest.mark.asyncio
async def test_backfill_rejects_echo_and_overlong() -> None:
    entries = [_pair("WM", "Working Memory"), _pair("RAG", "Retrieval-Augmented Generation")]

    async def fake_complete(system: str, user: str) -> str:
        long_tr = "超" * 120
        return f"WM = WM\nRAG = {long_tr}"

    result, filled = await backfill_abbreviation_translations(entries, fake_complete)
    assert filled == 0
    assert all(e["translation"] == "" for e in result)


@pytest.mark.asyncio
async def test_backfill_never_overwrites_existing_translations() -> None:
    entries = [
        {"source": "Working Memory", "translation": "已定译名", "aliases": ["WM"], "kind": "term"}
    ]

    async def fake_complete(system: str, user: str) -> str:
        return "WM = 不应生效"

    result, filled = await backfill_abbreviation_translations(entries, fake_complete)
    assert filled == 0
    assert result[0]["translation"] == "已定译名"


def test_build_backfill_prompt_lists_items_and_rules() -> None:
    system, user = build_backfill_prompt([_pair("WM", "Working Memory")], "zh")
    assert "zh" in system
    assert "1. WM — Working Memory" in user
    assert "KEY = <rendering>" in user
    assert "Acronym items" in user


@pytest.mark.asyncio
async def test_router_complete_raw_targets_draft_model() -> None:
    provider = MockModelProvider(default_response="ok")
    router = ModelRouter(provider=provider, draft_model="draft-x", repair_model="premium-x")

    out = await router.complete_raw("sys", "user prompt")

    assert out == "ok"
    assert provider.call_history[0]["model"] == "draft-x"


@pytest.mark.asyncio
async def test_pipeline_backfills_and_enforces_in_draft_prompts(tmp_path: Path) -> None:
    """Mined pairs get decided renderings via ONE call, then the term glossary
    carries them into every subsequent draft prompt."""
    md = tmp_path / "abbrev_book.md"
    md.write_text(
        "# Chapter 1: Memory\n\n"
        "Long Short-Term Memory (LSTM) networks are widely used.\n"
        "The LSTM cell gates information flow.\n",
        encoding="utf-8",
    )
    provider = MockModelProvider(
        default_response="这是默认翻译。",
        custom_responses={"Long Short-Term": "LSTM = 长短期记忆网络（LSTM）"},
    )
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
        job_id="job_backfill_e2e",
    ):
        pass

    # First LLM call is the bulk backfill (draft tier), containing the pair
    backfill_calls = [c for c in provider.call_history if "### Items" in c["prompt"]]
    assert len(backfill_calls) == 1
    assert backfill_calls[0]["model"] == "mock-draft"
    assert "LSTM — Long Short-Term Memory" in backfill_calls[0]["prompt"]

    # Decided rendering then flows into subsequent draft prompts as bible terms
    draft_calls = [c for c in provider.call_history if "### Items" not in c["prompt"]]
    assert draft_calls, "expected draft calls after backfill"
    assert any("长短期记忆网络" in c["prompt"] for c in draft_calls)


def test_backfilled_entry_validates_as_bible_entry() -> None:
    d = {
        "source": "Working Memory",
        "translation": "工作记忆（WM）",
        "aliases": ["WM"],
        "kind": "term",
    }
    e = BibleEntry.model_validate(d)
    assert e.translation == "工作记忆（WM）"


def test_abbreviation_backfill_non_latin_keys() -> None:
    from ubt.core.memory.abbreviation_backfill import _RESPONSE_LINE

    line1 = "1. 王小明 = Wang Xiaoming"
    line2 = "田中 = Tanaka"
    line3 = "- 李华 : Li Hua"

    matches = [m.groupdict() for m in _RESPONSE_LINE.finditer(f"{line1}\n{line2}\n{line3}")]
    assert len(matches) == 3, f"Expected 3 matches, got {len(matches)}: {matches}"
    assert matches[0]["acro"] == "王小明"
    assert matches[0]["tr"] == "Wang Xiaoming"
    assert matches[1]["acro"] == "田中"
    assert matches[1]["tr"] == "Tanaka"
    assert matches[2]["acro"] == "李华"
    assert matches[2]["tr"] == "Li Hua"
