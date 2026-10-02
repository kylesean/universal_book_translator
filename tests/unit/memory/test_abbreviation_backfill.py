"""Contract tests for the bulk abbreviation translation backfill channel."""

from __future__ import annotations

from typing import Any

import pytest

from ubt.core.memory.abbreviation_backfill import (
    _MAX_BATCH,
    _MAX_TRANSLATION_CHARS,
    _apply_response,
    _clean_translation,
    backfill_abbreviation_translations,
    build_backfill_prompt,
)

pytestmark = pytest.mark.fast


def _entry(
    source: str, aliases: list[str], *, translation: str = "", kind: str = "term"
) -> dict[str, Any]:
    return {"source": source, "aliases": list(aliases), "translation": translation, "kind": kind}


# --------------------------------------------------------------------------- #
# _clean_translation
# --------------------------------------------------------------------------- #


def test_clean_translation_strips_and_unquotes() -> None:
    assert _clean_translation("  《工作记忆》  ", "WM") == "工作记忆"


@pytest.mark.parametrize("raw", ["", "   ", "《》"])
def test_clean_translation_rejects_empty(raw: str) -> None:
    assert _clean_translation(raw, "WM") is None


def test_clean_translation_rejects_overlong() -> None:
    assert _clean_translation("x" * (_MAX_TRANSLATION_CHARS + 1), "WM") is None
    assert _clean_translation("x" * _MAX_TRANSLATION_CHARS, "WM") == "x" * _MAX_TRANSLATION_CHARS


@pytest.mark.parametrize("echo", ["WM", "wm", "Wm"])
def test_clean_translation_rejects_acronym_echo(echo: str) -> None:
    assert _clean_translation(echo, "WM") is None


def test_clean_translation_rejects_newline_or_equals() -> None:
    assert _clean_translation("a\nb", "WM") is None
    assert _clean_translation("a=b", "WM") is None


# --------------------------------------------------------------------------- #
# _apply_response
# --------------------------------------------------------------------------- #


def test_apply_response_equals_and_colon_forms() -> None:
    batch = [_entry("Working Memory", ["WM"]), _entry("Central Unit", ["CU"])]
    filled = _apply_response("WM = 工作记忆\nCU: 中央单元", batch)
    assert filled == 2
    assert batch[0]["translation"] == "工作记忆"
    assert batch[1]["translation"] == "中央单元"


def test_apply_response_numbered_and_bulleted_lines() -> None:
    batch = [_entry("A", ["A"]), _entry("B", ["B"]), _entry("C", ["C"])]
    filled = _apply_response("1. A = 甲\n2) B = 乙\n- C = 丙", batch)
    assert filled == 3


def test_apply_response_first_occurrence_wins() -> None:
    batch = [_entry("A", ["A"])]
    assert _apply_response("A = first\nA = second", batch) == 1
    assert batch[0]["translation"] == "first"


def test_apply_response_never_overwrites_existing() -> None:
    batch = [_entry("A", ["A"], translation="existing")]
    assert _apply_response("A = new", batch) == 0
    assert batch[0]["translation"] == "existing"


def test_apply_response_matches_any_alias_case_insensitively() -> None:
    batch = [_entry("Working Memory", ["WM", "W.M."])]
    assert _apply_response("w.m. = 工作记忆", batch) == 1
    assert batch[0]["translation"] == "工作记忆"


def test_apply_response_without_aliases_is_not_filled() -> None:
    batch = [_entry("Working Memory", [])]
    assert _apply_response("Working Memory = 工作记忆", batch) == 0
    assert batch[0]["translation"] == ""


def test_apply_response_ignores_unparseable_lines() -> None:
    batch = [_entry("A", ["A"])]
    assert _apply_response("no separator here\nA = 甲", batch) == 1


def test_apply_response_rejects_translation_containing_equals() -> None:
    batch = [_entry("A", ["A"])]
    assert _apply_response("A = x = y", batch) == 0


def test_apply_response_decimal_like_numbering_is_not_stripped() -> None:
    batch = [_entry("A", ["A"])]
    assert _apply_response("1.2. A = 甲", batch) == 0


def test_apply_response_empty_response() -> None:
    assert _apply_response("", [_entry("A", ["A"])]) == 0


# --------------------------------------------------------------------------- #
# build_backfill_prompt
# --------------------------------------------------------------------------- #


def test_build_prompt_mentions_target_language() -> None:
    system_prompt, _ = build_backfill_prompt([_entry("A", ["A"])], "zh")
    assert "zh" in system_prompt
    assert "EXACTLY" in system_prompt


def test_build_prompt_lists_items_with_alias_key() -> None:
    _, user_prompt = build_backfill_prompt([_entry("Working Memory", ["WM"])], "zh")
    assert "### Task" in user_prompt
    assert "### Items" in user_prompt
    assert "1. WM — Working Memory" in user_prompt


def test_build_prompt_marks_person_names() -> None:
    _, user_prompt = build_backfill_prompt(
        [_entry("Bingley", ["Mr. Bingley"], kind="person")], "zh"
    )
    assert "1. Mr. Bingley — Bingley (person name)" in user_prompt


def test_build_prompt_flattens_source_newlines() -> None:
    _, user_prompt = build_backfill_prompt([_entry("a\nb", ["K"])], "zh")
    assert "1. K — a b" in user_prompt


def test_build_prompt_missing_alias_uses_empty_key() -> None:
    _, user_prompt = build_backfill_prompt([_entry("Source", [])], "zh")
    assert "1.  — Source" in user_prompt


# --------------------------------------------------------------------------- #
# backfill_abbreviation_translations
# --------------------------------------------------------------------------- #


async def test_backfill_fills_empty_translations() -> None:
    entries = [_entry("Working Memory", ["WM"]), _entry("Central Unit", ["CU"])]

    async def complete(system: str, user: str) -> str:
        return "WM = 工作记忆\nCU = 中央单元"

    result, filled = await backfill_abbreviation_translations(entries, complete)
    assert filled == 2
    assert result[0]["translation"] == "工作记忆"
    assert result[1]["translation"] == "中央单元"


async def test_backfill_skips_already_translated_entries() -> None:
    entries = [_entry("Working Memory", ["WM"], translation="既有")]

    async def complete(system: str, user: str) -> str:
        return "WM = 新的"

    _, filled = await backfill_abbreviation_translations(entries, complete)
    assert filled == 0
    assert entries[0]["translation"] == "既有"


async def test_backfill_groups_variants_sharing_a_key() -> None:
    # Mr. Bingley / Miss Bingley share the "bingley" core through their aliases.
    entries = [
        _entry("Mr. Bingley", ["bingley"], kind="person"),
        _entry("Miss Bingley", ["bingley"], kind="person"),
    ]
    calls: list[str] = []

    async def complete(system: str, user: str) -> str:
        calls.append(user)
        return "bingley = 彬格莱"

    _, filled = await backfill_abbreviation_translations(entries, complete)
    assert filled == 2
    assert entries[0]["translation"] == "彬格莱"
    assert entries[1]["translation"] == "彬格莱"
    assert len(calls) == 1  # one representative per shared key


async def test_backfill_batches_and_applies_only_within_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ubt.core.memory.abbreviation_backfill._MAX_BATCH", 1)
    entries = [_entry("Alpha", ["AA"]), _entry("Beta", ["BB"])]
    calls = 0

    async def complete(system: str, user: str) -> str:
        nonlocal calls
        calls += 1
        # The first batch's response also names the *other* batch's acronym;
        # it must not fill Beta (that entry belongs to the second batch).
        return "AA = 甲\nBB = 乙" if calls == 1 else ""

    _, filled = await backfill_abbreviation_translations(entries, complete)
    assert calls == 2
    assert filled == 1
    assert entries[0]["translation"] == "甲"
    assert entries[1]["translation"] == ""


async def test_backfill_empty_entries_makes_no_call() -> None:
    calls = 0

    async def complete(system: str, user: str) -> str:
        nonlocal calls
        calls += 1
        return ""

    _, filled = await backfill_abbreviation_translations([], complete)
    assert filled == 0
    assert calls == 0


async def test_backfill_unparseable_response_leaves_empty() -> None:
    entries = [_entry("Alpha", ["AA"])]

    async def complete(system: str, user: str) -> str:
        return "garbage without separators"

    _, filled = await backfill_abbreviation_translations(entries, complete)
    assert filled == 0
    assert entries[0]["translation"] == ""


def test_max_batch_constant() -> None:
    assert _MAX_BATCH == 60


def test_max_translation_chars_constant() -> None:
    assert _MAX_TRANSLATION_CHARS == 80
