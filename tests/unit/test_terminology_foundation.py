"""Regression tests for the publication-grade terminology foundation.

Covers TM fuzzy context gate, bible frequency ranking,
and Aho-Corasick selection with CJK embedded-match guard.
"""

from pathlib import Path

import pytest

from ubt.core.memory.bible import BibleEntry, merge_bible_entries
from ubt.core.memory.cjk_matcher import select_terms_for_chunk
from ubt.core.memory.tm import (
    PROVENANCE_HUMAN_PE,
    TMPendingEntry,
    TranslationMemory,
)

pytestmark = pytest.mark.fast


def test_fuzzy_hit_requires_caller_context(tmp_path: Path) -> None:
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    tm.writeback(
        [
            TMPendingEntry(
                src_lang="en",
                tgt_lang="zh",
                source_text="The torque converter failed.",
                target_text="变矩器故障。",
                context_hash="CTX_A",
            )
        ]
    )
    assert (
        tm.lookup_fuzzy("en", "zh", "The torque converter failed badly.", context_hash="CTX_A")
        is not None
    )
    assert (
        tm.lookup_fuzzy("en", "zh", "The torque converter failed badly.", context_hash="CTX_B")
        is None
    )


def test_fuzzy_human_pe_survives_context_change(tmp_path: Path) -> None:
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    tm.writeback(
        [
            TMPendingEntry(
                src_lang="en",
                tgt_lang="zh",
                source_text="The stator is worn.",
                target_text="定子磨损。",
                provenance=PROVENANCE_HUMAN_PE,
                context_hash="CTX_A",
            )
        ]
    )
    hit = tm.lookup_fuzzy("en", "zh", "The stator is worn out.", context_hash="CTX_B")
    # Human post-edits are exempt from the context gate, so the *content* must
    # come back, not merely a non-None object: asserting existence alone would
    # pass for any row the lookup happened to return.
    assert hit is not None
    assert hit.target_text == "定子磨损。"
    assert hit.provenance == PROVENANCE_HUMAN_PE


def test_merge_keeps_max_frequency() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="LSTM", translation="长短期记忆", frequency=3),
            BibleEntry(source="lstm", translation="", aliases=["long"], frequency=9),
        ]
    )
    assert len(merged) == 1
    assert merged[0].frequency == 9
    assert merged[0].translation == "长短期记忆"


def test_frequency_drives_top_n_ranking() -> None:
    terms = [
        {"source": "zebra", "translation": "斑马", "aliases": [], "frequency": 1},
        {"source": "apple", "translation": "苹果", "aliases": [], "frequency": 99},
    ]
    selected = select_terms_for_chunk(terms, "unrelated chunk text", top_n=1)
    assert [t["source"] for t in selected] == ["apple"]


def test_embedded_short_cjk_deprioritized_not_dropped() -> None:
    terms = [
        {"source": "网络", "translation": "network", "aliases": [], "frequency": 5},
        {"source": "Transformer", "translation": "变换器", "aliases": [], "frequency": 50},
    ]
    selected = select_terms_for_chunk(terms, "Transformer 计算机网络模型", top_n=2)
    sources = [t["source"] for t in selected]
    assert sources[0] == "Transformer"
    assert "网络" in sources


def test_cjk_matcher_latin_terms_in_cjk_text() -> None:
    from ubt.core.memory.cjk_matcher import _compile_boundary_pattern, count_term_in_text

    # "LSTM" embedded inside Chinese text without surrounding ASCII spaces
    text = "针对LSTM网络进行结构优化与评估"
    # Must count 1, not 0
    count = count_term_in_text("LSTM", text)
    assert count == 1, f"Expected 1 hit for LSTM in {text!r}, got {count}"

    pattern = _compile_boundary_pattern("LSTM")
    assert pattern.search(text) is not None


def test_cjk_matcher_table_sanitizes_newlines() -> None:
    from ubt.core.memory.cjk_matcher import format_terms_markdown_table

    terms = [
        {
            "source": "multi\nline\rterm",
            "target": "多行\n译文",
            "aliases": ["alias\n1"],
        }
    ]
    table = format_terms_markdown_table(terms)
    # The rendered markdown table must not have raw newlines breaking row syntax
    # Count rows by splitting lines starting with '|'
    lines = [line.strip() for line in table.strip().splitlines() if line.strip().startswith("|")]
    # Header + separator + 1 row = exactly 3 lines
    assert len(lines) == 3, f"Expected exactly 3 table lines, got {len(lines)}:\n{table}"
