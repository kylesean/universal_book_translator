"""Unit tests for the target-side emphasis markup codec."""

from __future__ import annotations

import pytest

from ubt.core.ir.emphasis import (
    BOLD_CLOSE,
    BOLD_OPEN,
    mark_bold_spans,
    parse_bold,
    strip_emphasis_markers,
)

pytestmark = pytest.mark.fast


def test_no_markers_leaves_text_unchanged() -> None:
    clean, runs = parse_bold("图 1 | SoL-Pi 降低了 50.0%。")
    assert clean == "图 1 | SoL-Pi 降低了 50.0%。"
    assert runs == ()


def test_a_pair_becomes_one_bold_run() -> None:
    clean, runs = parse_bold(f"成本降低了 {BOLD_OPEN}50.0%{BOLD_CLOSE}；随后")
    assert clean == "成本降低了 50.0%；随后"
    assert [(r.text, r.bold) for r in runs] == [("50.0%", True)]


def test_two_pairs_become_two_runs() -> None:
    clean, runs = parse_bold(f"{BOLD_OPEN}甲{BOLD_CLOSE}和{BOLD_OPEN}乙{BOLD_CLOSE}")
    assert clean == "甲和乙"
    assert [r.text for r in runs] == ["甲", "乙"]


def test_nested_markers_are_one_run() -> None:
    clean, runs = parse_bold(f"{BOLD_OPEN}外{BOLD_OPEN}内{BOLD_CLOSE}外{BOLD_CLOSE}")
    assert clean == "外内外"
    assert [r.text for r in runs] == ["外内外"]


def test_unterminated_open_emphasizes_to_the_end() -> None:
    clean, runs = parse_bold(f"前{BOLD_OPEN}重点")
    assert clean == "前重点"
    assert [r.text for r in runs] == ["重点"]


def test_a_stray_close_is_dropped() -> None:
    clean, runs = parse_bold(f"前{BOLD_CLOSE}后")
    assert clean == "前后"
    assert runs == ()


def test_markers_with_inner_spacing_or_case_are_recognized() -> None:
    clean, runs = parse_bold("⟦ b ⟧重点⟦ / B ⟧")
    assert clean == "重点"
    assert [r.text for r in runs] == ["重点"]


def test_a_run_never_carries_edge_whitespace() -> None:
    clean, runs = parse_bold(f"{BOLD_OPEN} 重点 {BOLD_CLOSE}")
    assert clean == " 重点 "
    assert [r.text for r in runs] == ["重点"]


def test_mark_then_parse_round_trips() -> None:
    source = "SoL-Pi discovers a more token-efficient harness."
    marked = mark_bold_spans(source, ["SoL-Pi discovers"])
    assert marked == f"{BOLD_OPEN}SoL-Pi discovers{BOLD_CLOSE} a more token-efficient harness."
    clean, runs = parse_bold(marked)
    assert clean == source
    assert [r.text for r in runs] == ["SoL-Pi discovers"]


def test_mark_skips_missing_and_empty_spans() -> None:
    assert mark_bold_spans("abc", ["nope", "", "  "]) == "abc"


def test_mark_folds_a_dash_mismatch_between_extractor_and_reader() -> None:
    # The pdfium probe can report an en dash where the reader's source text has
    # a hyphen ("agent–environment" vs "agent-environment"); the span must still
    # be found and the *source* spelling preserved inside the markers.
    marked = mark_bold_spans("agent-environment loop", ["agent\u2013environment"])
    assert marked == f"{BOLD_OPEN}agent-environment{BOLD_CLOSE} loop"
    clean, runs = parse_bold(marked)
    assert clean == "agent-environment loop"
    assert [r.text for r in runs] == ["agent-environment"]


def test_mark_uses_each_occurrence_in_order() -> None:
    marked = mark_bold_spans("a X b X", ["X", "X"])
    assert marked == f"a {BOLD_OPEN}X{BOLD_CLOSE} b {BOLD_OPEN}X{BOLD_CLOSE}"


def test_strip_emphasis_markers_removes_every_marker() -> None:
    assert strip_emphasis_markers(f"a{BOLD_OPEN}b{BOLD_CLOSE}c") == "abc"


def test_parse_never_leaves_a_marker_in_the_clean_text() -> None:
    for raw in (
        f"{BOLD_OPEN}{BOLD_CLOSE}",
        f"{BOLD_OPEN}未闭合",
        f"孤立{BOLD_CLOSE}收尾",
        f"{BOLD_OPEN}a{BOLD_OPEN}b{BOLD_CLOSE}",
        "⟦ B ⟧x⟦ / B ⟧",
    ):
        clean, _ = parse_bold(raw)
        assert "⟦" not in clean
        assert "⟧" not in clean
