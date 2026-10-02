"""Glossary term matching: script-aware occurrence tests and chunk selection.

The matcher decides which glossary rows ride along with a block. Two failure
modes matter and pull in opposite directions: a term that occurs but is not
selected loses its enforced rendering (drift), while a 2-3 char CJK fragment
that only ever occurs embedded inside a longer word is a compound, not a hit,
and selecting it floods the prompt with noise. The contract here pins both the
match semantics (word-boundary for alphabetic scripts, substring for CJK, and
the deliberate single-CJK-character rejection) and the selection order.
"""

from __future__ import annotations

from typing import Any

import pytest

from ubt.core.memory.cjk_matcher import (
    _AHO_MIN_TERMS,
    _cjk_has_free_boundary,
    _term_matches,
    contains_cjk,
    count_term_in_text,
    format_terms_markdown_table,
    select_terms_for_chunk,
    term_appears_in_text,
)

pytestmark = pytest.mark.fast


def _ids(terms: list[dict[str, object]]) -> list[object]:
    return [term.get("id", term.get("source")) for term in terms]


# --------------------------------------------------------------------------- #
# contains_cjk
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", False),
        ("\u6a21\u578b", True),
        ("abc\u6f22", True),
        ("\ud55c\uad6d", True),
        ("", False),
    ],
)
def test_contains_cjk(text: str, expected: bool) -> None:
    assert contains_cjk(text) is expected


# --------------------------------------------------------------------------- #
# count_term_in_text
# --------------------------------------------------------------------------- #


def test_latin_count_is_case_insensitive() -> None:
    assert count_term_in_text("Attention", "attention and ATTENTION") == 2


def test_latin_count_respects_word_boundaries() -> None:
    # 'cat' must not match 'category'.
    assert count_term_in_text("cat", "cat category cat.") == 2


def test_cjk_count_is_a_substring_scan() -> None:
    assert count_term_in_text("\u6a21\u578b", "\u6a21\u578b\u548c\u6a21\u578b") == 2


def test_a_single_cjk_character_is_never_counted() -> None:
    # Over-matching a lone character would select the term for every block.
    assert count_term_in_text("\u6a21", "\u6a21\u578b\u6a21\u578b") == 0


def test_cyrillic_count_is_case_insensitive() -> None:
    assert (
        count_term_in_text("\u041c\u0438\u0440", "\u043c\u0438\u0440 \u0438 \u041c\u0438\u0440")
        == 2
    )


@pytest.mark.parametrize(("source", "text"), [("", "x"), ("x", "")])
def test_count_of_a_blank_side_is_zero(source: str, text: str) -> None:
    assert count_term_in_text(source, text) == 0


# --------------------------------------------------------------------------- #
# _term_matches / term_appears_in_text
# --------------------------------------------------------------------------- #


def test_term_matches_latin_uses_boundaries() -> None:
    assert _term_matches("cat", "a cat here") is True
    assert _term_matches("cat", "category") is False


def test_term_matches_cjk_uses_substring_and_rejects_single_char() -> None:
    assert _term_matches("\u6a21\u578b", "\u8fd9\u662f\u6a21\u578b") is True
    assert _term_matches("\u6a21", "\u6a21\u578b") is False


def test_term_appears_via_source_or_alias() -> None:
    term = {"source": "\u6a21\u578b", "aliases": ["model", "\u67b6\u6784"]}
    assert term_appears_in_text(term, "\u6a21\u578b") is True
    assert term_appears_in_text(term, "the model works") is True
    assert term_appears_in_text(term, "\u67b6\u6784") is True
    assert term_appears_in_text(term, "nothing here") is False


def test_term_without_aliases_only_matches_its_source() -> None:
    assert term_appears_in_text({"source": "\u6a21\u578b"}, "\u6a21\u578b") is True
    assert term_appears_in_text({"source": "\u6a21\u578b"}, "model") is False


# --------------------------------------------------------------------------- #
# _cjk_has_free_boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\u6a21\u578babc", True),  # free on the left
        ("abc\u6a21\u578b", True),  # free on the right
        ("\u5927\u6a21\u578b\u7cfb\u7edf", False),  # only ever embedded
        ("\u6a21\u578b", True),  # the whole text
    ],
)
def test_cjk_free_boundary(text: str, expected: bool) -> None:
    assert _cjk_has_free_boundary("\u6a21\u578b", text) is expected


@pytest.mark.parametrize(("source", "text"), [("", "x"), ("x", "")])
def test_free_boundary_of_a_blank_side_is_false(source: str, text: str) -> None:
    assert _cjk_has_free_boundary(source, text) is False


# --------------------------------------------------------------------------- #
# format_terms_markdown_table
# --------------------------------------------------------------------------- #


def test_formatting_no_terms_is_empty() -> None:
    assert format_terms_markdown_table([]) == ""


def test_formatting_renders_the_header_and_a_row() -> None:
    table = format_terms_markdown_table(
        [{"source": "\u6a21\u578b", "aliases": ["model"], "translation": "model"}]
    )
    assert table.splitlines() == [
        "| \u539f\u6587 | \u522b\u540d | \u8bd1\u6587 |",
        "| :--- | :--- | :--- |",
        "| \u6a21\u578b | model | model |",
    ]


def test_formatting_escapes_pipes() -> None:
    table = format_terms_markdown_table(
        [{"source": "a|b", "aliases": ["c|d"], "translation": "e|f"}]
    )
    assert "| a\\|b | c\\|d | e\\|f |" in table


def test_formatting_falls_back_to_the_target_key() -> None:
    assert "| x |  | y |" in format_terms_markdown_table([{"source": "x", "target": "y"}])


def test_formatting_collapses_whitespace_in_cells() -> None:
    assert "| a b |  |  |" in format_terms_markdown_table(
        [{"source": "  a   b ", "translation": ""}]
    )


# --------------------------------------------------------------------------- #
# select_terms_for_chunk
# --------------------------------------------------------------------------- #


def test_selection_of_no_terms_is_empty() -> None:
    assert select_terms_for_chunk([], "anything") == []


def test_selection_puts_local_hits_first_then_global_top_n() -> None:
    terms = [
        {"id": "a", "source": "alpha", "frequency": 5},
        {"id": "b", "source": "beta", "frequency": 9},
        {"id": "c", "source": "gamma", "frequency": 1},
    ]
    # 'gamma' is local; the two global slots go to the highest frequencies.
    assert _ids(select_terms_for_chunk(terms, "the gamma ray", top_n=2, max_terms=50)) == [
        "c",
        "b",
        "a",
    ]


def test_selection_respects_max_terms() -> None:
    terms = [
        {"id": "a", "source": "alpha", "frequency": 5},
        {"id": "b", "source": "beta", "frequency": 9},
        {"id": "c", "source": "gamma", "frequency": 1},
    ]
    assert _ids(select_terms_for_chunk(terms, "gamma", top_n=2, max_terms=2)) == ["c", "b"]


@pytest.mark.parametrize("cap", [0, -1])
def test_a_nonpositive_cap_disables_the_table(cap: int) -> None:
    assert select_terms_for_chunk([{"id": "a", "source": "alpha"}], "alpha", max_terms=cap) == []


def test_local_hits_are_truncated_to_the_cap() -> None:
    terms = [
        {"id": "a", "source": "alpha", "frequency": 5},
        {"id": "b", "source": "beta", "frequency": 9},
        {"id": "c", "source": "gamma", "frequency": 1},
    ]
    assert _ids(select_terms_for_chunk(terms, "alpha beta gamma", top_n=0, max_terms=2)) == [
        "b",
        "a",
    ]


def test_an_embedded_cjk_fragment_is_deprioritized_not_dropped() -> None:
    terms = [
        {"id": "clean", "source": "\u6a21\u578b", "frequency": 1},
        # '\u7f51\u7edc' occurs only inside '\u795e\u7ecf\u7f51\u7edc\u7cfb\u7edf', so it is a
        # compound fragment even though its frequency is far higher.
        {"id": "embedded", "source": "\u7f51\u7edc", "frequency": 100},
    ]
    assert _ids(
        select_terms_for_chunk(
            terms, "\u6a21\u578babc \u795e\u7ecf\u7f51\u7edc\u7cfb\u7edf", top_n=0
        )
    ) == [
        "clean",
        "embedded",
    ]


def test_selection_is_identical_above_the_automaton_threshold() -> None:
    # The Aho-Corasick prefilter is only a speed-up; semantics must not change.
    terms = [
        {"id": f"t{i}", "source": f"term{i}", "frequency": i} for i in range(_AHO_MIN_TERMS + 2)
    ]
    result = _ids(select_terms_for_chunk(terms, "term3 term7", top_n=3, max_terms=50))
    assert result == ["t7", "t3", "t9", "t8", "t6"]


def test_a_key_shared_between_a_source_and_an_alias_keeps_both_terms() -> None:
    # One key can belong to two terms ('neural' is both a source and an alias).
    # The automaton stores a per-key index list; collapsing it to one value
    # dropped every term but the last from the candidate set.
    terms: list[dict[str, Any]] = [
        {"id": f"f{i}", "source": f"filler{i}"} for i in range(_AHO_MIN_TERMS - 2)
    ]
    terms += [
        {"id": "src", "source": "neural"},
        {"id": "alias", "source": "\u522b\u7684", "aliases": ["neural"]},
    ]
    result = _ids(select_terms_for_chunk(terms, "neural networks", top_n=0))
    assert "src" in result
    assert "alias" in result
