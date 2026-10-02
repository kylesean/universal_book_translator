"""Glossary markdown tables: the retrieval table and the capped decision sheet.

The book-level sheet cannot be retrieved (an alias-only mention, a deliberate
non-obvious rendering), so it is truncated by rank: proper names first, then
attest frequency, then source text as a deterministic tie-break -- the sheet
the prompt carries and the sheet the TM context fingerprint records must be the
same bytes. The chunk table is the retrieval path and unions local hits with
the global top-N.
"""

from __future__ import annotations

import pytest

from ubt.core.memory.glossary_table import (
    build_chunk_glossary_table,
    build_global_glossary_table,
)

pytestmark = pytest.mark.fast


def _sources(table: str) -> list[str]:
    """The first cell of each data row (skipping the header and separator)."""
    rows = table.splitlines()[2:]
    return [line.split("|")[1].strip() for line in rows]


# --------------------------------------------------------------------------- #
# build_global_glossary_table
# --------------------------------------------------------------------------- #


def test_global_table_of_no_terms_is_empty() -> None:
    assert build_global_glossary_table([], 10) == ""


@pytest.mark.parametrize("cap", [0, -1])
def test_a_nonpositive_cap_disables_the_global_sheet(cap: int) -> None:
    assert build_global_glossary_table([{"source": "x"}], cap) == ""


def test_global_table_drops_rows_without_a_source() -> None:
    assert build_global_glossary_table([{"translation": "x"}], 10) == ""


def test_global_table_ranks_names_first_then_frequency() -> None:
    terms = [
        {"source": "common", "kind": "term", "frequency": 100, "translation": "c"},
        {"source": "Zeta", "kind": "person", "frequency": 1, "translation": "z"},
        {"source": "Alpha", "kind": "place", "frequency": 1, "translation": "a"},
        {"source": "Beta", "kind": "term", "frequency": 1, "translation": "b"},
    ]
    assert _sources(build_global_glossary_table(terms, 10)) == [
        "Alpha",
        "Zeta",
        "common",
        "Beta",
    ]


def test_the_cap_is_applied_after_ranking() -> None:
    terms = [
        {"source": "common", "kind": "term", "frequency": 100},
        {"source": "Zeta", "kind": "person", "frequency": 1},
        {"source": "Alpha", "kind": "place", "frequency": 1},
    ]
    assert _sources(build_global_glossary_table(terms, 2)) == ["Alpha", "Zeta"]


def test_source_text_breaks_frequency_ties_deterministically() -> None:
    terms = [{"source": "bbb", "frequency": 5}, {"source": "aaa", "frequency": 5}]
    assert _sources(build_global_glossary_table(terms, 10)) == ["aaa", "bbb"]


# --------------------------------------------------------------------------- #
# build_chunk_glossary_table
# --------------------------------------------------------------------------- #


def test_chunk_table_of_nothing_is_empty() -> None:
    assert build_chunk_glossary_table([], [], "x") == ""


def test_chunk_table_renders_terms_without_abbreviations() -> None:
    gloss = [{"id": "m", "source": "\u6a21\u578b", "translation": "model", "frequency": 3}]
    table = build_chunk_glossary_table(gloss, [], "\u8fd9\u662f\u6a21\u578b")
    assert "| \u6a21\u578b |  | model |" in table
    assert "\u7f29\u5199" not in table


def test_chunk_table_appends_a_local_abbreviation_block() -> None:
    gloss = [{"id": "m", "source": "\u6a21\u578b", "translation": "model", "frequency": 3}]
    abbrevs = [
        {
            "source": "Convolutional Neural Network",
            "translation": "",
            "aliases": ["CNN"],
            "kind": "term",
        }
    ]
    table = build_chunk_glossary_table(
        gloss, abbrevs, "\u8fd9\u662f\u6a21\u578b\u548c CNN \u7684\u4ecb\u7ecd"
    )
    assert "| CNN | Convolutional Neural Network |" in table
    # The abbreviation block follows the term table.
    assert table.index("\u6a21\u578b") < table.index("CNN")


def test_chunk_table_omits_abbreviations_absent_from_the_block() -> None:
    abbrevs = [{"source": "Convolutional Neural Network", "aliases": ["CNN"], "kind": "term"}]
    table = build_chunk_glossary_table([], abbrevs, "\u8fd9\u91cc\u6ca1\u6709\u7f29\u5199")
    assert "CNN" not in table
