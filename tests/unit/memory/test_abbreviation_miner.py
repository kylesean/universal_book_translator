"""The 0-token abbreviation miner: "Full Expansion (ACRONYM)" extraction.

These pairs are the highest-value glossary rows (author-attested, early,
recurring), so the miner is precision-first: the words immediately before the
parenthesised acronym must spell it by their initials, which admits mixed
initialisms (LSTM, SDPA, BERT) and rejects greedy false prefixes. Digit-bearing
and stopword acronyms are dropped as noise.
"""

from __future__ import annotations

import pytest

from ubt.core.memory.abbreviation_miner import (
    _find_expansion_for,
    format_abbreviations_markdown_table,
    mine_abbreviations,
    mine_abbreviations_stream,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# _find_expansion_for
# --------------------------------------------------------------------------- #


def test_a_plain_initialism_expands() -> None:
    assert _find_expansion_for("LSTM", "Long Short-Term Memory ") == "Long Short-Term Memory"


def test_hyphenated_compounds_split_into_initials() -> None:
    # 'dot-product' contributes dot + product to SDPA.
    assert (
        _find_expansion_for("SDPA", "scaled dot-product attention ")
        == "scaled dot-product attention"
    )


def test_stop_words_may_be_skipped_in_the_expansion() -> None:
    assert (
        _find_expansion_for("BERT", "Bidirectional Encoder Representations from Transformers ")
        == "Bidirectional Encoder Representations from Transformers"
    )


def test_a_greedy_false_prefix_is_rejected() -> None:
    # "We employ scaled dot-product attention" does not spell LSTM.
    assert _find_expansion_for("LSTM", "We employ scaled dot-product attention ") is None


def test_a_single_token_expansion_is_rejected() -> None:
    assert _find_expansion_for("AB", "Alpha ") is None


# --------------------------------------------------------------------------- #
# mine_abbreviations
# --------------------------------------------------------------------------- #


def test_mining_extracts_pairs_in_the_bible_entry_shape() -> None:
    text = "Long Short-Term Memory (LSTM) is used. Also scaled dot-product attention (SDPA) here."
    assert mine_abbreviations(text) == [
        {
            "source": "Long Short-Term Memory",
            "translation": "",
            "aliases": ["LSTM"],
            "kind": "term",
        },
        {
            "source": "scaled dot-product attention",
            "translation": "",
            "aliases": ["SDPA"],
            "kind": "term",
        },
    ]


def test_the_first_attestation_of_an_acronym_wins() -> None:
    # Both expansions spell LSTM, so the choice is a real tie-break.
    text = "Long Short-Term Memory (LSTM) and Lost Something Totally Meaningful (LSTM)."
    entries = mine_abbreviations(text)
    assert len(entries) == 1
    assert entries[0]["source"] == "Long Short-Term Memory"


@pytest.mark.parametrize(
    "text",
    [
        "Version G5 (G5) noise.",  # a digit-bearing acronym is version noise
        "Artificial Intelligence (AI).",  # a stopword acronym
        "Television (TV).",
        "AB (AB) here.",  # the expansion is too short
    ],
)
def test_noisy_acronyms_are_rejected(text: str) -> None:
    assert mine_abbreviations(text) == []


def test_mining_of_empty_text_is_empty() -> None:
    assert mine_abbreviations("") == []


def test_the_cap_keeps_the_earliest_then_sorts_by_acronym() -> None:
    # First-attested order (ZA, AB) differs from acronym order (AB, ZA), so the
    # sort is observable: cap the first two, then order them by acronym.
    text = "Zulu Alpha (ZA), Alpha Beta (AB), Charlie Delta (CD)"
    entries = mine_abbreviations(text, max_entries=2)
    assert [entry["aliases"][0] for entry in entries] == ["AB", "ZA"]


# --------------------------------------------------------------------------- #
# mine_abbreviations_stream
# --------------------------------------------------------------------------- #


def test_streaming_matches_a_single_pass_over_joined_blocks() -> None:
    blocks = [
        "Long Short-Term Memory (LSTM) is used.",
        "scaled dot-product attention (SDPA) works.",
    ]
    assert mine_abbreviations_stream(blocks) == mine_abbreviations("\n".join(blocks))


def test_streaming_of_no_blocks_is_empty() -> None:
    assert mine_abbreviations_stream([]) == []


def test_a_pair_spanning_a_block_boundary_is_still_mined() -> None:
    # The tail carry-over exists so a pair split across blocks is not lost.
    entries = mine_abbreviations_stream(
        ["scaled dot-product", " attention (SDPA) here"], chunk_chars=10
    )
    assert [entry["aliases"][0] for entry in entries] == ["SDPA"]


# --------------------------------------------------------------------------- #
# format_abbreviations_markdown_table
# --------------------------------------------------------------------------- #


def test_formatting_no_entries_is_empty() -> None:
    assert format_abbreviations_markdown_table([]) == ""


def test_formatting_renders_acronym_and_expansion() -> None:
    table = format_abbreviations_markdown_table(
        mine_abbreviations("Long Short-Term Memory (LSTM).")
    )
    assert "| LSTM | Long Short-Term Memory |" in table
    assert "\u4fdd\u6301\u7f29\u5199\u4e0d\u53d8" in table


def test_formatting_escapes_pipes_only_in_the_expansion() -> None:
    table = format_abbreviations_markdown_table([{"aliases": ["A|B"], "source": "x|y"}])
    assert "| A|B | x\\|y |" in table
