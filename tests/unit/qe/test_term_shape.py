"""Shared shape predicates: identifier detection and sentence counting.

``is_identifier_shaped`` decides whether the omission gate must *demand* a Latin
term survive translation; ``is_verbatim_carryover`` is its wider superset used to
strip legitimately-kept tokens from the target-language density residue. The
difference is load-bearing: ``IT``/``US``/``MHA`` kept verbatim must not read as
untranslated, yet the omission gate must not demand ``IT``/``OR`` survive.
``count_sentences`` is the one counter both the omission gate and the MT
admission gate share, so abbreviation periods must not inflate the count.
"""

from __future__ import annotations

import pytest

from ubt.core.qe.term_shape import (
    count_sentences,
    is_identifier_shaped,
    is_verbatim_carryover,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# is_identifier_shaped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "term",
    [
        "fig",
        "Fig",
        "TABLE",
        "eq",
        "no",
        "pp",
        "appendix",
        "Figure1",
        "Figure1a",
        "Fig.1",
        "Fig-2",
        "Table3",
        "Table-1",
        "Eq4",
        "Sec5.1",
    ],
)
def test_structural_document_labels_are_not_identifiers(term: str) -> None:
    assert is_identifier_shaped(term) is False


@pytest.mark.parametrize("term", ["GQA-8", "GPT4", "3D"])
def test_digit_bearing_terms_are_identifiers(term: str) -> None:
    assert is_identifier_shaped(term) is True


@pytest.mark.parametrize("term", ["PagedAttention", "iPhone"])
def test_camel_case_terms_are_identifiers(term: str) -> None:
    assert is_identifier_shaped(term) is True


@pytest.mark.parametrize("term", ["foo_bar", "a.b"])
def test_underscore_or_dot_terms_are_identifiers(term: str) -> None:
    assert is_identifier_shaped(term) is True


@pytest.mark.parametrize("term", ["MHA", "LSTM", "CPU"])
def test_long_all_caps_acronyms_are_identifiers(term: str) -> None:
    assert is_identifier_shaped(term) is True


@pytest.mark.parametrize("term", ["IT", "OR", "US", "AI"])
def test_short_all_caps_runs_are_not_identifiers(term: str) -> None:
    assert is_identifier_shaped(term) is False


@pytest.mark.parametrize("term", ["THE", "AND", "NEW"])
def test_common_uppercased_words_are_not_identifiers(term: str) -> None:
    assert is_identifier_shaped(term) is False


@pytest.mark.parametrize("term", ["hello", "Hello", "plain prose"])
def test_ordinary_prose_is_not_an_identifier(term: str) -> None:
    assert is_identifier_shaped(term) is False


# --------------------------------------------------------------------------- #
# is_verbatim_carryover
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("term", ["PagedAttention", "GQA-8", "MHA", "CPU"])
def test_carryover_accepts_identifiers(term: str) -> None:
    assert is_verbatim_carryover(term) is True


@pytest.mark.parametrize("term", ["IT", "OR", "US", "AI", "THE", "AND"])
def test_carryover_accepts_short_all_caps_unlike_identifier_shaped(term: str) -> None:
    assert is_verbatim_carryover(term) is True
    assert is_identifier_shaped(term) is False


@pytest.mark.parametrize("term", ["foo_bar", "a.b", "e.g."])
def test_carryover_rejects_underscore_and_dot_terms(term: str) -> None:
    # Underscore/dot are identifier-shaped for the omission gate but are not a
    # camelCase/acronym/digit run, so the density check does not strip them.
    assert is_identifier_shaped(term) is True
    assert is_verbatim_carryover(term) is False


@pytest.mark.parametrize("term", ["fig", "Fig", "TABLE"])
def test_carryover_rejects_structural_labels(term: str) -> None:
    assert is_verbatim_carryover(term) is False


def test_carryover_rejects_plain_prose() -> None:
    assert is_verbatim_carryover("hello") is False


# --------------------------------------------------------------------------- #
# count_sentences
# --------------------------------------------------------------------------- #


def test_empty_text_has_no_sentences() -> None:
    assert count_sentences("") == 0


def test_plain_sentences_are_counted() -> None:
    assert count_sentences("Hello world.") == 1
    assert count_sentences("One. Two. Three.") == 3


def test_abbreviation_periods_do_not_split() -> None:
    assert count_sentences("Eq. 3.14 shows x. Then y.") == 2
    assert count_sentences("Fig. 3.5 is here") == 1
    assert count_sentences("etc. and more") == 1


def test_reference_labels_are_masked_only_before_a_number_or_citation() -> None:
    # 'No.'/'Ref.'/'Eq.' introduce a number/citation -> not a sentence break.
    assert count_sentences("See No. 5 for details.") == 1
    assert count_sentences("Ref. [3] proves it.") == 1
    assert count_sentences("Eq. (3.11) gives the bound.") == 1


def test_ordinary_words_ending_a_sentence_are_still_counted() -> None:
    # These spellings share a prefix with reference labels but are plain words
    # here; masking their period would under-count the source and could drop it
    # below min_source_sentences, silently disarming the omission gate.
    assert count_sentences("This is a lab. We test the model. It works.") == 3
    assert count_sentences("We report the max. The min is lower.") == 2
    assert count_sentences("The answer is no. We disagree.") == 2
    assert count_sentences("They measured the var. The mean is stable.") == 2


def test_decimal_points_do_not_split() -> None:
    assert count_sentences("3.14 is pi") == 1
    assert count_sentences("a.b.c") == 1
    assert count_sentences("It rose by 19 . 6% versus 1 . 1% before.") == 1


def test_latin_bang_needs_whitespace_or_a_cjk_neighbour() -> None:
    assert count_sentences("word!word") == 1
    assert count_sentences("Caf\u00e9!Go") == 1
    assert count_sentences("Hello? World") == 2


def test_cjk_terminators_split_without_whitespace() -> None:
    assert count_sentences("\u4e2d\u6587\uff01\u4f60\u597d") == 2
    assert count_sentences("\u4e2d\u6587?\u4f60\u597d") == 2


def test_a_closing_quote_keeps_the_split() -> None:
    assert count_sentences('He said "Hi." Then left.') == 2
