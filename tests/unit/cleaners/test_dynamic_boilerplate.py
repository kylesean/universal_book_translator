"""Dynamic boilerplate cleaning: strip chrome, never strip content.

This cleaner runs per *block* and writes its result back into the delivered
file, so a false positive deletes a reader's text. The pattern comments record
the regressions that shaped the current rules:

- an unanchored page-marker rule ate the ``3`` from ``3. Preheat the oven…``,
  the year from ``1812 was…``, and the numeral from ``IV. On the Origin of
  Species``;
- an ``IGNORECASE`` roman-numeral class matched ``Mild``/``Civil``/``Dim``/``Mix``
  and deleted the first word of prose;
- a bare *prefix* match on a harvested disclaimer deleted every body sentence
  that merely opened with the same words.

These tests pin both directions: the chrome that must go, and the content that
must survive. The consensus harvester is pure (no IO), so it is exercised here
too.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.dynamic_boilerplate import (
    BoilerplateFingerprint,
    DynamicBoilerplateHarvester,
)

pytestmark = pytest.mark.fast

_DISCLAIMER = (
    "This book is a work of fiction. Names, characters, places and incidents "
    "are products of the author's imagination."
)


# --------------------------------------------------------------------------- #
# clean_head: page markers and running headers.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected_clean", "expected_stripped"),
    [
        ("Page 42\nThe real text", "The real text", "Page 42"),
        ("42\nThe real text", "The real text", "42"),
        ("iv\nBody starts here", "Body starts here", "iv"),
    ],
)
def test_leading_page_markers_are_stripped(
    text: str, expected_clean: str, expected_stripped: str
) -> None:
    assert BoilerplateFingerprint().clean_head(text) == (expected_clean, expected_stripped)


@pytest.mark.parametrize(
    "text",
    [
        "1812 was a year of change",  # a 4-digit year is content
        "2024\nChapter body",  # a dated heading is content
        "IV. On the Origin of Species",  # roman followed by '.' is a heading
    ],
)
def test_content_that_looks_like_a_marker_is_not_stripped(text: str) -> None:
    cleaned, stripped = BoilerplateFingerprint().clean_head(text)
    assert stripped == ""
    assert cleaned == text.strip()


@pytest.mark.parametrize("word", ["Mild weather today", "Civil war began", "Dim light"])
def test_words_spelled_from_roman_letters_are_not_markers(word: str) -> None:
    # The old IGNORECASE [ivxlcdm]+ class matched these and deleted the word.
    cleaned, stripped = BoilerplateFingerprint().clean_head(word)
    assert stripped == ""
    assert cleaned == word


def test_a_whole_text_that_is_only_a_number_is_not_stripped() -> None:
    # Stripping it would leave nothing; the guard keeps the block.
    assert BoilerplateFingerprint().clean_head("42") == ("42", "")


@pytest.mark.parametrize(
    "text",
    ["Some Title CHAPTER 3 42 Body text here", "42 CHAPTER 3 Some Title Body"],
)
def test_running_chapter_headers_are_stripped(text: str) -> None:
    cleaned, stripped = BoilerplateFingerprint().clean_head(text)
    assert stripped
    assert cleaned == "Body text here" or cleaned == "Some Title Body"


def test_empty_head_is_a_no_op() -> None:
    assert BoilerplateFingerprint().clean_head("") == ("", "")


# --------------------------------------------------------------------------- #
# clean_tail: photo credits and harvested disclaimers.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tail_text",
    ["Bettmann/Corbis", "Getty Images", "Smith \u00a9 2020"],
)
def test_trailing_photo_credits_are_stripped(tail_text: str) -> None:
    cleaned, stripped = BoilerplateFingerprint().clean_tail(f"Body text. {tail_text}")
    assert cleaned == "Body text."
    assert stripped == tail_text


def test_a_harvested_disclaimer_at_the_tail_is_stripped() -> None:
    fingerprint = BoilerplateFingerprint(footer_disclaimers=(_DISCLAIMER,))
    cleaned, stripped = fingerprint.clean_tail(f"Body paragraph here. {_DISCLAIMER}")
    assert cleaned == "Body paragraph here"
    assert stripped == _DISCLAIMER


def test_a_body_sentence_that_only_opens_like_the_disclaimer_is_kept() -> None:
    # The prefix anchor alone is not evidence; the tail must resemble the
    # disclaimer (>= 0.85). Otherwise the body sentence was truncated.
    fingerprint = BoilerplateFingerprint(footer_disclaimers=(_DISCLAIMER,))
    body = "Body paragraph here. This book is a work of fiction and I love it very much indeed."
    cleaned, stripped = fingerprint.clean_tail(body)
    assert cleaned == body
    assert stripped == ""


def test_a_disclaimer_shorter_than_the_minimum_is_ignored() -> None:
    fingerprint = BoilerplateFingerprint(footer_disclaimers=("short",))
    assert fingerprint.clean_tail("Body text. short") == ("Body text. short", "")


def test_a_disclaimer_whose_tail_varies_is_still_stripped() -> None:
    # A trailing page number / changed URL is why the anchor is a prefix.
    fingerprint = BoilerplateFingerprint(footer_disclaimers=(_DISCLAIMER,))
    cleaned, stripped = fingerprint.clean_tail(f"Body para. {_DISCLAIMER} Page 7")
    assert cleaned == "Body para"
    assert "Page 7" in stripped


def test_empty_tail_is_a_no_op() -> None:
    assert BoilerplateFingerprint().clean_tail("") == ("", "")


# --------------------------------------------------------------------------- #
# clean: tail first, then head.
# --------------------------------------------------------------------------- #


def test_clean_strips_both_ends() -> None:
    assert BoilerplateFingerprint().clean("Page 42\nBody text here.") == "Body text here."


# --------------------------------------------------------------------------- #
# The consensus harvester (pure).
# --------------------------------------------------------------------------- #


def test_fingerprint_defaults_have_no_patterns() -> None:
    fingerprint = BoilerplateFingerprint()
    assert fingerprint.footer_disclaimers == ()
    assert fingerprint.header_patterns == ()


def _footer_samples(count: int) -> list[str]:
    footer = "Copyright 2020 Publisher. All rights reserved worldwide."
    return [f"Page {i} content. " + "x" * 60 + " " + footer for i in range(count)]


def test_harvest_learns_a_recurring_footer() -> None:
    result = DynamicBoilerplateHarvester(min_sample_length=50, min_match_size=20).harvest(
        _footer_samples(6)
    )
    assert any("Copyright 2020 Publisher" in footer for footer in result.footer_disclaimers)


def test_harvest_needs_at_least_three_valid_samples() -> None:
    harvester = DynamicBoilerplateHarvester(min_sample_length=50)
    assert harvester.harvest([]).footer_disclaimers == ()
    assert harvester.harvest(["a" * 100, "b" * 100]).footer_disclaimers == ()


def test_harvest_does_not_learn_headers() -> None:
    # Header harvesting is not implemented; the field is reserved and stays empty.
    result = DynamicBoilerplateHarvester(min_sample_length=50, min_match_size=20).harvest(
        _footer_samples(6)
    )
    assert result.header_patterns == ()
