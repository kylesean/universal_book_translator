"""Unicode-soup math detection and masking for delimiter-free display equations.

Docling merges display equations into narrative blocks as delimiter-free unicode
soup (``βSI=e−ψpert``, ``whereQ0``). ``MathMasker`` cannot see them (no ``$``,
no backslash), so this module detects the spans precision-first — a span must
carry real math signal, never a bare identifier — and masks them behind
``⟦SOUP_MASK_…⟧`` tokens reusing the checksum-verified restore machinery.

The contracts pinned here:

* runs are maximal stretches of math-adjacent characters; CJK and whitespace
  break them, so a Latin equation embedded in Chinese prose is still one run;
* a run qualifies only with genuine signal: ≥2 math hits always qualify, a
  lone hit needs a subscript/superscript/paren context or a leading Greek letter
  or a digit, and bare identifiers (``whereQ0``, ``5CfinVtm``, ``x2``) never do;
* physical parameter assignments are matched as their own span, with a
  trailing guard so ``area = 5 square meters`` is not split into ``area = 5 s``;
* already-masked ``⟦…⟧`` token interiors are subtracted out (no nested tokens);
* ``SoupMathMasker.mask`` emits index-ordered, checksum-bound tokens whose
  ``unmask`` is the identity and whose faithful echo restores ``clean``.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.soup_math import (
    SoupMathMasker,
    _qualifies,
    _run_spans,
    find_soup_spans,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# _run_spans
# --------------------------------------------------------------------------- #


def test_run_spans_empty_text() -> None:
    assert _run_spans("") == []


def test_run_spans_splits_on_whitespace() -> None:
    assert _run_spans("abc def") == [(0, 3), (4, 7)]


def test_run_spans_keeps_operators_inside_a_run() -> None:
    assert _run_spans("x+y") == [(0, 3)]


def test_run_spans_breaks_on_cjk() -> None:
    assert _run_spans("\u4e2d\u6587abc\u4e2d\u6587") == [(2, 5)]


def test_run_spans_single_chars_are_their_own_runs() -> None:
    assert _run_spans("a b") == [(0, 1), (2, 3)]


def test_run_spans_include_a_trailing_run() -> None:
    assert _run_spans("a+b") == [(0, 3)]


# --------------------------------------------------------------------------- #
# _qualifies (precision-first)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        # Real math signal.
        ("\u03b2SI=e\u2212\u03c8pert", True),  # two greek + one op
        ("e\u2212\u03c8", True),  # greek + op
        ("\u03b1\u03b2", True),  # two greek
        ("\u03b2x", True),  # leading greek, single hit
        ("x_12", True),  # underscore + digit + length 4
        ("x^22", True),  # caret + digit + length 4
        ("m\u2080", True),  # subscript counts as a digit-bearing hit
        ("x\u00b2", True),  # superscript two
        ("\u2264x2", True),  # single op + digit
        ("(\u03b2", True),  # single greek hit + paren
        ("\u2264_x", True),  # single op hit + underscore
        # Bare identifiers / prose: never masked.
        ("whereQ0", False),
        ("5CfinVtm", False),
        ("x2", False),
        ("abc", False),
        ("x\u03b2", False),  # greek not at the start, no digit
        ("x_1", False),  # underscore but shorter than 4
        ("F\u226aF", False),  # spaced single operator
        ("\u2264x", False),  # single op, no digit
        ("\u00b0C", False),
        ("(x", False),  # a paren alone is not math signal
        ("\u03b2", False),  # lone character
    ],
)
def test_qualifies(run: str, expected: bool) -> None:
    assert _qualifies(run) is expected


# --------------------------------------------------------------------------- #
# find_soup_spans
# --------------------------------------------------------------------------- #


def test_find_soup_spans_empty_text() -> None:
    assert find_soup_spans("") == []


def test_find_soup_spans_ignores_bare_identifiers() -> None:
    assert find_soup_spans("whereQ0 and 5CfinVtm") == []


def test_find_soup_spans_finds_a_delimiter_free_equation() -> None:
    assert find_soup_spans("The value \u03b2SI=e\u2212\u03c8pert holds") == [(10, 21)]


def test_find_soup_spans_trims_trailing_punctuation() -> None:
    assert find_soup_spans("\u03b2SI=e\u2212\u03c8pert,") == [(0, 11)]


def test_find_soup_spans_matches_a_physical_assignment() -> None:
    assert find_soup_spans("v = 3 m/s") == [(0, 9)]


def test_find_soup_spans_does_not_split_a_unit_word() -> None:
    # Without the trailing guard this matched "area = 5 s" and split "square".
    assert find_soup_spans("area = 5 square meters") == [(0, 8)]
    assert find_soup_spans("area = 5 s") == [(0, 10)]


def test_find_soup_spans_finds_multiple_spans() -> None:
    assert find_soup_spans("E=mc\u00b2 and v=3 m/s") == [(0, 5), (10, 17)]


def test_find_soup_spans_absorbs_a_contained_assignment() -> None:
    assert find_soup_spans("x=5\u03b2") == [(0, 4)]


def test_find_soup_spans_subtracts_existing_tokens() -> None:
    assert find_soup_spans("\u27e6SOUP_MASK_0001-abc\u27e7") == []


def test_find_soup_spans_keeps_text_around_a_token() -> None:
    text = "\u03b2SI=e\u2212\u03c8 \u27e6SOUP_MASK_0001-abc\u27e7 x_12"
    assert find_soup_spans(text) == [(0, 7), (29, 33)]


# --------------------------------------------------------------------------- #
# SoupMathMasker
# --------------------------------------------------------------------------- #


def test_soup_masker_empty_text() -> None:
    assert SoupMathMasker().mask("") == ("", {})


def test_soup_masker_emits_a_checksum_bound_token() -> None:
    masked, mapping = SoupMathMasker().mask("The value \u03b2SI=e\u2212\u03c8pert holds")
    assert masked == "The value \u27e6SOUP_MASK_0001-4f7\u27e7 holds"
    ((token, original),) = mapping.items()
    assert token.startswith("\u27e6SOUP_MASK_") and token.endswith("\u27e7")
    assert original == "\u03b2SI=e\u2212\u03c8pert"


def test_soup_masker_numbers_spans_in_text_order() -> None:
    masked, mapping = SoupMathMasker().mask("E=mc\u00b2 and v=3 m/s")
    assert [t[: len("\u27e6SOUP_MASK_") + 4] for t in mapping] == [
        "\u27e6SOUP_MASK_0001",
        "\u27e6SOUP_MASK_0002",
    ]
    assert "E=mc\u00b2" not in masked and "v=3 m/s" not in masked


def test_soup_masker_roundtrip_is_identity() -> None:
    text = "The value \u03b2SI=e\u2212\u03c8pert holds for all x_12."
    masker = SoupMathMasker()
    masked, mapping = masker.mask(text)
    assert masker.unmask(masked, mapping) == text


def test_soup_masker_faithful_echo_restores_clean() -> None:
    text = "The value \u03b2SI=e\u2212\u03c8pert holds."
    masker = SoupMathMasker()
    masked, mapping = masker.mask(text)
    report = masker.unmask_checked(masked, mapping)
    assert report.text == text
    assert report.clean
    assert report.missing == []
    assert report.mismatched == []
    assert report.duplicated == []
    assert report.reordered == []


def test_soup_masker_leaves_prose_untouched() -> None:
    text = "whereQ0 and 5CfinVtm are bare identifiers"
    assert SoupMathMasker().mask(text) == (text, {})
