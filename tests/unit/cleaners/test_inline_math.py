"""Currency-aware inline ``$...$`` math detection.

A naive ``\\$[^\\$\\n]+\\$`` pairs the first ``$`` with the next one on the line,
so ``"$5 and $10"`` becomes a "math" span and every term inside it is shielded
from glossary enforcement and CJK spacing. Three callers used to carry their own
copy of the rule and drifted; this module is the single home, so its two-part
contract (delimiters must not touch whitespace; the interior must not be
currency/prose) is what keeps all of them consistent.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.inline_math import (
    inline_math_spans,
    is_math_content,
    iter_inline_math,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# is_math_content
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("content", ["", "   "])
def test_blank_content_is_not_math(content: str) -> None:
    assert is_math_content(content) is False


@pytest.mark.parametrize("content", ["5", "3.50", "1,000", "42"])
def test_a_plain_number_is_currency_not_math(content: str) -> None:
    assert is_math_content(content) is False


@pytest.mark.parametrize("content", ["10-20", "10-$20", "5\u201310", "10\u201420"])
def test_a_currency_range_is_not_math(content: str) -> None:
    assert is_math_content(content) is False


def test_bare_cjk_interior_is_not_math() -> None:
    assert is_math_content("\u4e2d\u6587") is False


def test_cjk_inside_a_latex_text_command_is_math() -> None:
    assert is_math_content("\\text{\u4e2d\u6587}") is True


@pytest.mark.parametrize("content", ["x", "abc", "a+b", "x^2", "\\alpha", "{x}", "(x)"])
def test_ordinary_math_interiors_are_math(content: str) -> None:
    assert is_math_content(content) is True


@pytest.mark.parametrize("content", [")prose", "prose("])
def test_bracket_glued_prose_is_not_math(content: str) -> None:
    # A stray dollar pairs a table header's prose row into a fake span.
    assert is_math_content(content) is False


def test_bracket_glued_content_with_a_latex_signal_is_still_math() -> None:
    # A shattered formula can end with '(' but carries \commands.
    assert is_math_content("\\psi = V \\ln(") is True


# --------------------------------------------------------------------------- #
# inline_math_spans / iter_inline_math
# --------------------------------------------------------------------------- #


def test_a_simple_span_is_found() -> None:
    assert inline_math_spans("a $x$ b") == [(2, 5)]


def test_a_latex_command_span_is_found() -> None:
    assert inline_math_spans("$\\alpha$") == [(0, 8)]


def test_a_currency_pair_is_not_a_span() -> None:
    # The second '$' is preceded by a space, so this is not a math pair.
    assert inline_math_spans("price $5 and $10 total") == []


@pytest.mark.parametrize("text", ["$5$", "$3.50$"])
def test_a_pure_currency_interior_is_not_a_span(text: str) -> None:
    assert inline_math_spans(text) == []


def test_a_bare_cjk_interior_is_not_a_span() -> None:
    assert inline_math_spans("$\u4e2d\u6587$") == []


def test_cjk_inside_text_is_a_span() -> None:
    assert inline_math_spans("$\\text{\u4e2d\u6587}$") == [(0, 11)]


@pytest.mark.parametrize("text", ["$ x $", "$x $", "$ x$"])
def test_whitespace_touching_delimiters_are_not_a_span(text: str) -> None:
    assert inline_math_spans(text) == []


def test_multiple_spans_are_all_returned() -> None:
    assert inline_math_spans("$a$ and $b$") == [(0, 3), (8, 11)]


def test_a_rejected_candidate_does_not_hide_a_later_span() -> None:
    # The currency '$5$' is rejected; scanning resumes and finds '$x$'.
    assert inline_math_spans("$5$ and $x$") == [(8, 11)]


def test_no_dollars_yields_no_spans() -> None:
    assert inline_math_spans("no math here") == []


def test_iter_inline_math_yields_the_same_intervals_as_spans() -> None:
    text = "$a$ and $b$"
    assert [(m.start(), m.end()) for m in iter_inline_math(text)] == inline_math_spans(text)
