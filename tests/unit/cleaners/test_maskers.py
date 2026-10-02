"""Masking cleaners: the protect-restore contract and its integrity verdict.

Math, code and citation maskers all follow one contract: replace a verbatim
span with a checksum-tagged token before drafting, then restore it afterwards
*and verify* the echo. The verification is what makes the roundtrip safe -- a
model that rewrites an index, drops a token, leaks the protected text, or swaps
two tokens must be caught, because the alternative is publishing corrupted
math/code/citations.

The math/code/citation families share :mod:`ubt.core.cleaners.mask_tokens`; the
roundtrip and integrity cases are parametrized over all three, and the
family-specific delimiters are tested per masker.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

import pytest

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.inline_math import inline_math_spans, is_math_content
from ubt.core.cleaners.mask_tokens import (
    UnmaskReport,
    find_reordered,
    order_by_position,
    position_of,
    token_checksum,
)
from ubt.core.cleaners.math_masker import MathMasker, extract_math_spans

pytestmark = pytest.mark.fast


class _Masker(Protocol):
    def mask(self, text: str) -> tuple[str, dict[str, str]]: ...
    def unmask(self, text: str, mapping: dict[str, str]) -> str: ...
    def unmask_checked(self, text: str, mapping: dict[str, str]) -> UnmaskReport: ...


_MASKER_CASES: list[tuple[str, Callable[[], _Masker], str]] = [
    ("math", MathMasker, "Energy is $E=mc^2$ and inline \\(x^2\\) too."),
    ("code", CodeMasker, "Use `print(x)` then:\n```py\nprint(1)\n```\ndone"),
    ("citation", CitationMasker, "See [12] and [3, 7, 21] and [5-7]."),
]


# --------------------------------------------------------------------------- #
# The shared contract: mask -> unmask is identity, and a faithful echo is clean.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "make_masker", "text"), _MASKER_CASES)
def test_mask_unmask_roundtrip_is_identity(
    name: str, make_masker: Callable[[], _Masker], text: str
) -> None:
    masker = make_masker()
    masked, mapping = masker.mask(text)
    assert mapping, f"{name} masked nothing"
    assert masker.unmask(masked, mapping) == text


@pytest.mark.parametrize(("name", "make_masker", "text"), _MASKER_CASES)
def test_faithful_echo_restores_clean(
    name: str, make_masker: Callable[[], _Masker], text: str
) -> None:
    masker = make_masker()
    masked, mapping = masker.mask(text)
    report = masker.unmask_checked(masked, mapping)
    assert report.text == text
    assert report.clean
    assert report.missing == []
    assert report.mismatched == []
    assert report.mutated == []
    assert report.reordered == []
    assert report.duplicated == []


# --------------------------------------------------------------------------- #
# Integrity verdicts: each corruption is caught by exactly one bucket.
# --------------------------------------------------------------------------- #


def test_rewritten_checksum_is_mismatched_and_never_restores_the_wrong_span() -> None:
    masker = MathMasker()
    masked, mapping = masker.mask("Value $a+b$ here")
    token = next(iter(mapping))
    tampered = re.sub(r"-[0-9a-z]{3}⟧$", "-zzz⟧", token)
    report = masker.unmask_checked(masked.replace(token, tampered), mapping)
    assert report.mismatched == [1]
    assert not report.clean
    assert "$a+b$" not in report.text  # the wrong span must not be published
    assert report.mutated  # the token survives as residue


def test_deleted_token_is_missing() -> None:
    masker = MathMasker()
    masked, mapping = masker.mask("A $x$ B $y$ C")
    first = list(mapping)[0]
    report = masker.unmask_checked(masked.replace(first, ""), mapping)
    assert report.missing == [1]
    assert not report.clean


def test_leaked_protected_text_is_duplicated() -> None:
    masker = MathMasker()
    masked, mapping = masker.mask("A $x$ B")
    original = next(iter(mapping.values()))
    report = masker.unmask_checked(masked + " leaked " + original, mapping)
    assert report.duplicated == [1]
    assert not report.clean


def test_swapped_tokens_are_reordered() -> None:
    masker = MathMasker()
    masked, mapping = masker.mask("A $x$ B $y$ C")
    first, second = list(mapping)
    swapped = masked.replace(first, "@@").replace(second, first).replace("@@", second)
    report = masker.unmask_checked(swapped, mapping)
    # Both spans restore intact, but they moved -- only the order check sees it.
    assert report.reordered == [1, 2]
    assert not report.clean
    assert report.text == "A $y$ B $x$ C"


def test_checksumless_echo_restores_for_math_but_not_for_citation() -> None:
    # A faithful model may drop the optional checksum suffix. Math restores it
    # (reported as unverified, which does not fail ``clean``); a citation refuses
    # to re-publish an unverifiable marker and records it as mutated.
    def _strip_checksum(token: str) -> str:
        return re.sub(r"-[0-9a-z]{3}⟧$", "⟧", token)

    math_masked, math_mapping = MathMasker().mask("x $y$ z")
    math_token = next(iter(math_mapping))
    math_report = MathMasker().unmask_checked(
        math_masked.replace(math_token, _strip_checksum(math_token)), math_mapping
    )
    assert math_report.text == "x $y$ z"
    assert math_report.unverified == [1]
    assert math_report.clean

    cite_masked, cite_mapping = CitationMasker().mask("See [12] here")
    cite_token = next(iter(cite_mapping))
    cite_stripped = cite_masked.replace(cite_token, _strip_checksum(cite_token))
    cite_report = CitationMasker().unmask_checked(cite_stripped, cite_mapping)
    assert cite_report.text == cite_stripped  # not restored
    assert cite_report.mutated
    assert not cite_report.clean


# --------------------------------------------------------------------------- #
# Family-specific delimiters.
# --------------------------------------------------------------------------- #


def test_math_masks_every_delimiter_form() -> None:
    text = r"d $$a+b$$ i $c$ p \(d\) b \[e\] env \begin{align}f\end{align}"
    masked, mapping = MathMasker().mask(text)
    assert len(mapping) == 5
    for construct in ("$$a+b$$", "$c$", r"\(d\)", r"\[e\]", r"\begin{align}"):
        assert construct not in masked
    assert MathMasker().unmask(masked, mapping) == text


def test_math_currency_is_not_masked() -> None:
    _, mapping = MathMasker().mask("It costs $5 and $3.50 today.")
    assert mapping == {}


def test_math_span_helpers() -> None:
    # ``count_math_spans`` is intentionally not tested: the review report flags it
    # as dead code (Q5) pending a delete-or-wire decision.
    assert extract_math_spans("a $x$ b $y$") == ["$x$", "$y$"]


def test_math_mapping_is_ordered_by_textual_position() -> None:
    # Delimiters are masked display-first, so index order != textual order; the
    # returned mapping must carry the order the model actually saw.
    masked, mapping = MathMasker().mask(r"\(first\) then $second$")
    positions = [masked.index(token) for token in mapping]
    assert positions == sorted(positions)


def test_code_masks_fenced_and_inline() -> None:
    text = "Use `print(x)` then:\n```py\nprint(1)\n```\ndone"
    masked, mapping = CodeMasker().mask(text)
    assert len(mapping) == 2
    assert "`print(x)`" not in masked
    assert "```py" not in masked
    assert CodeMasker().unmask(masked, mapping) == text


def test_citation_masks_only_numeric_markers() -> None:
    text = "See [12], [3, 7, 21], [5-7], but (Smith et al., 2021) and [see note] stay."
    masked, mapping = CitationMasker().mask(text)
    assert len(mapping) == 3
    assert "(Smith et al., 2021)" in masked
    assert "[see note]" in masked
    assert CitationMasker().unmask(masked, mapping) == text


def test_inline_math_guard_rejects_bare_numbers() -> None:
    assert not is_math_content("5")
    assert not is_math_content("3.50")
    assert is_math_content("x^2")
    assert inline_math_spans("a $x$ b") == [(2, 5)]


# --------------------------------------------------------------------------- #
# Shared helpers.
# --------------------------------------------------------------------------- #


def test_token_checksum_is_deterministic_and_input_bound() -> None:
    assert token_checksum(1, "a") == token_checksum(1, "a")
    assert token_checksum(1, "a") != token_checksum(2, "a")
    assert token_checksum(1, "a") != token_checksum(1, "b")


def test_position_of_falls_back_to_one_past_the_end() -> None:
    assert position_of("A", "xxAyy") == 2
    assert position_of("ZZZ", "abc") == 3


def test_order_by_position_sorts_by_textual_occurrence() -> None:
    mapping = {"B": "b", "A": "a"}
    assert list(order_by_position("...A...B", mapping)) == ["A", "B"]


def test_find_reordered_reports_both_sides_of_an_inversion() -> None:
    assert find_reordered([1, 2], [2, 1]) == [1, 2]
    assert find_reordered([1, 2], [1, 2]) == []
    assert find_reordered([1, 2], [9, 1, 2]) == []  # unknown index skipped


def test_unmask_report_clean_ignores_unverified_but_not_reordered_or_duplicated() -> None:
    assert UnmaskReport(text="x", unverified=[1]).clean
    assert not UnmaskReport(text="x", reordered=[1]).clean
    assert not UnmaskReport(text="x", duplicated=[1]).clean
    assert not UnmaskReport(text="x", missing=[1]).clean
    assert not UnmaskReport(text="x", mismatched=[1]).clean
    assert not UnmaskReport(text="x", mutated=["1"]).clean
