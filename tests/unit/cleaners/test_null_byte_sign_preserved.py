"""A NUL where a sign stood is a sign, not debris.

A publisher CMap that cannot map a glyph emits U+0000. The glyphs it fails on
are the sign-bearing ones, so deleting every NUL next to a number silently
changes the value. On the reference chapter that shipped:

    "1.58 - 10 6 %"        for "1.58 × 10⁻⁶ %"   (a million times too large)
    "3.31 - 10 5 %"        for "3.31 × 10⁻⁵ %"
    "gate voltage (0.2 …)" for "(−0.2 …)"         (a negative bound, positive)
    "100°C to 100°C"       for "(−100°C to 100°C)" (a range collapsed to a point)

Nothing downstream could recover it: the numeric validator compares against the
tokens the *source* block carries, and by the time it runs the source no longer
carries a sign.

Pinned here, together with the debris cases that must stay plain deletions — a
NUL between two words is not a sign, and inventing one would corrupt prose.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.lnds_pruner import normalize_academic_pdf_math

pytestmark = pytest.mark.fast

_NUL = "\x00"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A multiplication sign and a negative exponent, as one expression.
        (f"the RMS error is 1.58 {_NUL} 10 {_NUL} 6 % with a peak", "1.58 × 10^-6 %"),
        (f"peak error of 3.31 {_NUL} 10 {_NUL} 5 %.", "3.31 × 10^-5 %"),
        (f"values as low as 1 {_NUL} 10 {_NUL} 15 .", "1 × 10^-15"),
        # The flattened multiplication arrived as a hyphen, the exponent's sign
        # as a NUL: both must be read from one numeric expression.
        (f"error is 1.58 - 10 {_NUL} 6 %", "1.58 × 10^-6 %"),
        (f"peak error of 3.31 - 10 {_NUL} 5 %.", "3.31 × 10^-5 %"),
        # A leading minus on a range endpoint.
        (f"gate voltage ( {_NUL} 0.2 to 1.5 V)", "gate voltage (-0.2 to 1.5 V)"),
        (f"temperature ( {_NUL} 100 ° C to 100 ° C)", "(-100 ° C"),
    ],
)
def test_a_sign_bearing_nul_is_restored(text: str, expected: str) -> None:
    assert expected in normalize_academic_pdf_math(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Debris between words: dropped, and no sign invented.
        ("word" + _NUL + "word", "wordword"),
        ("trailing debris " + _NUL, "trailing debris "),
        (_NUL + "leading", "leading"),
        # A NUL before a word is not a minus on a number.
        (f"see {_NUL} Fig. 3", "see  Fig. 3"),
        # A genuine range is not a flattened scientific notation.
        ("thickness (0.5 - 5 nm)", "thickness (0.5 - 5 nm)"),
        ("width 1 - 10 nm", "width 1 - 10 nm"),
    ],
)
def test_plain_debris_is_still_just_dropped(text: str, expected: str) -> None:
    got = normalize_academic_pdf_math(text)
    assert got == expected
    assert _NUL not in got


def test_a_span_sentinel_survives_the_null_handling() -> None:
    """``\\x00SPAN0\\x00`` is live markup downstream, never debris."""
    sentinel = f"{_NUL}SPAN0{_NUL}"
    got = normalize_academic_pdf_math(f"alpha {sentinel} beta")
    assert sentinel in got


def test_the_rule_is_idempotent() -> None:
    """A second pass must not turn a repaired sign into something else."""
    once = normalize_academic_pdf_math(f"1.58 {_NUL} 10 {_NUL} 6 %")
    assert normalize_academic_pdf_math(once) == once
