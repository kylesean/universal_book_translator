"""LaTeX math accents must reach Typst as compilable function calls.

Translators re-encode a flattened accent (``x̂``) as the conventional LaTeX
form (``\\hat{x}``); ``typstify_math`` is the rigid route's converter, and an
accent it cannot map fails the whole math block closed to literal text. The
expected strings below use the spellings Typst accepts where the LaTeX name is
not one: ``\\bar`` becomes ``overline`` and ``\\ddot`` becomes ``diaer``.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.overlay_text import typstify_math

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (r"\hat{x}", "hat(x)"),
        (r"\hat x", "hat(x)"),  # bare-command form
        (r"\tilde{y}", "tilde(y)"),
        (r"\vec{v}", "vec(v)"),
        (r"\bar{y}", "overline(y)"),  # Typst has no "bar" accent
        (r"\overline{AB}", 'overline(#"AB")'),
        (r"\dot{x} + \ddot{y}", "dot(x) + diaer(y)"),  # Typst has no "ddot"
        (r"\vec{v} \cdot \hat{n}", "vec(v) dot hat(n)"),
    ],
)
def test_latex_accents_become_typst_calls(body: str, expected: str) -> None:
    assert typstify_math(body) == expected


@pytest.mark.parametrize("body", [r"\acute{x}", r"\breve{x}"])
def test_accents_with_no_typst_spelling_fail_closed(body: str) -> None:
    # None sends the block back to escaped literal text; a guessed spelling
    # would emit a Typst document that does not compile.
    assert typstify_math(body) is None
