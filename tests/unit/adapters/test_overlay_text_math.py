"""LaTeX math accents must reach Typst as compilable function calls.

Translators re-encode a flattened accent (``x̂``) as the conventional LaTeX
form (``\\hat{x}``); ``typstify_math`` is the rigid route's converter, and an
accent it cannot map fails the whole math block closed to literal text. The
expected strings below use the spellings Typst accepts where the LaTeX name is
not one: ``\\bar`` becomes ``overline`` and ``\\ddot`` becomes ``diaer``.
"""

from __future__ import annotations

import shutil

import pytest

from ubt.adapters.pdf.overlay_text import typstify_math
from ubt.adapters.pdf.typst_math_probe import TypstMathProbe
from ubt.adapters.pdf.typst_symbols import INLINE_SYMBOLS, SPECIAL_LATEX_COMMANDS

pytestmark = pytest.mark.fast

#: Symbol-table commands that take an argument and cannot be probed bare.
_ARG_TAKING = {"frac": r"\frac{a}{b}", "sqrt": r"\sqrt{a}"}

#: A compiling body for every command handled by a dedicated pre-pass.
_SPECIAL_BODIES = {
    "mathbb": r"\mathbb{R}",
    "mathfrak": r"\mathfrak{g}",
    "mathcal": r"\mathcal{L}",
    "text": r"\text{if}",
    "mathrm": r"\mathrm{d}",
    "hat": r"\hat{x}",
    "tilde": r"\tilde{x}",
    "vec": r"\vec{v}",
    "bar": r"\bar{x}",
    "overline": r"\overline{AB}",
    "dot": r"\dot{x}",
    "ddot": r"\ddot{x}",
    "quad": r"a \quad b",
    "qquad": r"a \qquad b",
    "enspace": r"a \enspace b",
    "thinspace": r"a \thinspace b",
    "thickspace": r"a \thickspace b",
    "medspace": r"a \medspace b",
    "negthinspace": r"a \negthinspace b",
    "big": r"\big( x \big)",
    "Big": r"\Big( x \Big)",
    "bigg": r"\bigg( x \bigg)",
    "Bigg": r"\Bigg( x \Bigg)",
    "left": r"\left( x \right)",
    "right": r"\left( x \right)",
}


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


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("typst") is None,
    reason="the typst compiler is not installed; the probe cannot run",
)
def test_every_inline_symbol_table_command_compiles() -> None:
    # The QE whitelist now derives from INLINE_SYMBOLS, so a symbol-table typo
    # would silently widen the gate to a command the renderer cannot compile.
    # Pin the table itself to reality: every mapped command converts and the
    # probe accepts the spelling.
    probe = TypstMathProbe()
    failures = []
    for command in sorted(INLINE_SYMBOLS):
        converted = typstify_math(_ARG_TAKING.get(command, "\\" + command))
        if converted is None or not probe.check(converted):
            failures.append((command, converted))
    assert failures == []


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("typst") is None,
    reason="the typst compiler is not installed; the probe cannot run",
)
def test_every_special_latex_command_renders() -> None:
    # SPECIAL_LATEX_COMMANDS feeds the QE whitelist too, but its members are
    # handled by regex pre-passes rather than the symbol table. Pin each to a
    # compiling body so a drift cannot widen the gate to a command the
    # converter no longer strips.
    probe = TypstMathProbe()
    assert set(_SPECIAL_BODIES) == set(SPECIAL_LATEX_COMMANDS)
    failures = []
    for command in sorted(SPECIAL_LATEX_COMMANDS):
        converted = typstify_math(_SPECIAL_BODIES[command])
        if converted is None or not probe.check(converted):
            failures.append((command, converted))
    assert failures == []


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("typst") is None,
    reason="the typst compiler is not installed; the probe cannot run",
)
def test_check_many_matches_per_body_check() -> None:
    # The batched probe lays every body out as a page of one document and
    # bisects on failure; a good body must never be poisoned by a bad one.
    bodies = ["x", "cal(L)", "frak(g)", "bogus_undefined_symbol", "x +", "⟦ x ⟧"]
    batched = TypstMathProbe()
    batched.check_many(bodies)
    individual = TypstMathProbe()
    assert {body: batched.check(body) for body in bodies} == {
        body: individual.check(body) for body in bodies
    }
