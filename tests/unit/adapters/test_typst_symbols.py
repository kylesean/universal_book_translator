from __future__ import annotations

import shutil

import pytest

from ubt.adapters.pdf.typst_math_probe import TypstMathProbe
from ubt.core.math_symbols import INLINE_SYMBOLS, SPECIAL_LATEX_COMMANDS
from ubt.render.typst.math_transpiler import typstify_math

_ARG_TAKING = {"frac": r"\frac{a}{b}", "sqrt": r"\sqrt{a}"}

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
        (r"\hat x", "hat(x)"),
        (r"\tilde{y}", "tilde(y)"),
        # Typst's `vec` builds a column vector, not an arrow accent: `vec(v)`
        # rendered as a 1x1 bracketed "(v)". The accent has to be spelled out.
        (r"\vec{v}", "accent(v, arrow)"),
        (r"\bar{y}", "overline(y)"),
        (r"\overline{AB}", 'overline(#"AB")'),
        (r"\dot{x} + \ddot{y}", "dot(x) + diaer(y)"),
        (r"\vec{v} \cdot \hat{n}", "accent(v, arrow) dot hat(n)"),
    ],
)
@pytest.mark.fast
def test_latex_accents_become_typst_calls(body: str, expected: str) -> None:
    assert typstify_math(body) == expected


@pytest.mark.fast
@pytest.mark.parametrize("body", [r"\acute{x}", r"\breve{x}"])
def test_accents_with_no_typst_spelling_fail_closed(body: str) -> None:
    assert typstify_math(body) is None


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("typst") is None,
    reason="the typst compiler is not installed; the probe cannot run",
)
def test_every_inline_symbol_table_command_compiles() -> None:
    probe = TypstMathProbe()
    converted = {
        command: typstify_math(_ARG_TAKING.get(command, "\\" + command))
        for command in sorted(INLINE_SYMBOLS)
    }
    probe.check_many(body for body in converted.values() if body is not None)
    failures = [
        (command, body)
        for command, body in converted.items()
        if body is None or not probe.check(body)
    ]
    assert failures == []


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("typst") is None,
    reason="the typst compiler is not installed; the probe cannot run",
)
def test_every_special_latex_command_renders() -> None:
    probe = TypstMathProbe()
    assert set(_SPECIAL_BODIES) == set(SPECIAL_LATEX_COMMANDS)
    converted = {
        command: typstify_math(_SPECIAL_BODIES[command])
        for command in sorted(SPECIAL_LATEX_COMMANDS)
    }
    probe.check_many(body for body in converted.values() if body is not None)
    failures = [
        (command, body)
        for command, body in converted.items()
        if body is None or not probe.check(body)
    ]
    assert failures == []


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("typst") is None,
    reason="the typst compiler is not installed; the probe cannot run",
)
def test_check_many_matches_per_body_check() -> None:
    bodies = ["x", "cal(L)", "frak(g)", "bogus_undefined_symbol", "x +", "⟦ x ⟧"]
    batched = TypstMathProbe()
    batched.check_many(bodies)
    individual = TypstMathProbe()
    assert {body: batched.check(body) for body in bodies} == {
        body: individual.check(body) for body in bodies
    }
