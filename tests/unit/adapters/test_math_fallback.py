"""Math fallback: balanced frac/sqrt must compile, degrades must stay readable."""

from __future__ import annotations

import pytest

from ubt.render.typst.document_renderer import render_prose
from ubt.render.typst.math_transpiler import clean_math_fallback_text, typstify_math

pytestmark = pytest.mark.fast


def test_typstify_math_balanced_fractions_and_nested_subscripts() -> None:
    latex = r"\text{面积} = \sum_{i=1}^{n} \frac{(x_{i+1} + x_i)}{2} (y_{i+1} - y_i)"
    typst = typstify_math(latex)
    assert typst is not None
    assert '#"面积"' in typst
    assert "frac((x_(i+1) + x_i), 2)" in typst
    assert r"\frac" not in typst


def test_typstify_math_cjk_in_text() -> None:
    latex = r"\text{总和} + \mathrm{差值}"
    typst = typstify_math(latex)
    assert typst is not None
    assert '#"总和"' in typst
    assert '#"差值"' in typst


def test_clean_math_fallback_text_strips_latex_macros() -> None:
    broken_latex = r"$\frac{1}{2} \phi s_2 \gamma s_1 \phi d$"
    cleaned = clean_math_fallback_text(broken_latex)
    assert "\\" not in cleaned
    assert "frac" not in cleaned
    assert "1/2" in cleaned
    assert "ϕ" in cleaned
    assert "γ" in cleaned


def test_clean_math_fallback_text_display_math() -> None:
    display_latex = r"$$\text{面积} = \sum_{i=1}^{n} \frac{(x_{i+1} + x_i)}{2}$$"
    cleaned = clean_math_fallback_text(display_latex)
    assert "\\" not in cleaned
    assert "面积" in cleaned
    assert "∑" in cleaned
    assert "(x_{i+1} + x_i)/2" in cleaned


def test_uncompilable_math_does_not_leak_latex_into_the_page() -> None:
    # A math span the transpiler cannot convert must degrade to readable Unicode
    # rather than leak raw LaTeX command names (\frac, \phi) into the rendered
    # text. ``render_prose`` is the single place that decides this now.
    line = r"公式为：$\frac{1}{2} \phi s_2 \gamma s_1 \phi d$。"
    rendered = render_prose(line)
    # This body *is* convertible, so it becomes real Typst math (better than a
    # Unicode approximation). What matters is that no LaTeX command name
    # survives into the page.
    assert r"\frac" not in rendered
    assert r"\phi" not in rendered
    assert r"\gamma" not in rendered
    assert "$frac(1, 2) phi s_2 gamma s_1 phi d$" in rendered


def test_an_unconvertible_formula_degrades_to_unicode_not_latex() -> None:
    # The case the fallback exists for: a command the transpiler does not know
    # cannot become math, and must not become LaTeX source or a command name
    # either — it degrades to the neutral ``⟨math⟩`` placeholder.
    rendered = render_prose(r"公式：$\unknowncmd{x} + \phi$。")
    assert "\\unknowncmd" not in rendered
    assert "unknowncmd" not in rendered, "the command name itself must not ship"
    assert "⟨math⟩" in rendered
    assert "ϕ" in rendered


def test_fallback_never_leaks_a_command_name_for_structure_commands() -> None:
    r"""The 2026-10-14 review measured 169 of 241 table commands leaking their
    names on the fallback path (``begin{cases}``, ``underline{x}``,
    ``tfrac{1}{2}``, ``otimes``...). The fallback must degrade every one of
    them to readable text: structure shells unwrap to their content, known
    symbols map to Unicode, and anything unmapped becomes the neutral
    ``⟨math⟩`` placeholder — never the command name.
    """
    cases = clean_math_fallback_text(r"\begin{cases} x & y \\ z & w \end{cases}")
    assert "begin" not in cases and "end" not in cases
    assert cases == "x, y ; z, w"

    underline = clean_math_fallback_text(r"\underline{x} + \tfrac{1}{2}")
    assert "underline" not in underline and "tfrac" not in underline
    assert underline == "x + (1)/(2)"

    assert clean_math_fallback_text(r"\otimes \ast \cdots") == "⊗ ∗ ⋯"

    sized = clean_math_fallback_text(r"a \big( x \big)")
    assert "big" not in sized
    assert sized == "a ( x )"

    unknown = clean_math_fallback_text(r"\unknowncmd{x}")
    assert "unknowncmd" not in unknown
    assert "⟨math⟩" in unknown
