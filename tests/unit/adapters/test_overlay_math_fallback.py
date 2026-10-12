"""Unit tests for overlay math fallback and balanced frac/sqrt compilation."""

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
    # cannot become math, and must not become LaTeX source either.
    rendered = render_prose(r"公式：$\unknowncmd{x} + \phi$。")
    # ``\{`` is Typst's escape for a literal brace, so the visible text is
    # "unknowncmd{x}" — the command name without its backslash, which is the
    # point: no LaTeX syntax survives, only the word.
    assert "\\unknowncmd" not in rendered
    assert r"unknowncmd\{x\}" in rendered
    assert "ϕ" in rendered
