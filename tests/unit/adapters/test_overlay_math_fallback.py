"""Unit tests for overlay math fallback and balanced frac/sqrt compilation."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.overlay_text import (
    clean_math_fallback_text,
    render_overlay_line,
    typstify_math,
)

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


def test_render_overlay_line_uncompilable_math_does_not_leak_latex() -> None:
    # When math_probe returns False (or None), render_overlay_line must use clean_math_fallback_text
    # and NEVER leak raw LaTeX commands like \frac or \phi into the rendered text.
    line = r"公式为：$\frac{1}{2} \phi s_2 \gamma s_1 \phi d$。"
    rendered = render_overlay_line(line, math_probe=lambda _: False)
    assert r"\frac" not in rendered
    assert r"\phi" not in rendered
    assert r"\gamma" not in rendered
    assert "1/2" in rendered
    assert "ϕ" in rendered
    assert "γ" in rendered
