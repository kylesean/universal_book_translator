from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from ubt.render.typst.math_transpiler import typstify_math, typstify_math_span

pytestmark = pytest.mark.fast

SWITCH_TRANSFORMER_FORMULAS = [
    r"p _ { i } ( x ) = \frac { e ^ { h ( x ) _ { i } } } { \sum _ { j } ^ { N } e ^ { h ( x ) _ { j } } } .",
    r"y = \sum _ { i \in \mathcal { T } } p _ { i } ( x ) E _ { i } ( x ) .",
    r"\exp { \, \cap c a p a c i t y } = \left ( \frac { \, \text {tokens per batch} } { \, \text {number of experts} } \right ) \times \, \text {capacity factor} .",
    r"\text {loss} = \alpha \cdot N \cdot \sum _ { i = 1 } ^ { N } f _ { i } \cdot P _ { i }",
    r"f _ { i } = \frac { 1 } { T } \sum _ { x \in \mathcal { B } } \mathbb { 1 } \{ \arg \max p ( x ) = i \} \\ \text {on of the router probability allocated for expert } i , \, 2",
    r"P _ { i } = \frac { 1 } { T } \sum _ { x \in \mathcal { B } } p _ { i } ( x ) .",
    r"\ e i n s u m ( [ n , B / n , d _ { m o d e l } ] , [ n , B / n , E , C ] , \text {dimension} = [ B / n ] )",
]


@pytest.mark.parametrize("latex", SWITCH_TRANSFORMER_FORMULAS)
def test_every_reference_formula_converts(latex: str) -> None:
    result = typstify_math(latex)
    assert result is not None, f"transpiler returned None for {latex[:60]!r}"


@pytest.mark.parametrize("latex", SWITCH_TRANSFORMER_FORMULAS)
def test_no_latex_survives_the_conversion(latex: str) -> None:
    result = typstify_math(latex)
    assert result is not None
    assert not re.search(r"\\[A-Za-z]", result), f"LaTeX command survived: {result!r}"


def test_docling_spaced_commands_are_normalized() -> None:
    assert typstify_math(r"\frac { a } { b }") == "frac(a, b)"
    assert typstify_math(r"\mathbb { R }") == "bb(R)"
    assert typstify_math(r"\mathcal { T }") == "cal(T)"


def test_font_command_does_not_fuse_with_a_preceding_command() -> None:
    assert typstify_math(r"i\in\mathcal{T}") == "i in cal(T)"
    assert typstify_math(r"\sum_{i\in\mathcal{T}} p_i(x) E_i(x)") is not None
    assert typstify_math(r"x\mathcal{Y}") == "x cal(Y)"
    assert typstify_math(r"i \in \mathcal{T}") == "i in cal(T)"


def test_accent_takes_the_whole_operand_not_its_first_letter() -> None:
    assert typstify_math(r"\hat\mathcal{T}") == "hat(cal(T))"
    assert typstify_math(r"\vec\mathcal{X}") == "vec(cal(X))"
    assert typstify_math(r"\bar\mathbb{R}") == "overline(bb(R))"
    assert typstify_math(r"\hat{\mathcal{T}}") == "hat(cal(T))"
    assert typstify_math(r"\hat x") == "hat(x)"


def test_accent_over_a_command_keeps_the_command() -> None:
    assert typstify_math(r"\hat\alpha") == "hat(alpha)"
    assert typstify_math(r"\hat\alpha\beta") == "hat(alpha)beta"
    assert typstify_math(r"\hat\max") == "hat(max)"


def test_a_single_unknown_operator_does_not_kill_the_formula() -> None:
    result = typstify_math(r"\arg \max p ( x )")
    assert result is not None
    assert "arg" in result
    assert "\\arg" not in result


def test_operatorname_keeps_its_argument_as_a_name() -> None:
    assert typstify_math(r"\operatorname{rank} ( A )") == '#"rank" ( A )'


def test_spacing_hints_are_dropped() -> None:
    with_limits = typstify_math(r"\sum \limits _ { i }")
    with_displaystyle = typstify_math(r"\displaystyle x")
    assert with_limits is not None and "limits" not in with_limits
    assert with_displaystyle is not None and "displaystyle" not in with_displaystyle


def test_mathrm_still_produces_an_upright_label() -> None:
    assert typstify_math(r"\mathrm{差值}") == '#"差值"'
    assert typstify_math(r"\text{总和} + \mathrm{差值}") == '#"总和" + #"差值"'


def test_text_with_spaces_becomes_one_string() -> None:
    assert typstify_math(r"\text{tokens per batch}") == '#"tokens per batch"'


def test_a_multiword_text_in_a_subscript_stays_one_string() -> None:
    result = typstify_math(r"I_{\text{off}}")
    assert result == 'I_#"off"'
    assert result.count('"') == 2


def test_span_form_unwraps_the_delimiters() -> None:
    assert typstify_math_span("$x^2$") == "x^2"
    assert typstify_math_span(r"\(x^2\)") == "x^2"
    assert typstify_math_span(r"\[x^2\]") == "x^2"
    assert typstify_math_span("$$x^2$$") == "x^2"


def test_a_bare_number_is_not_math() -> None:
    assert typstify_math_span("$5$") is None


@pytest.mark.parametrize("latex", SWITCH_TRANSFORMER_FORMULAS)
def test_the_output_actually_compiles(latex: str, tmp_path: Path) -> None:
    from ubt.adapters.pdf.typst_compile import resolve_typst_binary

    binary = resolve_typst_binary(os.environ.get("UBT_TYPST_BINARY", "typst"))
    if binary is None:
        pytest.skip("typst compiler not installed")

    body = typstify_math(latex)
    assert body is not None
    source = tmp_path / "f.typ"
    source.write_text(f"$ {body} $\n", encoding="utf-8")
    proc = subprocess.run(
        [binary, "compile", str(source), str(tmp_path / "f.pdf")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"typst rejected the formula:\n{proc.stderr}"


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\beta_t\sqrt{x}", "beta_t sqrt(x)"),
        (r"a_b\frac{1}{2}", "a_b frac(1, 2)"),
        (r"x_i\sqrt{y}", "x_i sqrt(y)"),
        (r"\alpha_i\frac{a}{b}", "alpha_i frac(a, b)"),
    ],
)
def test_a_bare_subscript_does_not_swallow_the_next_function_name(
    latex: str, expected: str
) -> None:
    assert typstify_math(latex) == expected


def test_a_spaced_command_still_converts_identically() -> None:
    assert typstify_math(r"\beta_t \sqrt{x}") == "beta_t sqrt(x)"
    assert typstify_math(r"\frac{a}{b}") == "frac(a, b)"


def test_font_style_nested_braces() -> None:
    assert typstify_math(r"\mathbf{x_{i}}") == "bold(x_(i))"
    assert typstify_math(r"\mathcal{O}(n^{2})") == "cal(O)(n^(2))"


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\downarrow", "arrow.b"),
        (r"\uparrow", "arrow.t"),
        (r"\updownarrow", "arrow.l.r"),
        (r"\ell", "ell"),
        (r"\ddots", "dots.down"),
        (r"\dagger", "dagger"),
        (r"\ddagger", "dagger.double"),
        (r"\checkmark", "checkmark"),
        (r"\star", "star"),
        (r"\#L", " hash L"),
        (r"\binom{K}{2}", "binom(K, 2)"),
        (r"2 \uparrow", "2 arrow.t"),
        (r"y_\ell", "y_ell"),
    ],
)
def test_a_previously_missing_symbol_now_converts(latex: str, expected: str) -> None:
    assert typstify_math(latex) == expected


def test_a_bare_hash_is_still_rejected() -> None:
    assert typstify_math("x # y") is None
    assert typstify_math('x " y') is None


def test_latex_command_names_never_reach_the_page_verbatim() -> None:
    assert typstify_math(r"p(x \mid y)") == "p(x divides y)"
    assert typstify_math(r"\|y - X\beta\|_2^2") == ("bar.v.double y - X beta bar.v.double _2^2")
    assert typstify_math(r"\sqrt[3]{x}") == "root(3, x)"
    assert typstify_math(r"\sqrt[n]{x+1}") == "root(n, x+1)"
    assert typstify_math(r"\sqrt{x}") == "sqrt(x)"


def test_a_cjk_label_with_an_ascii_word_is_not_requoted() -> None:
    assert typstify_math(r"\text{总 token 数}") == '#"总 token 数"'
    assert (
        typstify_math(r"\frac{\text{总 token 数}}{\text{专家数量}}")
        == 'frac(#"总 token 数", #"专家数量")'
    )


def test_a_cjk_label_keeps_its_ascii_words_verbatim() -> None:
    assert typstify_math(r"\text{参数}") == '#"参数"'
    assert typstify_math(r"\text{第 3 层}") == '#"第 3 层"'


@pytest.mark.parametrize(
    "latex",
    [r"a^", r"a_", r"x^", r"x_", r"a^ + b", r"x^{2}^", r"f(x)^", r"a^_b", r"A_"],
)
def test_a_dangling_script_does_not_emit_a_broken_body(latex: str) -> None:
    body = typstify_math(latex)
    assert body is not None
    assert not body.endswith(("^", "_")), body
    assert "_ _" not in body and "^ ^" not in body


def test_a_script_with_an_operand_is_untouched() -> None:
    assert typstify_math("x^2") == "x^2"
    assert typstify_math("x_i") == "x_i"
    assert typstify_math("a^{}") == "a^()"


def test_argmax_argmin_preserve_semantics() -> None:
    assert typstify_math(r"\argmax_{x} f(x)") == "arg max_(x) f(x)"
    assert typstify_math(r"\argmin_{\theta} \mathcal{L}(\theta)") == "arg min_(theta) cal(L)(theta)"


def test_unmapped_operator_dynamic_fallback() -> None:
    assert typstify_math(r"\tr(A)") == '#"tr" (A)'
    assert typstify_math(r"\diag(v)") == '#"diag" (v)'
    assert typstify_math(r"\sinc(x)") == '#"sinc" (x)'


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (r"\sum\limits_{i=1}^n i", "sum_(i=1)^n i"),
        (r"\sum\nolimits_i x_i", "sum_i x_i"),
        (r"E = mc^2 \nonumber", None),
        (r"E = mc^2 \notag", None),
        (r"\operatorname*{argmax}_x f(x)", '#"argmax"_x f(x)'),
        (r"\operatorname{rank} A", '#"rank" A'),
    ],
)
def test_layout_and_operator_commands_never_print_their_own_name(
    body: str, expected: str | None
) -> None:
    converted = typstify_math(body)
    assert converted is not None
    for command in ("limits", "nolimits", "nonumber", "notag", "operatorname"):
        assert command not in converted, f"{command} leaked into {converted!r}"
    if expected is not None:
        assert converted == expected
