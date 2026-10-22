"""The LaTeX → Typst math transpiler (RFC-2026-02 §5).

The transpiler is the one component the semantic reflow path cannot route
around: every display formula goes through it, and a failure is visible to the
reader as a literal ``\\frac`` on the page. These tests pin the three defects
found by measuring the arXiv 2101.03961 reference run, where the transpiler
returned ``None`` for **6 of 7** display formulas:

1. Docling emits spaced LaTeX (``\\frac { a } { b }``). Every pattern in the
   module requires the brace adjacent, so the whole transpiler failed closed.
   This was the single root cause of 15 failures in that one paper.
2. ``\\arg`` (and friends) are absent from the symbol table, and the
   unknown-command guard fails the *entire formula* closed over one name.
3. ``\\text{tokens per batch}`` was split into ``#"tokens" #"per" #"batch"``.
   Harmless in display position, but in a script position Typst renders
   ``A_(#"i" #"j")`` as ``A_i j`` with a space, not ``A_ij``.

Each test names the paper block it came from so a future failure can be traced
back to the source.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from ubt.render.typst.math_transpiler import typstify_math, typstify_math_span

pytestmark = pytest.mark.fast

#: The seven display formulas of arXiv 2101.03961 (Switch Transformers), exactly
#: as Docling emitted them — spaced commands and all.
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
    """All 7 convert; before the fixes only the 1 command-free formula did."""
    result = typstify_math(latex)
    assert result is not None, f"transpiler returned None for {latex[:60]!r}"


@pytest.mark.parametrize("latex", SWITCH_TRANSFORMER_FORMULAS)
def test_no_latex_survives_the_conversion(latex: str) -> None:
    """No backslash command may reach the page as literal text."""
    result = typstify_math(latex)
    assert result is not None
    # A trailing `\\` is a legitimate Typst line break inside math; a `\cmd`
    # or an escaped brace is not.
    assert not re.search(r"\\[A-Za-z]", result), f"LaTeX command survived: {result!r}"


def test_docling_spaced_commands_are_normalized() -> None:
    """`\\frac { a } { b }` — the root cause of 15 failures in one paper."""
    assert typstify_math(r"\frac { a } { b }") == "frac(a, b)"
    assert typstify_math(r"\mathbb { R }") == "bb(R)"
    assert typstify_math(r"\mathcal { T }") == "cal(T)"


def test_font_command_does_not_fuse_with_a_preceding_command() -> None:
    """An unspaced `\\in\\mathcal{T}` must not become the unknown `\\incal`.

    The font-style rewrite turns `\\mathcal{T}` into `cal(T)`, so an immediately
    preceding command ran into it (`\\in` + `cal` -> `\\incal`) and the whole
    formula failed closed to readable Unicode. Compact LaTeX has no space there.
    """
    assert typstify_math(r"i\in\mathcal{T}") == "i in cal(T)"
    assert typstify_math(r"\sum_{i\in\mathcal{T}} p_i(x) E_i(x)") is not None
    # A letter before the font command must keep its own spacing too.
    assert typstify_math(r"x\mathcal{Y}") == "x cal(Y)"
    # And the already-spaced spelling keeps working.
    assert typstify_math(r"i \in \mathcal{T}") == "i in cal(T)"


def test_accent_takes_the_whole_operand_not_its_first_letter() -> None:
    """`\\hat\\mathcal{T}` accented a bare `c`: `hat(c)al(T)`, a silent mis-render.

    LaTeX's `\\hat` takes the next *token*; once `\\mathcal{T}` became the Typst
    call `cal(T)`, the single-character bare-accent form matched only its first
    letter. The accent must wrap the whole call.
    """
    assert typstify_math(r"\hat\mathcal{T}") == "hat(cal(T))"
    assert typstify_math(r"\vec\mathcal{X}") == "vec(cal(X))"
    assert typstify_math(r"\bar\mathbb{R}") == "overline(bb(R))"
    # A braced operand keeps working, and so does the single-character form.
    assert typstify_math(r"\hat{\mathcal{T}}") == "hat(cal(T))"
    assert typstify_math(r"\hat x") == "hat(x)"


def test_accent_over_a_command_keeps_the_command() -> None:
    """`\\hat\\alpha` used to fail closed (None); LaTeX accents the whole command."""
    assert typstify_math(r"\hat\alpha") == "hat(alpha)"
    # `\hat` takes exactly one token, so the second command stays a sibling.
    assert typstify_math(r"\hat\alpha\beta") == "hat(alpha)beta"
    assert typstify_math(r"\hat\max") == "hat(max)"


def test_a_single_unknown_operator_does_not_kill_the_formula() -> None:
    """`\\arg` alone cost a whole display formula before the operator rescue."""
    result = typstify_math(r"\arg \max p ( x )")
    assert result is not None
    assert "arg" in result
    assert "\\arg" not in result


def test_operatorname_keeps_its_argument_as_a_name() -> None:
    r"""`\operatorname{rank}` must be quoted: bare ``rank`` is an unknown variable.

    Verified against the compiler — ``$ rank ( A ) $`` fails with "unknown
    variable: rank", while ``$ #"rank" ( A ) $`` renders the upright name.
    """
    assert typstify_math(r"\operatorname{rank} ( A )") == '#"rank" ( A )'


def test_spacing_hints_are_dropped() -> None:
    """`\\limits`/`\\displaystyle` carry no glyph."""
    with_limits = typstify_math(r"\sum \limits _ { i }")
    with_displaystyle = typstify_math(r"\displaystyle x")
    assert with_limits is not None and "limits" not in with_limits
    assert with_displaystyle is not None and "displaystyle" not in with_displaystyle


def test_mathrm_still_produces_an_upright_label() -> None:
    """The operator rescue must NOT swallow `\\mathrm` (a real regression)."""
    assert typstify_math(r"\mathrm{差值}") == '#"差值"'
    assert typstify_math(r"\text{总和} + \mathrm{差值}") == '#"总和" + #"差值"'


def test_text_with_spaces_becomes_one_string() -> None:
    """`\\text{tokens per batch}` is one label, not three adjacent strings.

    The `#` is load-bearing: a bare string is not a value in math mode, so
    dropping it produced `A_{"tokens per batch"}`, whose braces Typst then
    typesets as literal glyphs (measured: the page shows `A_{tokens per batch}`).
    """
    assert typstify_math(r"\text{tokens per batch}") == '#"tokens per batch"'


def test_a_multiword_text_in_a_subscript_stays_one_string() -> None:
    """In a script position the split form renders with a spurious space.

    Typst typesets ``A_(#"i" #"j")`` as ``A_i j`` but ``A_#"ij"`` as ``A_ij``,
    so the label must survive as a single string inside a script group.
    """
    result = typstify_math(r"I_{\text{off}}")
    assert result == 'I_#"off"'
    assert result.count('"') == 2


def test_span_form_unwraps_the_delimiters() -> None:
    assert typstify_math_span("$x^2$") == "x^2"
    assert typstify_math_span(r"\(x^2\)") == "x^2"
    assert typstify_math_span(r"\[x^2\]") == "x^2"
    assert typstify_math_span("$$x^2$$") == "x^2"


def test_a_bare_number_is_not_math() -> None:
    """`$5` is currency, not a formula."""
    assert typstify_math_span("$5$") is None


@pytest.mark.parametrize("latex", SWITCH_TRANSFORMER_FORMULAS)
def test_the_output_actually_compiles(latex: str, tmp_path: Path) -> None:
    """The end-to-end contract: Typst accepts every converted formula.

    Skipped when the compiler is absent, so the suite stays runnable without
    the toolchain — but a present compiler must accept all seven.

    The binary is resolved through the same ``resolve_typst_binary`` the render
    path uses (``UBT_TYPST_BINARY`` / ``typst`` on PATH), not a hardcoded
    ``$HOME/.cargo/bin`` path: a literal made the fast tier machine-specific and
    non-hermetic, silently skipping on every runner but the author's.
    """
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


# --------------------------------------------------------------------------- #
# A script immediately before a converted call must not merge into its name
# --------------------------------------------------------------------------- #


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
    r"""``\beta_t\sqrt{x}`` must not become ``beta_#"tsqrt" (x)``.

    ``\frac``/``\sqrt`` strip the backslash and emit a call, so with no
    separator the subscript's ``t`` merged with the function name. The word-run
    quoter then read ``tsqrt`` as one word and emitted ``#\"tsqrt\"``, losing the
    root or the fraction entirely. Measured on the gold corpus: 6 formulas.
    """
    assert typstify_math(latex) == expected


def test_a_spaced_command_still_converts_identically() -> None:
    # The separator must not change the already-correct spelling.
    assert typstify_math(r"\beta_t \sqrt{x}") == "beta_t sqrt(x)"
    assert typstify_math(r"\frac{a}{b}") == "frac(a, b)"


# --------------------------------------------------------------------------- #
# Symbols that were absent from the table, so whole formulas failed closed
# --------------------------------------------------------------------------- #


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
    r"""These all returned ``None``, degrading the **whole** formula to text.

    ``\downarrow``/``\uparrow`` are the "lower/higher is better" markers in
    nearly every results table, ``\ell`` is the standard index variable, and
    ``\dagger`` marks statistical significance. Measured on the gold corpus they
    accounted for 93 formulas that shipped as readable Unicode instead of
    typeset math; after adding them 1 remains.
    """
    assert typstify_math(latex) == expected


def test_a_bare_hash_is_still_rejected() -> None:
    r"""Only the LaTeX *escape* ``\#`` may pass; a bare ``#`` enters code mode.

    ``#`` is Typst's code-mode marker, so a formula containing one is not
    representable and must still fail closed rather than reaching the compiler.
    """
    assert typstify_math("x # y") is None
    assert typstify_math('x " y') is None


def test_latex_command_names_never_reach_the_page_verbatim() -> None:
    r"""The three zero-signal misprints the 2026-10-14 review measured.

    Each of these *compiled* (so every probe/gate stayed green) while printing
    wrong content: ``\mid`` as the letters "mid", ``\|`` as a single bar, and
    ``\sqrt[3]{x}`` as the literal text ``sqrt[3]{x}``. The assertions here
    check the emitted Typst body — "it compiles" is exactly the oracle that
    failed.
    """
    assert typstify_math(r"p(x \mid y)") == "p(x divides y)"
    assert typstify_math(r"\|y - X\beta\|_2^2") == ("bar.v.double y - X beta bar.v.double _2^2")
    assert typstify_math(r"\sqrt[3]{x}") == "root(3, x)"
    assert typstify_math(r"\sqrt[n]{x+1}") == "root(n, x+1)"
    # The plain root must be untouched by the degree branch.
    assert typstify_math(r"\sqrt{x}") == "sqrt(x)"


def test_a_cjk_label_with_an_ascii_word_is_not_requoted() -> None:
    r"""`\text{总 token 数}` is one label; its inner word must not be quoted again.

    The CJK branch emits the label already quoted, and the word-run quoter below
    then matched the ASCII word *inside* that string: the result was
    `#"总 #"token" 数"`, whose inner quotes close the label early. Typst rejected
    it with `unknown variable: token` and the whole document was lost (measured
    on switch-transformers, in a figure caption).
    """
    assert typstify_math(r"\text{总 token 数}") == '#"总 token 数"'
    assert (
        typstify_math(r"\frac{\text{总 token 数}}{\text{专家数量}}")
        == 'frac(#"总 token 数", #"专家数量")'
    )


def test_a_cjk_label_keeps_its_ascii_words_verbatim() -> None:
    # A pure-CJK label and a mixed one must both survive; the ASCII words keep
    # their spelling and spacing rather than becoming separate math identifiers.
    assert typstify_math(r"\text{参数}") == '#"参数"'
    assert typstify_math(r"\text{第 3 层}") == '#"第 3 层"'


# --------------------------------------------------------------------------- #
# A dangling script operator must not fail the whole document
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "latex",
    [r"a^", r"a_", r"x^", r"x_", r"a^ + b", r"x^{2}^", r"f(x)^", r"a^_b", r"A_"],
)
def test_a_dangling_script_does_not_emit_a_broken_body(latex: str) -> None:
    r"""``$a^$`` is ``error: expected expression`` in Typst, and one bad body
    fails the *whole* document (the publish path raises on a compile failure).

    The model emits these when it re-encodes a flattened superscript
    (``x²`` -> ``x^``) or drops a brace. A dangling ``^``/``_`` gets an empty
    string attached, the same spelling the leading-script guard uses.
    """
    body = typstify_math(latex)
    assert body is not None
    assert not body.endswith(("^", "_")), body
    assert "_ _" not in body and "^ ^" not in body


def test_a_script_with_an_operand_is_untouched() -> None:
    # The guard must not rewrite a legitimate script.
    assert typstify_math("x^2") == "x^2"
    assert typstify_math("x_i") == "x_i"
    assert typstify_math("a^{}") == "a^()"


def test_argmax_argmin_preserve_semantics() -> None:
    """\\argmax and \\argmin must render 'arg max' and 'arg min', not just 'arg'."""
    assert typstify_math(r"\argmax_{x} f(x)") == "arg max_(x) f(x)"
    assert typstify_math(r"\argmin_{\theta} \mathcal{L}(\theta)") == "arg min_(theta) cal(L)(theta)"


def test_unmapped_operator_dynamic_fallback() -> None:
    """Unmapped operators like \\tr, \\diag, \\sinc fallback to quoted Typst string operators."""
    assert typstify_math(r"\tr(A)") == '#"tr" (A)'
    assert typstify_math(r"\diag(v)") == '#"diag" (v)'
    assert typstify_math(r"\sinc(x)") == '#"sinc" (x)'


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # ``\limits`` is written glued to ``_`` in real LaTeX. ``\b`` never
        # matched there, so the command survived to the unknown-command guard
        # and printed ``#"limits"`` into the equation.
        (r"\sum\limits_{i=1}^n i", "sum_(i=1)^n i"),
        (r"\sum\nolimits_i x_i", "sum_i x_i"),
        # Annotation commands have no glyph; they must vanish, not print.
        (r"E = mc^2 \nonumber", None),
        (r"E = mc^2 \notag", None),
        # The starred operator form: ``*`` is part of the name.
        (r"\operatorname*{argmax}_x f(x)", '#"argmax"_x f(x)'),
        (r"\operatorname{rank} A", '#"rank" A'),
    ],
)
def test_layout_and_operator_commands_never_print_their_own_name(
    body: str, expected: str | None
) -> None:
    """A command Typst spells as an identifier must not reach the page as text.

    These all *compile*, which is why no render gate caught them: the reader
    saw ``∑ limits𝑛𝑖=1`` and ``E = mc² nonumber`` instead of a formula.
    """
    converted = typstify_math(body)
    assert converted is not None
    for command in ("limits", "nolimits", "nonumber", "notag", "operatorname"):
        assert command not in converted, f"{command} leaked into {converted!r}"
    if expected is not None:
        assert converted == expected
