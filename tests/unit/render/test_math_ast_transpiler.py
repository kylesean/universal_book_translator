"""Unit tests for the AST-based LaTeX math transpiler (Phase 2)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_compile import resolve_typst_binary
from ubt.render.typst.math_ast_transpiler import (
    has_latex_environments,
    transpile_latex_ast,
)
from ubt.render.typst.math_transpiler import typstify_math, typstify_math_span

pytestmark = pytest.mark.fast


def _compile_typst(math_snippet: str) -> tuple[bool, str]:
    """Helper to verify that Typst actually compiles the transpiled math."""
    try:
        binary = resolve_typst_binary(os.environ.get("UBT_TYPST_BINARY", "typst"))
        if not binary:
            pytest.skip("typst compiler not found on system")
    except Exception:
        pytest.skip("typst compiler not found on system")

    doc = f"$ {math_snippet} $\n"
    with tempfile.NamedTemporaryFile("w", suffix=".typ") as f:
        f.write(doc)
        f.flush()
        out_pdf = Path(f.name).with_suffix(".pdf")
        try:
            res = subprocess.run(
                [binary, "compile", f.name, str(out_pdf)],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0, res.stderr
        finally:
            if out_pdf.exists():
                out_pdf.unlink(missing_ok=True)


def test_has_latex_environments() -> None:
    assert has_latex_environments(r"\begin{pmatrix} a & b \end{pmatrix}")
    assert has_latex_environments(r"f(x) = \begin{cases} 1 \end{cases}")
    assert not has_latex_environments(r"\frac{a}{b} + x^2")


def test_transpile_latex_ast_direct() -> None:
    res = transpile_latex_ast(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")
    assert res == "mat(1, 2; 3, 4)"


def test_pmatrix_transpilation() -> None:
    latex = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
    result = typstify_math(latex)
    assert result == "mat(a, b; c, d)"

    ok, err = _compile_typst(result)
    assert ok, f"Typst failed to compile: {err}"


def test_bmatrix_transpilation() -> None:
    latex = r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}"
    result = typstify_math(latex)
    assert result == 'mat(delim: "[", 1, 0; 0, 1)'

    ok, err = _compile_typst(result)
    assert ok, f"Typst failed to compile: {err}"


def test_vmatrix_and_vmatrix_double() -> None:
    vmat = r"\begin{vmatrix} a & b \\ c & d \end{vmatrix}"
    res_v = typstify_math(vmat)
    assert res_v == 'mat(delim: "|", a, b; c, d)'

    vmat2 = r"\begin{Vmatrix} a & b \\ c & d \end{Vmatrix}"
    res_v2 = typstify_math(vmat2)
    assert res_v2 == 'mat(delim: "‖", a, b; c, d)'

    ok, err = _compile_typst(res_v)
    assert ok, err
    ok, err = _compile_typst(res_v2)
    assert ok, err


def test_matrix_none_delim() -> None:
    latex = r"\begin{matrix} x & y \\ z & w \end{matrix}"
    result = typstify_math(latex)
    assert result == "mat(delim: #none, x, y; z, w)"

    ok, err = _compile_typst(result)
    assert ok, err


def test_trailing_double_backslash_pruned() -> None:
    latex = r"\begin{pmatrix} a & b \\ c & d \\ \end{pmatrix}"
    result = typstify_math(latex)
    assert result == "mat(a, b; c, d)"


def test_matrix_with_complex_cells() -> None:
    latex = r"\begin{pmatrix} \frac{1}{2} & \alpha_1 \\ \sum_{i=1}^n x_i & e^{-t} \end{pmatrix}"
    result = typstify_math(latex)
    assert result is not None
    assert "frac(1, 2)" in result
    assert "alpha_1" in result

    ok, err = _compile_typst(result)
    assert ok, f"Typst failed to compile complex matrix: {err}"


def test_cases_transpilation() -> None:
    latex = r"\begin{cases} x^2 & \text{if } x > 0 \\ -x & \text{otherwise} \end{cases}"
    result = typstify_math(latex)
    assert result is not None
    assert result.startswith("cases(")
    assert 'x^2 & #"if" x > 0' in result or 'x^2 & #"if "' in result
    assert '-x & #"otherwise"' in result

    ok, err = _compile_typst(result)
    assert ok, f"Typst failed to compile cases: {err}"


def test_aligned_transpilation() -> None:
    latex = r"\begin{aligned} a &= b + c \\ d &= e + f \end{aligned}"
    result = typstify_math(latex)
    assert result is not None
    assert "&" in result
    assert "\\" in result

    ok, err = _compile_typst(result)
    assert ok, f"Typst failed to compile aligned: {err}"


def test_nested_matrices() -> None:
    latex = r"M = \begin{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} & 0 \\ 0 & 1 \end{pmatrix}"
    result = typstify_math(latex)
    assert result is not None
    assert "mat(" in result

    ok, err = _compile_typst(result)
    assert ok, f"Typst failed to compile nested matrix: {err}"


def test_formula_with_embedded_matrix() -> None:
    latex = r"\mathbf{y} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \mathbf{x} + \mathbf{b}"
    result = typstify_math(latex)
    assert result is not None
    assert 'mat(delim: "[", a, b; c, d)' in result
    assert "bold(y)" in result

    ok, err = _compile_typst(result)
    assert ok, f"Typst failed to compile embedded matrix formula: {err}"


def test_span_with_environment() -> None:
    span = r"$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$"
    result = typstify_math_span(span)
    assert result == "mat(1, 2; 3, 4)"


def test_a_blank_leading_cell_is_preserved() -> None:
    """A blank-corner matrix keeps its empty first cell (not collapsed).

    The lead-empty prune that was added for the ``array`` column spec actually
    ate a *real* empty cell: pylatexenc already consumes the spec, so the only
    thing that prune could drop was a legitimate blank corner.
    """
    result = typstify_math(r"\begin{pmatrix} & b \\ c & d \end{pmatrix}")
    assert result == "mat(, b; c, d)"
    ok, _ = _compile_typst(result)
    assert ok


def test_an_unstarred_matrix_position_arg_is_stripped() -> None:
    """``\\begin{matrix}[r] ...`` must not leak ``[r]`` into a cell."""
    result = typstify_math(r"\begin{matrix}[r] a & b \\ c & d \end{matrix}")
    assert result == "mat(delim: #none, a, b; c, d)"


def test_the_spaced_docling_dialect_reaches_the_ast() -> None:
    """The gate runs after normalization, so ``\\begin { pmatrix }`` still works."""
    result = typstify_math(r"\begin { pmatrix } 1 & 2 \\ 3 & 4 \end { pmatrix }")
    assert result == "mat(1, 2; 3, 4)"
