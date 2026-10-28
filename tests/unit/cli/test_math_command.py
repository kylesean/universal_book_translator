"""Unit tests for the 'ubt math' CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.fast


@pytest.fixture
def sample_doc_with_math(tmp_path: Path) -> Path:
    doc = tmp_path / "math_sample.md"
    content = (
        "# 量子力学导论\n\n"
        "我们首先回顾质能等价定理：\n\n"
        "$$\n"
        "E = m c^2\n"
        "$$\n\n"
        "随后引入含时薛定谔波动方程：\n\n"
        "$$\n"
        "i hbar frac(diff, diff t) psi(r, t) = - frac(hbar^2, 2m) nabla^2 psi(r, t) + V(r, t) psi(r, t) + sum_(k=1)^N lambda_k phi_k(r, t)\n"
        "$$\n"
    )
    doc.write_text(content, encoding="utf-8")
    return doc


def test_math_inspect_command(sample_doc_with_math: Path) -> None:
    # Terminal text output
    res = runner.invoke(app, ["math", "inspect", str(sample_doc_with_math), "--max-width", "40"])
    assert res.exit_code == 0
    assert "Mathematical Equation Layout Health" in res.output

    # JSON output
    json_res = runner.invoke(
        app, ["math", "inspect", str(sample_doc_with_math), "--max-width", "40", "--json"]
    )
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_equations"] >= 1
    assert data["oversized_equations"] >= 1


def test_math_break_command(tmp_path: Path) -> None:
    long_eq = "f(x) = a + b + c + d + e + f + g + h + i + j + k + l + m + n"
    # Break Typst formula with alignment
    res = runner.invoke(app, ["math", "break", long_eq, "--max-width", "25", "--format", "typst"])
    assert res.exit_code == 0
    assert "\\\n" in res.output

    # Break to output file
    out_file = tmp_path / "broken.typ"
    res_file = runner.invoke(
        app,
        ["math", "break", long_eq, "--max-width", "25", "-o", str(out_file)],
    )
    assert res_file.exit_code == 0
    assert out_file.exists()
    assert "\\\n" in out_file.read_text(encoding="utf-8")

    # Break LaTeX formula with JSON output
    latex_eq = r"L = A + B + C + D + E + F + G + H + I + J + K"
    json_res = runner.invoke(
        app,
        ["math", "break", latex_eq, "--max-width", "20", "--format", "latex", "--json"],
    )
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert "\\\\" in data["broken"]
    assert data["syntax"] == "latex"


def test_math_audit_command(sample_doc_with_math: Path) -> None:
    # Normal pass with lenient min-score
    res_pass = runner.invoke(
        app, ["math", "audit", str(sample_doc_with_math), "--min-score", "40.0"]
    )
    assert res_pass.exit_code == 0
    assert "PASSED" in res_pass.output

    # Strict failure due to oversized equation
    res_strict_fail = runner.invoke(
        app, ["math", "audit", str(sample_doc_with_math), "--strict", "--max-width", "30"]
    )
    assert res_strict_fail.exit_code == 1
    assert "FAILED" in res_strict_fail.output
