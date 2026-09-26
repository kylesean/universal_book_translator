"""Unit tests for TypstDiagnosticHealer.

Verifies:
- Direct fast-path compile without unnecessary probing.
- Precision Action Matrix:
  - UNKNOWN_VARIABLE quoting in math mode, scoped to the line the compiler named.
  - UNCLOSED_DELIMITER / UNCLOSED_STRING line-level healing.
  - SYNTAX error degradation to code spans and comment removal.
- Persistent comment error culprit isolation.
- Audit ledger tracking (last_syntax_fallbacks).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ubt.adapters.pdf.typst_healer import (
    TypstDiagnosticHealer,
    _quote_unknown_var_on_line,
)
from ubt.core.exceptions import DocumentParseError


def test_quote_unknown_var_on_line() -> None:
    line = "When $V_th = 0.5$ and $I_th = 1$, we observe saturation."
    repaired, modified = _quote_unknown_var_on_line(line, "th")
    assert modified
    assert '$V_"th" = 0.5$' in repaired
    assert '$I_"th" = 1$' in repaired

    # Outside math mode, variables are not touched
    prose_only = "The value of th is not in math."
    repaired_prose, modified_prose = _quote_unknown_var_on_line(prose_only, "th")
    assert not modified_prose
    assert repaired_prose == prose_only


def test_quote_unknown_var_dx_differential() -> None:
    line = "Evaluate $\\int f(x) dx = 1$."
    repaired, modified = _quote_unknown_var_on_line(line, "dx")
    assert modified
    assert "d x" in repaired


def test_quote_unknown_var_ignores_keywords_and_assets() -> None:
    line = '$let x = 1$ and #image("figure.png")'
    repaired, modified = _quote_unknown_var_on_line(line, "image")
    assert not modified
    assert repaired == line


def test_healer_fast_path(tmp_path: Path) -> None:
    """If compilation succeeds on attempt 0, no healing retries are performed."""
    fake_run = MagicMock(
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    )
    healer = TypstDiagnosticHealer()
    out_pdf = tmp_path / "fast.pdf"
    res = healer.heal_and_compile("= Title\nHello world.", out_pdf, run_typst_override=fake_run)
    assert res == out_pdf
    assert fake_run.call_count == 1
    assert len(healer.last_syntax_fallbacks) == 0


def test_healer_heals_unknown_variable(tmp_path: Path) -> None:
    """Healer catches unknown variable from diagnostic and quotes it."""
    error_stderr = """error: unknown variable: th
  ┌─ /tmp/test.typ:2:4
  │
2 │ $ V_th = 1 $
  │     ^^
"""
    call_count = 0

    def mock_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=error_stderr
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    healer = TypstDiagnosticHealer()
    out_pdf = tmp_path / "var_heal.pdf"
    source = "= Test\n$ V_th = 1 $\n"
    res = healer.heal_and_compile(source, out_pdf, run_typst_override=mock_runner)
    assert res == out_pdf
    assert call_count == 2
    # Verify the saved typ file has the quoted variable
    typ_content = out_pdf.with_suffix(".typ").read_text(encoding="utf-8")
    assert '$ V_"th" = 1 $' in typ_content


def test_healer_leaves_formulas_the_compiler_did_not_flag(tmp_path: Path) -> None:
    """Only the named line is repaired — no global sweep over every math block.

    A file-wide rewrite of the reported variable also corrupts unrelated
    formulas the compiler never complained about, so the healer must stay
    scoped to the diagnostic's line.
    """
    error_stderr = """error: unknown variable: th
  ┌─ /tmp/test.typ:2:4
  │
2 │ $ V_th = 1 $
  │     ^^
"""
    call_count = 0

    def mock_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=error_stderr
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    healer = TypstDiagnosticHealer()
    out_pdf = tmp_path / "scoped_heal.pdf"
    source = "= Test\n$ V_th = 1 $\n\n$ W_th = 2 $\n"
    healer.heal_and_compile(source, out_pdf, run_typst_override=mock_runner)

    typ_content = out_pdf.with_suffix(".typ").read_text(encoding="utf-8")
    assert '$ V_"th" = 1 $' in typ_content  # the line the compiler named
    assert "$ W_th = 2 $" in typ_content  # an unnamed line must be untouched


def test_healer_degrades_persistent_syntax_error(tmp_path: Path) -> None:
    """Healer falls back to code span and then emergency comment fallback."""
    error_stderr = """error: syntax error
  ┌─ /tmp/test.typ:2:1
  │
2 │ invalid syntax line
  │ ^
"""

    def mock_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=error_stderr)

    healer = TypstDiagnosticHealer(max_attempts=2)
    out_pdf = tmp_path / "syntax_fail.pdf"
    source = "= Section\ninvalid syntax line\n"
    # Even in emergency sweep, mock_runner keeps failing, so DocumentParseError is raised
    with pytest.raises(DocumentParseError, match="Typst compilation failed"):
        healer.heal_and_compile(source, out_pdf, run_typst_override=mock_runner)
    assert len(healer.last_syntax_fallbacks) > 0


def test_healer_version_probe_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="typst 0.15.1 (unknown commit)\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    healer = TypstDiagnosticHealer()
    assert healer.compiler_version() == "0.15.1"
    assert healer.compiler_version() == "0.15.1"
    assert len(calls) == 1


def test_persistent_comment_heal_reports_every_nullified_line() -> None:
    from ubt.adapters.pdf.typst_healer import _heal_persistent_comment_error

    lines = ["// already commented $x$", "", 'a = "unclosed', "", "", "// error line $y$"]
    audit: list[str] = []
    assert _heal_persistent_comment_error(lines, 5, audit)
    # The culprit (line 3) and the error line (line 6) both lose their content.
    assert audit == ['line 3: a = "unclosed', "line 6: // error line $y$"]


def test_probe_single_math_strips_multiline_footnote(monkeypatch: pytest.MonkeyPatch) -> None:
    healer = TypstDiagnosticHealer()
    written_text: list[str] = []

    def _intercept_write(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.name == "probe.typ":
            written_text.append(data)
        return len(data)

    monkeypatch.setattr(Path, "write_text", _intercept_write)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )

    math_with_multiline_footnote = "$ x = y #footnote[\n  multiline footnote\n  content\n] $"
    result = healer.probe_single_math(math_with_multiline_footnote)
    assert result is True
    assert len(written_text) == 1
    assert "footnote" not in written_text[0]
