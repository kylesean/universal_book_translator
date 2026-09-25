"""Golden + live tests for the Typst stderr diagnostic parser.

The string samples below are captured from Typst 0.15.1 — the series CI pins.
:mod:`ubt.adapters.pdf.typst_diagnostics` is pinned to that dialect, so if Typst
changes its stderr layout these goldens fail and the parser is updated
deliberately instead of the self-healing loop silently misreading errors.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_diagnostics import (
    DIALECT_TYPST_SERIES,
    TypstErrorKind,
    error_line_numbers,
    parse_typst_stderr,
)

UNKNOWN_VAR = """\
error: unknown variable: th
  ┌─ unknown_var.typ:3:3
  │
3 │ $V_th = 1$
  │    ^^
  │
  = hint: if you meant to display multiple letters as is, try adding spaces between each letter: `t h`
  = hint: or if you meant to display this as text, try placing it in quotes: `"th"`
"""

UNCLOSED_DELIMITER = """\
error: unclosed delimiter
  ┌─ unclosed_delim.typ:3:5
  │
3 │ #text("abc
  │      ^

error: unclosed string
  ┌─ unclosed_delim.typ:3:6
  │
3 │   #text("abc
  │ ╭───────^
4 │ │
"""

UNCLOSED_MATH = """\
error: unclosed delimiter
  ┌─ unclosed_math.typ:3:0
  │
3 │ $V_th = 1
  │ ^
"""

SYNTAX_EXPR = """\
error: expected expression
  ┌─ syntax_expr.typ:3:5
  │
3 │ #(1 + )
  │      ^
"""


def test_dialect_is_pinned_to_a_known_series() -> None:
    """A pin bump must be a deliberate edit, not a silent drift."""
    assert DIALECT_TYPST_SERIES == ("0.15",)


def test_unknown_variable_is_classified_with_its_name() -> None:
    (diagnostic,) = parse_typst_stderr(UNKNOWN_VAR)
    assert diagnostic.kind is TypstErrorKind.UNKNOWN_VARIABLE
    assert (diagnostic.line, diagnostic.column) == (3, 3)
    assert diagnostic.variable == "th"
    assert diagnostic.message == "unknown variable: th"


def test_unclosed_stderr_yields_one_diagnostic_per_error() -> None:
    diagnostics = parse_typst_stderr(UNCLOSED_DELIMITER)
    assert [d.kind for d in diagnostics] == [
        TypstErrorKind.UNCLOSED_DELIMITER,
        TypstErrorKind.UNCLOSED_STRING,
    ]
    assert [d.line for d in diagnostics] == [3, 3]


def test_column_zero_is_accepted() -> None:
    (diagnostic,) = parse_typst_stderr(UNCLOSED_MATH)
    assert diagnostic.kind is TypstErrorKind.UNCLOSED_DELIMITER
    assert (diagnostic.line, diagnostic.column) == (3, 0)


def test_expected_expression_is_syntax() -> None:
    (diagnostic,) = parse_typst_stderr(SYNTAX_EXPR)
    assert diagnostic.kind is TypstErrorKind.SYNTAX
    assert diagnostic.line == 3


def test_message_without_a_location_is_ignored() -> None:
    assert parse_typst_stderr("error: something went very wrong\n") == []


def test_error_line_numbers_dedup_and_keep_first_seen_order() -> None:
    assert error_line_numbers(UNCLOSED_DELIMITER) == [3]
    assert error_line_numbers(UNKNOWN_VAR + UNCLOSED_MATH) == [3]


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_live_typst_stderr_is_read_by_the_parser() -> None:
    """Real compiler output must still parse — the dialect guard that matters."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "live.typ"
        source.write_text("= Test\n\n$V_th = 1$\n", encoding="utf-8")
        proc = subprocess.run(
            ["typst", "compile", "--root", tmp, str(source), str(Path(tmp) / "live.pdf")],
            capture_output=True,
            text=True,
            # Typst emits UTF-8 regardless of console codepage; the box-drawing
            # characters in the location line are what _LOCATION_RE matches, so a
            # cp1252 decode on the Windows runner silently yields no diagnostics
            # (the same reason 49de922 fixed the production call sites).
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    assert proc.returncode != 0
    diagnostics = parse_typst_stderr(proc.stderr)
    assert diagnostics, f"parser found no diagnostic in real typst stderr:\n{proc.stderr}"
    assert any(d.kind is TypstErrorKind.UNKNOWN_VARIABLE and d.line == 3 for d in diagnostics)
