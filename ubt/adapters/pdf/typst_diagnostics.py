"""Structured parsing of the Typst compiler's stderr diagnostic dialect.

The self-healing loop in :class:`~ubt.adapters.pdf.typst_reconstructor.TypstReconstructor`
reacts to compiler failures by *line number* and, for math, by *variable name*
(``unknown variable: th``). That reaction is a contract with a specific stderr
dialect, so the parsing lives here — one place, pinned by golden tests built
from captured real output — instead of being re-derived with inline regexes at
each call site.

Dialect (Typst 0.15.x, the series CI pins)::

    error: <message>
      ┌─ <file>:<line>:<col>
      │
    <line> │ <source>
      │ <caret>

Only two structural anchors are relied on: the ``error:`` message line and the
``┌─ <file>:<line>:<col>`` location line. Caret art, gutters and ``= hint:``
lines are ignored. A single error message may be followed by more than one
caret block; the first location wins (:func:`parse_typst_stderr`).

When the pinned Typst series changes, re-capture the samples and bump
:data:`DIALECT_TYPST_SERIES` if the anchors moved; the dialect check fails
loudly instead of the self-healing loop silently degrading content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

#: Typst release series whose stderr this parser is pinned to. CI installs a
#: matching pinned compiler; a different series should re-run the dialect test.
DIALECT_TYPST_SERIES = ("0.15",)

#: ``┌─ /abs/path/file.typ:12:3`` (path may contain spaces or a Windows drive).
_LOCATION_RE = re.compile(r"┌─\s*(?P<file>.+?):(?P<line>\d+):(?P<col>\d+)\s*$")

_MESSAGE_PREFIX = "error:"
_UNKNOWN_VARIABLE_RE = re.compile(r"unknown variable:\s*([A-Za-z0-9_]+)")


class TypstErrorKind(StrEnum):
    """Coarse classes the self-healing loop reacts to differently."""

    UNKNOWN_VARIABLE = "unknown_variable"
    UNCLOSED_DELIMITER = "unclosed_delimiter"
    UNCLOSED_STRING = "unclosed_string"
    SYNTAX = "syntax"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TypstDiagnostic:
    """One compiler error: where it was reported and what kind it is."""

    kind: TypstErrorKind
    line: int  # 1-based, as Typst reports it
    column: int  # 0-based (Typst may report column 0)
    message: str
    variable: str | None = None


def classify_message(message: str) -> TypstErrorKind:
    """Map a Typst error message to a :class:`TypstErrorKind`."""
    lowered = message.lower()
    if lowered.startswith("unknown variable"):
        return TypstErrorKind.UNKNOWN_VARIABLE
    if lowered.startswith("unclosed delimiter"):
        return TypstErrorKind.UNCLOSED_DELIMITER
    if lowered.startswith("unclosed string"):
        return TypstErrorKind.UNCLOSED_STRING
    if "syntax" in lowered or "expected" in lowered:
        return TypstErrorKind.SYNTAX
    return TypstErrorKind.OTHER


def parse_typst_stderr(stderr: str) -> list[TypstDiagnostic]:
    """Parse Typst stderr into one diagnostic per error message.

    A message with no following location line is dropped: the self-healing loop
    cannot act on it. A message with several caret blocks keeps the first
    location (the others are the same error re-drawn).
    """
    diagnostics: list[TypstDiagnostic] = []
    message: str | None = None
    for raw in stderr.splitlines():
        stripped = raw.strip()
        if stripped.startswith(_MESSAGE_PREFIX):
            message = stripped[len(_MESSAGE_PREFIX) :].strip()
            continue
        if message is None:
            continue
        location = _LOCATION_RE.search(raw)
        if location is None:
            continue
        variable_match = _UNKNOWN_VARIABLE_RE.search(message)
        diagnostics.append(
            TypstDiagnostic(
                kind=classify_message(message),
                line=int(location.group("line")),
                column=int(location.group("col")),
                message=message,
                variable=variable_match.group(1) if variable_match else None,
            )
        )
        message = None
    return diagnostics


def error_line_numbers(stderr: str) -> list[int]:
    """Distinct 1-based line numbers Typst reported, in first-seen order."""
    seen: dict[int, None] = {}
    for diagnostic in parse_typst_stderr(stderr):
        seen.setdefault(diagnostic.line, None)
    return list(seen)
