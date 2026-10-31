"""Guard against ``except`` handlers that discard the error and say nothing.

The audit counted ~350 ``except Exception`` handlers in the package. Most are
correct: a fallback assignment (``except Exception: return []``) is visible to
the next reader, and probing code legitimately reports "not available" through
its return value. Replacing them wholesale would be noise, not a fix.

The handler this test rejects is the one with nothing left to inspect: its whole
body is ``pass``, ``continue`` or ``break``. When one of those fires, the
process keeps running with no record anywhere that something failed — no log
line, no re-raise, no fallback value — which is how a stale cache, an unreadable
page or a corrupt report turns into an unexplained wrong result.

The escape hatch is a same-line comment on the ``except`` clause: it is the
cheapest possible statement of *why* silence is right, and it puts that reason
where the next editor will see it. That is what this gate asks for, not a
rewrite of the handler.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"

#: Exception bases that also swallow programming errors (``AttributeError``,
#: ``TypeError``, ...). A handler narrowing to ``ValueError`` is stating
#: something, so it is out of scope.
_BROAD_EXCEPTIONS = frozenset({"Exception", "BaseException"})


def _caught_names(handler: ast.ExceptHandler) -> list[str]:
    if handler.type is None:
        return ["bare"]
    parts = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names: list[str] = []
    for part in parts:
        if isinstance(part, ast.Name):
            names.append(part.id)
        elif isinstance(part, ast.Attribute):
            names.append(part.attr)
    return names


def _body_is_a_silent_exit(handler: ast.ExceptHandler) -> bool:
    """True when the handler does nothing but pass/continue/break.

    A bare string statement (a docstring) is not an action, so it is ignored —
    but it is also not a justification: the author still has to say why the
    error is being dropped, and the failure message asks for the comment.
    """
    statements = [
        node
        for node in handler.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    return len(statements) == 1 and isinstance(statements[0], (ast.Pass, ast.Continue, ast.Break))


def _is_justified(source_lines: list[str], handler: ast.ExceptHandler) -> bool:
    """Whether the ``except`` line carries a trailing comment."""
    line = source_lines[handler.lineno - 1]
    return "#" in line


def _violation_lines(source: str) -> list[int]:
    tree = ast.parse(source)
    lines = source.splitlines()
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not any(name in _BROAD_EXCEPTIONS or name == "bare" for name in _caught_names(node)):
            continue
        if not _body_is_a_silent_exit(node) or _is_justified(lines, node):
            continue
        found.append(node.lineno)
    return found


def test_the_gate_detects_a_silent_handler() -> None:
    """The detector itself must not be asleep: a vacuous gate passes forever."""
    silent = "try:\n    x()\nexcept Exception:\n    pass\n"
    assert _violation_lines(silent) == [3]

    justified = "try:\n    x()\nexcept Exception:  # pdfium is optional here\n    pass\n"
    assert _violation_lines(justified) == []

    narrowed = "try:\n    x()\nexcept ValueError:\n    pass\n"
    assert _violation_lines(narrowed) == []

    recorded = "try:\n    x()\nexcept Exception:\n    logger.warning('failed')\n"
    assert _violation_lines(recorded) == []


def test_no_broad_except_swallows_an_error_silently() -> None:
    violations: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        violations.extend(
            f"{path.relative_to(_REPO_ROOT)}:{line}"
            for line in _violation_lines(path.read_text(encoding="utf-8"))
        )

    assert not violations, (
        "These handlers catch a broad exception and do nothing with it:\n  "
        + "\n  ".join(violations)
        + "\nLog the error, re-raise it, return an explicit fallback, or state on the "
        "`except` line why silence is correct."
    )
