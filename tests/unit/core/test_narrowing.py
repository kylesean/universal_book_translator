"""``narrow``: the invariant check that survives ``python -O``.

``assert`` is stripped from the bytecode under ``-O``, so an invariant stated
that way goes silent in an optimized run and the ``None`` it was guarding
surfaces later as an ``AttributeError`` with no trace of which check failed.
The project relies on this for type narrowing in a dozen places; ``narrow``
replaces those asserts so the check and the actionable message both survive.

The companion check pins the *policy*: library code under ``ubt/`` must not
use a bare ``assert`` for this, because the failure is invisible under ``-O``
and nothing else in the tree would notice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ubt.core.narrowing import narrow

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"


def test_narrow_returns_a_present_value() -> None:
    assert narrow(0, what="zero") == 0  # falsy is present, not absent
    assert narrow("", what="empty") == ""
    assert narrow(False, what="false") is False


def test_narrow_raises_on_none_with_the_name_in_the_message() -> None:
    with pytest.raises(AssertionError, match="fragment form is None"):
        narrow(None, what="fragment form")


def test_narrow_is_not_stripped_by_optimization() -> None:
    """The whole point: the guard must still fire under ``-O``.

    ``assert`` disappears under ``-O``; ``narrow`` raises explicitly, so this
    passes with and without ``PYTHONOPTIMIZE``.
    """
    with pytest.raises(AssertionError):
        narrow(None, what="value")


def test_library_code_has_no_bare_assert() -> None:
    """``assert`` in ``ubt/`` is a silent no-op under ``-O``.

    The project documents this hazard for the EPUB adapter and the job queue,
    where an explicit raise replaced it. Every other site was narrowing, and
    now goes through :func:`ubt.core.narrowing.narrow`. Keep it that way: an
    assert that guards a real invariant fails silently in an optimized run,
    and no other gate can see it.
    """
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "`assert` is stripped under `python -O`; raise explicitly, or use "
        "`ubt.core.narrowing.narrow` for None-narrowing:\n" + "\n".join(offenders)
    )
