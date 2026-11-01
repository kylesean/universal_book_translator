from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ubt.core.narrowing import narrow

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"


def test_narrow_returns_a_present_value() -> None:
    assert narrow(0, what="zero") == 0
    assert narrow("", what="empty") == ""
    assert narrow(False, what="false") is False


def test_narrow_raises_on_none_with_the_name_in_the_message() -> None:
    with pytest.raises(AssertionError, match="fragment form is None"):
        narrow(None, what="fragment form")


def test_narrow_is_not_stripped_by_optimization() -> None:
    with pytest.raises(AssertionError):
        narrow(None, what="value")


def test_library_code_has_no_bare_assert() -> None:
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
