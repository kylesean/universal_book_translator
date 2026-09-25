"""Comments that name a file or symbol must point at something that exists.

A docstring saying "see ``ubt/adapters/pdf/visual_gate.py``" or naming a symbol
that has since moved is drift no type checker sees. This gate resolves
backticked repo-path and dotted ``ubt.*`` references in source comments and
docstrings, so a rename cannot silently leave a comment pointing at a file or
symbol that no longer exists.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAN_DIRS = (_REPO_ROOT / "ubt", _REPO_ROOT / "tests")
#: Only unambiguous repo-relative paths are checked; a bare ``foo/bar.py`` may
#: be relative to anywhere and would produce false positives.
_FILE_REF_ROOTS = ("ubt/", "tests/", "scripts/", "docs/")
_FILE_REF = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|toml|json|sql))`")
_DOT_REF = re.compile(r"`(ubt(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _resolve_module(parts: list[str]) -> tuple[Path | None, list[str]]:
    """Longest ``ubt.*`` prefix that is a module, plus the trailing attributes."""
    for i in range(len(parts), 0, -1):
        candidate = _REPO_ROOT.joinpath(*parts[:i])
        if candidate.with_suffix(".py").is_file():
            return candidate.with_suffix(".py"), parts[i:]
        if (candidate / "__init__.py").is_file():
            return candidate / "__init__.py", parts[i:]
    return None, []


def test_backticked_references_resolve() -> None:
    problems: list[str] = []
    for path in itertools.chain.from_iterable(root.rglob("*.py") for root in _SCAN_DIRS):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(_REPO_ROOT)
        for match in _FILE_REF.finditer(text):
            ref = match.group(1)
            if ref.startswith(_FILE_REF_ROOTS) and not (_REPO_ROOT / ref).exists():
                problems.append(f"{rel}: file reference `{ref}` does not exist")
        for match in _DOT_REF.finditer(text):
            ref = match.group(1)
            module, attrs = _resolve_module(ref.split("."))
            if module is None:
                problems.append(f"{rel}: module reference `{ref}` resolves to no module")
                continue
            identifiers = set(_IDENT.findall(module.read_text(encoding="utf-8", errors="replace")))
            for attr in attrs:
                if attr not in identifiers:
                    problems.append(
                        f"{rel}: symbol reference `{ref}` — {attr!r} not found in "
                        f"{module.relative_to(_REPO_ROOT)}"
                    )
                    break
    assert not problems, "stale code references:\n" + "\n".join(problems)
