"""Guard against dead module-level definitions.

`test_no_orphaned_modules.py` catches a whole *module* nobody imports. It cannot
see the other half of the same problem: a function or constant that survives
inside a live module after its last caller is deleted. The 2026-10 audit found
2,500 lines of that shape, and a second sweep found 25 more (16 functions, 8
constants, 1 class) left behind by refactors — including a `contains_point` with
a `bisect` body, an `as_typst_tuple` renderer, and a `validate_contract` that
documented invariants nothing enforced.

This test is the same shape as the module guard: pure AST, no imports executed,
milliseconds to run. It reports a definition that appears exactly once in the
whole tree — its own `def`/`class`/assignment line.

Deliberately conservative, because a false positive deletes live code:

- any decorated definition is skipped (Typer commands, pytest fixtures,
  ``@property``, ``@field_validator`` are reached through the framework);
- any ``__dunder__`` name is skipped (protocol hooks are called by the runtime);
- names starting with ``test_`` are skipped (collected by pytest, not imported);
- the search covers ``tests/``, ``scripts/`` and the documentation/config files,
  not just the package, so a symbol named only in a docstring or a ``.toml``
  still counts as referenced.

A hit means "nothing in this repository mentions this name", which is the
actionable signal. It is not "safe to delete" — a public API consumed by a
downstream package would also hit — so the failure message says to delete it
*or* document why it stays.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"

#: Roots whose text keeps a name live. ``tests/`` and ``scripts/`` matter: a
#: slow-tier harness runs as a subprocess and is not collected by pytest, so a
#: symbol reached only through one of them is live but invisible to the suite.
_TEXT_ROOTS = ("ubt", "tests", "scripts")
_TEXT_SUFFIXES = (".py", ".md", ".toml", ".json", ".yaml", ".yml", ".typ")

#: Definitions that are reached without ever being named. Each entry must say
#: how it is reached, so the allowlist cannot become a dumping ground.
_UNNAMEABLE_BY_DESIGN: frozenset[str] = frozenset(
    {
        # HTMLParser protocol hooks: the stdlib calls these by attribute name.
        "handle_comment",
        "handle_decl",
        "handle_pi",
        "unknown_decl",
    }
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in _TEXT_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in _TEXT_SUFFIXES:
                files.append(path)
    return files


def _definition_names(tree: ast.Module) -> list[tuple[str, int]]:
    """(name, lineno) for every module-level definition that could be dead."""
    found: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.decorator_list:
                continue
            name = node.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.append((target.id, node.lineno))
            continue
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            continue
        if name.startswith("__") and name.endswith("__"):
            continue
        if name.startswith("test_"):
            continue
        found.append((name, node.lineno))
    return found


def _exported_names(tree: ast.Module) -> list[str]:
    """The string entries of every ``__all__`` in a module.

    An ``__all__`` entry is *not* a use: the module's own re-export list is the
    one place a dead definition can hide, because the naive counter counts the
    string as a mention. ``page_sizes``, ``VlmEvidence`` and
    ``resolve_cli_adaptive_dual_mode`` each survived this way -- defined once,
    named only in their own ``__all__`` beside them, imported by nobody. Their
    entries are subtracted from the reference count below so the guard sees
    through the re-export.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            names += [
                e.value
                for e in value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    return names


def test_no_dead_definitions() -> None:
    """Every module-level definition is used somewhere besides its own ``def`` line.

    Counts identifiers once across the whole tree (a ``Counter``), rather than
    regex-searching the corpus per name: the naive sweep is
    O(definitions x corpus) and measured at 110s, which is not a fast test.
    Counting is required, not mere membership — the definition's own line always
    contains its name, so presence alone would make the check vacuous. A count of
    one is exactly that self-reference.

    An ``__all__`` entry is subtracted from every count: a module naming a
    definition only in its own re-export list has no consumer, and counting the
    string hid exactly that (see :func:`_exported_names`).
    """
    mentions: Counter[str] = Counter()
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        mentions.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    exported: Counter[str] = Counter()
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        exported.update(_exported_names(tree))

    dead: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for name, lineno in _definition_names(tree):
            if name in _UNNAMEABLE_BY_DESIGN:
                continue
            if mentions[name] - exported[name] <= 1:
                dead.append(f"{path.relative_to(_REPO_ROOT)}:{lineno} {name}")

    assert not dead, (
        "dead definition(s) -- nothing in the repository uses them. Delete, or "
        "document why they stay: " + ", ".join(sorted(dead))
    )
