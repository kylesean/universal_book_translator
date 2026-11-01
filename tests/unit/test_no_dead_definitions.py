from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"

_TEXT_ROOTS = ("ubt", "tests", "scripts")
_TEXT_SUFFIXES = (".py", ".md", ".toml", ".json", ".yaml", ".yml", ".typ")

_UNNAMEABLE_BY_DESIGN: frozenset[str] = frozenset(
    {
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


_FRAMEWORK_DECORATORS = frozenset(
    {
        "command",
        "callback",
        "fixture",
        "pytest.fixture",
        "property",
        "cached_property",
        "setter",
        "getter",
        "deleter",
        "field_validator",
        "model_validator",
        "field_serializer",
        "model_serializer",
        "validator",
        "root_validator",
        "lru_cache",
        "cache",
        "functools.lru_cache",
        "functools.cache",
    }
)


def _decorator_name(node: ast.AST) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts))


def _is_framework_reached(decorator_list: Sequence[ast.AST]) -> bool:
    for decorator in decorator_list:
        name = _decorator_name(decorator)
        if name in _FRAMEWORK_DECORATORS:
            return True
        if name.rsplit(".", 1)[-1] in _FRAMEWORK_DECORATORS:
            return True
    return False


def _definition_names(tree: ast.Module) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _is_framework_reached(node.decorator_list):
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
