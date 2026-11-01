from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ALLOWLIST: dict[str, str] = {}

_CONVENTION_DECORATORS = {
    "field_validator",
    "validator",
    "model_validator",
    "computed_field",
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "websocket",
    "head",
    "options",
    "exception_handler",
    "middleware",
    "on_event",
    "command",
    "callback",
    "property",
    "cached_property",
    "staticmethod",
    "classmethod",
    "abstractmethod",
    "overload",
    "contextmanager",
    "asynccontextmanager",
    "lru_cache",
    "cache",
    "dataclass",
    "fixture",
    "tool",
    "prompt",
    "resource",
}


def _census() -> Counter[str]:
    tokens: Counter[str] = Counter()
    files = [
        *(_REPO_ROOT / "ubt").rglob("*.py"),
        *(_REPO_ROOT / "tests").rglob("*.py"),
        *(_REPO_ROOT / "scripts").glob("*.py"),
    ]
    files.append(_REPO_ROOT / "pyproject.toml")
    for path in files:
        if not path.exists() or "__pycache__" in path.parts:
            continue
        tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", path.read_text(errors="ignore")))
    return tokens


def _convention_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        rendered = (
            ast.unparse(decorator.func)
            if isinstance(decorator, ast.Call)
            else ast.unparse(decorator)
        )
        if rendered in _CONVENTION_DECORATORS or rendered.split(".")[-1] in _CONVENTION_DECORATORS:
            return True
    return False


def test_no_definition_is_orphaned_in_production() -> None:
    tokens = _census()
    orphans: list[str] = []
    for path in sorted((_REPO_ROOT / "ubt").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:  # pragma: no cover - the lint gate owns syntax
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = node.name
            if (name.startswith("__") and name.endswith("__")) or name in _ALLOWLIST:
                continue
            if _convention_decorated(node):
                continue
            if tokens[name] <= 1:
                rel = path.relative_to(_REPO_ROOT)
                orphans.append(f"{rel}:{node.lineno} {type(node).__name__} {name}")
    assert orphans == [], (
        "definitions with zero references anywhere (delete, wire up, or add an "
        "entry to _ALLOWLIST with the reason it is retained):\n  " + "\n  ".join(orphans)
    )
