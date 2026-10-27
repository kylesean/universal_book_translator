"""Guard against orphaned *definitions*: a function, method or class nothing
calls.

``test_no_orphaned_modules`` catches a module nothing imports; this is the
same guard one level down, for the definition the module imports but never
uses — a private helper left behind by a refactor, a public method only its
own test still reaches. The 2026-10 round found nine of those (~180 lines,
among them a superseded batch-API pair and a checkpoint writer the flusher
had replaced); each was deleted, and this test keeps the class dead.

Like the module gate, the check is pure AST plus one identifier census — no
imports are executed, it costs milliseconds, and it errs toward silence:

* the census counts *every* token occurrence, so a name mentioned in a
  docstring, an ``__all__`` list, a pyproject entry point or a test counts as
  referenced (a false negative, never a false alarm);
* definitions carrying a convention decorator are skipped — pydantic
  validators, FastAPI/Typer handlers, properties and friends are invoked by
  their framework, not by name;
* anything the census still flags must earn a line in ``_ALLOWLIST`` with the
  reason it is retained. The house rule is the module gate's: delete it, wire
  it up, or allowlist it with a reason — silence is not an option.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Definitions the census flags but that are known-live. Every entry carries
#: the reason; an entry whose reason has evaporated is deletable.
_ALLOWLIST: dict[str, str] = {}

#: Decorators whose target is invoked by a framework rather than by name.
_CONVENTION_DECORATORS = {
    # pydantic / pydantic-settings
    "field_validator",
    "validator",
    "model_validator",
    "computed_field",
    # FastAPI routing (bare and namespaced forms both land via the last part)
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
    # Typer commands
    "command",
    "callback",
    # stdlib / house idioms
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
    """Every identifier token in the trees the gate considers live."""
    tokens: Counter[str] = Counter()
    files = [
        *(_REPO_ROOT / "ubt").rglob("*.py"),
        *(_REPO_ROOT / "tests").rglob("*.py"),
        *(_REPO_ROOT / "scripts").glob("*.py"),
    ]
    files.append(_REPO_ROOT / "pyproject.toml")  # console-script entry points
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
            # The def site itself is one token occurrence; a definition the
            # census never sees again is referenced by nothing anywhere.
            if tokens[name] <= 1:
                rel = path.relative_to(_REPO_ROOT)
                orphans.append(f"{rel}:{node.lineno} {type(node).__name__} {name}")
    assert orphans == [], (
        "definitions with zero references anywhere (delete, wire up, or add an "
        "entry to _ALLOWLIST with the reason it is retained):\n  " + "\n  ".join(orphans)
    )
