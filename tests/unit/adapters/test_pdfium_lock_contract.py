"""Call-site contract for libpdfium: every native call holds ``PDFIUM_LOCK``.

A file-level check is false-green: ``visual_gate.py`` imports the gate module and
still called ``pdfium.PdfDocument`` directly, which is exactly the race that
crashes the process. This test walks the AST and inspects each *call*, requiring
it to sit inside a ``@pdfium_serialized`` function or a ``with PDFIUM_LOCK``
block.

Tests are allowed to call pdfium directly (each test is single-threaded and owns
its fixture document); they opt out with a module-level
``# ubt-pdfium-allow: <reason>`` marker so the exemption is explicit and
greppable.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOW_MARKER = "# ubt-pdfium-allow:"


def _loaded_aliases(tree: ast.Module) -> set[str]:
    """Names that resolve to a pypdfium2 module or symbol in this file."""
    aliases: set[str] = {"pypdfium2"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "pypdfium2" or item.name.startswith("pypdfium2."):
                    aliases.add(item.asname or item.name.split(".")[0])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "pypdfium2" or node.module.startswith("pypdfium2."))
        ):
            for item in node.names:
                aliases.add(item.asname or item.name)
    return aliases


def _dotted(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_guarded_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _dotted(decorator).split(".")[-1] == "pdfium_serialized"
        for decorator in node.decorator_list
    )


def _is_lock_context(node: ast.expr) -> bool:
    return _dotted(node).split(".")[-1] == "PDFIUM_LOCK"


class _CallCollector(ast.NodeVisitor):
    def __init__(self, aliases: set[str]) -> None:
        self._aliases = aliases
        self._depth = 0
        self.unprotected: list[tuple[int, str]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        guarded = _is_guarded_decorator(node)
        if guarded:
            self._depth += 1
        self.generic_visit(node)
        if guarded:
            self._depth -= 1

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        locked = any(_is_lock_context(item.context_expr) for item in node.items)
        if locked:
            self._depth += 1
        self.generic_visit(node)
        if locked:
            self._depth -= 1

    visit_With = _visit_with
    visit_AsyncWith = _visit_with

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        root = name.split(".")[0]
        if root in self._aliases and self._depth == 0:
            self.unprotected.append((node.lineno, name))
        self.generic_visit(node)


def _scan(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    if any(ALLOW_MARKER in line for line in source.splitlines()[:20]):
        return []
    tree = ast.parse(source)
    collector = _CallCollector(_loaded_aliases(tree))
    collector.visit(tree)
    display = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    return [f"{display}:{line}: {name}" for line, name in collector.unprotected]


def test_every_pdfium_call_site_is_serialized() -> None:
    offenders: list[str] = []
    for base in ("ubt", "tests"):
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            if path.name == "pdfium_gate.py":
                # The gate owns the lock; its own acquisition sites are the
                # implementation of the contract, not a consumer of it.
                continue
            offenders.extend(_scan(path))
    assert not offenders, "pdfium calls outside @pdfium_serialized / PDFIUM_LOCK:\n" + "\n".join(
        offenders
    )


def test_scan_is_call_level_not_file_level(tmp_path: Path) -> None:
    """A module that imports the gate elsewhere must still be flagged."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import pypdfium2 as pdfium\n\n"
        "from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK  # imports the gate\n\n"
        "def leaky():\n"
        "    return pdfium.PdfDocument('x')\n",
        encoding="utf-8",
    )
    assert _scan(bad), "an unprotected call in a gate-importing file went unnoticed"

    good = tmp_path / "good.py"
    good.write_text(
        "import pypdfium2 as pdfium\n\n"
        "from ubt.adapters.pdf.pdfium_gate import pdfium_serialized\n\n"
        "@pdfium_serialized\n"
        "def clean():\n"
        "    return pdfium.PdfDocument('x')\n",
        encoding="utf-8",
    )
    assert not _scan(good)


def test_pdfium_serialized_is_reentrant() -> None:
    """Nested decorated calls are normal (reader -> textgeom); they must not deadlock."""
    state = {"inner": False}

    @pdfium_serialized
    def inner() -> bool:
        state["inner"] = True
        return True

    @pdfium_serialized
    def outer() -> bool:
        return inner()

    result: dict[str, bool] = {}
    worker = threading.Thread(target=lambda: result.setdefault("ok", outer()), daemon=True)
    worker.start()
    # A plain Lock would leave the second acquisition blocked forever.
    worker.join(timeout=2.0)
    assert not worker.is_alive(), "pdfium_serialized deadlocked on a nested call"
    assert result.get("ok") is True
    assert state["inner"] is True
