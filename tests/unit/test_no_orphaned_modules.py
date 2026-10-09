"""Guard against orphaned modules: a source file no other module imports.

The 2026-10 audit found ~2,500 lines of dead modules (a retired display-math
converter, an unwired witness cache, an unused advisor, an orphaned asset
extractor) that survived a large refactor because nothing checked for them:
lint, types and tests all pass on a module nobody calls.

This test builds the intra-package import graph and fails on a module that no
other module imports. It is deliberately conservative -- a module counts as
live if anything imports it *or any submodule of it*, and the allowlist names
the few modules reached only through a string (``python -m`` subprocess
entry points), which no static import can see.

It is a fast, pure-AST test: no imports are executed, so it cannot be fooled by
import side effects and costs milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"

#: Roots whose imports keep a module live. ``scripts/`` matters: a slow-tier
#: corpus harness runs as a subprocess and is *not* collected by pytest, so a
#: module reached only through one of them is live but invisible to the test
#: suite alone. Missing this root once deleted a live module.
_LIVE_IMPORT_ROOTS = ("ubt", "scripts", "tests")

#: Modules reached only through a string, never an import: subprocess entry
#: points launched as ``python -m ...`` or as a script path. Each must name why
#: it cannot be imported statically.
_DYNAMIC_ENTRY_POINTS: frozenset[str] = frozenset(
    {
        # Launched by ``SubprocessQERunner`` as ``python <script> --serve``.
        "ubt.core.qe.comet_score_ipc",
        # Launched as ``python -m ubt.adapters.pdf.vlm.drivers.deepseek_worker``.
        "ubt.adapters.pdf.vlm.drivers.deepseek_worker",
    }
)

#: Package ``__init__`` files and ``__main__`` are entry points by design.
_ENTRY_MODULE_SUFFIXES = ("__init__", "__main__")


def _module_name(path: Path) -> str:
    parts = list(path.with_suffix("").relative_to(_REPO_ROOT).parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_ubt_modules(path: Path, known: set[str]) -> set[str]:
    """Every ``ubt.*`` module this file imports, at module or function level.

    ``from ubt.pkg import sub`` may name either an object or a submodule; it is
    resolved as a module when ``ubt.pkg.sub`` is one of ``known``, so a
    ``from ubt.adapters.pdf import pdf_struct`` marks ``pdf_struct`` live.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ubt"):
                found.add(node.module)
                for alias in node.names:
                    candidate = f"{node.module}.{alias.name}"
                    if candidate in known:
                        found.add(candidate)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ubt"):
                    found.add(alias.name)
    return found


def test_no_orphaned_modules() -> None:
    """Every ``ubt`` module is imported by another, or allowlisted as dynamic."""
    package_files = [p for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    module_of = {p: _module_name(p) for p in package_files}
    known = set(module_of.values())

    # Every import site that can keep a module alive: the package, the scripts
    # run as subprocesses, and the tests themselves.
    importer_files: list[Path] = list(package_files)
    for root in _LIVE_IMPORT_ROOTS:
        importer_files.extend(
            p for p in (_REPO_ROOT / root).rglob("*.py") if "__pycache__" not in p.parts
        )

    imported: set[str] = set()
    for path in importer_files:
        imported |= _imported_ubt_modules(path, known)

    # A submodule import implies its parent packages are reachable too.
    live = set(imported)
    for module in list(imported):
        parts = module.split(".")
        for depth in range(1, len(parts)):
            live.add(".".join(parts[:depth]))

    orphans: list[str] = []
    for module in module_of.values():
        if module.endswith(_ENTRY_MODULE_SUFFIXES):
            continue
        if module in live or module in _DYNAMIC_ENTRY_POINTS:
            continue
        orphans.append(module)

    assert not orphans, (
        "orphaned module(s) no other module imports -- delete them or add a "
        f"documented entry to _DYNAMIC_ENTRY_POINTS: {sorted(orphans)}"
    )
