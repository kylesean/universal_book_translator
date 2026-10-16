"""Guard against orphaned modules: a source file no production code imports.

The 2026-10 audit found ~2,500 lines of dead modules (a retired display-math
converter, an unwired witness cache, an unused advisor, an orphaned asset
extractor) that survived a large refactor because nothing checked for them:
lint, types and tests all pass on a module nobody calls.

Two checks run here, and the second is the one that catches the class the first
structurally cannot:

1. **Reachability** -- a module nothing imports at all (production, scripts or
   tests) is dead. ``scripts/`` counts as an importer because a slow-tier corpus
   harness runs as a subprocess and is not collected by pytest, so a module
   reached only through one of them is live but invisible to the suite alone.

2. **Production reachability** -- a module that *only* tests import has no
   production consumer. This is the failure the first check hides: the gate
   originally counted ``tests/`` as an importer, so a module kept alive solely
   by its own unit test read as live forever. The retired per-box breakage
   solver and the math compile probe were exactly this -- both orphaned when
   the overlay engine was deleted, both with a passing test file, neither
   reached by any shipped code path. A module in this state is dead code with a
   guard rail; it must be deleted or wired up, and either way listed in
   ``_PRODUCTION_ORPHAN_ALLOWLIST`` with the reason it is retained.

Both checks are pure-AST: no imports are executed, so they cannot be fooled by
import side effects and cost milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"

#: Roots whose imports keep a module live for the *reachability* check.
#: ``scripts/`` matters: a slow-tier corpus harness runs as a subprocess and is
#: *not* collected by pytest, so a module reached only through one of them is
#: live but invisible to the test suite alone. Missing this root once deleted a
#: live module. ``tests`` is included here on purpose -- a module imported by a
#: test is reachable (check 1 passes) but may still be a production orphan
#: (check 2).
_LIVE_IMPORT_ROOTS = ("ubt", "scripts", "tests")

#: Roots whose imports count as a *production* consumer for check 2. A module
#: nothing here imports is dead shipped code even if a test imports it.
_PRODUCTION_IMPORT_ROOTS = ("ubt", "scripts")

#: Modules reached only through a string, never an import: subprocess entry
#: points launched as ``python -m ...``. Each must name why it cannot be
#: imported statically.
_DYNAMIC_ENTRY_POINTS: frozenset[str] = frozenset(
    {
        # Launched as ``python -m ubt.adapters.pdf.vlm.drivers.deepseek_worker``.
        "ubt.adapters.pdf.vlm.drivers.deepseek_worker",
    }
)

#: Production orphans (no importer outside ``tests/``) that are deliberately
#: retained. Every entry must say what keeps it, so the list cannot become a
#: silent graveyard. Empty is the healthy state.
_PRODUCTION_ORPHAN_ALLOWLIST: frozenset[str] = frozenset()

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


def _with_parents(imported: set[str]) -> set[str]:
    """A submodule import implies its parent packages are reachable too."""
    live = set(imported)
    for module in list(imported):
        parts = module.split(".")
        for depth in range(1, len(parts)):
            live.add(".".join(parts[:depth]))
    return live


def _importers(root: str, package_files: list[Path]) -> list[Path]:
    """Every ``.py`` file under ``root`` (plus the package itself for ``ubt``)."""
    files: list[Path] = list(package_files) if root == "ubt" else []
    base = _REPO_ROOT / root
    if base.exists():
        files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def test_no_orphaned_modules() -> None:
    """Every ``ubt`` module is imported by another, or allowlisted as dynamic."""
    package_files = [p for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    module_of = {p: _module_name(p) for p in package_files}
    known = set(module_of.values())

    # Every import site that can keep a module alive: the package, the scripts
    # run as subprocesses, and the tests themselves.
    importer_files: list[Path] = []
    for root in _LIVE_IMPORT_ROOTS:
        importer_files.extend(_importers(root, package_files))

    imported: set[str] = set()
    for path in importer_files:
        imported |= _imported_ubt_modules(path, known)
    live = _with_parents(imported)

    orphans: list[str] = []
    for module in module_of.values():
        if module.endswith(_ENTRY_MODULE_SUFFIXES):
            continue
        if module in live or module in _DYNAMIC_ENTRY_POINTS:
            continue
        orphans.append(module)

    assert not orphans, (
        "orphaned module(s) no module imports -- delete them or add a "
        f"documented entry to _DYNAMIC_ENTRY_POINTS: {sorted(orphans)}"
    )


def test_no_test_only_modules() -> None:
    """No module is kept alive *only* by the test suite.

    A module whose sole importer is a test has no production consumer: it is
    dead shipped code with a guard rail. Counting ``tests/`` as an importer (as
    the reachability check must, to avoid deleting live script-only modules)
    hides exactly this, so the production graph is checked on its own.
    """
    package_files = [p for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    known = {_module_name(p) for p in package_files}

    production_imported: set[str] = set()
    for root in _PRODUCTION_IMPORT_ROOTS:
        for path in _importers(root, package_files):
            production_imported |= _imported_ubt_modules(path, known)
    production_live = _with_parents(production_imported)

    test_only: list[str] = []
    for module in sorted(known):
        if module.endswith(_ENTRY_MODULE_SUFFIXES):
            continue
        if module in production_live or module in _DYNAMIC_ENTRY_POINTS:
            continue
        if module in _PRODUCTION_ORPHAN_ALLOWLIST:
            continue
        test_only.append(module)

    assert not test_only, (
        "module(s) no production code imports (only tests do) -- wire them up, "
        "delete them, or add a reasoned entry to _PRODUCTION_ORPHAN_ALLOWLIST: "
        f"{test_only}"
    )


def test_production_orphan_allowlist_has_no_stale_entries() -> None:
    """Every allowlist entry must still be a production orphan.

    Otherwise a module wired back up leaves a stale exemption that would hide
    its next orphaning.
    """
    package_files = [p for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    known = {_module_name(p) for p in package_files}
    production_imported: set[str] = set()
    for root in _PRODUCTION_IMPORT_ROOTS:
        for path in _importers(root, package_files):
            production_imported |= _imported_ubt_modules(path, known)
    production_live = _with_parents(production_imported)

    stale = sorted(
        entry
        for entry in _PRODUCTION_ORPHAN_ALLOWLIST
        if entry in production_live or entry not in known
    )
    assert not stale, (
        "stale _PRODUCTION_ORPHAN_ALLOWLIST entries (now imported by production, "
        f"or no longer a module) -- remove them: {stale}"
    )
