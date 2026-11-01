from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "ubt"

_LIVE_IMPORT_ROOTS = ("ubt", "scripts", "tests")

_PRODUCTION_IMPORT_ROOTS = ("ubt", "scripts")

_DYNAMIC_ENTRY_POINTS: frozenset[str] = frozenset(
    {
        "ubt.adapters.pdf.vlm.drivers.unlimited_ocr2_worker",
    }
)

_PRODUCTION_ORPHAN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ubt.adapters.pdf.cels_formula",
    }
)

_ENTRY_MODULE_SUFFIXES = ("__init__", "__main__")


def _module_name(path: Path) -> str:
    parts = list(path.with_suffix("").relative_to(_REPO_ROOT).parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_ubt_modules(path: Path, known: set[str]) -> set[str]:
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
    live = set(imported)
    for module in list(imported):
        parts = module.split(".")
        for depth in range(1, len(parts)):
            live.add(".".join(parts[:depth]))
    return live


def _importers(root: str, package_files: list[Path]) -> list[Path]:
    files: list[Path] = list(package_files) if root == "ubt" else []
    base = _REPO_ROOT / root
    if base.exists():
        files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def test_no_orphaned_modules() -> None:
    package_files = [p for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    module_of = {p: _module_name(p) for p in package_files}
    known = set(module_of.values())

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
