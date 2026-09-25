"""Guard: never swap or mutate ``sys.modules`` to stub a heavy dependency.

Root cause of the historical exit-139 SIGSEGV: ``test_docling_mock_conversion``
used ``patch.dict(sys.modules, {...})`` to fake the ``docling`` package. On
exit, patch.dict restored the real torch-backed ``docling`` stack, and pytest's
teardown ``gc_collect_harder()`` then traversed a dangling ``PyMethodObject``
that the module swap had left behind — crashing the interpreter *after* a green
run (exit 139, no traceback). The fix is a lazy import seam
(``_docling_symbols``) that tests patch directly.

The same class of mutation came back on 2026-09-21 as exit-134 SIGABRT, from a
direction no test authored: ``--cov=ubt.adapters.pdf.typst_reconstructor`` makes
coverage import that dotted source just to locate it, then remove everything the
import touched (``coverage.misc.sys_modules_saved``). ``ubt/adapters/pdf/__init__``
pulls pikepdf in on the way, so ``pikepdf._core`` — a nanobind extension whose
type registry is process-global and survives the removal — is *re-executed* by
the next real import, and nanobind fail-fasts with ``abort()``. ``del sys.modules
["pikepdf"]`` reproduces it with no pytest and no coverage involved. A module
that owns native state is not a dict entry: both failures were only diagnosable
from a core dump, and neither produced a traceback or a test result.

This guard keeps the authorable half of that pattern from returning. It flags,
anywhere under ``ubt/`` or ``tests/``:

- ``patch.dict(sys.modules, ...)`` / ``patch.object(sys, "modules", ...)``;
- direct mutation: ``sys.modules[...] = ...``, ``del sys.modules[...]``,
  ``sys.modules.pop/clear/update/setdefault(...)``.

``monkeypatch.setitem`` is deliberately not flagged: it puts the previous value
back (or deletes only the entry it added), so it cannot leave a real module
looking un-imported, and it is how tests stub packages that are not installed at
all (``rapidocr``, ``pdf_oxide``).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCAN_DIRS = ("ubt", "tests")
_MUTATORS = frozenset({"pop", "clear", "update", "setdefault"})


def _is_sys_modules(node: ast.expr) -> bool:
    """True for the ``sys.modules`` attribute expression itself."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _mocked_by_patch(node: ast.Call) -> bool:
    """True for the two mock forms that restore the real module on teardown."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr in {"dict", "object"}):
        return False
    if func.attr == "dict":
        return bool(node.args) and _is_sys_modules(node.args[0])
    return (
        len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "sys"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "modules"
    )


def _mutates_sys_modules(node: ast.stmt | ast.expr) -> bool:
    if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
        targets: list[ast.expr] = node.targets if isinstance(node, ast.Assign) else [node.target]
    elif isinstance(node, ast.Delete):
        targets = node.targets
    elif isinstance(node, ast.Call):
        if _mocked_by_patch(node):
            return True
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr in _MUTATORS
            and _is_sys_modules(func.value)
        )
    else:
        return False
    return any(
        isinstance(target, ast.Subscript) and _is_sys_modules(target.value) for target in targets
    )


def test_no_sys_modules_swapping_or_mutation() -> None:
    """Exit-139 / exit-134 regression: add a lazy import seam in the production
    module and patch that callable instead."""
    violations: list[str] = []
    for dirname in SCAN_DIRS:
        for py_file in sorted((REPO_ROOT / dirname).rglob("*.py")):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - a file this suite cannot parse cannot run
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.stmt | ast.expr) and _mutates_sys_modules(node):
                    violations.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not violations, (
        "sys.modules must not be swapped or mutated — removing or replacing an "
        "entry can re-trigger native module initialisation, which aborts the "
        "interpreter instead of raising (see this file's docstring). Stub a "
        "lazy import seam in the production module instead. Found: " + ", ".join(violations)
    )
