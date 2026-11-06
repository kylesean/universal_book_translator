from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_INNER_LOOP_MARK = "fast"
_HEAVY_MARK = "slow"
_CI_TIER_MARKS = frozenset({_INNER_LOOP_MARK, _HEAVY_MARK})
_ADDITIVE_MARKS = frozenset({"network", "legacy_drift"})


def _mark_names(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.List, ast.Tuple)):
            for item in sub.elts:
                found |= _mark_names(item)
            continue
        target = sub.func if isinstance(sub, ast.Call) else sub
        if not isinstance(target, ast.Attribute):
            continue
        owner = target.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "mark"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "pytest"
        ):
            found.add(target.attr)
    return found


def _module_marks(tree: ast.Module) -> set[str]:
    marks: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets
        ):
            marks |= _mark_names(stmt.value)
    return marks


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def test_no_test_carries_both_fast_and_slow() -> None:
    offenders: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        inherited = _module_marks(tree)
        for func in _test_functions(tree):
            effective = inherited | _mark_names(ast.Module(body=[func], type_ignores=[]))
            if _INNER_LOOP_MARK in effective and _HEAVY_MARK in effective:
                rel = path.relative_to(_TESTS_ROOT.parent)
                offenders.append(f"{rel}::{func.name}")
    assert offenders == [], (
        "a test marked both `fast` and `slow` is pulled into the pre-push `-m fast` "
        "tier by the module-level mark, dragging a heavy subprocess loop into the "
        "inner loop; drop the inherited `fast` (per-test marks, not `pytestmark`):\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_every_test_carries_a_tier_mark() -> None:
    offenders: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        inherited = _module_marks(tree)
        for func in _test_functions(tree):
            effective = inherited | _mark_names(ast.Module(body=[func], type_ignores=[]))
            if not (effective & _CI_TIER_MARKS):
                rel = path.relative_to(_TESTS_ROOT.parent)
                offenders.append(f"{rel}::{func.name}")
    assert offenders == [], (
        "test(s) carry no CI tier mark, so no gate runs them (CI runs only "
        "`-m fast` and `-m slow`); `network`/`legacy_drift` are additive, not "
        "tiers. Add `pytestmark = pytest.mark.fast` to the module if the tests "
        "are offline and cheap, or `@pytest.mark.slow` per test if not:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_legacy_drift_never_joins_the_inner_loop() -> None:
    offenders: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        inherited = _module_marks(tree)
        for func in _test_functions(tree):
            effective = inherited | _mark_names(ast.Module(body=[func], type_ignores=[]))
            if "legacy_drift" in effective and _INNER_LOOP_MARK in effective:
                rel = path.relative_to(_TESTS_ROOT.parent)
                offenders.append(f"{rel}::{func.name}")
    assert offenders == [], (
        "test(s) are both `legacy_drift` and `fast`: a quarantined check cannot "
        "live in the inner loop it is quarantined from; use `slow`:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


_FAST_SUBPROCESS_ALLOWLIST: dict[str, str] = {
    "tests/unit/adapters/test_visual_gate.py": "one-page typst render for the gate's type tier",
    "tests/unit/adapters/test_facsimile_table_preservation.py": (
        "single-page facsimile render + pdftotext, skipped when tools are absent"
    ),
    "tests/unit/render/test_academic_template.py": "single-snippet typst compile",
    "tests/unit/render/test_document_renderer.py": "single-snippet typst compile",
    "tests/unit/render/test_math_ast_transpiler.py": "single-snippet typst compile",
    "tests/unit/render/test_math_transpiler.py": "single-snippet typst compile",
    "tests/unit/render/test_typst_lang.py": "single-snippet typst compile",
    "tests/unit/render/test_typst_list_items.py": (
        "single-snippet typst compile + pdftotext, skipped when typst is absent"
    ),
    "tests/unit/core/test_import_isolation.py": "subprocess imports the package to prove isolation",
    "tests/unit/test_writer_lock.py": "subprocess holds the lock to prove cross-process exclusion",
    "tests/unit/test_config_construction.py": "fresh interpreter per case to observe a deadlock",
    "tests/unit/cli/test_console_dev.py": "throwaway parent+child to prove SIGTERM reaches the child",
    "tests/unit/test_docker_entrypoint.py": (
        "one bash invocation of the container entrypoint to assert its fail-closed exit"
    ),
}

_SUBPROCESS_CALLS = frozenset({"run", "Popen", "check_output", "check_call", "call"})


def _subprocess_calls(tree: ast.Module) -> int:
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _SUBPROCESS_CALLS
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        ):
            total += 1
    return total


def test_a_fast_module_that_shells_out_is_declared() -> None:
    undeclared: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _INNER_LOOP_MARK not in _module_marks(tree) and not any(
            _INNER_LOOP_MARK
            in (_module_marks(tree) | _mark_names(ast.Module(body=[func], type_ignores=[])))
            for func in _test_functions(tree)
        ):
            continue
        if _subprocess_calls(tree) == 0:
            continue
        rel = str(path.relative_to(_TESTS_ROOT.parent))
        if rel not in _FAST_SUBPROCESS_ALLOWLIST:
            undeclared.append(rel)

    assert undeclared == [], (
        "fast-marked test module(s) spawn a subprocess without declaring it; a "
        "`fast` module that shells out drags the process into the inner loop. "
        "Either move the spawning tests to `slow`, or add the module to "
        "_FAST_SUBPROCESS_ALLOWLIST with the reason it is cheap enough:\n"
        + "\n".join(f"  {o}" for o in undeclared)
    )


def test_the_subprocess_allowlist_has_no_stale_entries() -> None:
    stale: list[str] = []
    for rel in _FAST_SUBPROCESS_ALLOWLIST:
        path = _TESTS_ROOT.parent / rel
        if not path.is_file():
            stale.append(f"{rel} (missing)")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        fast_here = _INNER_LOOP_MARK in _module_marks(tree) or any(
            _INNER_LOOP_MARK
            in (_module_marks(tree) | _mark_names(ast.Module(body=[func], type_ignores=[])))
            for func in _test_functions(tree)
        )
        if _subprocess_calls(tree) == 0 or not fast_here:
            stale.append(rel)

    assert stale == [], (
        "allowlist entries that no longer describe reality; drop them:\n"
        + "\n".join(f"  {s}" for s in stale)
    )
