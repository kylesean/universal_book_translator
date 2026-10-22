"""Guard the ``fast``/``slow`` marker contract for the test tree.

``fast`` is the pre-push inner-loop contract (``pytest -m fast``), and the
pre-push hook runs it with ``pass_filenames: false`` -- the whole tier on every
push. ``slow`` means a heavy subprocess (typst/pandoc/pdftoppm). Three failures
are possible, and all are silent:

1. **Both marks on one test.** A contradiction: ``-m fast`` pulls a
   multi-second subprocess loop into the tier that is supposed to be seconds.
   The 2026-10 regression: ``test_overlay_text_math.py`` set a module-level
   ``pytestmark = pytest.mark.fast`` and then marked three typst-probe tests
   ``slow``. pytest ANDs a module mark onto every test in the file, so all
   three joined the fast tier: 196 per-body ``typst`` invocations, ~115s of a
   ~50s gate.

2. **Neither mark.** The ``addopts`` filter is ``-m 'not slow and not
   legacy_drift'``, but both CI steps pass an explicit ``-m``, which *replaces*
   it: CI runs ``-m fast`` and ``-m slow`` and nothing else. An unmarked test is
   therefore in neither tier and is executed by no gate at all -- it still
   passes when run by hand, so nothing goes red; it just stops protecting the
   tree. Twelve files (77 tests) sat this way after the unit-suite rebuild,
   including the ``tests/integration/test_pipeline_smoke.py`` end-to-end
   rehearsal the strategy doc described as the default tier.

3. **A ``fast`` module that shells out.** The first two checks reason about
   marks only, so a module that is simply *marked* ``fast`` while spawning
   typst/pandoc keeps the subprocess in the inner loop with no contradiction to
   detect. Check 3 rejects a ``fast``-marked module that calls ``subprocess``
   directly, unless it names the process it needs in a ``slow``-tier comment.

All checks are fast and pure-AST: no imports are executed, so they cannot be
fooled by import side effects and cost milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_INNER_LOOP_MARK = "fast"
_HEAVY_MARK = "slow"
#: The two tiers a gate actually runs. CI's pytest steps pass an explicit ``-m``
#: (``-m fast``, ``-m slow``), and ``-m`` *replaces* the ``addopts`` filter, so
#: these are the only marks that put a test in front of a gate.
_CI_TIER_MARKS = frozenset({_INNER_LOOP_MARK, _HEAVY_MARK})
#: Marks that modify a test without choosing its tier. ``network`` lifts the
#: offline socket block; ``legacy_drift`` says "not the inner loop". Neither is
#: a tier: a test carrying only one of them is executed by *no* gate, which is
#: exactly the hole this module exists to close. They are therefore additive --
#: a test must still carry ``fast`` or ``slow`` to be run at all.
_ADDITIVE_MARKS = frozenset({"network", "legacy_drift"})


def _mark_names(node: ast.AST) -> set[str]:
    """Names of every ``pytest.mark.<name>`` reachable from a decorator/assignment.

    Handles the three spellings pytest accepts for a mark: a bare
    ``@pytest.mark.fast``, a call ``@pytest.mark.fast()``, and the module-level
    ``pytestmark = pytest.mark.fast`` / ``pytestmark = [pytest.mark.fast, ...]``
    (where the value is a bare mark, a call, or a list of either).
    """
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.List, ast.Tuple)):
            for item in sub.elts:
                found |= _mark_names(item)
            continue
        # Unwrap ``pytest.mark.fast(...)`` -> ``pytest.mark.fast``.
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
    """Every test must be in a gate's tier, not merely carry *a* mark.

    Both CI steps pass an explicit ``-m``, which replaces the ``addopts``
    filter rather than narrowing it, so the tiers CI actually runs are exactly
    ``fast`` and ``slow``. ``network`` and ``legacy_drift`` are additive
    modifiers, not tiers: a test marked only ``network`` (to lift the socket
    block) or only ``legacy_drift`` (to leave the inner loop) is executed by no
    gate at all -- it passes when run by hand and silently stops protecting the
    tree. Every test must therefore carry ``fast`` or ``slow``.
    """
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
    """A quarantined test must be ``slow``, never ``fast``.

    ``legacy_drift`` means "kept out of the inner loop"; ``fast`` is the inner
    loop. A test carrying both is a contradiction -- it would run in the very
    tier it is quarantined from, which is the regression this module's own
    docstring records.
    """
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


#: A ``fast`` module may shell out only to a process it names here, with the
#: reason. This is the escape hatch for the tier's deliberate exceptions: a
#: single-snippet typst compile is cheap (well under a second) and is exactly
#: the render contract the tier exists to protect, so it belongs in `fast` as
#: long as CI installs typst first (`.github/workflows/ci.yml` does).
_FAST_SUBPROCESS_ALLOWLIST: dict[str, str] = {
    "tests/unit/adapters/test_visual_gate.py": "one-page typst render for the gate's type tier",
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
    # A hang cannot be observed in-process (the wedged thread keeps the lock
    # forever), so each case runs in a throwaway interpreter with a hard
    # timeout. Three short ``python -c`` starts, ~0.3s total.
    "tests/unit/test_config_construction.py": "fresh interpreter per case to observe a deadlock",
}

_SUBPROCESS_CALLS = frozenset({"run", "Popen", "check_output", "check_call", "call"})


def _subprocess_calls(tree: ast.Module) -> int:
    """Count ``subprocess.<spawn>(...)`` calls (not mentions in strings)."""
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
    """A `fast` module spawning a subprocess must be a known, justified exception.

    Checks 1 and 2 reason about *marks*, so a module that is simply marked
    `fast` while spawning typst/pandoc keeps the subprocess in the inner loop
    with no contradiction to detect -- which is how 196 typst invocations leaked
    into a ~50s gate. The allowlist makes each exception an explicit, reviewable
    decision with a reason instead of an accident of where the mark landed.
    """
    undeclared: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # A module is in the inner loop if *any* of its tests is. Checking only
        # the module-level ``pytestmark`` missed the modules that mark tests
        # individually on purpose — ``test_typst_symbols`` does exactly that so
        # its three ``slow`` probe tests are not dragged into ``-m fast`` — and
        # those were the ones shelling out.
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
    """An allowlist entry that no longer spawns anything is dead weight."""
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
