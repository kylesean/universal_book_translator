"""Guard the ``fast``/``slow`` marker contract for the test tree.

``fast`` is the pre-push inner-loop contract (``pytest -m fast``), and the
pre-push hook runs it with ``pass_filenames: false`` -- the whole tier on every
push. ``slow`` means a heavy subprocess (typst/pandoc/pdftoppm). Two failures
are possible, and both are silent:

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

Both checks are fast and pure-AST: no imports are executed, so they cannot be
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
#: Marks that put a test in some tier, for the "is it marked at all" check.
_TIER_MARKS = _CI_TIER_MARKS | _ADDITIVE_MARKS


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
