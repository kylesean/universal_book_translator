"""When a report has no average QE, it must also carry the reason.

The quality-gate UI rendered one bare word — "QE not measured" — for a state
that has two very different causes:

* the run used the **heuristic** engine, which emits discrete defect bands and
  never a continuous score, so by construction there is nothing to average; or
* the LLM judge **was** enabled, but no block reached the scorer (everything
  took the fast path, hit translation memory, or kept its source text).

The report already records ``qe_engine``; it did not record the judge flag, so
the console could not tell the two apart and could not say what would fix it.
These tests pin that the snapshot carries both, on the pipeline's own path and
on the reporter's fallback path (a ledger written before the field existed).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO = Path(__file__).resolve().parents[3]


def _snapshot_keys_in(path: Path) -> set[str]:
    """String keys of every ``config_snapshot = {…}`` literal in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Attribute) and target.attr == "config_snapshot"):
                continue
            if isinstance(node.value, ast.Dict):
                keys |= {
                    str(k.value)
                    for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
    return keys


def test_the_pipeline_snapshot_records_the_judge_switch() -> None:
    """``qe_engine`` alone cannot explain a missing average."""
    keys = _snapshot_keys_in(_REPO / "ubt" / "core" / "engine" / "pipeline.py")
    assert "qe_engine" in keys, "the snapshot lost the engine label"
    assert "qe_judge_enabled" in keys, (
        "config_snapshot must record qe_judge_enabled: without it the console "
        "cannot explain a missing average QE or tell the two causes apart"
    )


def test_the_reporter_fallback_reads_the_same_keys() -> None:
    """A ledger read through the metadata fallback must carry them too.

    The reporter only falls back to ``manifest.metadata`` for old ledgers, which
    is exactly the population a diagnostic screen is most likely to be looking
    at. A key present in one path and absent from the other makes the
    explanation appear or vanish depending on the report's age.
    """
    reporter = (_REPO / "ubt" / "core" / "engine" / "reporter.py").read_text(encoding="utf-8")
    pipeline_keys = _snapshot_keys_in(_REPO / "ubt" / "core" / "engine" / "pipeline.py")

    # The fallback is a tuple of names inside ``for k in (...):``.
    tree = ast.parse(reporter)
    fallback: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple):
            names = {str(e.value) for e in node.elts if isinstance(e, ast.Constant)}
            if "qe_engine" in names:
                fallback = names
    assert fallback, "the reporter's config_snapshot fallback key list was not found"

    # ``qe_threshold``/``max_repair_rounds`` are informational; the two that
    # explain a missing average must be present on both paths.
    for key in ("qe_engine", "qe_judge_enabled"):
        assert key in fallback, f"the fallback does not read {key}"
        assert key in pipeline_keys, f"the pipeline does not write {key}"
