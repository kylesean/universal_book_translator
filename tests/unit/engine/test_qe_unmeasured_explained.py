from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO = Path(__file__).resolve().parents[3]


def _snapshot_keys_in(path: Path) -> set[str]:
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
    keys = _snapshot_keys_in(_REPO / "ubt" / "core" / "engine" / "pipeline.py")
    assert "qe_engine" in keys, "the snapshot lost the engine label"
    assert "qe_judge_enabled" in keys, (
        "config_snapshot must record qe_judge_enabled: without it the console "
        "cannot explain a missing average QE or tell the two causes apart"
    )


def test_the_reporter_fallback_reads_the_same_keys() -> None:
    reporter = (_REPO / "ubt" / "core" / "engine" / "reporter.py").read_text(encoding="utf-8")
    pipeline_keys = _snapshot_keys_in(_REPO / "ubt" / "core" / "engine" / "pipeline.py")

    tree = ast.parse(reporter)
    fallback: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple):
            names = {str(e.value) for e in node.elts if isinstance(e, ast.Constant)}
            if "qe_engine" in names:
                fallback = names
    assert fallback, "the reporter's config_snapshot fallback key list was not found"

    for key in ("qe_engine", "qe_judge_enabled"):
        assert key in fallback, f"the fallback does not read {key}"
        assert key in pipeline_keys, f"the pipeline does not write {key}"
