from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "semantic_gold" / "manifest.json"

_KNOWN_CATEGORIES = frozenset(
    {
        "formula-dense",
        "complex-tables",
        "two-column-figures",
        "long-mixed",
        "toc-outline",
        "no-outline",
    }
)


def test_the_corpus_manifest_names_real_documents() -> None:
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "duplicate case id"
    for case in cases:
        assert set(case["category"]) <= _KNOWN_CATEGORIES, f"{case['id']}: unknown category"


def test_every_manifest_case_declares_its_measured_shape() -> None:
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        measured = case.get("measured") or {}
        assert "pages" in measured, f"{case['id']}: no measured page count"
        assert "bookmarks" in measured, f"{case['id']}: no measured bookmark count"
