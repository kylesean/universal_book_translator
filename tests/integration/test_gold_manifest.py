"""The semantic-gold manifest: ids and categories must stay well-formed.

The manifest drives the corpus sweep (``test_publish_corpus_sweep``) and the
baseline scorer (``scripts/eval_gold_corpus``). A typo'd id or an unknown
category silently shrinks coverage instead of failing, so this cheap guard runs
in the ``fast`` tier rather than beside the slow sweep.
"""

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
    """Every manifest case has a distinct id and a known category."""
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "duplicate case id"
    for case in cases:
        assert set(case["category"]) <= _KNOWN_CATEGORIES, f"{case['id']}: unknown category"


def test_every_manifest_case_declares_its_measured_shape() -> None:
    """A case must record what was measured, so the README table can be checked.

    The manifest's ``measured`` block is how the corpus README's page/bookmark
    counts stay honest; a case added without it would make the table silently
    incomplete.
    """
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        measured = case.get("measured") or {}
        assert "pages" in measured, f"{case['id']}: no measured page count"
        assert "bookmarks" in measured, f"{case['id']}: no measured bookmark count"
