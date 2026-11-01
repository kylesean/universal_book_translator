from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REVIEW = _REPO_ROOT / "web" / "src" / "views" / "review" / "ReviewWorkbench.tsx"


def _source(path: Path) -> str:
    if not path.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    return path.read_text(encoding="utf-8")


def test_the_review_editor_is_as_tall_as_the_text_it_replaces() -> None:
    source = _source(_REVIEW)

    assert "AutoGrowTextarea" in source, "the editor must size itself to its text"
    assert "scrollHeight" in source
    assert not re.search(r"^\s*rows=\{", source, re.MULTILINE), (
        "a fixed-row textarea clips the translation"
    )


def test_opening_the_editor_does_not_reflow_the_card_out_from_under_it() -> None:
    source = _source(_REVIEW)

    marker = "onDoubleClick={() =>"
    edits = [i for i in range(len(source)) if source.startswith(marker, i)]
    assert edits, "the editor's double-click affordance disappeared"
    assert "onClick={(event) => event.stopPropagation()}" in source


def test_the_review_labels_follow_the_selected_language() -> None:
    view = _source(_REVIEW)
    zh = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "zh.ts")
    en = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts")
    dictionary = _source(_REPO_ROOT / "web" / "src" / "i18n" / "types.ts")

    for key in ("issueKinds", "blockTypes", "blockStatuses", "mqmSeverities"):
        assert f"{key}: Record<string, string>" in dictionary, f"{key} is not typed"
        assert key in zh and key in en, f"{key} is missing from a locale"
    assert "issueLabel(t," in view
    assert "blockTypeLabel(t," in view
    assert "blockStatusLabel(t," in view
    assert "mqmSeverityLabel(t," in view

    assert "ISSUE_LABELS" not in view
    assert "{seg.block_type}" not in view
    assert "{inspectorSegment.block_type}" not in view
    assert "seg.status.toUpperCase()" not in view
    assert "> 0 issues" not in view
    for identifier in ("narrative", "list_item", "needs_human", "mtqe_passed"):
        assert identifier in zh or identifier in en
