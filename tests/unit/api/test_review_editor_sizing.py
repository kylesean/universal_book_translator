"""Guard: the review editor must be the size of the text it replaces.

The console has no frontend test runner, so a layout regression ships silently.
Both sizes below were measured in a real browser against the reference job
(`pdf_main#b0005`, 351px of translation at 1280px).
"""

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
    """`rows={3}` showed three of a paragraph's eleven lines.

    Measured on `pdf_main#b0005` at 1280px: the translation occupies 351px and
    the editor replaced it with a 58.5px box whose own content was 351px tall,
    so reviewing a paragraph meant scrolling inside a sixth of it.
    """
    source = _source(_REVIEW)

    assert "AutoGrowTextarea" in source, "the editor must size itself to its text"
    assert "scrollHeight" in source
    # The old fixed-row editor must not come back. Match the JSX attribute, not
    # the prose: the comments legitimately quote the value this replaced.
    assert not re.search(r"^\s*rows=\{", source, re.MULTILINE), (
        "a fixed-row textarea clips the translation"
    )


def test_opening_the_editor_does_not_reflow_the_card_out_from_under_it() -> None:
    """At 1280px the editor never opened at all: 0 of 2 attempts.

    The card's onClick opens the inspector, which narrows the target column from
    446px to 286px. A double-click fires a click first, so the text moved before
    the second click landed; measured 0/2 at 1280 and 1366, 2/2 at 1600.
    """
    source = _source(_REVIEW)

    marker = "onDoubleClick={() =>"
    edits = [i for i in range(len(source)) if source.startswith(marker, i)]
    assert edits, "the editor's double-click affordance disappeared"
    # The target pane's own gesture must be isolated from the card's.
    assert "onClick={(event) => event.stopPropagation()}" in source


def test_the_review_labels_follow_the_selected_language() -> None:
    """The ribbon read "Terminology / Formula / Numeric" in every locale.

    Those names, the issue-kind chips, the block-type chip, the status badge,
    the page chip and the inspector's severity were all hardcoded English (or a
    raw engine enum upper-cased), so a Chinese operator got a页面 with an
    English ruler across the top of every card.
    """
    view = _source(_REVIEW)
    zh = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "zh.ts")
    en = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts")
    dictionary = _source(_REPO_ROOT / "web" / "src" / "i18n" / "types.ts")

    # The labels come from the dictionary now, through one helper each, so a new
    # call site cannot quietly reintroduce a literal.
    for key in ("issueKinds", "blockTypes", "blockStatuses", "mqmSeverities"):
        assert f"{key}: Record<string, string>" in dictionary, f"{key} is not typed"
        assert key in zh and key in en, f"{key} is missing from a locale"
    assert "issueLabel(t," in view
    assert "blockTypeLabel(t," in view
    assert "blockStatusLabel(t," in view
    assert "mqmSeverityLabel(t," in view

    # The hardcoded map is gone, and the raw-enum renders with it.
    assert "ISSUE_LABELS" not in view
    assert "{seg.block_type}" not in view
    assert "{inspectorSegment.block_type}" not in view
    assert "seg.status.toUpperCase()" not in view
    assert "> 0 issues" not in view
    # The engine identifiers the pane renders are all covered by the dictionary.
    for identifier in ("narrative", "list_item", "needs_human", "mtqe_passed"):
        assert identifier in zh or identifier in en
