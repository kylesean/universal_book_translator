"""The review workbench must be able to name every defect the gate rejects for.

``_ISSUE_MARKERS`` (``ubt/api/review.py``) is a *view* over the engine's defect
vocabulary: it groups a flag into the issue kind the reviewer filters by. When
the two drifted, a block the quality gate refused arrived in the workbench with
no issue type at all — the reviewer saw a segment flagged as broken and nothing
saying why. Measured before this guard: nine of the 35 structural markers had no
kind (the five masker namespaces, ``script density``, ``Foreign-script text``,
the consistency constraint prefix and ``Heading label dropped``).

The two lifecycle prefixes (``Drafting error``, ``Repair error``) are correctly
absent: they describe a failed *call*, not a defect in the text, and the
workbench already surfaces them through the block's status.
"""

from __future__ import annotations

import pytest

from ubt.api.review import _ISSUE_MARKERS, ISSUE_KINDS
from ubt.core.qe.defect_taxonomy import (
    DRAFTING_ERROR_PREFIX,
    REPAIR_ERROR_PREFIX,
    STRUCTURAL_DEFECT_MARKERS,
)

pytestmark = pytest.mark.fast

#: Structural markers that are deliberately not workbench issue kinds: they are
#: lifecycle notes about a failed provider call, not defects in the target text.
#: The taxonomy's constants carry a trailing colon (they are used with
#: ``startswith``); the marker tables list the same phrases without it.
_LIFECYCLE_ONLY: frozenset[str] = frozenset(
    {DRAFTING_ERROR_PREFIX.rstrip(":"), REPAIR_ERROR_PREFIX.rstrip(":")}
)


def _covered(marker: str) -> bool:
    """Whether some issue kind's markers would match this flag."""
    return any(marker in candidate for markers in _ISSUE_MARKERS.values() for candidate in markers)


def test_every_structural_marker_has_an_issue_kind() -> None:
    missing = [
        marker
        for marker in STRUCTURAL_DEFECT_MARKERS
        if marker not in _LIFECYCLE_ONLY and not _covered(marker)
    ]
    assert missing == [], (
        "these markers can reject a draft but the workbench cannot classify it, "
        "so the reviewer sees a flagged segment with no issue type:\n"
        + "\n".join(f"  {m!r}" for m in missing)
    )


def test_the_lifecycle_exemption_is_not_a_loophole() -> None:
    """The two exempt prefixes must really be lifecycle, not quality, markers."""
    from ubt.core.qe.defect_taxonomy import is_transient_failure

    for prefix in (DRAFTING_ERROR_PREFIX, REPAIR_ERROR_PREFIX):
        assert is_transient_failure([f"{prefix} provider 500"]), prefix


def test_every_issue_kind_is_reachable() -> None:
    """A kind no marker can produce is a dead filter in the UI."""
    for kind, markers in _ISSUE_MARKERS.items():
        assert markers, f"issue kind {kind!r} has no markers"
    assert "render" in ISSUE_KINDS


def test_a_known_contamination_flag_reaches_the_workbench() -> None:
    """Spot check on the gap that motivated the guard."""
    from ubt.api.review import segment_issue_kinds
    from ubt.core.ir.models import BlockType, IRBlock, make_element

    block = IRBlock(
        element=make_element(
            id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="x"
        )
    )
    block.error_flags = ["Foreign-script text in a Chinese target: '기' is not Chinese"]
    assert "structure" in segment_issue_kinds(block)
