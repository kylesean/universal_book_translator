"""Track 1 (uncalibrated repair) must not promote a still-defective candidate.

``RepairLoop`` has two tracks. The *calibrated* track re-scores with a neural
runner and re-runs the numeric/glossary validators before it will call a block
REPAIRED. The *uncalibrated* track (heuristic runner) decides on structural
validity alone: a tier-1 in-place repair only has to pass ``_structural_check``
(the 0-Token invariants — no HTML drift, no length blow-up), which says nothing
about whether the defect that *triggered* the repair is gone.

The hole: a draft alters a source number, FastPass flags it, repair is asked to
restore it, and the model returns a splice that changes the number to a *third*
value. The splice is structurally sound and preserves all out-of-span content,
so ``t1_chosen_flags`` is ``[]`` and the old code set ``status = REPAIRED`` —
shipping a translation whose defect the repair was specifically invoked to fix,
and (because REPAIRED is terminal) never re-checking it. These tests pin the
residual-defect re-check that closes it.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.core.qe.base import BaseQERunner
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


class _UncalibratedRunner(BaseQERunner):
    """A runner whose scores are defect classes, not quality (the heuristic)."""

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        return [0.9 for _ in pairs]

    def is_calibrated(self) -> bool:
        return False


class _FakeRouter:
    """Returns a fixed repair payload; no provider is contacted."""

    repair_reasoning_effort = "low"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def repair(self, **_kw: Any) -> str:
        return self._reply


def _loop(reply: str) -> RepairLoop:
    return RepairLoop(router=cast(ModelRouter, _FakeRouter(reply)), qe_runner=_UncalibratedRunner())


def _block(*, source: str, draft: str, flags: list[str]) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=source,
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
        ),
        target_text=draft,
        draft_text=draft,
        status=BlockStatus.REPAIR_PENDING,
        error_flags=flags,
    )


# The draft states "43 days"; the source says 42. The annotator marks the "43"
# as span id "1", so a `<correction id="1">…</correction>` reply is spliced in
# place — the tier-1 path that never re-runs the numeric validator itself.
_SOURCE = "The trial ran for 42 days."
_DRAFT = "该试验持续了43天。"
_FLAGS = ["Numeric fidelity: mismatched number '43', expected '42'"]


@pytest.mark.asyncio
async def test_in_place_splice_to_a_wrong_number_is_not_promoted() -> None:
    """A splice that swaps one wrong number for another keeps the block REPAIR_PENDING."""
    block = _block(source=_SOURCE, draft=_DRAFT, flags=_FLAGS)
    # The model "fixes" 43 -> 41: still not the source's 42.
    loop = _loop('<correction id="1">41</correction>')

    result = await loop.repair_single_block(block)

    assert result.status is not BlockStatus.REPAIRED, (
        "a candidate that still does not state the source number must not ship as REPAIRED"
    )
    assert any("repair_structural_failure" in f for f in result.error_flags)


@pytest.mark.asyncio
async def test_in_place_splice_restoring_the_number_is_promoted() -> None:
    """The control: the same path promotes normally once the number is correct."""
    block = _block(source=_SOURCE, draft=_DRAFT, flags=_FLAGS)
    loop = _loop('<correction id="1">42</correction>')

    result = await loop.repair_single_block(block)

    assert result.status is BlockStatus.REPAIRED
    assert result.error_flags == []


@pytest.mark.asyncio
async def test_residual_check_adds_no_flag_on_a_clean_candidate() -> None:
    """A candidate the numeric validator accepts gains no spurious flag."""
    block = _block(
        source="Chapter 3 lists 12 species.",
        draft="第3章列出了11个物种。",
        flags=["Numeric fidelity: mismatched number '11', expected '12'"],
    )
    loop = _loop('<correction id="1">12</correction>')

    result = await loop.repair_single_block(block)

    assert result.error_flags == []
    assert result.status is BlockStatus.REPAIRED
