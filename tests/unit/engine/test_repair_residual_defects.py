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


# --------------------------------------------------------------------------- #
# Glossary drift is repaired deterministically, before the paid model
#
# The enforcer substitutes an untranslated source term for its canonical
# rendering at zero token cost, and the same validator that flagged the drift
# verifies the result -- the "one rule both rejects and accepts" property that
# makes a repair checkable. On the ATC'25 run all 13 glossary candidates went to
# the paid model instead, every call 500'd, and all 13 were quarantined as
# NEEDS_HUMAN while the export stage fixed the same drift deterministically.
# --------------------------------------------------------------------------- #

_GLOSSARY: list[dict[str, Any]] = [
    {"source": "framekernel", "translation": "帧内核", "aliases": [], "inflected_variants": []}
]
_DRIFT = "Glossary term violation: Glossary terms drifted: ['framekernel -> 帧内核']"


def _pending(draft: str, flags: list[str]) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="We propose framekernel.",
        )
    )
    block.target_text = draft
    block.draft_text = draft
    block.error_flags = list(flags)
    block.status = BlockStatus.REPAIR_PENDING
    return block


def _bare_loop() -> RepairLoop:
    # The deterministic pre-repair reads no instance state, so a bare instance
    # keeps the test free of router/QE construction.
    return RepairLoop.__new__(RepairLoop)


def test_glossary_drift_is_fixed_without_calling_the_model() -> None:
    block = _pending("我们提出了 framekernel。", [_DRIFT])
    result = _bare_loop()._try_deterministic_glossary(
        block, block.draft_text or "", _GLOSSARY, "zh"
    )
    assert result is not None
    # The enforcer applies the CJK/Latin spacing rule, so the substituted term
    # is padded: 我们提出了 帧内核。 — the drift is gone either way.
    assert "framekernel" not in (result.target_text or "")
    assert "帧内核" in (result.target_text or "")
    assert result.status is BlockStatus.REPAIRED
    assert result.error_flags == []


def test_a_drift_that_leaves_another_defect_still_reaches_the_model() -> None:
    """Only a full resolution may skip the paid repair."""
    block = _pending("我们提出了 framekernel。", [_DRIFT, "Omission suspected: verbatim residue"])
    assert (
        _bare_loop()._try_deterministic_glossary(block, block.draft_text or "", _GLOSSARY, "zh")
        is None
    )


def test_a_triage_verdict_is_never_cleared_by_the_enforcer() -> None:
    """`needs_human_review` is a permanent verdict, not a repairable flag."""
    block = _pending("我们提出了 framekernel。", [_DRIFT, "needs_human_review"])
    assert (
        _bare_loop()._try_deterministic_glossary(block, block.draft_text or "", _GLOSSARY, "zh")
        is None
    )


def test_an_already_canonical_draft_is_not_a_repair() -> None:
    block = _pending("我们提出了帧内核。", [_DRIFT])
    assert (
        _bare_loop()._try_deterministic_glossary(block, block.draft_text or "", _GLOSSARY, "zh")
        is None
    )
