"""Bi-modal Quality Gate tests: Track 1 (Structural Integrity) vs Track 2 (Semantic Quality).

Validates that:
1. Uncalibrated runners (HeuristicQERunner) do not fabricate continuous float scores.
2. In the absence of calibrated neural/judge scorers, mtqe_score remains None.
3. Repair candidate lifecycle is governed strictly by structural integrity (error_flags resolved -> REPAIRED).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast


def _make_block(block_id: str, src: str, draft: str, flags: list[str] | None = None) -> IRBlock:
    el = Paragraph(id=block_id, spine_index=0, span=Span(page=1), text=src)
    block = IRBlock(element=el)
    block.draft_text = draft
    block.target_text = draft
    block.error_flags = flags or []
    return block


def test_heuristic_runner_is_uncalibrated() -> None:
    runner = HeuristicQERunner()
    assert runner.is_calibrated() is False


@pytest.mark.asyncio
async def test_repair_loop_uncalibrated_runner_adopts_structurally_valid_candidate() -> None:
    runner = HeuristicQERunner()
    router = MagicMock()
    # Mock router to return a clean repair
    router.repair = AsyncMock(return_value="这是干净的修复文本。")

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85)

    block = _make_block(
        "b1",
        "This is source text.",
        "这是草稿文本。",
        flags=["html_attr_mismatch"],
    )
    block.status = BlockStatus.REPAIR_PENDING

    repaired = await loop.repair_single_block(block)
    # Track 1 passes: structural integrity is sound and error_flags cleared
    assert repaired.status == BlockStatus.REPAIRED
    assert repaired.error_flags == []
    assert repaired.target_text == "这是干净的修复文本。"
    assert repaired.mtqe_score is None


@pytest.mark.asyncio
async def test_repair_loop_uncalibrated_runner_fails_after_max_rounds_when_structural_defect_persists() -> (
    None
):
    runner = HeuristicQERunner()
    router = MagicMock()
    # Mock router to return empty candidate (fails structural integrity)
    router.repair = AsyncMock(return_value="")

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85, max_rounds=2)

    block = _make_block(
        "b2",
        "Source with <b>tag</b>",
        "草稿",
        flags=["html_attr_mismatch"],
    )
    block.repair_rounds = 1  # will become 2 (rounds_cap)
    block.status = BlockStatus.REPAIR_PENDING

    repaired = await loop.repair_single_block(block)
    assert repaired.status == BlockStatus.FAILED
    assert len(repaired.error_flags) > 0
    assert repaired.mtqe_score is None


@pytest.mark.asyncio
async def test_a_raising_repair_still_consumes_a_round() -> None:
    # A repair call that raises (provider outage) must still advance the round
    # counter: the round is consumed before the call. Otherwise a persistently
    # failing repair stays at round 0, the transient-failure reset requeues it to
    # REPAIR_PENDING on every resume, and the circuit breaker (gated on
    # repair_rounds < max_rounds) never fires -- re-billing the model forever.
    runner = HeuristicQERunner()
    router = MagicMock()
    router.repair = AsyncMock(side_effect=RuntimeError("provider outage"))

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85)

    block = _make_block(
        "b3", "This is source text.", "这是草稿文本。", flags=["html_attr_mismatch"]
    )
    block.status = BlockStatus.REPAIR_PENDING
    assert block.repair_rounds == 0

    with pytest.raises(RuntimeError):
        await loop.repair_single_block(block)

    assert block.repair_rounds == 1, "a failed repair attempt must consume a round"
