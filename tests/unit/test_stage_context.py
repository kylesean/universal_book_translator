"""Unit tests for StageContext methods and caching behavior."""

from pathlib import Path

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockStatus, FlowID, IRBlock


@pytest.mark.asyncio
async def test_current_blocks_cache_and_invalidation(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "stage_ctx_test.db")
    blocks = [
        IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="Source 1"),
    ]
    doc = SeedDoc(doc_id="test_doc", source_path="test.txt", format_type="txt", blocks=blocks)
    seed_job(ledger, "job_ctx", doc, target_lang="zh")

    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_ctx",
    )

    # Initial fetch: caches 1 block with status PENDING
    initial_blocks = await ctx.current_blocks()
    assert len(initial_blocks) == 1
    assert initial_blocks[0].status == BlockStatus.PENDING

    # A ledger write moves the block revision, so the snapshot above is no
    # longer answerable: the next default read comes from the database. This is
    # the 2026-09 review P1-11 contract — a stage must not need to remember to
    # refresh after another stage wrote blocks.
    ledger.save_checkpoint("b1", status=BlockStatus.MTQE_PASSED, target_text="Target 1")
    refreshed = await ctx.current_blocks()
    assert refreshed[0].status == BlockStatus.MTQE_PASSED
    assert refreshed[0].target_text == "Target 1"

    # With nothing written in between, the snapshot is reused, not re-read.
    again = await ctx.current_blocks()
    assert again is refreshed

    # force_refresh remains for in-memory edits the ledger was never told about,
    # and so does the explicit invalidation.
    refreshed[0].status = BlockStatus.REPAIR_PENDING
    forced = await ctx.current_blocks(force_refresh=True)
    assert forced[0].status == BlockStatus.MTQE_PASSED
    assert forced is not again

    ctx.invalidate_blocks_cache()
    assert ctx._blocks is None
    refetched = await ctx.current_blocks()
    assert refetched[0].status == BlockStatus.MTQE_PASSED
    assert refetched is not forced
