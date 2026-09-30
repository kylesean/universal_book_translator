"""The repair stage's no-triage finalization floor tracks the QE threshold."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.repair import run_repair_stage
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock

pytestmark = pytest.mark.fast


def _pending_block(block_id: str, score: float) -> IRBlock:
    return IRBlock(
        id=block_id,
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Source sentence.",
        target_text="译文。",
        draft_text="译文。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=score,
        repair_rounds=0,
        error_flags=[],
    )


@pytest.mark.asyncio
async def test_floor_finalization_uses_the_configured_threshold(tmp_path: Path) -> None:
    """A 0.85-score leftover ships at the default 0.75 gate but not at 0.9.

    The fallback floor hardcoded 0.5, so a stricter ``qe_threshold`` still
    shipped leftover REPAIR_PENDING blocks the quality gate would never have
    passed — two different definitions of "shippable" on two code paths.
    """
    ledger = SQLiteJobLedger(tmp_path / "job_floor.sqlite")
    job_id = "job_floor"
    seed_job(
        ledger,
        job_id,
        SeedDoc(
            doc_id="doc",
            source_path=str(tmp_path / "b.md"),
            blocks=[_pending_block("b1", 0.85)],
        ),
    )
    ctx = build_stage_ctx(
        tmp_path,
        config=UBTConfig(db_dir=tmp_path, qe_threshold=0.9),
        ledger=ledger,
        job_id=job_id,
    )

    async for _ in run_repair_stage(ctx, defer_unresolved_to_triage=False):
        pass

    failed = ledger.fetch_blocks_by_status(job_id, BlockStatus.FAILED)
    assert [b.id for b in failed] == ["b1"]
