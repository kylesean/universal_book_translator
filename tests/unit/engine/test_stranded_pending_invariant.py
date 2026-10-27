"""The stranded-pending invariant behind the draft loop's keyset cursor.

The interactive draft loop claims blocks with a monotone ``(spine_index,
block_id)`` cursor: once a batch is consumed the cursor never returns. Every
consumed block must therefore have left ``pending`` by the time the batch's
checkpoint writes are flushed — drafted, passed, failed, or requeued.
``_assert_no_stranded_pending`` is the fail-loud tripwire: a future early-exit
path that returns without checkpointing would otherwise strand the block
behind the cursor until export force-FAILED it, after the money was spent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.draft import _assert_no_stranded_pending
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast


def _block(index: int) -> IRBlock:
    element = make_element(
        id=f"b{index:03d}",
        spine_index=index,
        block_type=BlockType.NARRATIVE,
        source_text=f"Paragraph {index}.",
        bbox=BoundingBox(page=1, x0=50.0, y0=700.0 - index * 20, x1=350.0, y1=712.0 - index * 20),
    )
    return IRBlock(element=element)


def _ledger(tmp_path: Path, blocks: list[IRBlock]) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job_from_manifest("job", BookManifest(doc_id="doc", title="t", source_path="s.pdf"))
    from ubt.core.ir.models import ChapterIR

    ledger.append_chapter(
        "job",
        ChapterIR(doc_id="doc", chapter_id="c1", title="C1", spine_index=0, blocks=blocks),
    )
    return ledger


def test_a_batch_fully_checkpointed_passes(tmp_path: Path) -> None:
    blocks = [_block(1), _block(2)]
    ledger = _ledger(tmp_path, blocks)
    try:
        # Simulate the loop's writes: one drafted, one TM-passed.
        ledger.save_checkpoints_batch(
            [
                {"block_id": "b001", "status": BlockStatus.DRAFTED, "target_text": "T1"},
                {
                    "block_id": "b002",
                    "status": BlockStatus.MTQE_PASSED,
                    "target_text": "T2",
                    "mtqe_score": 1.0,
                },
            ],
            job_id="job",
        )
        _assert_no_stranded_pending(ledger, "job", blocks)  # no raise
    finally:
        ledger.close()


def test_a_block_left_pending_fails_loud(tmp_path: Path) -> None:
    """The regression the tripwire exists for: a future early-exit path that
    returns without checkpointing."""
    blocks = [_block(1), _block(2)]
    ledger = _ledger(tmp_path, blocks)
    try:
        # Only the first block is checkpointed; b002 stays pending.
        ledger.save_checkpoints_batch(
            [{"block_id": "b001", "status": BlockStatus.DRAFTED, "target_text": "T1"}],
            job_id="job",
        )
        with pytest.raises(Exception, match="left block 'b002' pending"):
            _assert_no_stranded_pending(ledger, "job", blocks)
    finally:
        ledger.close()


def test_blocks_from_other_jobs_do_not_count(tmp_path: Path) -> None:
    """The COUNT is scoped to the job AND the batch's ids."""
    blocks = [_block(1), _block(2)]
    ledger = _ledger(tmp_path, blocks)
    try:
        _assert_no_stranded_pending(ledger, "job", blocks[:0])  # empty batch: no-op
        # job's own blocks are checkpointed; the OTHER job's b001 stays
        # pending — the batch-id scoping must not count it against 'job'.
        ledger.save_checkpoints_batch(
            [{"block_id": "b001", "status": BlockStatus.DRAFTED, "target_text": "T1"}],
            job_id="job",
        )
        other = SQLiteJobLedger(tmp_path / "job.sqlite")
        other.init_job_from_manifest(
            "other", BookManifest(doc_id="doc", title="t", source_path="s.pdf")
        )
        from ubt.core.ir.models import ChapterIR

        other.append_chapter(
            "other",
            ChapterIR(
                doc_id="doc",
                chapter_id="c1",
                title="C1",
                spine_index=0,
                blocks=[_block(1)],  # same block id, different job: still pending
            ),
        )
        other.close()
        _assert_no_stranded_pending(ledger, "job", blocks[:1])  # no raise
    finally:
        ledger.close()
