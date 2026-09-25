"""Unit tests for CheckpointBatchFlusher."""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher
from ubt.core.ir.models import BlockStatus, DocumentIR, FlowID, IRBlock


def _make_test_doc() -> DocumentIR:
    blocks = [
        IRBlock(
            id=f"b_{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"Sentence {i}",
        )
        for i in range(1, 11)
    ]
    return DocumentIR(
        doc_id="test_flusher_doc",
        source_path="/tmp/test.epub",
        format_type="epub",
        metadata={},
        blocks=blocks,
    )


@pytest.mark.asyncio
async def test_flusher_commits_enqueued_checkpoints(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    ledger.init_job("job_flush", _make_test_doc(), target_lang="zh")

    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.05, max_batch_size=5)

    # Enqueue 5 block checkpoints
    for i in range(1, 6):
        await flusher.enqueue(
            {
                "block_id": f"b_{i:03d}",
                "target_text": f"译文 {i}",
                "status": BlockStatus.DRAFTED,
                "draft_text": f"译文 {i}",
            }
        )

    # Wait for the background worker to flush
    await asyncio.sleep(0.15)
    await flusher.close()

    blocks = {b.id: b for b in ledger.get_all_blocks("job_flush")}
    for i in range(1, 6):
        b = blocks[f"b_{i:03d}"]
        assert b.status == BlockStatus.DRAFTED
        assert b.target_text == f"译文 {i}"


@pytest.mark.asyncio
async def test_flusher_flush_all_drains_immediately(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "ledger2.sqlite")
    ledger.init_job("job_flush2", _make_test_doc(), target_lang="zh")

    # Long flush interval so timer doesn't trigger
    flusher = CheckpointBatchFlusher(ledger, flush_interval=60.0, max_batch_size=100)

    for i in range(1, 4):
        await flusher.enqueue(
            {
                "block_id": f"b_{i:03d}",
                "target_text": f"实时刷新 {i}",
                "status": BlockStatus.MTQE_PASSED,
                "mtqe_score": 0.95,
            }
        )

    # Calling flush_all forces immediate commit
    await flusher.flush_all()
    await flusher.close()

    blocks = {b.id: b for b in ledger.get_all_blocks("job_flush2")}
    for i in range(1, 4):
        b = blocks[f"b_{i:03d}"]
        assert b.status == BlockStatus.MTQE_PASSED
        assert b.target_text == f"实时刷新 {i}"
        assert b.mtqe_score == 0.95


@pytest.mark.asyncio
async def test_flusher_survives_transient_save_failure(tmp_path: Path) -> None:
    """One failed batch goes back to the queue and is persisted on retry."""
    ledger = SQLiteJobLedger(tmp_path / "ledger_transient.sqlite")
    ledger.init_job("job_transient", _make_test_doc(), target_lang="zh")
    original = ledger.save_checkpoints_batch

    attempts: list[int] = []

    def flaky(updates: list[dict[str, Any]], **kwargs: Any) -> int:
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is busy")
        return original(updates, **kwargs)

    ledger.save_checkpoints_batch = flaky  # type: ignore[method-assign]

    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.02, max_batch_size=50)
    for i in range(1, 4):
        await flusher.enqueue(
            {
                "block_id": f"b_{i:03d}",
                "target_text": f"重试后落盘 {i}",
                "status": BlockStatus.DRAFTED,
            }
        )
    await flusher.close()

    assert len(attempts) >= 2
    blocks = {b.id: b for b in ledger.get_all_blocks("job_transient")}
    for i in range(1, 4):
        assert blocks[f"b_{i:03d}"].target_text == f"重试后落盘 {i}"


@pytest.mark.asyncio
async def test_flusher_persistent_failure_raises_at_close_without_loss(tmp_path: Path) -> None:
    """Repeated save failures must surface (never a silent drop); the queued
    checkpoints survive for a retry once the ledger is healthy again."""
    ledger = SQLiteJobLedger(tmp_path / "ledger_fatal.sqlite")
    ledger.init_job("job_fatal", _make_test_doc(), target_lang="zh")
    original = ledger.save_checkpoints_batch

    def boom(updates: list[dict[str, Any]], **kwargs: Any) -> int:
        raise sqlite3.OperationalError("disk on fire")

    ledger.save_checkpoints_batch = boom  # type: ignore[method-assign]

    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.02, max_batch_size=50)
    for i in range(1, 4):
        await flusher.enqueue(
            {
                "block_id": f"b_{i:03d}",
                "target_text": f"已计费译文 {i}",
                "status": BlockStatus.DRAFTED,
            }
        )

    with pytest.raises(RuntimeError, match="flush failed"):
        await flusher.close()

    # Nothing was lost on the raise path: with the ledger working again the
    # buffered checkpoints still land on a second close.
    ledger.save_checkpoints_batch = original  # type: ignore[method-assign]
    await flusher.close()
    blocks = {b.id: b for b in ledger.get_all_blocks("job_fatal")}
    for i in range(1, 4):
        assert blocks[f"b_{i:03d}"].target_text == f"已计费译文 {i}"


@pytest.mark.asyncio
async def test_close_is_final_start_does_not_resurrect(tmp_path: Path) -> None:
    """A cancelled worker resuming late must not be able to start() a flusher
    that close() already finalized: the resurrected task writes to a ledger
    closed upstream and its failure goes unretrieved, dropping checkpoints
    silently. close() is terminal; enqueue() falls back to a sync commit."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    ledger.init_job("job_final", _make_test_doc(), target_lang="zh")
    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.05, max_batch_size=5)

    await flusher.enqueue({"block_id": "b_001", "status": BlockStatus.DRAFTED.value})
    await flusher.close()
    assert flusher._flusher_task is None and flusher._closed is True

    # A late start() after close must be a no-op, not a resurrection.
    flusher.start()
    assert flusher._flusher_task is None, "start() resurrected a closed flusher"
    assert flusher._closed is True

    # And a late enqueue still lands (synchronous fallback), no silent drop.
    await flusher.enqueue({"block_id": "b_002", "status": BlockStatus.DRAFTED.value})
    reopened = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    try:
        b2 = reopened.get_block("b_002")
        assert b2 is not None and b2.status == BlockStatus.DRAFTED
    finally:
        reopened.close()
    ledger.close()


async def test_close_does_not_wait_out_the_flush_interval(tmp_path: Path) -> None:
    """Shutdown must not pay the configured flush interval.

    The worker parks in ``wait_for(queue.get(), timeout=flush_interval)`` and
    ``close()`` only flips ``_closed``, so the loop re-reads its exit condition
    after that wait expires. ``ledger_flush_interval`` is user-facing with no
    upper bound (``config.py`` ``ledger_flush_interval``), so every job end
    blocked for one whole interval — ~10s below, ~300s for an operator who set it
    to 300.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger_close.sqlite")
    ledger.init_job("job_close", _make_test_doc(), target_lang="zh")
    flusher = CheckpointBatchFlusher(ledger, flush_interval=10.0, max_batch_size=10)
    await flusher.enqueue(
        {
            "block_id": "b_001",
            "target_text": "关停校验",
            "status": BlockStatus.MTQE_PASSED,
            "mtqe_score": 0.9,
        }
    )
    await flusher.flush_all()

    started = time.monotonic()
    await flusher.close()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"close() blocked {elapsed:.2f}s waiting out a 10s flush interval"
    blocks = {b.id: b for b in ledger.get_all_blocks("job_close")}
    assert blocks["b_001"].target_text == "关停校验"
    ledger.close()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_ledger_flusher_persists_drained_checkpoints_when_task_cancelled() -> None:
    from unittest.mock import MagicMock

    mock_ledger = MagicMock()
    mock_ledger.save_checkpoints_batch = MagicMock()

    flusher = CheckpointBatchFlusher(
        ledger=mock_ledger,
        flush_interval=0.1,
        max_batch_size=5,
    )
    flusher.start()

    # Submit an item to the flusher
    mock_checkpoint = {"block_id": "b1", "status": "drafted"}
    await flusher.enqueue(mock_checkpoint)

    # Cancel the flusher close task
    async def cancel_close() -> None:
        task = asyncio.create_task(flusher.close())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await cancel_close()

    # The checkpoint batch MUST have been saved because it was shielded
    assert mock_ledger.save_checkpoints_batch.called
    saved_batch = mock_ledger.save_checkpoints_batch.call_args[0][0]
    assert mock_checkpoint in saved_batch
