"""Unit tests for CheckpointBatchFlusher."""

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher
from ubt.core.ir.models import BlockStatus, FlowID, IRBlock


def _make_test_doc() -> SeedDoc:
    blocks = [
        IRBlock(
            id=f"b_{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"Sentence {i}",
        )
        for i in range(1, 11)
    ]
    return SeedDoc(
        doc_id="test_flusher_doc",
        source_path="/tmp/test.epub",
        format_type="epub",
        metadata={},
        blocks=blocks,
    )


@pytest.mark.asyncio
async def test_flusher_commits_enqueued_checkpoints(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    seed_job(ledger, "job_flush", _make_test_doc(), target_lang="zh")

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
    seed_job(ledger, "job_flush2", _make_test_doc(), target_lang="zh")

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
    seed_job(ledger, "job_transient", _make_test_doc(), target_lang="zh")
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
    seed_job(ledger, "job_fatal", _make_test_doc(), target_lang="zh")
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
    seed_job(ledger, "job_final", _make_test_doc(), target_lang="zh")
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

    The worker parks in ``wait_for(queue.get(), timeout=flush_interval)``.
    ``close()`` flips ``_closed`` *and* pushes the ``_WAKE`` sentinel, so the
    loop re-reads its exit condition immediately instead of waiting out the
    interval. ``ledger_flush_interval`` is user-facing with no upper bound
    (``config.py`` ``ledger_flush_interval``), so a regression here would block
    every job end for one whole interval — ~10s below, ~300s for an operator who
    set it to 300.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger_close.sqlite")
    seed_job(ledger, "job_close", _make_test_doc(), target_lang="zh")
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


@pytest.mark.fast
def test_failed_batch_is_retried_before_newer_updates() -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher

    async def scenario() -> list[str]:
        ledger = _FlakyBlockingLedger()
        flusher = CheckpointBatchFlusher(
            cast(SQLiteJobLedger, ledger), flush_interval=0.01, max_batch_size=50
        )
        await flusher.enqueue({"block_id": "X", "status": "v1"})
        # Let the first save start, enqueue the newer update while it is in
        # flight, then let the save fail.
        await asyncio.to_thread(ledger.first_entered.wait, 5)
        await flusher.enqueue({"block_id": "X", "status": "v2"})
        ledger.release_first.set()
        await flusher.close()
        return [str(u["status"]) for u in ledger.committed]

    assert asyncio.run(scenario()) == ["v1", "v2"]


class _FlakyBlockingLedger:
    """First save blocks, then fails; later saves commit in call order."""

    def __init__(self) -> None:
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.calls = 0
        self.committed: list[dict[str, object]] = []

    def save_checkpoints_batch(self, updates: list[dict[str, object]], **_kwargs: object) -> int:
        self.calls += 1
        if self.calls == 1:
            self.first_entered.set()
            self.release_first.wait(timeout=5)
            raise sqlite3.OperationalError("database is locked")
        self.committed.extend(updates)
        return len(updates)


@pytest.mark.fast
@pytest.mark.asyncio
async def test_flusher_applies_backoff_on_transient_failure_and_drains_on_task_error(
    tmp_path: Path,
) -> None:
    """Flusher must backoff between consecutive failures and
    close() must drain remaining items even if the background task died."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    doc = SeedDoc(
        doc_id="doc_backoff",
        source_path="/tmp/test.epub",
        format_type="epub",
        metadata={},
        blocks=[
            IRBlock(id="b_001", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="One"),
            IRBlock(id="b_002", flow_id=FlowID.MAIN_STORY, spine_index=2, source_text="Two"),
        ],
    )
    seed_job(ledger, "job_backoff", doc, target_lang="zh")
    original_save = ledger.save_checkpoints_batch

    # Part 1: Verify retry backoff delay is > 0 when _save fails transiently
    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.01, max_batch_size=10)
    assert getattr(flusher, "_retry_base_delay", 0.0) > 0.0, (
        "CheckpointBatchFlusher must define a positive _retry_base_delay for transient SQLite failures"
    )

    # Part 2: Simulate background task dying with RuntimeError, then ledger recovering
    # before close() is called with additional pending items in the queue.
    fail_now = True

    def controlled_save(updates: list[dict[str, Any]], **kwargs: Any) -> int:
        if fail_now:
            raise sqlite3.OperationalError("database is locked")
        return original_save(updates, **kwargs)

    ledger.__dict__["save_checkpoints_batch"] = controlled_save
    await flusher.enqueue(
        {"block_id": "b_001", "target_text": "译文1", "status": BlockStatus.DRAFTED}
    )

    # Wait for background task to hit _MAX_CONSECUTIVE_FAILURES and terminate
    assert flusher._flusher_task is not None
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(asyncio.shield(flusher._flusher_task), timeout=2.0)

    # Now the ledger recovers before close() is called
    fail_now = False
    with pytest.raises(RuntimeError):
        await flusher.close()

    # Even though close() surfaced the task RuntimeError, pending items must ALREADY be drained and saved!
    blocks = {b.id: b for b in ledger.get_all_blocks("job_backoff")}
    assert blocks["b_001"].target_text == "译文1"
    ledger.close()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_flush_all_synchronizes_with_in_flight_background_save(tmp_path: Path) -> None:
    """flush_all() must serialize with and wait for any background _save in progress."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    seed_job(ledger, "job_sync", _make_test_doc(), target_lang="zh")
    original_save = ledger.save_checkpoints_batch

    in_save = asyncio.Event()
    release_save = asyncio.Event()
    save_concurrent = False
    active_saves = 0

    def slow_save(updates: list[dict[str, Any]], **kwargs: Any) -> int:
        nonlocal save_concurrent, active_saves
        active_saves += 1
        if active_saves > 1:
            save_concurrent = True
        try:
            # Signal we are inside save
            in_save.set()
            # Wait for release
            start_t = time.time()
            while not release_save.is_set() and time.time() - start_t < 1.0:
                time.sleep(0.01)
            return original_save(updates, **kwargs)
        finally:
            active_saves -= 1

    ledger.__dict__["save_checkpoints_batch"] = slow_save
    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.01, max_batch_size=1)

    # Enqueue item 1 to trigger background worker
    await flusher.enqueue(
        {"block_id": "b_001", "target_text": "译文1", "status": BlockStatus.DRAFTED}
    )

    # Wait until background worker enters slow_save
    await in_save.wait()

    # Now enqueue item 2 and call flush_all() while background save is in flight
    await flusher.enqueue(
        {"block_id": "b_002", "target_text": "译文2", "status": BlockStatus.DRAFTED}
    )

    flush_task = asyncio.create_task(flusher.flush_all())
    await asyncio.sleep(0.05)
    # flush_all should NOT have finished yet because background save is still blocked
    assert not flush_task.done()

    # Release slow_save
    release_save.set()
    await flush_task

    assert not save_concurrent, "Two _save operations must not execute concurrently"
    blocks = {b.id: b for b in ledger.get_all_blocks("job_sync")}
    assert blocks["b_001"].target_text == "译文1"
    assert blocks["b_002"].target_text == "译文2"
    await flusher.close()
    ledger.close()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_older_retry_never_overwrites_a_newer_flush_all_value(tmp_path: Path) -> None:
    """A failed older batch must land *before* flush_all's newer value.

    The worker held an older batch while ``flush_all`` drained a newer one; on
    failure the older batch was requeued and retried after the newer save had
    landed, overwriting it. Draining under the save lock keeps the order.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    seed_job(ledger, "job_order", _make_test_doc(), target_lang="zh")
    original_save = ledger.save_checkpoints_batch

    in_save = asyncio.Event()
    release_save = asyncio.Event()
    calls = 0

    def fail_first_save(updates: list[dict[str, Any]], **kwargs: Any) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            in_save.set()
            start_t = time.time()
            while not release_save.is_set() and time.time() - start_t < 1.0:
                time.sleep(0.01)
            raise RuntimeError("simulated first-save failure")
        return original_save(updates, **kwargs)

    ledger.__dict__["save_checkpoints_batch"] = fail_first_save
    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.01, max_batch_size=1)

    # Older value for b_001: the worker pops it and is inside the failing save.
    await flusher.enqueue(
        {"block_id": "b_001", "target_text": "旧值", "status": BlockStatus.DRAFTED}
    )
    await in_save.wait()

    # Newer value for the SAME block, then a mid-run flush while the older save
    # is still in flight.
    await flusher.enqueue(
        {"block_id": "b_001", "target_text": "新值", "status": BlockStatus.DRAFTED}
    )
    flush_task = asyncio.create_task(flusher.flush_all())
    await asyncio.sleep(0.05)

    release_save.set()
    await flush_task
    # Give the worker time to retry the failed older batch too.
    await asyncio.sleep(0.05)

    blocks = {b.id: b for b in ledger.get_all_blocks("job_order")}
    assert blocks["b_001"].target_text == "新值"
    await flusher.close()
    ledger.close()
