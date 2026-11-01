from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

import pytest

from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher

pytestmark = pytest.mark.fast


class _FailingLedger:
    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self._lock = threading.Lock()
        self.calls = 0

    def save_checkpoints_batch(self, updates: list[dict[str, Any]], **_: Any) -> int:
        with self._lock:
            self.calls += 1
            call = self.calls
        if call == 1:
            self.first_started.set()
            self.release_first.wait(timeout=5)
        raise RuntimeError("ledger unavailable")


async def test_enqueue_does_not_hang_when_the_worker_dies_while_the_queue_is_full() -> None:
    ledger = _FailingLedger()
    flusher = CheckpointBatchFlusher(
        ledger,  # type: ignore[arg-type]
        flush_interval=0.01,
        max_batch_size=1,
        max_queue_size=1,
    )
    flusher.start()
    loop = asyncio.get_running_loop()
    try:
        await flusher.enqueue({"block_id": "a"})
        await loop.run_in_executor(None, ledger.first_started.wait, 5)
        await flusher.enqueue({"block_id": "b"})
        ledger.release_first.set()
        with pytest.raises(RuntimeError, match="abandoning batched writes"):
            await asyncio.wait_for(flusher.enqueue({"block_id": "c"}), timeout=5)
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(flusher.close(), timeout=5)


class _TransientlyFailingLedger:
    def __init__(self) -> None:
        self.calls = 0

    def save_checkpoints_batch(self, updates: list[dict[str, Any]], **_: Any) -> int:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("database is locked")
        return len(updates)


async def test_a_transient_failure_is_visible_as_a_pending_retry() -> None:
    ledger = _TransientlyFailingLedger()
    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.01)  # type: ignore[arg-type]
    try:
        await flusher.enqueue({"block_id": "a"})
        await flusher.flush_all()
        assert flusher.has_pending_retries, "the failed batch was not retained"
        assert flusher.pending_retry_count == 1

        await asyncio.sleep(0.1)
        assert not flusher.has_pending_retries
        assert flusher.pending_retry_count == 0
    finally:
        with contextlib.suppress(Exception):
            await flusher.close()


async def test_a_clean_flusher_reports_no_pending_retry() -> None:
    ledger = _TransientlyFailingLedger()
    ledger.calls = 1
    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.01)  # type: ignore[arg-type]
    try:
        await flusher.enqueue({"block_id": "a"})
        await flusher.flush_all()
        assert not flusher.has_pending_retries
        assert flusher.pending_retry_count == 0
    finally:
        await flusher.close()
