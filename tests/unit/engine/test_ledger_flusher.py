"""The write-behind checkpoint flusher must fail loudly, never hang.

``CheckpointBatchFlusher.enqueue`` blocks while its bounded queue is full,
waiting for the background worker to free a slot. If that worker dies of its
retry budget in the same window, nobody will ever drain the queue and the
producer — a translation stage — parks forever. The class's whole contract is
"a checkpoint is never silently dropped; a spent retry budget fails the job
honestly", so a deadlock is worse than an error: the job neither completes nor
fails, it just stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

import pytest

from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher

pytestmark = pytest.mark.fast


class _FailingLedger:
    """A ledger whose first save blocks, then every save raises.

    ``save_checkpoints_batch`` runs in a worker thread (``asyncio.to_thread``),
    so the gates are ``threading`` primitives: the test parks the worker inside
    its first save while it fills the queue, then releases it to fail its way
    through the retry budget and die.
    """

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
        # "a": the worker pops it immediately and parks inside save(a).
        await flusher.enqueue({"block_id": "a"})
        await loop.run_in_executor(None, ledger.first_started.wait, 5)
        # "b": the queue was emptied by the pop, so this fills it to maxsize.
        await flusher.enqueue({"block_id": "b"})
        # Let the worker resume: it fails save(a) three times and raises.
        ledger.release_first.set()
        # "c" finds the queue full and parks in ``put``; the worker dies while
        # it waits. Without the fix this never returns.
        with pytest.raises(RuntimeError, match="abandoning batched writes"):
            await asyncio.wait_for(flusher.enqueue({"block_id": "c"}), timeout=5)
    finally:
        # The worker already died and the ledger still fails, so the final
        # drain in ``close`` raises too — that is the point.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(flusher.close(), timeout=5)
