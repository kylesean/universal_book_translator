"""Asynchronous write-behind buffer for SQLite ledger checkpoints.

Buffers block checkpoint updates in memory and commits them to SQLite in
batched transactions via ``save_checkpoints_batch``. This eliminates SQLite
WAL write-lock contention and thread synchronization overhead under high
concurrency during translation stages.

Durability contract: a checkpoint handed to the flusher is never silently
dropped. Failed batches go back to the queue for bounded retries; once the
retry budget is spent the flusher dies loudly (the error surfaces at the next
``enqueue`` and at ``close``), so the job fails honestly instead of reporting
"completed" with paid translations missing from the ledger.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ubt.core.engine.ledger import SQLiteJobLedger

logger = logging.getLogger(__name__)

#: Consecutive failed saves tolerated before the flusher gives up and raises.
_MAX_CONSECUTIVE_FAILURES = 3


class _Wake:
    """Sentinel pushed by ``close()`` to unpark a worker idling in ``get()``."""


_WAKE = _Wake()


class CheckpointBatchFlusher:
    """Buffers checkpoint updates in memory and flushes in batches."""

    def __init__(
        self,
        ledger: SQLiteJobLedger,
        flush_interval: float = 0.25,
        max_batch_size: int = 50,
    ) -> None:
        self.ledger = ledger
        self.flush_interval = max(0.01, flush_interval)
        self.max_batch_size = max(1, max_batch_size)
        self._queue: asyncio.Queue[dict[str, Any] | _Wake] = asyncio.Queue()
        # Failed batches have priority over updates queued while the database
        # was unavailable. Retrying an older batch before newer checkpoints for
        # the same block preserves the ledger's event order.
        self._retry_batches: list[list[dict[str, Any]]] = []
        self._flusher_task: asyncio.Task[None] | None = None
        self._closed = False
        self._failures = 0
        self._retry_base_delay = 0.02

    def start(self) -> None:
        """Start the background flusher task if not already running.

        ``close`` is final: a cancelled worker resuming late must not be
        able to resurrect the session and hand checkpoints to a flusher
        nobody will ever join or close (its writes hit a closed ledger and
        its death goes unretrieved). ``enqueue`` covers the closed case
        with its synchronous fallback.
        """
        self._raise_if_task_died()
        if self._closed:
            return
        if self._flusher_task is None or self._flusher_task.done():
            self._flusher_task = asyncio.create_task(self._run_flusher())

    def _raise_if_task_died(self) -> None:
        """Re-surface a fatal save error rather than restarting on a clean task.

        Without this, a worker that died on repeated DB failures would be
        silently recreated by the next ``enqueue`` and the pending checkpoints
        would sit in the queue forever.
        """
        if self._flusher_task is not None and self._flusher_task.done():
            exc = self._flusher_task.exception()
            if exc is not None:
                raise exc

    async def enqueue(self, update: dict[str, Any]) -> None:
        """Enqueue a block checkpoint update.

        If the flusher has been closed, commits synchronously in a thread.
        """
        self._raise_if_task_died()
        if self._closed:
            await asyncio.to_thread(self.ledger.save_checkpoints_batch, [update])
            return
        self.start()
        await self._queue.put(update)

    async def _save(self, batch: list[dict[str, Any]]) -> bool:
        """Persist *batch*; on failure retain it for the next retry window.

        Returns True when the batch landed. ``asyncio.CancelledError`` is *not*
        caught here — it passes through so the worker's cancel handler can
        re-queue the batch (a cancelled ``to_thread`` that had not started
        would otherwise drop it).
        """
        try:
            await asyncio.to_thread(self.ledger.save_checkpoints_batch, batch)
        except Exception as exc:
            self._failures += 1
            logger.error(
                "Failed to flush %d checkpoint(s) to ledger (consecutive failure %d/%d): %s",
                len(batch),
                self._failures,
                _MAX_CONSECUTIVE_FAILURES,
                exc,
            )
            self._retry_batches.insert(0, batch)
            return False
        self._failures = 0
        return True

    async def _run_flusher(self) -> None:
        """Continuously collect and flush queued updates to SQLite."""
        batch: list[dict[str, Any]] = []
        try:
            while not self._closed or not self._queue.empty() or self._retry_batches:
                batch = []
                if self._retry_batches:
                    # Preserve per-block checkpoint order: an older failed
                    # batch must land before updates that arrived while it was
                    # being retried.
                    batch = self._retry_batches.pop(0)
                else:
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=self.flush_interval
                        )
                        self._queue.task_done()
                        if isinstance(item, _Wake):
                            # close() is waiting on this task; the loop
                            # condition below is what actually ends it.
                            continue
                        batch.append(item)
                        while len(batch) < self.max_batch_size:
                            try:
                                extra = self._queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            self._queue.task_done()
                            if not isinstance(extra, _Wake):
                                batch.append(extra)
                    except TimeoutError:
                        pass

                if batch:
                    await self._save(batch)
                    batch = []
                    if self._failures >= _MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"Ledger checkpoint flush failed {self._failures} times in a row; "
                            "abandoning batched writes"
                        ) from None
                    if self._failures > 0:
                        await asyncio.sleep(self._retry_base_delay * (2 ** (self._failures - 1)))
        except asyncio.CancelledError:
            # External cancellation (job cancel) between "dequeue" and "save
            # started": hand the batch back so close()'s final drain persists
            # it instead of losing up to max_batch_size paid checkpoints.
            if batch:
                self._retry_batches.insert(0, batch)
            raise

    def _drain_nowait(self) -> list[dict[str, Any]]:
        """Pop retries and queued items, dropping wake sentinels."""
        items: list[dict[str, Any]] = [item for batch in self._retry_batches for item in batch]
        self._retry_batches.clear()
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return items
            self._queue.task_done()
            if not isinstance(item, _Wake):
                items.append(item)

    async def flush_all(self) -> None:
        """Drain and commit all currently enqueued items immediately.

        A failure retains the batch for the background worker (which owns the
        retry budget), so this mid-run checkpoint never aborts the stage.
        """
        batch = self._drain_nowait()
        if batch:
            await self._save(batch)

    async def close(self) -> None:
        """Stop the background worker after all pending checkpoints landed.

        Never cancels the worker mid-batch: a cancelled ``to_thread`` whose
        executor job had not started would drop an already-dequeued batch
        without a trace. The worker drains the queue and exits on its own;
        any remaining items are persisted here, and any failure propagates to
        the caller so the job is not marked completed on lost work.
        """
        self._closed = True
        task, self._flusher_task = self._flusher_task, None
        task_exc: BaseException | None = None
        if task is not None:
            self._queue.put_nowait(_WAKE)
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                task_exc = exc
        pending = self._drain_nowait()
        if pending:
            saved = await asyncio.shield(self._save(pending))
            if not saved and task_exc is None:
                task_exc = RuntimeError(
                    f"Ledger checkpoint flush failed ({len(pending)} pending item(s)); "
                    "abandoning batched writes"
                )
        if task_exc is not None:
            raise task_exc
