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
import contextlib
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
        max_queue_size: int = 1000,
    ) -> None:
        self.ledger = ledger
        self.flush_interval = max(0.01, flush_interval)
        self.max_batch_size = max(1, max_batch_size)
        self.max_queue_size = max(self.max_batch_size, max_queue_size)
        self._queue: asyncio.Queue[dict[str, Any] | _Wake] = asyncio.Queue(
            maxsize=self.max_queue_size
        )
        # Failed batches have priority over updates queued while the database
        # was unavailable. Retrying an older batch before newer checkpoints for
        # the same block preserves the ledger's event order.
        self._retry_batches: list[list[dict[str, Any]]] = []
        self._flusher_task: asyncio.Task[None] | None = None
        self._closed = False
        self._failures = 0
        self._retry_base_delay = 0.02
        self._save_lock = asyncio.Lock()
        #: Set by ``enqueue``/``close`` to unpark an idle worker. The worker
        #: waits on it WITHOUT holding ``_save_lock``, then drains + saves under
        #: the lock; a blocking ``queue.get()`` would hold a popped batch
        #: outside the lock and let ``flush_all`` commit newer items first.
        self._wake = asyncio.Event()

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
        await self._put(update)
        self._wake.set()

    async def _put(self, update: dict[str, Any]) -> None:
        """Hand *update* to the worker without hanging if the worker dies first.

        ``Queue.put`` blocks while the queue is full, waiting for the worker to
        free a slot. If the worker dies of its retry budget in that window,
        nobody will ever drain the queue and the producer hangs for the rest of
        the job — a silent deadlock where the class promises a loud failure.
        The liveness check at the top of ``enqueue`` ran *before* the put, so it
        cannot see a worker that dies while we are parked.

        Fast path: a non-blocking put. Only when the queue is full do we race
        the blocking put against the worker task, so a dead worker surfaces its
        error (or, for a clean exit, is restarted) instead of parking us.
        """
        try:
            self._queue.put_nowait(update)
            return
        except asyncio.QueueFull:
            pass
        task = self._flusher_task
        if task is None:
            # Closed concurrently between the check above and here: the queue
            # is being drained by ``close``; write straight to the ledger.
            await asyncio.to_thread(self.ledger.save_checkpoints_batch, [update])
            return
        put_task = asyncio.ensure_future(self._queue.put(update))
        try:
            done, _ = await asyncio.wait({put_task, task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            put_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await put_task
            raise
        if put_task in done:
            return
        # The worker finished before a slot opened up: re-surface its error (or
        # restart a cleanly-exited one) and retry, rather than block forever.
        put_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await put_task
        self._raise_if_task_died()
        if self._closed:
            await asyncio.to_thread(self.ledger.save_checkpoints_batch, [update])
            return
        self.start()
        await self._put(update)

    async def _save(self, batch: list[dict[str, Any]]) -> bool:
        """Persist *batch* under the save lock (see :meth:`_save_locked`)."""
        async with self._save_lock:
            return await self._save_locked(batch)

    async def _save_locked(self, batch: list[dict[str, Any]]) -> bool:
        """Persist *batch*; on failure retain it for the next retry window.

        Caller must hold ``_save_lock``. Returns True when the batch landed.
        ``asyncio.CancelledError`` is *not* caught here — it passes through so
        the worker's cancel handler can re-queue the batch (a cancelled
        ``to_thread`` that had not started would otherwise drop it).
        """
        if not batch:
            return True
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
                # Wait for work WITHOUT the lock, then drain + save atomically
                # under it. Holding the lock across the whole drain+save is what
                # stops ``flush_all`` from committing a newer queue item that an
                # older retried batch would then overwrite; it also keeps
                # the two save paths from ever running concurrently.
                if self._queue.empty() and not self._retry_batches:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._wake.wait(), timeout=self.flush_interval)
                    self._wake.clear()
                async with self._save_lock:
                    batch = self._drain_nowait(limit=self.max_batch_size)
                    if batch:
                        await self._save_locked(batch)
                # The batch was saved, or requeued inside ``_save_locked``. Only
                # a cancellation *during* the save leaves it unaccounted for.
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

    def _drain_nowait(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Pop retries and queued items, dropping wake sentinels.

        ``limit`` caps how many *queued* items are pulled (retry batches are
        always taken whole, so their order is preserved); ``None`` drains
        everything.
        """
        items: list[dict[str, Any]] = [item for batch in self._retry_batches for item in batch]
        self._retry_batches.clear()
        while limit is None or len(items) < limit:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return items
            self._queue.task_done()
            if not isinstance(item, _Wake):
                items.append(item)
        return items

    async def flush_all(self) -> None:
        """Drain and commit all currently enqueued items immediately.

        A failure retains the batch for the background worker (which owns the
        retry budget), so this mid-run checkpoint never aborts the stage. The
        drain runs under ``_save_lock`` so the worker cannot requeue an older
        failed batch on top of this one.
        """
        async with self._save_lock:
            batch = self._drain_nowait()
            if batch:
                await self._save_locked(batch)

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
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(_WAKE)
            self._wake.set()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(task)
            except BaseException as exc:
                task_exc = exc

        pending = self._drain_nowait()
        if pending:
            save_task = asyncio.ensure_future(self._save(pending))
            try:
                saved = await asyncio.shield(save_task)
            except asyncio.CancelledError:
                # ``shield`` keeps the save running when the awaiter is
                # cancelled, so an unobserved verdict would drop a failed final
                # flush behind the cancellation. Observe
                # the shielded task, surface a failed flush, then re-raise the
                # cancellation only when the flush actually succeeded.
                try:
                    saved = await save_task
                except BaseException:
                    saved = False
                if not saved and task_exc is None:
                    task_exc = RuntimeError(
                        f"Ledger checkpoint flush failed ({len(pending)} pending item(s)); "
                        "abandoning batched writes"
                    )
                if task_exc is not None:
                    raise task_exc from None
                raise
            if not saved and task_exc is None:
                task_exc = RuntimeError(
                    f"Ledger checkpoint flush failed ({len(pending)} pending item(s)); "
                    "abandoning batched writes"
                )
        if task_exc is not None:
            raise task_exc
