"""One poller for every queue-mode SSE subscriber (replaces the poll thread pool).

The old design gave each open stream its own polling loop and ran the blocking
``job_queue.get`` on a shared ``ThreadPoolExecutor(max_workers=4)``. With up to
64 concurrent subscribers that pool was oversubscribed 16:1, and a slow SQLite
read (contention with a writer) blocked the pool for *every* stream and for the
``/status`` endpoints that shared the default executor — one contended read
stalled the whole push surface.

This class inverts that: a single background coroutine polls the queue once per
tick (one batched query for every subscribed job), fans the snapshot out to each
subscriber's in-memory ``asyncio.Queue``, and terminates on a job's terminal
state. Subscribers never touch SQLite; they await their own queue. The number of
polling threads is therefore one, independent of subscriber count.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, QueuedJob

if TYPE_CHECKING:
    from ubt.core.engine.job_queue import JobQueue

logger = logging.getLogger(__name__)

#: Poll cadence. Matches the old per-stream sleep, so latency is unchanged.
DEFAULT_POLL_INTERVAL = 1.0

#: Per-subscriber buffer. Deep enough that a briefly-stalled client (a slow
#: write) does not lose the intermediate frames it will be caught up on anyway;
#: a snapshot stream converges, so dropping under sustained backpressure is
#: acceptable and bounded.
_SUBSCRIBER_QUEUE_MAXSIZE = 64

#: ``None`` on a subscriber queue signals "this job reached a terminal state".
_TERMINAL = None


@dataclass(eq=False)
class _Subscriber:
    job_id: str
    queue: asyncio.Queue[QueuedJob | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAXSIZE)
    )


class QueueSseBroadcaster:
    """Single-poller fan-out of queue-mode job snapshots to SSE subscribers."""

    def __init__(
        self,
        job_queue: JobQueue,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._job_queue = job_queue
        self._poll_interval = max(0.01, poll_interval)
        # Keyed by the subscriber object (``eq=False`` gives identity hashing),
        # so two subscribers for the same job are distinct entries.
        self._subscribers: dict[_Subscriber, None] = {}
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> _Subscriber | None:
        """Register a subscriber; ``None`` when the job does not exist.

        Starts the poller on the first subscriber and returns the subscriber's
        queue. The caller owns releasing it (``unsubscribe``) in a ``finally``.
        """
        exists = await asyncio.to_thread(self._job_queue.get, job_id)
        if exists is None:
            return None
        sub = _Subscriber(job_id=job_id)
        async with self._lock:
            self._subscribers[sub] = None
            self._ensure_poller()
        return sub

    async def unsubscribe(self, sub: _Subscriber) -> None:
        """Drop a subscriber; stops the poller once none remain."""
        async with self._lock:
            self._subscribers.pop(sub, None)
            if not self._subscribers and self._task is not None:
                self._task.cancel()
                self._task = None

    def _ensure_poller(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Poll subscribed jobs once per tick and fan out their snapshots."""
        try:
            while True:
                async with self._lock:
                    job_ids = [sub.job_id for sub in self._subscribers]
                if not job_ids:
                    return
                try:
                    jobs = await asyncio.to_thread(self._job_queue.get_many, job_ids)
                except Exception as exc:  # a transient DB error must not kill the poller
                    logger.warning("SSE broadcast poll failed: %s", exc)
                    await asyncio.sleep(self._poll_interval)
                    continue
                async with self._lock:
                    for sub in list(self._subscribers):
                        job = jobs.get(sub.job_id)
                        if job is None:
                            continue
                        _offer(sub.queue, job)
                        if job.status in TERMINAL_JOB_STATUSES:
                            _offer(sub.queue, _TERMINAL)
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            raise

    async def close(self) -> None:
        """Stop the poller (app shutdown)."""
        async with self._lock:
            self._subscribers.clear()
            if self._task is not None:
                self._task.cancel()
                self._task = None


def _offer(queue: asyncio.Queue[QueuedJob | None], item: QueuedJob | None) -> None:
    """Non-blocking put with drop-oldest backpressure (the queue never blocks the poller)."""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(item)


async def subscriber_frames(
    sub: _Subscriber,
    *,
    initial: QueuedJob,
) -> AsyncIterator[QueuedJob | None]:
    """Yield snapshots for a subscriber until its job reaches a terminal state.

    The first yield is *initial* (the row the caller already read to validate the
    job), so a subscriber does not wait a full poll tick for its first frame.
    ``None`` marks the terminal frame.
    """
    yield initial
    if initial.status in TERMINAL_JOB_STATUSES:
        yield _TERMINAL
        return
    while True:
        item = await sub.queue.get()
        yield item
        if item is _TERMINAL:
            return


__all__ = ["DEFAULT_POLL_INTERVAL", "QueueSseBroadcaster", "subscriber_frames"]
