"""The ``/stream`` surface: per-job SSE progress streams.

Split out of :mod:`ubt.api.app` verbatim (see :mod:`ubt.api.routes`); the
subscriber accounting (``QueueSubscriberCounter`` and the process-wide
ceiling) moved with it, since only this surface uses it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ubt.api.manager import JobManager
from ubt.api.routes.shared import (
    cross_tenant_404,
    progress_frame,
    public_artifact,
    terminal_frame,
)
from ubt.api.scope import ApiScope
from ubt.api.security import validate_job_id
from ubt.api.sse_broadcaster import subscriber_frames
from ubt.api.sse_replay import SseReplayBuffer
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES
from ubt.core.engine.progress import ProgressSnapshot

# Strong references to fire-and-forget tasks: the event loop holds only weak
# references, so an unreferenced task can be garbage-collected mid-flight and
# its work silently never happen.
background_tasks: set[asyncio.Task[None]] = set()


class QueueSubscriberCounter:
    """Per-job SSE subscriber accounting for queue mode.

    Embedded mode counts subscribers on the in-memory ``JobRecord``; queue mode
    has no such record but needs the same ceiling. Each open stream is a slot in
    the fan-out (see :class:`~ubt.api.sse_broadcaster.QueueSseBroadcaster`), so
    the ceiling bounds memory per subscriber as well as the read surface.
    Counting is per job id, not global.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(1, int(limit))
        self._counts: dict[str, int] = {}

    def acquire(self, job_id: str) -> bool:
        """Reserve a slot; ``False`` when the job already has ``limit`` of them."""
        current = self._counts.get(job_id, 0)
        if current >= self._limit:
            return False
        self._counts[job_id] = current + 1
        return True

    def release(self, job_id: str) -> None:
        """Give a slot back; never goes negative for an unbalanced release."""
        remaining = self._counts.get(job_id, 1) - 1
        if remaining > 0:
            self._counts[job_id] = remaining
        else:
            self._counts.pop(job_id, None)


#: Process-wide ceiling on simultaneously open SSE streams, across *all* jobs.
#:
#: The per-job ceilings above do not bound the process: with N jobs each at
#: ``_MAX_SUBSCRIBERS_PER_JOB`` the fan-out still grows linearly, and every
#: open stream polls from a worker thread once a second — enough subscribers
#: stall the shared default executor and every other endpoint with it
#: under high concurrency.
MAX_GLOBAL_STREAM_SUBSCRIBERS = 64

#: Single key the global stream budget is booked under, so the per-job counter
#: doubles as a process-wide total (one shared slot bucket, one ceiling).
GLOBAL_SUBSCRIBER_KEY = "__global__"


def register_stream_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the stream surface."""
    manager = scope.manager
    job_queue = scope.job_queue
    sse_broadcaster = scope.sse_broadcaster
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir

    # Queue mode has no in-memory ``JobRecord`` to count subscribers on, but it
    # needs the same ceiling (see QueueSubscriberCounter).
    queue_subscribers = QueueSubscriberCounter(JobManager._MAX_SUBSCRIBERS_PER_JOB)
    # ...and neither counter bounds the *process*: the global budget below is
    # what caps the fan-out across every job (see MAX_GLOBAL_STREAM_SUBSCRIBERS).
    # Booked on ``app.state`` so a caller (or a test) can exhaust it deliberately.
    global_subscribers = QueueSubscriberCounter(MAX_GLOBAL_STREAM_SUBSCRIBERS)
    api_app.state.global_stream_subscribers = global_subscribers

    def _acquire_global_stream_slot(per_job_key: str | None = None) -> None:
        """Reserve the process-wide stream slot or fail with 429.

        ``per_job_key`` is released back when the global budget is exhausted,
        so a refused subscriber never leaks the per-job slot it took first.
        """
        if global_subscribers.acquire(GLOBAL_SUBSCRIBER_KEY):
            return
        if per_job_key is not None:
            queue_subscribers.release(per_job_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many concurrent stream subscribers: the global limit of "
                f"{MAX_GLOBAL_STREAM_SUBSCRIBERS} is in use. Close a stream or "
                "retry after a job finishes."
            ),
        )

    # Bounded per-job frame history for Last-Event-ID replay (PRD §9.2).
    sse_replay = SseReplayBuffer()
    api_app.state.sse_replay = sse_replay
    api_app.state.sse_broadcaster = sse_broadcaster

    def _last_event_id(request: Request) -> int | None:
        """The client's replay cursor, from the SSE header or a query fallback.

        A fresh ``EventSource`` cannot set ``Last-Event-ID`` by hand, so a
        manually-reconnecting client passes it as ``?last_event_id=``.
        """
        raw = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @api_app.get(
        "/jobs/{job_id}/stream",
        response_class=StreamingResponse,
        tags=["Jobs"],
    )
    async def stream_progress(
        job_id: str, request: Request, x_ubt_tenant: str | None = Header(default=None)
    ) -> StreamingResponse:
        """SSE: ``event: progress`` snapshots until one ``event: <status>`` terminal frame.

        Both deployment modes emit that one contract (see ``progress_frame`` /
        ``terminal_frame``); the payload never depends on how the server was
        started. Every frame carries an ``id:`` cursor, so a client reconnecting
        with ``Last-Event-ID`` (header or ``?last_event_id=``) receives the
        frames it missed before the live stream resumes (PRD §9.2).
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        last_event_id = _last_event_id(request)
        if job_queue is not None:
            initial_job = await asyncio.to_thread(job_queue.get, valid_id)
            if initial_job is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job not found: {valid_id}",
                )
            if not queue_subscribers.acquire(valid_id):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many concurrent stream subscribers for this job",
                )
            _acquire_global_stream_slot(per_job_key=valid_id)
            queue_slot_released = False
            # One shared poller fans snapshots out to every subscriber's queue;
            # this stream never touches SQLite itself (see QueueSseBroadcaster).
            # A queue implies a broadcaster (both built together in create_app);
            # checked explicitly, not with ``assert``, so ``python -O`` cannot
            # strip it into an AttributeError on None.
            if sse_broadcaster is None:  # pragma: no cover - construction invariant
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="SSE broadcaster is not configured for queue mode",
                )
            sub = await sse_broadcaster.subscribe(valid_id)
            if sub is None:
                queue_subscribers.release(valid_id)
                global_subscribers.release(GLOBAL_SUBSCRIBER_KEY)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job not found: {valid_id}",
                )

            def _release_queue_stream_slots() -> None:
                nonlocal queue_slot_released
                if queue_slot_released:
                    return
                queue_slot_released = True
                queue_subscribers.release(valid_id)
                global_subscribers.release(GLOBAL_SUBSCRIBER_KEY)
                unsub = asyncio.ensure_future(sse_broadcaster.unsubscribe(sub))
                background_tasks.add(unsub)
                unsub.add_done_callback(background_tasks.discard)

            async def _queue_events() -> AsyncIterator[str]:
                last: dict[str, Any] | None = None
                try:
                    for frame in sse_replay.replay(valid_id, last_event_id):
                        yield frame
                    async for job in subscriber_frames(sub, initial=initial_job):
                        if await request.is_disconnected():
                            break
                        if job is None:
                            # Terminal: emit the final frame from the last row we
                            # saw. ``subscriber_frames`` only yields None after a
                            # terminal row, so ``last`` holds that row's payload.
                            break
                        progress_payload = {**ProgressSnapshot().to_payload(), **job.progress}
                        snapshot = {**progress_payload, "status": job.status.value}
                        if snapshot != last:
                            yield sse_replay.emit(
                                valid_id, progress_frame(progress_payload, job.status.value)
                            )
                            last = snapshot
                        if job.status in TERMINAL_JOB_STATUSES:
                            yield sse_replay.emit(
                                valid_id,
                                terminal_frame(
                                    job.status.value,
                                    public_artifact(job.progress.get("output_file")),
                                    job.error,
                                    counters=job.progress,
                                ),
                            )
                            break
                finally:
                    _release_queue_stream_slots()

            from starlette.background import BackgroundTask

            return StreamingResponse(
                _queue_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
                background=BackgroundTask(_release_queue_stream_slots),
            )
        record = manager.get_job(valid_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {valid_id}",
            )

        if len(record.subscribers) >= JobManager._MAX_SUBSCRIBERS_PER_JOB:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many concurrent stream subscribers for this job",
            )
        _acquire_global_stream_slot()
        queue: asyncio.Queue[TranslationProgressEvent | None] = asyncio.Queue(
            maxsize=JobManager._SUBSCRIBER_QUEUE_MAXSIZE
        )
        record.subscribers.append(queue)
        mem_slot_released = False

        def _release_mem_stream_slots() -> None:
            nonlocal mem_slot_released
            if mem_slot_released:
                return
            mem_slot_released = True
            global_subscribers.release(GLOBAL_SUBSCRIBER_KEY)
            if queue in record.subscribers:
                record.subscribers.remove(queue)

        async def event_generator() -> AsyncIterator[str]:
            try:
                for frame in sse_replay.replay(valid_id, last_event_id):
                    yield frame
                # A late subscriber gets one current snapshot, not a replay of
                # the raw event log: the queue-mode branch emits this exact
                # shape, and one endpoint must not answer two schemas depending
                # on the deployment mode.
                yield sse_replay.emit(
                    valid_id, progress_frame(record.progress.model_dump(), record.status)
                )

                if record.status in TERMINAL_JOB_STATUSES and queue.empty():
                    yield sse_replay.emit(
                        valid_id,
                        terminal_frame(
                            record.status,
                            public_artifact(record.progress.output_file),
                            record.error,
                            counters=record.progress.model_dump(),
                        ),
                    )
                    return

                while True:
                    if await request.is_disconnected():
                        break

                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        if record.status in TERMINAL_JOB_STATUSES:
                            yield sse_replay.emit(
                                valid_id,
                                terminal_frame(
                                    record.status,
                                    public_artifact(record.progress.output_file),
                                    record.error,
                                    counters=record.progress.model_dump(),
                                ),
                            )
                            break
                        continue

                    if event is None:
                        yield sse_replay.emit(
                            valid_id,
                            terminal_frame(
                                record.status,
                                public_artifact(record.progress.output_file),
                                record.error,
                            ),
                        )
                        break

                    yield sse_replay.emit(
                        valid_id, progress_frame(record.progress.model_dump(), record.status)
                    )
            finally:
                _release_mem_stream_slots()

        from starlette.background import BackgroundTask

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(_release_mem_stream_slots),
        )
