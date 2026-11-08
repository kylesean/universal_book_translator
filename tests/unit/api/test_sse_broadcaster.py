from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ubt.api.sse_broadcaster import (
    _TERMINAL,
    QueueSseBroadcaster,
    subscriber_frames,
)
from ubt.core.engine.job_queue import JobQueue, JobStatus

pytestmark = pytest.mark.fast


def _queue(tmp_path: Path) -> JobQueue:
    return JobQueue(tmp_path / "queue.db")


async def test_subscribe_returns_none_for_an_unknown_job(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    broadcaster = QueueSseBroadcaster(q, poll_interval=0.01)
    try:
        assert await broadcaster.subscribe("nosuchjob") is None
    finally:
        await broadcaster.close()
        q.close()


async def test_a_subscriber_receives_progress_then_terminal(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("job1", {"input_path": "x"}, now=1.0)
    claimed = q.claim("w1", now=2.0)
    assert claimed is not None and claimed.job_id == "job1"
    q.update_progress("job1", "w1", {"stage": "draft"}, now=3.0)

    broadcaster = QueueSseBroadcaster(q, poll_interval=0.01)
    try:
        sub = await broadcaster.subscribe("job1")
        assert sub is not None
        frames = subscriber_frames(sub, initial=claimed)

        assert await anext(frames) is claimed
        running = await asyncio.wait_for(anext(frames), timeout=2.0)
        assert running is not _TERMINAL
        assert running.status is JobStatus.RUNNING
        assert running.progress.get("stage") == "draft"

        q.complete("job1", "w1", status=JobStatus.COMPLETED, now=4.0)
        item = await asyncio.wait_for(anext(frames), timeout=2.0)
        while item is not _TERMINAL:
            assert item.status is JobStatus.COMPLETED
            item = await asyncio.wait_for(anext(frames), timeout=2.0)
        assert item is _TERMINAL
    finally:
        await broadcaster.close()
        q.close()


async def test_a_terminal_initial_snapshot_terminates_immediately(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("job1", {"input_path": "x"}, now=1.0)
    claimed = q.claim("w1", now=2.0)
    assert claimed is not None
    q.complete("job1", "w1", status=JobStatus.FAILED, error="boom", now=3.0)
    terminal = q.get("job1")
    assert terminal is not None

    broadcaster = QueueSseBroadcaster(q, poll_interval=5.0)
    try:
        sub = await broadcaster.subscribe("job1")
        assert sub is not None
        frames = [item async for item in subscriber_frames(sub, initial=terminal)]
        assert frames[-1] is _TERMINAL
        assert len(frames) <= 2
    finally:
        await broadcaster.close()
        q.close()


async def test_one_poller_serves_many_subscribers(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    for i in range(5):
        q.enqueue(f"job{i}", {"input_path": "x"}, now=1.0)
    broadcaster = QueueSseBroadcaster(q, poll_interval=0.01)
    try:
        subs = [await broadcaster.subscribe(f"job{i}") for i in range(5)]
        assert all(s is not None for s in subs)
        assert broadcaster._task is not None
        assert len(broadcaster._subscribers) == 5
    finally:
        await broadcaster.close()
        q.close()


async def test_unsubscribing_the_last_subscriber_stops_the_poller(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue("job1", {"input_path": "x"}, now=1.0)
    broadcaster = QueueSseBroadcaster(q, poll_interval=0.01)
    try:
        sub = await broadcaster.subscribe("job1")
        assert sub is not None
        assert broadcaster._task is not None
        await broadcaster.unsubscribe(sub)
        assert broadcaster._task is None
    finally:
        await broadcaster.close()
        q.close()


async def test_a_job_row_deleted_mid_stream_ends_the_stream(tmp_path: Path) -> None:
    """A deleted row can never produce another snapshot: send the sentinel.

    The poller used to ``continue`` on a missing row, hanging the subscriber —
    and its global stream slot — until the client disconnected by itself.
    """
    q = _queue(tmp_path)
    q.enqueue("job1", {"input_path": "x"}, now=1.0)
    broadcaster = QueueSseBroadcaster(q, poll_interval=0.01)
    try:
        sub = await broadcaster.subscribe("job1")
        assert sub is not None
        initial_row = q.get("job1")
        assert initial_row is not None
        frames = subscriber_frames(sub, initial=initial_row)
        assert q.delete("job1")

        # The first frame is the replayed initial snapshot, not the sentinel.
        first_frame = await asyncio.wait_for(anext(frames), timeout=2.0)
        assert first_frame is not _TERMINAL
        item = await asyncio.wait_for(anext(frames), timeout=2.0)
        assert item is _TERMINAL

        # The dead subscription is dropped, so nothing keeps polling the row.
        async with broadcaster._lock:
            assert all(s.job_id != "job1" for s in broadcaster._subscribers)
    finally:
        await broadcaster.close()
        q.close()
