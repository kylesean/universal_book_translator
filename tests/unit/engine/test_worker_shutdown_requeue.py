from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.job_queue import JobQueue, JobStatus
from ubt.core.engine.job_worker import JobWorker

pytestmark = pytest.mark.fast


class _Job:
    job_id = "job-shutdown"
    payload: dict[str, Any] = {"input_path": "/tmp/book.md"}
    status = None


def _worker(tmp_path: Path) -> JobWorker:
    queue = JobQueue(tmp_path / "queue.sqlite")
    return JobWorker(
        queue,
        UBTConfig(db_dir=tmp_path / "ledgers", tm_enabled=False),
        worker_id="w-test",
        concurrency=1,
    )


async def _run_interrupted(
    worker: JobWorker, *, shutting_down: bool, cancel_requested: bool = False
) -> list[str]:
    # ``cancel_requested`` must only apply AFTER ``task.cancel()``: the event
    # loop checks the same flag on every event (job_worker's cancel-between-
    # events path), and arming it early takes the JobInterruptedError route
    # instead of the shutdown-race route this test exercises.
    calls: list[str] = []
    worker._shutting_down = shutting_down
    armed = {"v": False}

    async def fake_q(fn: Any, *args: Any, **kwargs: Any) -> Any:
        name = getattr(fn, "__name__", "")
        if name == "complete":
            calls.append(f"complete:{kwargs['status'].value}")
            return True
        if name == "release_claim":
            calls.append("release_claim")
            return True
        if name == "update_progress":
            return True
        if name == "is_cancel_requested":
            return cancel_requested and armed["v"]
        return None

    async def fake_source(*args: Any, **kwargs: Any) -> Any:
        while True:
            await asyncio.sleep(0.01)
            yield _event()

    worker._q = fake_q  # type: ignore[method-assign]
    worker._event_source = fake_source

    task = asyncio.create_task(worker.execute(_Job(), "w-test"))  # type: ignore[arg-type]
    await asyncio.sleep(0.05)
    task.cancel()
    armed["v"] = True
    with pytest.raises(asyncio.CancelledError):
        await task
    return calls


def _event() -> Any:
    from ubt.core.engine.events import EventType, TranslationProgressEvent

    return TranslationProgressEvent(
        event_type=EventType.DRAFT_BATCH_COMPLETED,
        job_id="job-shutdown",
        total_blocks=10,
        completed_blocks=1,
        message="progress",
    )


@pytest.mark.asyncio
async def test_a_shutdown_requeues_the_job(tmp_path: Path) -> None:
    calls = await _run_interrupted(_worker(tmp_path), shutting_down=True)
    assert calls == ["release_claim"], calls


@pytest.mark.asyncio
async def test_a_user_cancel_still_marks_the_job_cancelled(tmp_path: Path) -> None:
    calls = await _run_interrupted(_worker(tmp_path), shutting_down=False)
    assert calls == ["complete:cancelled"], calls


def test_stop_sets_the_shutdown_flag(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    assert worker._shutting_down is False
    worker.stop()
    assert worker._shutting_down is True


@pytest.mark.asyncio
async def test_a_signal_handler_runs_stop_before_the_slots_unwind(tmp_path: Path) -> None:
    from ubt.cli.commands.worker import _run_worker_until_signalled

    worker = _worker(tmp_path)
    observed: list[bool] = []

    async def fake_slot(worker_id: str, *, drain: bool) -> int:
        await worker._stop.wait()
        observed.append(worker._shutting_down)
        return 0

    worker._slot = fake_slot  # type: ignore[method-assign]

    async def stop_soon() -> None:
        await asyncio.sleep(0.05)
        worker.stop()

    asyncio.create_task(stop_soon())
    await asyncio.wait_for(_run_worker_until_signalled(worker), timeout=5.0)
    assert observed == [True]


def test_the_queue_reclaims_a_requeued_row(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue.sqlite")
    queue.enqueue("job-a", {"input_path": "/tmp/a.md"})
    claimed = queue.claim("w1")
    assert claimed is not None
    assert claimed.status is JobStatus.RUNNING

    assert queue.release_claim("job-a", "w1") is True
    again = queue.claim("w2")
    assert again is not None and again.job_id == "job-a"

    queue.request_cancel("job-a")
    assert queue.claim("w3") is None


async def test_worker_writer_lock_conflict_backs_off_using_lease_expiry(tmp_path: Path) -> None:
    import time
    import unittest.mock

    from ubt.core.exceptions import LedgerWriterLockConflictError

    worker = _worker(tmp_path)
    job = _Job()
    slept: list[float] = []

    async def fake_sleep(duration: float) -> None:
        slept.append(duration)

    expiry = time.time() + 45.0
    conflict = LedgerWriterLockConflictError(
        "held", details={"job_id": job.job_id, "lease_expires_at": expiry}
    )

    async def throwing_event_source(*args: Any, **kwargs: Any) -> Any:
        raise conflict
        if False:
            yield

    worker._event_source = throwing_event_source

    with unittest.mock.patch("asyncio.sleep", side_effect=fake_sleep):
        result = await worker.execute(job)  # type: ignore[arg-type]

    assert result is False
    assert any(40.0 <= s <= 46.0 for s in slept)


@pytest.mark.asyncio
async def test_a_user_cancel_that_raced_the_shutdown_still_wins(tmp_path: Path) -> None:
    """``stop()`` must not resurrect a job the cancel API already answered.

    Both events arrive as a task cancellation; the queue row's
    ``cancel_requested`` flag is the one durable distinction, so the shutdown
    requeue defers to it instead of re-arming a job the operator just killed.
    """
    calls = await _run_interrupted(_worker(tmp_path), shutting_down=True, cancel_requested=True)
    assert calls == ["complete:cancelled"], calls
