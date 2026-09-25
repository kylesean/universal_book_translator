"""Durable SQLite job queue: claim, lease, fairness, reclaim, cancellation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.job_queue import (
    JobQueue,
    JobStatus,
    QueuedJob,
)
from ubt.core.exceptions import QueueDepthExceededError


@pytest.fixture
def queue(tmp_path: Path) -> Iterator[JobQueue]:
    q = JobQueue(
        tmp_path / "job_queue.sqlite",
        global_max_running=2,
        default_tenant_max_running=2,
    )
    yield q
    q.close()


def _enqueue(
    q: JobQueue, job_id: str, *, tenant: str = "default", priority: int = 0, now: float = 0.0
) -> QueuedJob:
    return q.enqueue(
        job_id, {"input": f"/x/{job_id}.pdf"}, tenant_id=tenant, priority=priority, now=now
    )


def _claim(q: JobQueue, worker: str, *, now: float) -> QueuedJob:
    job = q.claim(worker, now=now)
    assert job is not None
    return job


def test_enqueue_is_idempotent_on_job_id(queue: JobQueue) -> None:
    first = _enqueue(queue, "j1")
    assert first.status is JobStatus.QUEUED
    assert first.attempts == 0
    assert first.payload == {"input": "/x/j1.pdf"}

    queue.claim("w1", now=1.0)
    again = _enqueue(queue, "j1", now=2.0)
    # Resubmission must not reset a job that already moved on.
    assert again.status is JobStatus.RUNNING
    assert again.worker_id == "w1"


def test_claim_orders_by_priority_then_fifo(queue: JobQueue) -> None:
    _enqueue(queue, "low", priority=0, now=1.0)
    _enqueue(queue, "high", priority=5, now=2.0)
    _enqueue(queue, "high2", priority=5, now=3.0)

    assert _claim(queue, "w1", now=10.0).job_id == "high"
    assert _claim(queue, "w2", now=11.0).job_id == "high2"
    # remaining is the low-priority job once a slot frees up
    assert queue.claim("w3", now=12.0) is None  # global cap 2 reached


def test_global_limit_blocks_claim(queue: JobQueue) -> None:
    _enqueue(queue, "a")
    _enqueue(queue, "b")
    _enqueue(queue, "c")
    assert queue.claim("w1", now=1.0) is not None
    assert queue.claim("w2", now=2.0) is not None
    assert queue.claim("w3", now=3.0) is None


def test_per_tenant_cap_lets_other_tenants_through(tmp_path: Path) -> None:
    q = JobQueue(
        tmp_path / "q.sqlite",
        global_max_running=10,
        tenant_max_running={"a": 1},
        default_tenant_max_running=1,
    )
    try:
        _enqueue(q, "a1", tenant="a", now=1.0)
        _enqueue(q, "a2", tenant="a", now=2.0)
        _enqueue(q, "b1", tenant="b", now=3.0)

        assert _claim(q, "w1", now=10.0).job_id == "a1"
        # a is at its cap: the next slot must go to b, not a2.
        assert _claim(q, "w2", now=11.0).job_id == "b1"
        assert q.claim("w3", now=12.0) is None

        q.complete("a1", "w1", status=JobStatus.COMPLETED, now=13.0)
        assert _claim(q, "w4", now=14.0).job_id == "a2"
    finally:
        q.close()


def test_heartbeat_extends_only_for_owner(queue: JobQueue) -> None:
    _enqueue(queue, "a")
    job = queue.claim("w1", lease_seconds=60.0, now=100.0)
    assert job is not None and job.lease_expires_at == 160.0
    assert queue.heartbeat("a", "w1", lease_seconds=30.0, now=120.0) is True
    beat = queue.get("a")
    assert beat is not None and beat.lease_expires_at == 150.0
    assert queue.heartbeat("a", "other", now=121.0) is False


def test_stale_lease_is_requeued_then_failed_at_max_attempts(queue: JobQueue) -> None:
    queue.enqueue("a", {}, max_attempts=2, now=0.0)
    assert queue.claim("w1", lease_seconds=10.0, now=0.0) is not None
    assert queue.claim("w2", lease_seconds=10.0, now=1.0) is None  # cap; still running

    # Lease (expires at 10) lapses: attempt 1 of 2 -> requeued.
    assert queue.reclaim_stale(now=11.0) == (1, 0)
    requeued = queue.get("a")
    assert requeued is not None
    assert requeued.status is JobStatus.QUEUED
    assert requeued.attempts == 1

    # Claim again, lapse again: attempt 2 of 2 -> dead-lettered.
    assert queue.claim("w3", lease_seconds=10.0, now=12.0) is not None
    assert queue.reclaim_stale(now=23.0) == (0, 1)
    dead = queue.get("a")
    assert dead is not None
    assert dead.status is JobStatus.FAILED
    assert "lease expired" in (dead.error or "")


def test_cancel_queued_vs_running(queue: JobQueue) -> None:
    _enqueue(queue, "q")
    cancelled = queue.request_cancel("q", now=1.0)
    assert cancelled is not None and cancelled.status is JobStatus.CANCELLED
    assert queue.claim("w1", now=2.0) is None

    _enqueue(queue, "r", now=3.0)
    job = queue.claim("w1", now=4.0)
    assert job is not None and job.cancel_requested is False
    assert queue.is_cancel_requested("r", "w1") is False
    running = queue.request_cancel("r", now=5.0)
    assert running is not None and running.status is JobStatus.RUNNING
    assert queue.is_cancel_requested("r", "w1") is True
    # A different worker cannot observe the flag (it does not own the job).
    assert queue.is_cancel_requested("r", "w2") is False


def test_cancel_while_running_then_worker_dies_requeues_as_cancelled(
    queue: JobQueue,
) -> None:
    """§10.3-#3: a cancel during a crashed run must not be silently dropped.

    The user cancels while the job is RUNNING (flag set), then the worker dies
    and its lease lapses. Reclaiming must honour the cancel — the old path
    requeued the row and ``claim`` reset ``cancel_requested`` to 0, so the job
    was re-run and re-billed against an intent nobody ever acted on.
    """
    _enqueue(queue, "z")
    _claim(queue, "w1", now=1.0)
    assert queue.request_cancel("z", now=2.0) is not None
    assert queue.is_cancel_requested("z", "w1") is True

    # Lease (expires at ~61 with default) lapses without the worker finishing.
    requeued, failed = queue.reclaim_stale(now=10_000.0)
    assert (requeued, failed) == (0, 0)  # neither requeued nor dead-lettered
    row = queue.get("z")
    assert row is not None and row.status is JobStatus.CANCELLED

    # And it is not claimable again — no re-run, no re-bill.
    assert queue.claim("w2", now=10_001.0) is None


def test_complete_and_progress(queue: JobQueue) -> None:
    _enqueue(queue, "a")
    job = queue.claim("w1", now=1.0)
    assert job is not None
    # A non-owner cannot write progress over the running row.
    assert queue.update_progress("a", "other", {"completed_blocks": 1}, now=2.0) is False
    untouched = queue.get("a")
    assert untouched is not None and untouched.progress == {}
    assert queue.update_progress("a", "w1", {"completed_blocks": 3}, now=2.0) is True
    progressed = queue.get("a")
    assert progressed is not None and progressed.progress == {"completed_blocks": 3}

    assert queue.complete("a", "other", status=JobStatus.COMPLETED, now=3.0) is False
    assert (
        queue.complete(
            "a",
            "w1",
            status=JobStatus.COMPLETED,
            error=None,
            progress={"completed_blocks": 9},
            now=4.0,
        )
        is True
    )
    done = queue.get("a")
    assert done is not None
    assert done.status is JobStatus.COMPLETED
    assert done.worker_id is None
    assert done.finished_at == 4.0
    assert done.progress == {"completed_blocks": 9}


def test_position_and_depth(queue: JobQueue) -> None:
    _enqueue(queue, "first", priority=0, now=1.0)
    _enqueue(queue, "urgent", priority=9, now=2.0)
    _enqueue(queue, "last", priority=0, now=3.0)
    assert queue.queue_position("urgent") == 1
    assert queue.queue_position("first") == 2
    assert queue.queue_position("last") == 3

    assert queue.depth() == {"queued": 3}
    claim = queue.claim("w1", now=10.0)
    assert claim is not None and claim.job_id == "urgent"
    assert queue.depth() == {"queued": 2, "running": 1}
    assert queue.depth(tenant_id="nosuch") == {}
    assert queue.queue_position("first") == 1
    assert queue.queue_position("urgent") is None  # running, no longer queued


def test_enqueue_refuses_past_the_depth_cap(tmp_path: Path) -> None:
    """Intake is bounded; resubmitting a known id stays idempotent (review-2 X6).

    ``claim`` caps how many jobs run at once, not how many pile up, so an
    uncapped ``enqueue`` let ``POST /jobs/submit`` in a loop grow the queue's
    SQLite file without bound while workers drained at LLM speed.
    """
    q = JobQueue(tmp_path / "capped.sqlite", max_queued=2)
    try:
        _enqueue(q, "a")
        _enqueue(q, "b")
        with pytest.raises(QueueDepthExceededError):
            _enqueue(q, "c")
        # A resubmit of an already-queued id is a no-op, not a new row, so the
        # depth check must not refuse it.
        assert _enqueue(q, "a").job_id == "a"
        assert q.depth() == {"queued": 2}
        # Draining one slot lets the next submission through.
        assert q.claim("w1", now=1.0) is not None
        _enqueue(q, "c")
        assert q.depth() == {"queued": 2, "running": 1}
    finally:
        q.close()


def test_release_claim_honours_a_pending_cancel(queue: JobQueue) -> None:
    """A cancel the user already got an answer for cannot be dropped.

    Writer-lock contention released the claim back to QUEUED with
    ``cancel_requested`` still set — and the next ``claim`` zeroes the flag, so
    the job re-ran and re-billed itself against an acknowledged cancellation.
    ``reclaim_stale`` already resolves that state to CANCELLED; release_claim
    must do the same.
    """
    _enqueue(queue, "j_cancel_release")
    _claim(queue, "w1", now=1.0)
    cancelled = queue.request_cancel("j_cancel_release")
    assert cancelled is not None and cancelled.cancel_requested

    assert queue.release_claim("j_cancel_release", "w1", error="writer lock") is True
    job = queue.get("j_cancel_release")
    assert job is not None and job.status is JobStatus.CANCELLED

    # Nothing is left to hand out: the intent survived the requeue path.
    assert queue.claim("w2", now=2.0) is None


def test_release_claim_without_a_cancel_still_requeues(queue: JobQueue) -> None:
    _enqueue(queue, "j_requeue")
    _claim(queue, "w1", now=1.0)
    assert queue.release_claim("j_requeue", "w1", error="writer lock") is True
    job = queue.get("j_requeue")
    assert job is not None and job.status is JobStatus.QUEUED
    assert _claim(queue, "w2", now=2.0).job_id == "j_requeue"


@pytest.mark.fast
def test_job_queue_release_claim_is_atomic(tmp_path: Path) -> None:
    queue_db = tmp_path / "queue.sqlite"
    queue = JobQueue(queue_db)
    job = queue.enqueue("job_test", {"input_path": "a.txt"})
    claimed = queue.claim("w1")
    assert claimed is not None

    executed_sqls: list[str] = []

    class TracingConnection:
        def __init__(self, real_conn: Any) -> None:
            self.real_conn = real_conn

        def __getattr__(self, name: str) -> Any:
            return getattr(self.real_conn, name)

        def execute(self, sql: str, *args: Any) -> Any:
            executed_sqls.append(sql)
            return self.real_conn.execute(sql, *args)

    from contextlib import contextmanager
    from unittest.mock import patch

    with patch.object(queue, "_get_conn") as mock_conn_ctx:

        @contextmanager
        def _tracing() -> Iterator[Any]:
            yield TracingConnection(queue._conn)

        mock_conn_ctx.side_effect = _tracing

        queue.release_claim(job.job_id, "w1")
        assert any("BEGIN IMMEDIATE" in sql for sql in executed_sqls)
        assert any("COMMIT" in sql for sql in executed_sqls)
