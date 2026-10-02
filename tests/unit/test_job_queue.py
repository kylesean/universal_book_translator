"""The durable job queue: submission idempotency, atomic claims, and leases.

The queue is the service front door's durability guarantee: a submitted job must
survive a restart, two workers must never claim the same row, and a dead worker's
job must be reclaimed -- or, past its attempt budget, failed -- without losing a
user's cancel. These tests pin those contracts against a real SQLite file.

Time is always passed explicitly via ``now`` so every case is deterministic; no
wall-clock sleeping, no threads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.job_queue import JobQueue, JobStatus
from ubt.core.exceptions import QueueDepthExceededError

pytestmark = pytest.mark.fast


def _queue(tmp_path: Path, **kwargs: Any) -> JobQueue:
    return JobQueue(tmp_path / "queue.db", **kwargs)


# --------------------------------------------------------------------------- #
# Submission: persist once, then be idempotent or re-run a dead id.
# --------------------------------------------------------------------------- #


def test_enqueue_persists_a_queued_row(tmp_path: Path) -> None:
    job = _queue(tmp_path).enqueue("j1", {"prompt": "hi"}, priority=3, now=10.0)
    assert job.status is JobStatus.QUEUED
    assert job.attempts == 0
    assert job.max_attempts == 3
    assert job.payload == {"prompt": "hi"}
    assert job.tenant_id == "default"
    assert job.priority == 3
    assert job.enqueued_at == 10.0


def test_enqueue_honours_an_explicit_max_attempts(tmp_path: Path) -> None:
    job = _queue(tmp_path).enqueue("j1", {}, max_attempts=7, now=1.0)
    assert job.max_attempts == 7


def test_enqueue_is_idempotent_for_a_live_id(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    original = queue.enqueue("j1", {"v": 1}, now=1.0)
    again = queue.enqueue("j1", {"v": 999}, now=2.0)
    assert again.status is JobStatus.QUEUED
    assert again.payload == {"v": 1}
    assert again.enqueued_at == original.enqueued_at


def test_enqueue_is_idempotent_for_a_completed_id(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {"v": 1}, now=1.0)
    queue.claim("w", now=2.0)
    queue.complete("j1", "w", status=JobStatus.COMPLETED, now=3.0)
    again = queue.enqueue("j1", {"v": 2}, now=4.0)
    assert again.status is JobStatus.COMPLETED
    assert again.payload == {"v": 1}


def test_enqueue_requeues_a_dead_id_in_place(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {"v": 1}, now=1.0)
    queue.claim("w", now=2.0)
    queue.update_progress("j1", "w", {"done": 1})
    queue.complete("j1", "w", status=JobStatus.FAILED, error="boom", now=3.0)

    revived = queue.enqueue("j1", {"v": 2}, now=4.0)
    assert revived.status is JobStatus.QUEUED
    assert revived.attempts == 0
    assert revived.payload == {"v": 2}
    assert revived.error is None
    assert revived.worker_id is None
    assert revived.started_at is None
    assert revived.finished_at is None
    assert revived.lease_expires_at is None
    assert revived.heartbeat_at is None
    assert revived.cancel_requested is False
    assert revived.progress == {}


def test_enqueue_does_not_requeue_a_dead_id_across_tenants(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {"v": 1}, tenant_id="tenant-a", now=1.0)
    queue.claim("w", now=2.0)
    queue.complete("j1", "w", status=JobStatus.FAILED, error="boom", now=3.0)

    # A different tenant resubmitting the same id must not hijack the dead row.
    again = queue.enqueue("j1", {"v": 2}, tenant_id="tenant-b", now=4.0)
    assert again.status is JobStatus.FAILED
    assert again.tenant_id == "tenant-a"
    assert again.payload == {"v": 1}


def test_enqueue_rejects_a_new_row_past_the_depth_cap(tmp_path: Path) -> None:
    queue = _queue(tmp_path, max_queued=2)
    queue.enqueue("j1", {}, now=1.0)
    queue.enqueue("j2", {}, now=2.0)
    with pytest.raises(QueueDepthExceededError):
        queue.enqueue("j3", {}, now=3.0)


def test_enqueue_resubmitting_a_live_id_when_full_does_not_raise(tmp_path: Path) -> None:
    # The cap guards *new* rows; an idempotent resubmit adds none.
    queue = _queue(tmp_path, max_queued=1)
    queue.enqueue("j1", {}, now=1.0)
    again = queue.enqueue("j1", {}, now=2.0)
    assert again.job_id == "j1"


def test_depth_cap_counts_only_queued_rows(tmp_path: Path) -> None:
    queue = _queue(tmp_path, max_queued=1)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w", now=2.0)  # j1 is RUNNING now, so nothing is QUEUED
    queue.enqueue("j2", {}, now=3.0)  # allowed
    assert queue.depth().get("queued") == 1


# --------------------------------------------------------------------------- #
# Claim: atomic selection, ordering, and running caps.
# --------------------------------------------------------------------------- #


def test_claim_returns_none_when_nothing_is_queued(tmp_path: Path) -> None:
    assert _queue(tmp_path).claim("w", now=1.0) is None


def test_claim_prefers_priority_then_fifo(tmp_path: Path) -> None:
    queue = _queue(tmp_path, global_max_running=10)
    queue.enqueue("low", {}, priority=0, now=1.0)
    queue.enqueue("high-late", {}, priority=5, now=2.0)
    queue.enqueue("high-early", {}, priority=5, now=0.5)
    order = [queue.claim("w", now=100.0).job_id for _ in range(3)]  # type: ignore[union-attr]
    assert order == ["high-early", "high-late", "low"]


def test_claim_marks_running_and_starts_the_lease(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    job = queue.claim("worker-1", lease_seconds=30.0, now=10.0)
    assert job is not None
    assert job.status is JobStatus.RUNNING
    assert job.attempts == 1
    assert job.worker_id == "worker-1"
    assert job.started_at == 10.0
    assert job.heartbeat_at == 10.0
    assert job.lease_expires_at == 40.0
    assert job.cancel_requested is False


def test_claim_increments_attempts_on_each_claim(tmp_path: Path) -> None:
    queue = _queue(tmp_path, default_max_attempts=3)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", lease_seconds=5.0, now=10.0)
    queue.reclaim_stale(now=16.0)
    second = queue.claim("w2", now=17.0)
    assert second is not None
    assert second.attempts == 2


def test_claim_enforces_the_global_running_cap(tmp_path: Path) -> None:
    queue = _queue(tmp_path, global_max_running=1)
    queue.enqueue("j1", {}, now=1.0)
    queue.enqueue("j2", {}, now=2.0)
    assert queue.claim("w1", now=10.0) is not None
    assert queue.claim("w2", now=11.0) is None


def test_claim_skips_a_tenant_at_its_cap(tmp_path: Path) -> None:
    queue = _queue(
        tmp_path,
        tenant_max_running={"tenant-a": 1},
        default_tenant_max_running=1,
        global_max_running=10,
    )
    queue.enqueue("a1", {}, tenant_id="tenant-a", now=1.0)
    queue.enqueue("a2", {}, tenant_id="tenant-a", now=2.0)
    queue.enqueue("b1", {}, tenant_id="tenant-b", now=3.0)

    assert queue.claim("w", now=10.0).job_id == "a1"  # type: ignore[union-attr]
    # tenant-a is at its cap, so the next claim takes tenant-b's job instead.
    assert queue.claim("w", now=11.0).job_id == "b1"  # type: ignore[union-attr]
    assert queue.claim("w", now=12.0) is None


def test_claim_reclaims_an_expired_lease_before_choosing(tmp_path: Path) -> None:
    queue = _queue(tmp_path, default_max_attempts=3)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("dead-worker", lease_seconds=5.0, now=10.0)  # lease expires at 15
    reclaimer = queue.claim("live-worker", lease_seconds=5.0, now=16.0)
    assert reclaimer is not None
    assert reclaimer.job_id == "j1"
    assert reclaimer.worker_id == "live-worker"
    assert reclaimer.attempts == 2


# --------------------------------------------------------------------------- #
# Lease ownership: only the current owner may act on a running job.
# --------------------------------------------------------------------------- #


def test_heartbeat_extends_only_for_the_owner(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", lease_seconds=5.0, now=10.0)
    assert queue.heartbeat("j1", "w2", now=11.0) is False
    assert queue.heartbeat("j1", "w1", lease_seconds=5.0, now=11.0) is True
    assert queue.get("j1").lease_expires_at == 16.0  # type: ignore[union-attr]


def test_update_progress_only_for_the_owner(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    assert queue.update_progress("j1", "w2", {"p": 1}) is False
    assert queue.update_progress("j1", "w1", {"p": 1}) is True
    assert queue.get("j1").progress == {"p": 1}  # type: ignore[union-attr]


def test_is_cancel_requested_only_for_the_owner(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    queue.request_cancel("j1", now=3.0)
    assert queue.is_cancel_requested("j1", "w1") is True
    assert queue.is_cancel_requested("j1", "w2") is False
    assert queue.is_cancel_requested("missing", "w1") is False


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUBMITTED])
def test_complete_requires_a_terminal_status(tmp_path: Path, status: JobStatus) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    with pytest.raises(ValueError):
        queue.complete("j1", "w", status=status, now=2.0)


def test_complete_finishes_and_clears_the_lease(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    assert queue.complete("j1", "w1", status=JobStatus.FAILED, error="nope", now=9.0) is True
    job = queue.get("j1")
    assert job is not None
    assert job.status is JobStatus.FAILED
    assert job.error == "nope"
    assert job.finished_at == 9.0
    assert job.worker_id is None
    assert job.lease_expires_at is None
    assert job.heartbeat_at is None


def test_complete_refuses_when_the_lease_was_lost(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    assert queue.complete("j1", "w2", status=JobStatus.COMPLETED, now=3.0) is False
    assert queue.get("j1").status is JobStatus.RUNNING  # type: ignore[union-attr]


def test_complete_keeps_progress_when_none_is_passed(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    queue.update_progress("j1", "w1", {"done": 7})
    queue.complete("j1", "w1", status=JobStatus.COMPLETED, now=3.0)
    assert queue.get("j1").progress == {"done": 7}  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Releasing a claim: transient contention must not burn the attempt budget.
# --------------------------------------------------------------------------- #


def test_release_claim_requeues_and_refunds_the_attempt(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    assert queue.release_claim("j1", "w1", error="lock contention", now=3.0) is True
    job = queue.get("j1")
    assert job is not None
    assert job.status is JobStatus.QUEUED
    assert job.attempts == 0
    assert job.worker_id is None
    assert job.error == "lock contention"


def test_release_claim_can_keep_the_attempt(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    queue.release_claim("j1", "w1", decrement_attempt=False, now=3.0)
    assert queue.get("j1").attempts == 1  # type: ignore[union-attr]


def test_release_claim_honours_a_pending_cancel(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    queue.request_cancel("j1", now=3.0)
    queue.release_claim("j1", "w1", now=4.0)
    assert queue.get("j1").status is JobStatus.CANCELLED  # type: ignore[union-attr]


def test_release_claim_refuses_a_foreign_worker(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    assert queue.release_claim("j1", "w2", now=3.0) is False
    assert queue.get("j1").status is JobStatus.RUNNING  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Stale-lease reclamation: requeue, fail, or honour a cancel.
# --------------------------------------------------------------------------- #


def test_reclaim_stale_requeues_below_max_attempts(tmp_path: Path) -> None:
    queue = _queue(tmp_path, default_max_attempts=3)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", lease_seconds=5.0, now=10.0)  # lease expires at 15
    assert queue.reclaim_stale(now=16.0) == (1, 0)
    job = queue.get("j1")
    assert job is not None
    assert job.status is JobStatus.QUEUED
    assert job.attempts == 1
    assert job.worker_id is None
    assert job.started_at is None


def test_reclaim_stale_fails_at_max_attempts(tmp_path: Path) -> None:
    queue = _queue(tmp_path, default_max_attempts=1)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", lease_seconds=5.0, now=10.0)  # attempts reaches max
    assert queue.reclaim_stale(now=16.0) == (0, 1)
    job = queue.get("j1")
    assert job is not None
    assert job.status is JobStatus.FAILED
    assert job.error == "lease expired after max attempts"


def test_reclaim_stale_honours_a_pending_cancel(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", lease_seconds=5.0, now=10.0)
    queue.request_cancel("j1", now=11.0)
    assert queue.reclaim_stale(now=16.0) == (0, 0)
    assert queue.get("j1").status is JobStatus.CANCELLED  # type: ignore[union-attr]


def test_reclaim_stale_leaves_a_live_lease_alone(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", lease_seconds=10.0, now=100.0)  # lease expires at 110
    assert queue.reclaim_stale(now=110.0) == (0, 0)  # boundary is strict '<'
    assert queue.get("j1").status is JobStatus.RUNNING  # type: ignore[union-attr]
    assert queue.reclaim_stale(now=110.001) == (1, 0)


# --------------------------------------------------------------------------- #
# Cancellation.
# --------------------------------------------------------------------------- #


def test_request_cancel_cancels_a_queued_job(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    cancelled = queue.request_cancel("j1", now=2.0)
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.finished_at == 2.0


def test_request_cancel_flags_a_running_job(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    flagged = queue.request_cancel("j1", now=3.0)
    assert flagged is not None
    assert flagged.status is JobStatus.RUNNING
    assert flagged.cancel_requested is True


def test_request_cancel_returns_none_for_an_unknown_job(tmp_path: Path) -> None:
    assert _queue(tmp_path).request_cancel("missing", now=1.0) is None


def test_request_cancel_leaves_a_terminal_job_alone(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.claim("w1", now=2.0)
    queue.complete("j1", "w1", status=JobStatus.COMPLETED, now=3.0)
    still = queue.request_cancel("j1", now=4.0)
    assert still is not None
    assert still.status is JobStatus.COMPLETED


# --------------------------------------------------------------------------- #
# Queries and lifecycle.
# --------------------------------------------------------------------------- #


def test_queue_position_is_one_based_by_priority(tmp_path: Path) -> None:
    queue = _queue(tmp_path, global_max_running=10)
    queue.enqueue("a", {}, priority=0, now=1.0)
    queue.enqueue("b", {}, priority=5, now=2.0)
    queue.enqueue("c", {}, priority=0, now=3.0)
    assert queue.queue_position("b") == 1
    assert queue.queue_position("a") == 2
    assert queue.queue_position("c") == 3
    queue.claim("w", now=10.0)  # b leaves the queue
    assert queue.queue_position("b") is None
    assert queue.queue_position("missing") is None


def test_depth_counts_by_status_and_tenant(tmp_path: Path) -> None:
    queue = _queue(tmp_path, global_max_running=10)
    queue.enqueue("a", {}, tenant_id="tenant-a", now=1.0)
    queue.enqueue("b", {}, tenant_id="tenant-b", now=2.0)
    queue.claim("w", now=3.0)
    assert queue.depth() == {"queued": 1, "running": 1}
    assert queue.depth(tenant_id="tenant-b") == {"queued": 1}


def test_list_jobs_filters_by_tenant_and_status(tmp_path: Path) -> None:
    queue = _queue(tmp_path, global_max_running=10)
    queue.enqueue("a", {}, tenant_id="tenant-a", now=1.0)
    queue.enqueue("b", {}, tenant_id="tenant-b", now=2.0)
    queue.enqueue("c", {}, tenant_id="tenant-a", now=3.0)
    assert {j.job_id for j in queue.list_jobs(tenant_id="tenant-a")} == {"a", "c"}
    assert {j.job_id for j in queue.list_jobs(status=JobStatus.QUEUED)} == {"a", "b", "c"}
    assert [j.job_id for j in queue.list_jobs(limit=1)] == ["c"]


def test_close_is_idempotent_and_reopens_lazily(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("j1", {}, now=1.0)
    queue.close()
    queue.close()  # must never raise
    assert queue.get("j1").job_id == "j1"  # type: ignore[union-attr]


def test_rows_persist_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "queue.db"
    with JobQueue(db_path) as queue:
        queue.enqueue("j1", {"v": 1}, now=1.0)
    with JobQueue(db_path) as reopened:
        job = reopened.get("j1")
        assert job is not None
        assert job.payload == {"v": 1}
        assert job.status is JobStatus.QUEUED
