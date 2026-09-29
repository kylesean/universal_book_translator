"""Job worker: claim → run → terminal queue state, with cancel and failure paths."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterator
from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.job_queue import JobQueue, JobStatus, QueuedJob
from ubt.core.engine.job_worker import JobWorker
from ubt.core.exceptions import JobInterruptedError


@pytest.fixture
def queue(tmp_path: Path) -> Iterator[JobQueue]:
    q = JobQueue(
        tmp_path / "q.sqlite",
        global_max_running=10,
        default_tenant_max_running=10,
    )
    yield q
    q.close()


def _event(
    job_id: str,
    *,
    kind: EventType = EventType.DRAFT_BATCH_COMPLETED,
    total: int = 10,
    completed: int = 5,
    artifact: str | None = None,
) -> TranslationProgressEvent:
    return TranslationProgressEvent(
        event_type=kind,
        job_id=job_id,
        total_blocks=total,
        completed_blocks=completed,
        artifact_path=artifact,
    )


def _source(
    events: list[TranslationProgressEvent],
    *,
    fail: Exception | None = None,
    cancel_at: int | None = None,
    queue: JobQueue | None = None,
) -> Callable[[QueuedJob, UBTConfig], AsyncGenerator[TranslationProgressEvent, None]]:
    async def _gen(
        job: QueuedJob, config: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        for index, event in enumerate(events):
            if cancel_at is not None and index == cancel_at and queue is not None:
                queue.request_cancel(job.job_id)
            yield event
        if fail is not None:
            raise fail

    return _gen


@pytest.mark.asyncio
async def test_completes_job_and_persists_progress(queue: JobQueue, tmp_path: Path) -> None:
    queue.enqueue("j1", {"input_path": "x.pdf"})
    artifact = str(tmp_path / "out.pdf")
    worker = JobWorker(
        queue,
        UBTConfig(),
        worker_id="w",
        event_source=_source(
            [
                _event("j1", completed=4),
                _event("j1", kind=EventType.EXPORT_COMPLETED, completed=10, artifact=artifact),
            ]
        ),
    )
    assert await worker.run_until_idle() == 1

    job = queue.get("j1")
    assert job is not None
    assert job.status is JobStatus.COMPLETED
    assert job.progress["completed_blocks"] == 10
    assert job.progress["total_blocks"] == 10
    assert job.progress["output_file"] == artifact
    assert job.worker_id is None


@pytest.mark.asyncio
async def test_failed_job_is_terminal_with_error(queue: JobQueue) -> None:
    queue.enqueue("j1", {"input_path": "x.pdf"})
    worker = JobWorker(
        queue,
        UBTConfig(),
        worker_id="w",
        event_source=_source([_event("j1")], fail=RuntimeError("provider exploded")),
    )
    await worker.run_until_idle()

    job = queue.get("j1")
    assert job is not None
    assert job.status is JobStatus.FAILED
    # The raw message must not leak to clients: provider error bodies can carry
    # source book text or host paths. The class name + a server-log pointer is
    # the product guarantee (same as the embedded API manager).
    assert "RuntimeError" in (job.error or "")
    assert "provider exploded" not in (job.error or "")


@pytest.mark.asyncio
async def test_cancel_request_stops_a_running_job(queue: JobQueue) -> None:
    queue.enqueue("j1", {"input_path": "x.pdf"})
    worker = JobWorker(
        queue,
        UBTConfig(),
        worker_id="w",
        event_source=_source(
            [_event("j1"), _event("j1"), _event("j1")],
            cancel_at=0,
            queue=queue,
        ),
    )
    await worker.run_until_idle()

    job = queue.get("j1")
    assert job is not None
    assert job.status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_concurrency_drains_every_queued_job(queue: JobQueue) -> None:
    for index in range(3):
        queue.enqueue(f"j{index}", {"input_path": f"{index}.pdf"})
    worker = JobWorker(
        queue,
        UBTConfig(),
        worker_id="w",
        concurrency=3,
        event_source=_source([_event("any")]),
    )
    assert await worker.run_until_idle() == 3
    for index in range(3):
        job = queue.get(f"j{index}")
        assert job is not None and job.status is JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_process_one_is_false_on_empty_queue(queue: JobQueue) -> None:
    worker = JobWorker(queue, UBTConfig(), worker_id="w")
    assert await worker.process_one() is False


@pytest.mark.asyncio
async def test_lease_loss_stops_the_worker_without_a_terminal_write(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that lost its lease must not finish the job it no longer owns.

    Before, losing the lease only logged: the job kept drafting under a second
    worker that ``reclaim_stale`` had handed it to, so the book was paid for
    twice and both writers checkpointed the same blocks.
    """
    import asyncio

    emitted: list[str] = []
    closed = False
    queue.enqueue("job_lease", {"input_path": "x.pdf"}, tenant_id="default")
    claimed = queue.claim("w1")
    assert claimed is not None
    monkeypatch.setattr(queue, "heartbeat", lambda *a, **k: False)

    async def _slow(
        job: QueuedJob, _config: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        nonlocal closed
        try:
            for index in range(5):
                await asyncio.sleep(0.02)
                emitted.append(f"e{index}")
                yield _event(job.job_id)
        finally:
            # Stands in for the orchestrator's ``finally`` (ledger/router
            # teardown): aclosing must run it the instant we leave the loop, not
            # only when the generator is GC'd, so no further blocks are drafted
            # (and billed) against a job we no longer own.
            closed = True

    worker = JobWorker(queue, UBTConfig(), worker_id="w1", event_source=_slow)
    worker._heartbeat_interval = 0.005
    await worker.execute(claimed)

    assert len(emitted) < 5, "the event loop should have stopped early"
    assert closed, "the event source must be aclose()d on early exit"
    row = queue.get("job_lease")
    assert row is not None and row.status is JobStatus.RUNNING, row
    assert row.error is None


@pytest.mark.asyncio
async def test_writer_lock_conflict_yields_back_to_queued(queue: JobQueue) -> None:
    from ubt.core.exceptions import LedgerWriterLockConflictError

    queue.enqueue("job_conflict", {"input_path": "x.pdf"})
    claimed = queue.claim("w2")
    assert claimed is not None
    assert claimed.attempts == 1

    worker = JobWorker(
        queue,
        UBTConfig(),
        worker_id="w2",
        event_source=_source([], fail=LedgerWriterLockConflictError("locked by another process")),
    )
    await worker.execute(claimed)

    row = queue.get("job_conflict")
    assert row is not None
    assert row.status is JobStatus.QUEUED
    assert row.attempts == 0  # rolled back attempt
    assert row.worker_id is None
    assert row.error is not None and "writer lock held" in row.error
    assert "LedgerWriterLockConflictError" in row.error


@pytest.mark.asyncio
async def test_writer_lock_conflict_backs_off_before_reclaiming(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A yielded job must not be re-claimed in the same breath.

    ``release_claim`` puts the job back to QUEUED and ``_slot`` claims
    immediately, so without a backoff the slot grabbed the same job, hit the
    same still-held lock, and spun with no delay. Attempts are never consumed
    (``decrement_attempt=True``), so ``run_until_idle`` never returned and the
    job ran a second time as soon as the lock freed.
    """
    from ubt.core.exceptions import LedgerWriterLockConflictError

    queue.enqueue("job_backoff", {"input_path": "x.pdf"})
    claimed = queue.claim("w1")
    assert claimed is not None

    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _record_sleep(delay: float) -> None:
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _record_sleep)
    worker = JobWorker(
        queue,
        UBTConfig(),
        worker_id="w1",
        poll_interval=7.5,
        event_source=_source([], fail=LedgerWriterLockConflictError("locked")),
    )
    await worker.execute(claimed)

    assert worker.poll_interval in sleeps, "the slot must back off before re-claiming"


@pytest.mark.asyncio
async def test_cooperative_cancel_token_stops_worker(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue.sqlite")
    queue.enqueue("job_cancel_coop", {"input_path": "x.pdf"})
    claimed = queue.claim("w1")
    assert claimed is not None

    async def _long_stage(
        job: QueuedJob, job_config: UBTConfig, cancel_token: asyncio.Event | None = None
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        yield _event(job.job_id)
        # Simulate a long stage parking on the cancel token.
        if cancel_token is not None:
            await cancel_token.wait()
        if cancel_token is not None and cancel_token.is_set():
            raise JobInterruptedError("cancelled via token")
        yield _event(job.job_id)

    worker = JobWorker(
        queue,
        UBTConfig(),
        worker_id="w1",
        lease_seconds=0.1,  # fast heartbeat
        event_source=_long_stage,
    )

    # Request cancellation
    queue.request_cancel("job_cancel_coop")

    await worker.execute(claimed)

    row = queue.get("job_cancel_coop")
    assert row is not None
    assert row.status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_heartbeat_keeps_renewing_lease_after_cancel_requested(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling must not abandon the lease while the pipeline unwinds.

    The loop used to ``return`` right after setting ``cancel_token``, so no
    further heartbeats were sent: a slow unwind outlived the lease and
    ``reclaim_stale`` handed the still-tearing-down job to a second worker.
    """
    from contextlib import suppress

    queue.enqueue("job_hb_cancel", {"input_path": "x.pdf"})
    claimed = queue.claim("w1")
    assert claimed is not None
    queue.request_cancel("job_hb_cancel")

    heartbeats = 0
    real_heartbeat = queue.heartbeat

    def _counting(job_id: str, worker_id: str, *, lease_seconds: float = 60.0) -> bool:
        nonlocal heartbeats
        heartbeats += 1
        return real_heartbeat(job_id, worker_id, lease_seconds=lease_seconds)

    monkeypatch.setattr(queue, "heartbeat", _counting)

    worker = JobWorker(queue, UBTConfig(), worker_id="w1")
    worker._heartbeat_interval = 0.005
    lost = asyncio.Event()
    cancel_token = asyncio.Event()
    task = asyncio.create_task(worker._heartbeat_loop("job_hb_cancel", "w1", lost, cancel_token))
    await asyncio.sleep(0.05)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert cancel_token.is_set()
    assert not lost.is_set()
    assert heartbeats >= 2, "the lease must keep being renewed after cancel is requested"


@pytest.mark.asyncio
async def test_poison_payload_fails_job_not_worker(queue: JobQueue) -> None:
    """A queue row persists across versions, so a payload no current
    ``overrides_from_request`` accepts (drifted enum) must land the job in
    FAILED through the normal terminal path — not raise out of execute()
    before the try and crash-loop the whole worker process on reclaim."""

    async def _never(
        job: QueuedJob, config: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        return
        yield _event(job.job_id)  # pragma: no cover - keeps this an async generator

    queue.enqueue("j-poison", {"input_path": "x.pdf", "preset": "ultra"})
    worker = JobWorker(queue, UBTConfig(), worker_id="w", event_source=_never)
    await worker.run_until_idle()

    job = queue.get("j-poison")
    assert job is not None
    assert job.status is JobStatus.FAILED
    # The raw validation message (which can quote payload/paths) must not leak;
    # the class name + server-log pointer is the client contract.
    assert "see server logs" in (job.error or "")
    assert "ultra" not in (job.error or "")


@pytest.mark.asyncio
async def test_lease_loss_writes_failed_ledger_with_reason(
    queue: JobQueue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost lease is infrastructure failure, not a user cancel.

    The pipeline can only write a bare ``failed`` when its generator closes
    early; the worker knows the real reason and must persist it instead of
    leaving the misclassification as ``cancelled``.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    job_id = "job_lease_ledger"
    with SQLiteJobLedger(db_dir / f"{job_id}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            job_id, BookManifest(doc_id="d", title="t", source_path="x.pdf")
        )

    config = UBTConfig(db_dir=db_dir)
    queue.enqueue(job_id, {"input_path": "x.pdf"}, tenant_id="default")
    claimed = queue.claim("w1")
    assert claimed is not None
    monkeypatch.setattr(queue, "heartbeat", lambda *a, **k: False)

    async def _slow(
        job: QueuedJob, _config: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        for _ in range(5):
            await asyncio.sleep(0.02)
            yield _event(job.job_id)

    worker = JobWorker(queue, config, worker_id="w1", event_source=_slow)
    worker._heartbeat_interval = 0.005
    await worker.execute(claimed)

    with SQLiteJobLedger(db_dir / f"{job_id}.sqlite") as ledger:
        assert ledger.get_job_status(job_id) == "failed"
        assert ledger.get_job_metadata_value(job_id, "abort_reason") == "lease_lost"


@pytest.mark.asyncio
async def test_lease_loss_does_not_signal_cancel_token(
    queue: JobQueue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost lease must NOT be signalled through ``cancel_token``.

    Setting it made the pipeline raise ``JobInterruptedError``, which marked the
    ledger ``cancelled``; ``finalize_job('completed')`` then refuses to
    overwrite ``cancelled``, so the reclaiming worker's delivered book read
    cancelled. Lease loss travels through the worker's own lease check between
    events, so ``cancel_token`` must stay clear.
    """
    job_id = "job_lease_cancel"
    queue.enqueue(job_id, {"input_path": "x.pdf"}, tenant_id="default")
    claimed = queue.claim("w1")
    assert claimed is not None
    monkeypatch.setattr(queue, "heartbeat", lambda *a, **k: False)

    token_was_set = False

    async def _inspect_token(
        job: QueuedJob, _config: UBTConfig, cancel_token: asyncio.Event | None = None
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        nonlocal token_was_set
        for _ in range(10):
            await asyncio.sleep(0.01)
            if cancel_token is not None and cancel_token.is_set():
                token_was_set = True
                break
        yield _event(job.job_id)

    worker = JobWorker(
        queue, UBTConfig(db_dir=tmp_path), worker_id="w1", event_source=_inspect_token
    )
    worker._heartbeat_interval = 0.005
    await worker.execute(claimed)

    assert token_was_set is False, "cancel_token must stay clear when only the lease is lost"


def test_should_rehearse_uses_the_worker_process_key() -> None:
    """A keyed worker runs an API-auto-set rehearsal for real."""
    from pydantic import SecretStr

    from ubt.core.config import UBTConfig
    from ubt.core.engine.job_worker import _should_rehearse

    keyed = UBTConfig(api_key=SecretStr("sk-real"))
    keyless = UBTConfig(api_key=SecretStr(""))
    auto = {"dry_run": True, "rehearsal_auto": True}
    explicit = {"dry_run": True}

    assert _should_rehearse(auto, keyed) is False  # keyed worker runs for real
    assert _should_rehearse(auto, keyless) is True
    assert _should_rehearse(explicit, keyed) is True  # user asked for rehearsal
    assert _should_rehearse({}, keyed) is False
