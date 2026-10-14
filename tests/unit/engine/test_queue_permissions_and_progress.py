"""P2-13 / P2-14: queue-file permissions and terminal-progress preservation.

Two defects in the durable queue / worker terminal path:

* **P2-13** — the queue SQLite file was created ``0666 & ~umask`` (0644 under a
  normal umask) while per-job ledgers were tightened to 0600, so every queued
  job's payload (source/target paths, config overrides) was readable by any
  local user. The queue now applies the same ``ensure_private_dir`` +
  ``restrict_sqlite_family`` the ledger uses.
* **P2-14** — the worker wrote its *local* ``progress`` snapshot on every
  terminal transition. A second attempt that failed before emitting its first
  event left that snapshot empty, and ``complete()``'s ``COALESCE(?, ...)``
  overwrote the previous attempt's 90%-complete snapshot with ``{}`` — a failed
  retry reported the job as 0% done. Empty snapshots are now sent as ``None`` so
  the stored snapshot survives.
"""

from __future__ import annotations

import stat
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_queue import JobQueue, JobStatus, QueuedJob
from ubt.core.engine.job_worker import JobWorker

pytestmark = pytest.mark.fast

#: Windows has no POSIX mode bits; the permission assertions are skipped there.
_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")


@_posix_only
def test_the_queue_file_and_its_wal_sidecars_are_owner_only(tmp_path: Path) -> None:
    """The queue holds every job's payload; it must not be world-readable."""
    db = tmp_path / "queue.db"
    queue = JobQueue(db)
    queue.enqueue("j1", {"input_path": "/home/alice/book.epub"}, now=1.0)
    # Touch the WAL so the sidecar exists and can be inspected.
    queue.claim("w1", now=2.0)
    queue.close()

    mode = stat.S_IMODE(db.stat().st_mode)
    assert mode == 0o600, f"queue.db is {oct(mode)}, expected 0o600"
    for sidecar in (db.with_name("queue.db-wal"), db.with_name("queue.db-shm")):
        if sidecar.exists():
            side_mode = stat.S_IMODE(sidecar.stat().st_mode)
            assert side_mode == 0o600, f"{sidecar.name} is {oct(side_mode)}, expected 0o600"


@_posix_only
def test_a_freshly_created_queue_directory_is_private(tmp_path: Path) -> None:
    """A db_dir UBT creates for the queue is tightened; an existing one is not."""
    fresh = tmp_path / "created-by-ubt"
    JobQueue(fresh / "queue.db").close()
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o700


async def _events_without_any(
    job: QueuedJob, cfg: UBTConfig, **kwargs: object
) -> AsyncGenerator[TranslationProgressEvent, None]:
    """An event source that yields nothing (dies before its first event)."""
    if False:  # pragma: no cover - makes this an async generator
        yield TranslationProgressEvent(job_id=job.job_id, stage="never")


@pytest.mark.asyncio
async def test_a_second_attempt_keeps_the_prior_progress_snapshot(tmp_path: Path) -> None:
    """P2-14: an empty local snapshot must not COALESCE the stored one to zero."""
    queue = JobQueue(tmp_path / "queue.db")
    queue.enqueue("j1", {"input_path": "book.epub"}, now=1.0)
    # First attempt reached 90% and was requeued (lease lost).
    queue.claim("w1", now=2.0)
    queue.update_progress("j1", "w1", {"progress_percent": 90.0, "processed_blocks": 90})
    queue.release_claim("j1", "w1", now=3.0)
    # Second attempt claims, then dies before emitting a single event.
    claimed = queue.claim("w2", now=4.0)
    assert claimed is not None

    worker = JobWorker(
        queue=queue,
        config=UBTConfig(provider="openai"),
        worker_id="w2",
        event_source=_events_without_any,
    )
    assert await worker.execute(claimed) is True

    final = queue.get("j1")
    assert final is not None
    assert final.status is JobStatus.COMPLETED
    # The 90% snapshot from the first attempt survives rather than resetting.
    assert final.progress.get("progress_percent") == 90.0


@pytest.mark.asyncio
async def test_a_failed_first_attempt_persists_an_empty_snapshot_safely(tmp_path: Path) -> None:
    """The first attempt with no events writes no progress, not a corrupt one."""
    queue = JobQueue(tmp_path / "queue.db")
    queue.enqueue("j1", {"input_path": "book.epub"}, now=1.0)
    claimed = queue.claim("w1", now=2.0)
    assert claimed is not None

    async def failing(
        job: QueuedJob, cfg: UBTConfig, **kwargs: object
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        raise RuntimeError("died before the first event")
        if False:  # pragma: no cover
            yield TranslationProgressEvent(job_id=job.job_id, stage="never")

    worker = JobWorker(
        queue=queue, config=UBTConfig(provider="openai"), worker_id="w1", event_source=failing
    )
    assert await worker.execute(claimed) is True

    final = queue.get("j1")
    assert final is not None
    assert final.status is JobStatus.FAILED
    assert final.progress == {}
