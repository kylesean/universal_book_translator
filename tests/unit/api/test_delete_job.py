"""Contract for DELETE /jobs/{job_id} — console history management.

The queue list is a durable scan of ``db_dir`` ledgers, so a finished job
stays visible until its ledger is removed. These tests pin that deletion
removes exactly the ledger family (``.sqlite`` + ``-shm``/``-wal``/
``.sqlite-wal``) and the per-job deliverable directory, leaves sibling jobs
and the translation memory untouched, refuses a job that has not reached a
terminal status, and answers 404 for an unknown id.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.core.config import UBTConfig
from ubt.core.engine.job_queue import JobStatus
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast

_API_KEY = "delete-test-key"
_AUTH = {"X-API-Key": _API_KEY}
_JOB = "deletejob00001"
_SIBLING = "deletesib00002"


def _seed(tmp_path: Path, *, status: JobStatus = JobStatus.COMPLETED) -> UBTConfig:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    for job_id in (_JOB, _SIBLING):
        with SQLiteJobLedger(config.db_dir / f"{job_id}.sqlite") as ledger:
            ledger.init_job_from_manifest(
                job_id,
                BookManifest(doc_id=job_id, title="t", source_path=str(tmp_path / "in.md")),
            )
            ledger.finalize_job(job_id, status=status)
    # A deliverable directory as the default-output layout would leave it.
    outputs = config.db_dir / "outputs" / _JOB
    outputs.mkdir(parents=True)
    (outputs / "book_bilingual.pdf").write_bytes(b"%PDF-1.4 fake")
    return config


def test_delete_removes_ledger_family_and_outputs(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    client = TestClient(create_app(config))
    # A leftover WAL sidecar (the lock now lives inside the ledger DB, so this
    # is the realistic sibling a crash leaves behind).
    (config.db_dir / f"{_JOB}.sqlite-wal").write_bytes(b"")

    res = client.delete(f"/jobs/{_JOB}", headers=_AUTH)
    assert res.status_code == 200
    assert res.json() == {
        "job_id": _JOB,
        "removed_ledger": True,
        "removed_outputs": True,
    }

    assert not (config.db_dir / f"{_JOB}.sqlite").exists()
    assert not (config.db_dir / f"{_JOB}.sqlite-wal").exists()
    assert not (config.db_dir / "outputs" / _JOB).exists()
    # Siblings and the outputs root itself survive.
    assert (config.db_dir / f"{_SIBLING}.sqlite").exists()
    assert (config.db_dir / "outputs").is_dir()


def test_delete_unknown_job_is_404(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    res = client.delete("/jobs/job_doesnotexist", headers=_AUTH)
    assert res.status_code == 404


def test_delete_is_404_once_files_are_gone(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    client = TestClient(create_app(config))
    assert client.delete(f"/jobs/{_JOB}", headers=_AUTH).status_code == 200
    res = client.delete(f"/jobs/{_JOB}", headers=_AUTH)
    assert res.status_code == 404


def test_delete_rejects_malformed_job_id(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    # A dot is outside [A-Za-z0-9_-]+ but keeps the URL one path segment, so
    # the request reaches the route and is refused by validate_job_id (400).
    res = client.delete("/jobs/bad.id", headers=_AUTH)
    assert res.status_code == 400


def test_delete_does_not_touch_a_shared_service_database(tmp_path: Path) -> None:
    """A job id that names a service file must not delete that file.

    ``validate_job_id`` constrains the charset, not the meaning: ``tm`` and
    ``job_queue`` are legal ids whose ledgers would collide with the shared
    translation memory (``db_dir/tm.sqlite``) and the live queue database
    (``db_dir/job_queue.sqlite``). The delete route used to glob-unlink
    ``db_dir/{id}.sqlite*`` unconditionally, so ``DELETE /jobs/tm`` destroyed
    every human_pe pair and ``DELETE /jobs/job_queue`` destroyed the queue the
    API held open. Only a file that is verifiably a job ledger (a ``job_meta``
    row for this id) may be removed.
    """
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    # A real deployment's shared files, with NO job of that id ever created.
    tm = config.db_dir / "tm.sqlite"
    tm.write_bytes(b"real translation memory")
    queue = config.db_dir / "job_queue.sqlite"
    queue.write_bytes(b"real job queue")
    backup = config.db_dir / "backup_2026.sqlite"
    backup.write_bytes(b"operator backup")

    client = TestClient(create_app(config))
    for service_id in ("tm", "job_queue", "backup_2026"):
        res = client.delete(f"/jobs/{service_id}", headers=_AUTH)
        assert res.status_code == 404, (service_id, res.status_code)

    assert tm.exists() and tm.read_bytes() == b"real translation memory"
    assert queue.exists() and queue.read_bytes() == b"real job queue"
    assert backup.exists()


def test_delete_still_removes_a_verified_job_ledger(tmp_path: Path) -> None:
    """The guard must not stop a real terminal job's ledger from being deleted."""
    config = _seed(tmp_path)
    client = TestClient(create_app(config))
    res = client.delete(f"/jobs/{_JOB}", headers=_AUTH)
    assert res.status_code == 200
    assert not (config.db_dir / f"{_JOB}.sqlite").exists()


# --------------------------------------------------------------------------- #
# The queue row is part of a job's history.
# --------------------------------------------------------------------------- #


def test_delete_also_drops_the_queue_row(tmp_path: Path) -> None:
    """A surviving queue row made a deleted job reappear as COMPLETED.

    ``DELETE`` removed the ledger, sidecars and output directory but never the
    ``job_queue`` row, so ``/jobs`` kept listing the job with a download button
    that resolved to a 409. Re-submitting the same id then hit ``enqueue``'s
    "already completed" branch and returned the dead row — the job silently
    never ran again.
    """
    from ubt.core.engine.job_queue import JobQueue

    config = _seed(tmp_path)
    # The app only builds a queue in queue mode; pass the same instance the app
    # will use so the test observes the rows the running server does.
    queue = JobQueue(config.db_dir / "job_queue.sqlite")
    config = config.model_copy(update={"job_mode": "queue"})
    try:
        queue.enqueue(
            _JOB,
            {"input_path": str(tmp_path / "in.md"), "target_lang": "zh"},
        )
        # Drive it to a terminal state the way a worker does: claim, then complete.
        claimed = queue.claim("worker-1")
        assert claimed is not None and claimed.job_id == _JOB
        assert queue.complete(_JOB, "worker-1", status=JobStatus.COMPLETED)
        assert queue.get(_JOB) is not None

        client = TestClient(create_app(config, queue=queue))
        res = client.delete(f"/jobs/{_JOB}", headers=_AUTH)
        assert res.status_code == 200, res.json()
        assert queue.get(_JOB) is None, "the queue row must not outlive the job"
    finally:
        queue.close()


def test_queue_rows_projection_lists_a_job_before_its_ledger_exists() -> None:
    """``/jobs`` must show a queued job, which has no ledger until a worker claims it.

    The list was a glob over ``db_dir/*.sqlite`` only, so submitting ten jobs left
    the console empty while ``/status`` for the same ids reported a queue
    position. Operators re-submitted, producing duplicates and duplicate spend.
    """
    from ubt.api.job_catalog import list_job_summaries, summarize_queued_job

    class _Row:
        job_id = "queued000001"
        status = JobStatus.QUEUED
        enqueued_at = 1_700_000_000.0
        payload = {
            "input_path": "/books/chapter-3.pdf",
            "target_lang": "zh",
            "pages": [1, 2, 3, 5],
        }
        progress: dict[str, object] = {}

    row = summarize_queued_job(_Row())
    assert row["job_id"] == "queued000001"
    assert row["file_name"] == "chapter-3.pdf"
    assert row["status"] == JobStatus.QUEUED.value
    assert row["page_scope"] == "1-3,5"
    assert row["has_output"] is False
    # The absolute path must not leak into the summary (same rule as summaries
    # read from a ledger).
    assert "/books/" not in str(row)

    # A ledger row for the same id wins, and the queued row fills the gap only.
    empty_dir = __import__("tempfile").mkdtemp()
    listed = list_job_summaries(__import__("pathlib").Path(empty_dir), queued=[_Row()])
    assert [item["job_id"] for item in listed] == ["queued000001"]


# --------------------------------------------------------------------------- #
# /status must not invent a failure for a job someone is actively writing.
# --------------------------------------------------------------------------- #


def test_status_reports_running_while_a_live_writer_holds_the_lease(tmp_path: Path) -> None:
    """A live lease means the job is running, not that "the server restarted".

    Embedded mode has no startup reconcile, so any non-terminal ledger row used
    to be rewritten to FAILED. The CLI and the API share ``db_dir``, so a
    ``ubt translate`` running in another terminal was reported as failed — and
    the suggested "re-run it" collides with the writer lock or pays twice.
    """
    from ubt.core.engine.writer_lock import LedgerWriterLock

    config = _seed(tmp_path, status=JobStatus.RUNNING)
    db_path = config.db_dir / f"{_JOB}.sqlite"

    # A second process (the CLI) holds the writer lease for this ledger.
    lock = LedgerWriterLock(db_path, _JOB)
    lock.acquire()
    try:
        client = TestClient(create_app(config))
        live = client.get(f"/jobs/{_JOB}/status", headers=_AUTH)
        assert live.status_code == 200
        assert live.json()["status"] == JobStatus.RUNNING.value, live.json()
        assert live.json()["error"] is None
    finally:
        lock.release()

    # Nobody holds the lease now: the row really is a crash leftover.
    client = TestClient(create_app(config))
    dead = client.get(f"/jobs/{_JOB}/status", headers=_AUTH)
    assert dead.status_code == 200
    assert dead.json()["status"] == JobStatus.FAILED.value, dead.json()
    assert "server restarted" in (dead.json()["error"] or "")


def test_status_prefers_the_persisted_terminal_error(tmp_path: Path) -> None:
    config = _seed(tmp_path, status=JobStatus.FAILED)
    db_path = config.db_dir / f"{_JOB}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.set_job_metadata_value(
            _JOB, "last_error", "RuntimeError (see server logs; job_id=deletejob00001)"
        )

    client = TestClient(create_app(config))
    response = client.get(f"/jobs/{_JOB}/status", headers=_AUTH)

    assert response.status_code == 200
    assert response.json()["status"] == JobStatus.FAILED.value
    assert response.json()["error"] == ("RuntimeError (see server logs; job_id=deletejob00001)")
