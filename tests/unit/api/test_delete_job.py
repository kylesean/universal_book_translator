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
