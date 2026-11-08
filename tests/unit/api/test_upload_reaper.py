from __future__ import annotations

import asyncio
import logging
import os
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api import app as app_module
from ubt.api import uploads as uploads_module
from ubt.api.app import create_app
from ubt.api.uploads import (
    prune_stale_uploads,
    reap_stale_uploads,
    resumable_uploads,
    staged_uploads_in_use,
)
from ubt.core.config import UBTConfig
from ubt.core.engine.job_queue import JobQueue, JobStatus
from ubt.core.engine.ledger import SQLiteJobLedger

pytestmark = pytest.mark.fast

_DAY = 86400.0
_API_KEY = "unit-test-key"
_AUTH = {"X-API-Key": _API_KEY}
_WINDOW_DAYS = 7.0


def _config(tmp_path: Path, **overrides: object) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
        **overrides,  # type: ignore[arg-type]
    )


def _stage(directory: Path, name: str, *, age_days: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"%PDF-1.4 staged upload")
    stamp = time.time() - age_days * _DAY
    os.utime(path, (stamp, stamp))
    return path


def test_an_upload_past_the_window_is_removed(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    stale = _stage(uploads, "old-book.pdf", age_days=30)
    fresh = _stage(uploads, "yesterdays-book.pdf", age_days=1)

    assert prune_stale_uploads(uploads, max_age_days=_WINDOW_DAYS) == 1

    assert not stale.exists()
    assert fresh.exists()


def test_the_window_is_measured_from_now_not_from_the_newest_file(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    old = _stage(uploads, "old.pdf", age_days=8)
    also_old = _stage(uploads, "also-old.pdf", age_days=30)

    assert prune_stale_uploads(uploads, max_age_days=_WINDOW_DAYS) == 2

    assert not old.exists()
    assert not also_old.exists()


def test_a_window_of_zero_disables_the_sweep(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    kept = _stage(uploads, "ancient.pdf", age_days=3650)

    assert prune_stale_uploads(uploads, max_age_days=0) == 0

    assert kept.exists()


def test_a_missing_staging_directory_is_not_created(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"

    assert prune_stale_uploads(uploads, max_age_days=_WINDOW_DAYS) == 0

    assert not uploads.exists()


def test_a_directory_in_the_staging_area_is_left_alone(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    subdir = uploads / "someone-elses-tree"
    subdir.mkdir(parents=True)
    old = time.time() - 30 * _DAY
    os.utime(subdir, (old, old))

    assert prune_stale_uploads(uploads, max_age_days=_WINDOW_DAYS) == 0

    assert subdir.is_dir()


def test_a_symlink_is_left_alone(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True)
    target = _stage(tmp_path, "outside.pdf", age_days=30)
    link = uploads / "link.pdf"
    link.symlink_to(target)

    assert prune_stale_uploads(uploads, max_age_days=_WINDOW_DAYS) == 0

    assert link.is_symlink()
    assert target.exists()


def test_an_unreadable_entry_does_not_stop_the_sweep(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    _stage(uploads, "first.pdf", age_days=30)
    _stage(uploads, "second.pdf", age_days=30)

    assert prune_stale_uploads(uploads, max_age_days=_WINDOW_DAYS) == 2

    assert list(uploads.iterdir()) == []


def test_a_queued_job_spares_the_upload_it_will_read(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    claimed = _stage(uploads, "queued.pdf", age_days=30)
    orphan = _stage(uploads, "abandoned.pdf", age_days=30)

    with JobQueue(tmp_path / "db" / "job_queue.sqlite") as queue:
        queue.enqueue("job-queued", {"input_path": str(claimed)})
        assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, job_queue=queue) == 1

    assert claimed.exists()
    assert not orphan.exists()


def test_a_finished_jobs_upload_is_reaped(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    done = _stage(uploads, "done.pdf", age_days=30)

    with JobQueue(tmp_path / "db" / "job_queue.sqlite") as queue:
        queue.enqueue("job-done", {"input_path": str(done)})
        assert queue.claim("worker-1") is not None
        assert queue.complete("job-done", "worker-1", status=JobStatus.COMPLETED)
        assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, job_queue=queue) == 1

    assert not done.exists()


def test_a_queue_job_without_an_input_path_protects_nothing(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    old = _stage(uploads, "inline.pdf", age_days=30)

    with JobQueue(tmp_path / "db" / "job_queue.sqlite") as queue:
        queue.enqueue("job-inline", {"glossary_path": "/tmp/terms.json"})
        assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, job_queue=queue) == 1

    assert not old.exists()


def test_embedded_mode_protects_nothing(tmp_path: Path) -> None:
    assert staged_uploads_in_use(None) == set()


def test_a_failed_embedded_job_protects_its_source_for_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    uploads = config.db_dir / "uploads"
    resumable = _stage(uploads, "resumable.pdf", age_days=30)
    orphan = _stage(uploads, "orphan.pdf", age_days=30)
    job_id = "embedded-resume"
    with SQLiteJobLedger(config.db_dir / f"{job_id}.sqlite") as ledger:
        ledger.init_job_placeholder(job_id, str(resumable), "zh")
        ledger.finalize_job(job_id, status=JobStatus.FAILED)

    assert resumable_uploads(config.db_dir) == {str(resumable.resolve())}
    assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, db_dir=config.db_dir) == 1
    assert resumable.exists()
    assert not orphan.exists()


def test_a_completed_embedded_job_does_not_pin_its_source(tmp_path: Path) -> None:
    config = _config(tmp_path)
    uploads = config.db_dir / "uploads"
    done = _stage(uploads, "done.pdf", age_days=30)
    job_id = "embedded-done"
    with SQLiteJobLedger(config.db_dir / f"{job_id}.sqlite") as ledger:
        ledger.init_job_placeholder(job_id, str(done), "zh")
        ledger.finalize_job(job_id, status=JobStatus.COMPLETED)

    assert resumable_uploads(config.db_dir) == set()
    assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, db_dir=config.db_dir) == 1
    assert not done.exists()


def test_a_service_database_is_not_mistaken_for_a_job_ledger(tmp_path: Path) -> None:
    import sqlite3

    config = _config(tmp_path)
    config.db_dir.mkdir(parents=True)
    service_db = config.db_dir / "tm.sqlite"
    with sqlite3.connect(service_db) as conn:
        conn.execute("CREATE TABLE translation_memory (id INTEGER PRIMARY KEY)")

    assert resumable_uploads(config.db_dir) == set()
    with sqlite3.connect(service_db) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"translation_memory"}


def test_in_use_paths_are_resolved_not_spelled(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    staged = _stage(uploads, "relative.pdf", age_days=30)
    spelled = f"{tmp_path}/./uploads/../uploads/relative.pdf"

    with JobQueue(tmp_path / "db" / "job_queue.sqlite") as queue:
        queue.enqueue("job-rel", {"input_path": spelled})
        assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, job_queue=queue) == 0

    assert staged.exists()


def test_the_api_sweeps_the_staging_area_at_startup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    uploads = config.db_dir / "uploads"
    stale = _stage(uploads, "old.pdf", age_days=30)
    fresh = _stage(uploads, "new.pdf", age_days=1)

    with TestClient(create_app(config=config)):
        assert not stale.exists(), "startup must reap the upload that aged out"
        assert fresh.exists(), "startup must not touch an upload inside the window"


def test_startup_does_not_sweep_when_the_retention_window_is_off(tmp_path: Path) -> None:
    config = _config(tmp_path, upload_retention_days=0.0)
    uploads = config.db_dir / "uploads"
    stale = _stage(uploads, "old.pdf", age_days=30)

    with TestClient(create_app(config=config)):
        assert stale.exists()


def test_the_retention_window_is_configurable(tmp_path: Path) -> None:
    config = _config(tmp_path, upload_retention_days=30.0)
    uploads = config.db_dir / "uploads"
    ten_days = _stage(uploads, "ten-days.pdf", age_days=10)

    with TestClient(create_app(config=config)):
        assert ten_days.exists(), "a 30-day window keeps a 10-day-old upload"

    assert config.upload_retention_days == 30.0


#: Free-standing clock for the periodic-sweep tests. The staging files are
#: stamped relative to it and ``ubt.api.uploads.time`` is replaced with it, so
#: the age decision is deterministic instead of racing wall-clock time.
_FIXED_NOW = 1_800_000_000.0


def _stage_at(directory: Path, name: str, *, age_days: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"%PDF-1.4 staged upload")
    stamp = _FIXED_NOW - age_days * _DAY
    os.utime(path, (stamp, stamp))
    return path


async def _run_one_periodic_sweep(monkeypatch: pytest.MonkeyPatch, uploads: Path) -> list[float]:
    """Let the timer fire once, then stop it at the next tick.

    ``asyncio.sleep`` is the loop's only clock: the fake records the interval
    and raises ``CancelledError`` on the second call, which is exactly the
    shutdown path the lifespan takes. No test waits an hour.
    """
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await app_module._reap_uploads_periodically(
            uploads, max_age_days=_WINDOW_DAYS, job_queue=None, db_dir=None
        )
    return delays


async def test_the_periodic_sweep_reaps_expired_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads = tmp_path / "uploads"
    stale = _stage_at(uploads, "old.pdf", age_days=30)
    fresh = _stage_at(uploads, "new.pdf", age_days=1)
    monkeypatch.setattr(uploads_module, "time", types.SimpleNamespace(time=lambda: _FIXED_NOW))

    delays = await _run_one_periodic_sweep(monkeypatch, uploads)

    assert len(delays) == 2, "the timer must ask to sleep again after one sweep"
    assert delays[0] == app_module._UPLOAD_SWEEP_INTERVAL_S
    assert not stale.exists(), "the sweep must remove an upload past the window"
    assert fresh.exists(), "the sweep must keep an upload inside the window"


async def test_a_failed_periodic_sweep_is_logged_and_the_timer_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(*args: object, **kwargs: object) -> int:
        raise OSError("staging area is read-only")

    monkeypatch.setattr(app_module, "reap_stale_uploads", explode)

    with caplog.at_level(logging.WARNING, logger="ubt.api.app"):
        delays = await _run_one_periodic_sweep(monkeypatch, tmp_path / "uploads")

    assert len(delays) == 2, "a failed sweep must not kill the timer"
    assert "periodic upload sweep failed" in caplog.text
    assert "read-only" in caplog.text


def test_a_live_embedded_run_spares_its_staged_source(tmp_path: Path) -> None:
    """Queue rows cannot see an embedded run; the manager's records must.

    A retention window shorter than a long book's runtime deleted the staged
    source mid-run; the failure surfaced as an unrelated read error blocks
    later, with nothing pointing at the sweep.
    """
    staged = _stage(tmp_path / "uploads", "long_book.pdf", age_days=_WINDOW_DAYS + 1)
    records = types.SimpleNamespace(
        jobs={
            "job-live": types.SimpleNamespace(
                status=JobStatus.RUNNING,
                request=types.SimpleNamespace(input_path=str(staged)),
            ),
            "job-done": types.SimpleNamespace(
                status=JobStatus.COMPLETED,
                request=types.SimpleNamespace(input_path=str(staged)),
            ),
        }
    )
    removed = reap_stale_uploads(
        tmp_path / "uploads",
        max_age_days=_WINDOW_DAYS,
        manager=records,
        now=time.time(),
    )
    assert removed == 0
    assert staged.exists()

    # Without a manager the sweep behaves exactly as before (queue rows plus
    # the failed/cancelled ledger set).
    removed = reap_stale_uploads(
        tmp_path / "uploads",
        max_age_days=_WINDOW_DAYS,
        now=time.time(),
    )
    assert removed == 1


def test_live_embedded_inputs_resolves_paths_and_skips_terminal(tmp_path: Path) -> None:
    staged = tmp_path / "uploads" / "book.pdf"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"%PDF-1.4")
    records = types.SimpleNamespace(
        jobs={
            "live": types.SimpleNamespace(
                status=JobStatus.RUNNING,
                request=types.SimpleNamespace(input_path=str(staged)),
            ),
            "settling": types.SimpleNamespace(
                status=JobStatus.CANCELLED,
                request=types.SimpleNamespace(input_path=str(staged)),
            ),
        }
    )
    from ubt.api.uploads import live_embedded_inputs

    assert live_embedded_inputs(records) == {str(staged.resolve())}
