"""The upload staging area must not grow forever.

``/jobs/upload`` stages a copy of every document a console submits under
``<db_dir>/uploads`` and hands back its path; that copy is the job's input.
No other code path ever removed one, so an instance used for a while keeps
every document ever uploaded. The API sweeps the directory at startup.

The sweep is deliberately conservative: a file is only deleted when it is a
regular file, older than the window, and not named as the ``input_path`` of a
queue job that has not finished.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.uploads import prune_stale_uploads, reap_stale_uploads, staged_uploads_in_use
from ubt.core.config import UBTConfig
from ubt.core.engine.job_queue import JobQueue, JobStatus

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
    """Write a staged upload whose mtime says how long ago it arrived."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"%PDF-1.4 staged upload")
    stamp = time.time() - age_days * _DAY
    os.utime(path, (stamp, stamp))
    return path


# --------------------------------------------------------------------------- #
# The sweep itself
# --------------------------------------------------------------------------- #


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
    # The knob an operator turns to keep every staged document (a submission
    # archive) rather than have the API decide when it is done with it.
    uploads = tmp_path / "uploads"
    kept = _stage(uploads, "ancient.pdf", age_days=3650)

    assert prune_stale_uploads(uploads, max_age_days=0) == 0

    assert kept.exists()


def test_a_missing_staging_directory_is_not_created(tmp_path: Path) -> None:
    # The sweep is housekeeping, not setup: an operator who never uploaded
    # anything must not find an empty uploads/ tree planted in db_dir.
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
    # A link's mtime is the link's, not its target's, and following it would
    # turn an old-looking link into a delete of whatever it points at.
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


# --------------------------------------------------------------------------- #
# What a live job protects
# --------------------------------------------------------------------------- #


def test_a_queued_job_spares_the_upload_it_will_read(tmp_path: Path) -> None:
    # Age says nothing about whether the input is still wanted: a job can sit
    # QUEUED across a restart, and its staged copy is still what it will read.
    uploads = tmp_path / "uploads"
    claimed = _stage(uploads, "queued.pdf", age_days=30)
    orphan = _stage(uploads, "abandoned.pdf", age_days=30)

    with JobQueue(tmp_path / "db" / "job_queue.sqlite") as queue:
        queue.enqueue("job-queued", {"input_path": str(claimed)})
        assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, job_queue=queue) == 1

    assert claimed.exists()
    assert not orphan.exists()


def test_a_finished_jobs_upload_is_reaped(tmp_path: Path) -> None:
    # Only a job that still has work to do protects its input. Once the job is
    # COMPLETED the staged copy is by definition no longer needed.
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
    # Without a queue there is no durable record of a pending run to consult,
    # and the sweep must not invent one.
    assert staged_uploads_in_use(None) == set()


def test_in_use_paths_are_resolved_not_spelled(tmp_path: Path) -> None:
    # A relative or dot-filled spelling in a payload must still match the
    # resolved path the directory walk produces, or the file gets reaped.
    uploads = tmp_path / "uploads"
    staged = _stage(uploads, "relative.pdf", age_days=30)
    spelled = f"{tmp_path}/./uploads/../uploads/relative.pdf"

    with JobQueue(tmp_path / "db" / "job_queue.sqlite") as queue:
        queue.enqueue("job-rel", {"input_path": spelled})
        assert reap_stale_uploads(uploads, max_age_days=_WINDOW_DAYS, job_queue=queue) == 0

    assert staged.exists()


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #


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
