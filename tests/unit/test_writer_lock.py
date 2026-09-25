"""Tests for the job-level single-writer ledger lock (schema-v7 lease successor)."""

from pathlib import Path

import pytest

from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import LedgerError


def test_second_holder_conflicts_until_release(tmp_path: Path) -> None:
    db = tmp_path / "job.sqlite"
    lock = LedgerWriterLock(db, "job_a")
    lock.acquire()

    competing = LedgerWriterLock(db, "job_a")
    with pytest.raises(LedgerError, match="already being written"):
        competing.acquire()

    lock.release()
    competing.acquire()
    competing.release()


def test_release_is_idempotent_and_acquire_twice_is_noop(tmp_path: Path) -> None:
    db = tmp_path / "job.sqlite"
    lock = LedgerWriterLock(db, "job_b")
    lock.acquire()
    lock.acquire()  # re-entrant call on the held lock must not conflict
    lock.release()
    lock.release()
    # The two calls above only prove nothing raised: a `release()` that leaked the
    # lock and returned quietly passed them. What matters is that the lock is
    # really free afterwards, so a fresh holder can take it.
    competing = LedgerWriterLock(db, "job_b")
    competing.acquire()
    competing.release()


def test_holder_pid_is_recorded(tmp_path: Path) -> None:
    import os

    db = tmp_path / "job.sqlite"
    lock = LedgerWriterLock(db, "job_c")
    lock.acquire()
    try:
        assert lock.lock_path.read_text(encoding="ascii").strip() == str(os.getpid())
    finally:
        lock.release()


def test_busy_message_names_the_holder_pid(tmp_path: Path) -> None:
    """A competing writer must be told *which* process to wait for.

    The lock used to sit on byte 0 — the same byte the pid is written to — and the
    CRT enforces `msvcrt.locking` against other handles in the same process. So on
    Windows the read behind this message raised PermissionError, was suppressed,
    and the user got "already being written" with no pid: the one number that says
    what to kill. POSIX never hit it because `fcntl.flock` locks the file, not a
    byte range.
    """
    import os

    db = tmp_path / "job.sqlite"
    holder = LedgerWriterLock(db, "job_d")
    holder.acquire()
    try:
        assert holder._read_holder_pid() == os.getpid()
        competing = LedgerWriterLock(db, "job_d")
        with pytest.raises(LedgerError, match=r"pid \d+"):
            competing.acquire()
    finally:
        holder.release()


def test_distinct_job_ids_do_not_conflict(tmp_path: Path) -> None:
    a = LedgerWriterLock(tmp_path / "job_a.sqlite", "job_a")
    b = LedgerWriterLock(tmp_path / "job_b.sqlite", "job_b")
    a.acquire()
    b.acquire()
    a.release()
    b.release()


@pytest.mark.fast
def test_writer_lock_context_manager_and_del(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    lock = LedgerWriterLock(db_path, "job_1")
    # Context manager support
    with lock as acquired_lock:
        assert acquired_lock is lock
        assert lock._fd is not None
        assert lock.lock_path.exists()
    assert lock._fd is None

    # __del__ cleanup support
    lock2 = LedgerWriterLock(db_path, "job_1")
    lock2.acquire()
    fd = lock2._fd
    assert fd is not None
    # Calling __del__ should release the fd
    lock2.__del__()
    assert lock2._fd is None
