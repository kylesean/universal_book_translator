"""The ledger single-writer lock: one writer per ledger, enforced across processes.

``LedgerWriterLock`` is the only thing between two processes and a corrupted
ledger: both would fetch the same pending blocks, pay the provider twice, and
overwrite each other's checkpoints. These tests pin the *contract* of that
guard:

- the lock lives next to the ledger at ``<db>.writer.lock`` and is keyed on the
  ledger path, not the job id;
- acquiring records the holder pid, and a refused challenger is told who holds
  it;
- ``acquire``/``release`` are idempotent and the context manager always releases;
- exclusion holds *between processes*, which is the whole point.

Every check runs against a throwaway ``tmp_path`` ledger.
"""

from __future__ import annotations

import contextlib
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import LedgerError, LedgerWriterLockConflictError

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: A child process that takes the lock, announces readiness, then holds it until
#: the parent creates ``release``. The deadline keeps a broken parent from
#: leaving the child wedged forever.
_CHILD_HOLDER = """
import sys, time
from pathlib import Path
from ubt.core.engine.writer_lock import LedgerWriterLock

db_path = Path(sys.argv[1])
release = Path(sys.argv[2])
lock = LedgerWriterLock(db_path, "child-job")
lock.acquire()
print("READY", flush=True)
deadline = time.time() + 30
while not release.exists() and time.time() < deadline:
    time.sleep(0.02)
lock.release()
"""


@contextlib.contextmanager
def _child_holding_the_lock(db_path: Path, release: Path) -> Iterator[subprocess.Popen[str]]:
    """Run ``_CHILD_HOLDER`` until it owns the lock; release it on exit."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_HOLDER, str(db_path), str(release)],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = proc.stdout.readline() if proc.stdout else ""
        if ready.strip() != "READY":
            proc.kill()
            _, stderr = proc.communicate()
            raise AssertionError(f"child lock holder did not start: {stderr}")
        yield proc
    finally:
        release.write_text("go", encoding="ascii")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# --------------------------------------------------------------------------- #
# Path and file contract.
# --------------------------------------------------------------------------- #


def test_lock_path_is_derived_from_the_ledger_path(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    lock = LedgerWriterLock(db_path, "job-a")
    assert lock.lock_path == Path(str(db_path) + ".writer.lock")
    assert lock.job_id == "job-a"


def test_acquire_creates_parent_dirs_and_a_private_lock_file(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deeper" / "job.db"
    lock = LedgerWriterLock(db_path, "job-a")
    lock.acquire()
    try:
        assert lock.lock_path.exists()
        if sys.platform != "win32":  # Windows does not honor the open() mode bits
            assert stat.S_IMODE(lock.lock_path.stat().st_mode) == 0o600
    finally:
        lock.release()


def test_acquire_records_the_holder_pid(tmp_path: Path) -> None:
    lock = LedgerWriterLock(tmp_path / "job.db", "job-a")
    lock.acquire()
    try:
        assert lock.lock_path.read_text(encoding="ascii").strip() == str(os.getpid())
    finally:
        lock.release()


# --------------------------------------------------------------------------- #
# Mutual exclusion and the refusal message.
# --------------------------------------------------------------------------- #


def test_second_writer_to_the_same_ledger_is_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    holder = LedgerWriterLock(db_path, "holder-job")
    holder.acquire()
    try:
        with pytest.raises(LedgerWriterLockConflictError) as excinfo:
            LedgerWriterLock(db_path, "challenger-job").acquire()
        error = excinfo.value
        assert isinstance(error, LedgerError)
        # ``job_id`` is the refused challenger; ``holder_pid`` names the owner.
        assert error.details["job_id"] == "challenger-job"
        assert error.details["holder_pid"] == os.getpid()
        assert "challenger-job" in str(error)
        assert "(pid" in str(error)
    finally:
        holder.release()


def test_guard_is_keyed_on_the_ledger_not_the_job_id(tmp_path: Path) -> None:
    # Two different job ids still share one ledger file, so the second is refused
    # -- the resource being protected is the ledger, not the job name.
    db_path = tmp_path / "job.db"
    holder = LedgerWriterLock(db_path, "job-a")
    holder.acquire()
    try:
        with pytest.raises(LedgerWriterLockConflictError):
            LedgerWriterLock(db_path, "job-b").acquire()
    finally:
        holder.release()


def test_conflict_without_a_readable_holder_pid_still_refuses(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    holder = LedgerWriterLock(db_path, "holder-job")
    holder.acquire()
    try:
        # A holder that could not record its pid leaves an empty file; the lock
        # must still refuse, just without naming a pid.
        holder.lock_path.write_text("", encoding="ascii")
        with pytest.raises(LedgerWriterLockConflictError) as excinfo:
            LedgerWriterLock(db_path, "challenger-job").acquire()
        assert excinfo.value.details["holder_pid"] is None
        assert "(pid" not in str(excinfo.value)
    finally:
        holder.release()


def test_locks_on_different_ledgers_are_independent(tmp_path: Path) -> None:
    first = LedgerWriterLock(tmp_path / "one.db", "job-a")
    second = LedgerWriterLock(tmp_path / "two.db", "job-b")
    first.acquire()
    second.acquire()
    try:
        assert first.lock_path != second.lock_path
    finally:
        second.release()
        first.release()


# --------------------------------------------------------------------------- #
# Idempotency, release, and the context manager.
# --------------------------------------------------------------------------- #


def test_release_before_acquire_and_twice_after_are_no_ops(tmp_path: Path) -> None:
    lock = LedgerWriterLock(tmp_path / "job.db", "job-a")
    lock.release()  # never acquired
    lock.acquire()
    lock.release()
    lock.release()  # already released


def test_acquire_is_idempotent_on_one_instance(tmp_path: Path) -> None:
    lock = LedgerWriterLock(tmp_path / "job.db", "job-a")
    lock.acquire()
    lock.acquire()  # must not try to take the lock a second time and conflict with itself
    try:
        assert lock.lock_path.read_text(encoding="ascii").strip() == str(os.getpid())
    finally:
        lock.release()


def test_release_frees_the_lock_for_a_new_writer(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    holder = LedgerWriterLock(db_path, "holder-job")
    holder.acquire()
    holder.release()
    successor = LedgerWriterLock(db_path, "successor-job")
    successor.acquire()
    successor.release()


def test_context_manager_holds_then_releases(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    with LedgerWriterLock(db_path, "job-a"), pytest.raises(LedgerWriterLockConflictError):
        LedgerWriterLock(db_path, "job-b").acquire()
    # Released on exit, so a fresh writer can take it.
    successor = LedgerWriterLock(db_path, "job-b")
    successor.acquire()
    successor.release()


def test_context_manager_releases_on_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    with pytest.raises(ValueError), LedgerWriterLock(db_path, "job-a"):
        raise ValueError("boom")
    successor = LedgerWriterLock(db_path, "job-b")
    successor.acquire()
    successor.release()


# --------------------------------------------------------------------------- #
# The real thing: exclusion between processes.
# --------------------------------------------------------------------------- #


def test_exclusion_holds_between_processes(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    release = tmp_path / "release"
    with _child_holding_the_lock(db_path, release) as child:
        with pytest.raises(LedgerWriterLockConflictError) as excinfo:
            LedgerWriterLock(db_path, "parent-job").acquire()
        assert excinfo.value.details["holder_pid"] == child.pid
    # The child has exited and released; the lock is free again.
    successor = LedgerWriterLock(db_path, "parent-job")
    successor.acquire()
    successor.release()
