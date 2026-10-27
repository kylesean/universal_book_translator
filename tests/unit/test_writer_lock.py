"""The ledger single-writer lock: one writer per job, enforced across processes.

``LedgerWriterLock`` is the only thing between two processes and a corrupted
ledger: both would fetch the same pending blocks, pay the provider twice, and
overwrite each other's checkpoints. The guard is a ``job_locks`` row **inside
the ledger database** (not an advisory file lock), so it works on NFS/K8s PVCs
where ``fcntl``/``msvcrt`` byte-offset locks cannot be taken at all. These tests
pin the contract:

- the lock is keyed on ``(db, job_id)`` and held as an expiring lease;
- a live holder refuses a challenger and names the holder pid;
- ``acquire``/``release`` are idempotent and the context manager always releases;
- a crashed writer's lease expires and can be reclaimed;
- exclusion holds *between processes*, which is the whole point.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import subprocess
import sys
import time
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
lock = LedgerWriterLock(db_path, "child-job", lease_seconds=60.0)
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
# Keying and the row contract.
# --------------------------------------------------------------------------- #


def test_the_lock_is_keyed_on_db_and_job(tmp_path: Path) -> None:
    lock = LedgerWriterLock(tmp_path / "job.db", "job-a")
    assert lock.job_id == "job-a"
    assert lock.db_path == tmp_path / "job.db"


def test_acquire_creates_the_job_locks_row_with_the_holder_pid(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    lock = LedgerWriterLock(db_path, "job-a")
    lock.acquire()
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT owner_pid FROM job_locks WHERE job_id = ?", ("job-a",)
        ).fetchone()
        conn.close()
        assert row is not None
        assert int(row[0]) == __import__("os").getpid()
    finally:
        lock.release()


# --------------------------------------------------------------------------- #
# Mutual exclusion and the refusal message.
# --------------------------------------------------------------------------- #


def test_second_writer_to_the_same_job_is_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    holder = LedgerWriterLock(db_path, "same-job")
    holder.acquire()
    try:
        with pytest.raises(LedgerWriterLockConflictError) as excinfo:
            LedgerWriterLock(db_path, "same-job").acquire()
        error = excinfo.value
        assert isinstance(error, LedgerError)
        assert error.details["job_id"] == "same-job"
        assert error.details["holder_pid"] == __import__("os").getpid()
        assert error.lease_expires_at is not None
        assert error.lease_expires_at > time.time()
        assert "same-job" in str(error)
        assert "(pid" in str(error)
    finally:
        holder.release()


def test_different_jobs_in_one_ledger_are_independent(tmp_path: Path) -> None:
    # Each job owns its own row, so two jobs may proceed concurrently.
    db_path = tmp_path / "job.db"
    first = LedgerWriterLock(db_path, "job-a")
    second = LedgerWriterLock(db_path, "job-b")
    first.acquire()
    second.acquire()
    try:
        assert first.job_id != second.job_id
    finally:
        second.release()
        first.release()


def test_locks_on_different_ledgers_are_independent(tmp_path: Path) -> None:
    first = LedgerWriterLock(tmp_path / "one.db", "job-a")
    second = LedgerWriterLock(tmp_path / "two.db", "job-a")
    first.acquire()
    second.acquire()
    try:
        assert first.db_path != second.db_path
    finally:
        second.release()
        first.release()


# --------------------------------------------------------------------------- #
# Idempotency, release, lease expiry, and the context manager.
# --------------------------------------------------------------------------- #


def _lock_row(db_path: Path, job_id: str = "job-a") -> sqlite3.Row | None:
    """The ``job_locks`` row for ``job_id``, or ``None`` (the holder probe)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT * FROM job_locks WHERE job_id = ?", (job_id,)
        ).fetchone()
        return row
    finally:
        conn.close()


def _held_row(db_path: Path, job_id: str = "job-a") -> sqlite3.Row:
    """The row, asserting a holder exists (narrows the type for the caller)."""
    row = _lock_row(db_path, job_id)
    assert row is not None, "expected a live job_locks row"
    return row


def test_release_before_acquire_and_twice_after_are_no_ops(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    lock = LedgerWriterLock(db_path, "job-a")
    lock.release()  # never acquired: must not create a row
    assert not db_path.exists() or _lock_row(db_path) is None

    lock.acquire()
    assert _held_row(db_path) is not None
    lock.release()
    assert _lock_row(db_path) is None, "release must delete the row"
    lock.release()  # already released: still no row, still no raise
    assert _lock_row(db_path) is None


def test_acquire_is_idempotent_on_one_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    lock = LedgerWriterLock(db_path, "job-a")
    lock.acquire()
    token = _held_row(db_path)["owner_token"]
    lock.acquire()  # must not conflict with itself, and must not change owner
    assert _held_row(db_path)["owner_token"] == token
    lock.release()


def test_release_frees_the_lock_for_a_new_writer(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    holder = LedgerWriterLock(db_path, "job-a")
    holder.acquire()
    holder.release()

    successor = LedgerWriterLock(db_path, "job-a")
    successor.acquire()  # the whole point: a released lock is acquirable
    try:
        assert _held_row(db_path)["owner_token"] == successor._token
    finally:
        successor.release()


def test_a_dead_holders_expired_lease_is_reclaimed(tmp_path: Path) -> None:
    # A writer that crashed without releasing must not wedge the job forever:
    # once its lease expires, a new process reclaims it atomically.
    db_path = tmp_path / "job.db"
    stale = LedgerWriterLock(db_path, "job-a")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE job_locks (job_id TEXT PRIMARY KEY, owner_pid INTEGER NOT NULL, "
        "owner_token TEXT NOT NULL, lease_expires_at REAL NOT NULL, acquired_at REAL NOT NULL);"
    )
    conn.execute(
        "INSERT INTO job_locks VALUES ('job-a', 999999, 'dead-token', ?, ?);",
        (time.time() - 1.0, time.time() - 100.0),  # already expired
    )
    conn.commit()
    conn.close()

    stale.acquire()  # must take over the expired lease
    try:
        row = _lock_row(db_path)
        assert row is not None
        assert row["owner_token"] != "dead-token", "the expired holder still owns the row"
        assert int(row["owner_pid"]) == os.getpid()
        assert float(row["lease_expires_at"]) > time.time()
    finally:
        stale.release()


def test_context_manager_holds_then_releases(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    with LedgerWriterLock(db_path, "job-a"), pytest.raises(LedgerWriterLockConflictError):
        LedgerWriterLock(db_path, "job-a").acquire()
    # The context manager released on exit, so the next writer can take it.
    assert _lock_row(db_path) is None
    successor = LedgerWriterLock(db_path, "job-a")
    successor.acquire()
    try:
        assert _held_row(db_path) is not None
    finally:
        successor.release()


def test_context_manager_releases_on_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "job.db"
    with pytest.raises(ValueError), LedgerWriterLock(db_path, "job-a"):
        raise ValueError("boom")
    successor = LedgerWriterLock(db_path, "job-a")
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
            LedgerWriterLock(db_path, "child-job").acquire()
        assert excinfo.value.details["holder_pid"] == child.pid
    # The child has exited and released; the lock is free again.
    successor = LedgerWriterLock(db_path, "child-job")
    successor.acquire()
    successor.release()


def test_renew_extends_the_lease_so_a_long_run_keeps_exclusivity(tmp_path: Path) -> None:
    """A holder past its original lease window must still exclude a challenger.

    The lock is an expiring lease (so a crashed writer is reclaimable), and a
    real book run outlives one window. Without ``renew`` the lease would lapse
    mid-run and a second process would take the job — both would then bill the
    provider. ``renew`` pushes the expiry out; a challenger must still be refused.
    """
    db_path = tmp_path / "job.db"
    holder = LedgerWriterLock(db_path, "job-a", lease_seconds=0.2)
    holder.acquire()
    try:
        # Wind the recorded expiry into the past to simulate a long run whose
        # original window elapsed, then renew as the pipeline's renewer does.
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE job_locks SET lease_expires_at = ? WHERE job_id = 'job-a'", (0.0,))
        conn.commit()
        conn.close()

        holder.renew()

        with pytest.raises(LedgerWriterLockConflictError):
            LedgerWriterLock(db_path, "job-a", lease_seconds=0.2).acquire()
    finally:
        holder.release()
