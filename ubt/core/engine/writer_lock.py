"""Job-level single-writer lock, backed by the ledger's own SQLite database.

Two processes running the same ``job_id`` both fetch the same pending blocks,
both pay the provider, and their checkpoints overwrite one another — so a
job-level mutual-exclusion guard is required. This guard lives in a
``job_locks`` table **inside the ledger database**, not in a separate lock file
with ``msvcrt``/``fcntl`` byte offsets:

* an advisory file lock cannot be taken at all on NFS/overlayfs/K8s PVCs
  (``EOPNOTSUPP``) and behaves differently under Docker PID 1, so the old
  file-offset lock either wedged or misreported contention there;
* SQLite already serializes writers through its own WAL + transaction locks, so
  a table row is the natural place to hold "who owns this job right now".

The lock is an **expiring lease**: a row records the owner pid and an expiry, so
a crashed process that never released its lock does not wedge the job forever —
a later run reclaims an expired lease by an atomic ``UPDATE``. Acquisition and
renewal are single statements, so they are safe under concurrent processes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ubt.core.exceptions import LedgerWriterLeaseLostError, LedgerWriterLockConflictError
from ubt.core.fs_perms import ensure_private_dir

logger = logging.getLogger(__name__)

#: How long a lease is valid before another process may reclaim it. Long enough
#: to cover a normal run's lifetime between renewals, short enough that a crashed
#: writer does not wedge the job for long.
DEFAULT_LEASE_SECONDS = 300.0


class LedgerWriterLock:
    """Exclusive single-writer guard for one job, stored in ``job_locks``.

    ``acquire()`` raises :class:`LedgerWriterLockConflictError` when another
    live process already holds the job; ``release()`` is idempotent. The
    interface is unchanged from the file-lock implementation so every call site
    (pipeline, worker, API, MCP, CLI, re-export) keeps working.
    """

    def __init__(
        self,
        db_path: Path,
        job_id: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.job_id = job_id
        self.lease_seconds = lease_seconds
        self._held = False
        self._pid = os.getpid()
        #: Per-instance owner token. Distinguishes two ``LedgerWriterLock``
        #: instances in the *same* process (which share a pid): the same
        #: instance re-acquires idempotently, a second instance conflicts.
        self._token = f"{self._pid}:{id(self)}:{time.time_ns()}"

    def _connect(self) -> sqlite3.Connection:
        # The lock is taken before the ledger creates its file on a fresh run,
        # so the directory may not exist yet. Create it owner-only, matching the
        # ledger's own policy (the lock row lives inside a file that also carries
        # the manuscript).
        ensure_private_dir(self.db_path.parent)
        # A short busy timeout: the row write is a single statement, and under
        # WAL a concurrent reader never blocks it. The timeout only covers
        # another writer's brief transaction.
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000;")
        return conn

    def _table_exists(self, conn: sqlite3.Connection) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='job_locks';"
            ).fetchone()
            is not None
        )

    def acquire(self) -> None:
        if self._held:
            return
        now = time.time()
        expires = now + self.lease_seconds
        conn = self._connect()
        try:
            if not self._table_exists(conn):
                # The ledger has not created its schema yet (a lock is taken
                # before the first run on a fresh db in some call paths). Create
                # just this table; the ledger's own init is idempotent and adds
                # nothing conflicting.
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS job_locks ("
                    "job_id TEXT PRIMARY KEY, owner_pid INTEGER NOT NULL, "
                    "owner_token TEXT NOT NULL, lease_expires_at REAL NOT NULL, "
                    "acquired_at REAL NOT NULL);"
                )
            # Atomic claim: insert a fresh row, or take over a row whose lease
            # has expired, or renew our own instance. Any of these is one
            # statement, so two processes (or two instances) racing cannot both
            # win. Ownership is the per-instance token, not the pid: two
            # instances in one process share a pid and must still exclude.
            cur = conn.execute(
                """
                INSERT INTO job_locks
                    (job_id, owner_pid, owner_token, lease_expires_at, acquired_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    owner_pid = excluded.owner_pid,
                    owner_token = excluded.owner_token,
                    lease_expires_at = excluded.lease_expires_at,
                    acquired_at = excluded.acquired_at
                WHERE job_locks.lease_expires_at < ? OR job_locks.owner_token = ?
                """,
                (self.job_id, self._pid, self._token, expires, now, now, self._token),
            )
            if cur.rowcount == 0:
                holder = conn.execute(
                    "SELECT owner_pid FROM job_locks WHERE job_id = ?", (self.job_id,)
                ).fetchone()
                holder_pid = int(holder["owner_pid"]) if holder else None
                raise LedgerWriterLockConflictError(
                    f"Job {self.job_id} is already being written by another process"
                    + (f" (pid {holder_pid})" if holder_pid else "")
                    + ": refusing to run it twice — a second writer would pay the "
                    "provider again for blocks the first is drafting and overwrite "
                    "its checkpoints. Wait for the running job, or pick a distinct "
                    "--job-id.",
                    details={"job_id": self.job_id, "holder_pid": holder_pid},
                )
            self._held = True
        except sqlite3.OperationalError as exc:
            # A holder whose transaction outlives busy_timeout surfaces as
            # ``database is locked`` rather than as rowcount 0. Without this
            # mapping the losing claimant reports a generic failure, the worker
            # burns one of its attempts on it, and the typed conflict path
            # (free retry — the other writer owns the job) is skipped.
            if "locked" in str(exc).lower():
                raise LedgerWriterLockConflictError(
                    f"Job {self.job_id}: the ledger stayed locked by another "
                    f"writer past the busy timeout — refusing to contend: {exc}",
                    details={"job_id": self.job_id},
                ) from exc
            raise
        finally:
            conn.close()

    def renew(self) -> None:
        """Extend this process's lease.

        Raises :class:`LedgerWriterLeaseLostError` when the lease no longer
        belongs to this instance. The ``UPDATE`` is conditional on our token, so
        a reclaimed lease updates zero rows — and returning quietly there was
        the bug that let a stalled process keep writing a ledger another process
        had already taken over (both paying the provider, both overwriting the
        other's checkpoints). A reclaim is a stop, not a warning: the work this
        process already paid for is durable in the ledger, and the new owner
        continues from it.
        """
        if not self._held:
            return
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE job_locks SET lease_expires_at = ? WHERE job_id = ? AND owner_token = ?",
                (time.time() + self.lease_seconds, self.job_id, self._token),
            )
            if cur.rowcount == 0:
                self._held = False
                raise LedgerWriterLeaseLostError(
                    f"Writer lease for job {self.job_id} was reclaimed by another "
                    "process (this process stalled past its lease). Stopping rather "
                    "than writing a ledger that now has a different owner; resume "
                    "the job id to continue from the checkpoints already on disk."
                )
        finally:
            conn.close()

    @staticmethod
    def lease_holder(db_path: Path, job_id: str) -> int | None:
        """The pid holding a *live* lease for ``job_id``, or ``None``.

        Read-only liveness probe for surfaces that must distinguish "a writer is
        working on this" from "nothing owns it any more". The status endpoint
        used to treat any non-terminal ledger row as crashed and report FAILED,
        so opening a job that a CLI process was actively translating showed a
        failure and invited the operator to re-run it — which then lost the
        writer-lock race or paid twice.

        An expired lease is not a live writer: it is exactly what a crashed
        process leaves behind, and reporting it as running would be the opposite
        error. Returns ``None`` on any read problem — the caller must not invent
        a holder it cannot see.
        """
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error:
            return None
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT owner_pid, lease_expires_at FROM job_locks WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        except sqlite3.Error:
            # No ``job_locks`` table (a ledger written by an older build) or a
            # corrupt file: no evidence of a holder.
            return None
        finally:
            conn.close()
        if row is None:
            return None
        if float(row["lease_expires_at"]) <= time.time():
            return None
        return int(row["owner_pid"])

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            conn = self._connect()
        except sqlite3.Error:
            return
        try:
            if self._table_exists(conn):
                conn.execute(
                    "DELETE FROM job_locks WHERE job_id = ? AND owner_token = ?",
                    (self.job_id, self._token),
                )
        except sqlite3.Error:  # pragma: no cover - release must never raise
            logger.debug("writer-lock release failed for job %s", self.job_id)
        finally:
            conn.close()

    def __enter__(self) -> LedgerWriterLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


async def keep_lease_fresh(
    lock: LedgerWriterLock,
    *,
    on_lost: Callable[[str], None] | None = None,
) -> None:
    """Renew ``lock`` until cancelled, reporting a reclaim through ``on_lost``.

    Every writer that holds a lease for longer than one window must call this —
    a run and a re-export alike. It is one implementation on purpose: renewal
    had been written only inside the pipeline, so ``re_export_job`` held an
    unrenewed 300 s lease across a whole-book Typst render plus the visual gate,
    and a page-heavy re-export could outlive the lease and be legally reclaimed
    while both processes wrote the same ledger and output file.

    Returns (does not raise) on lease loss: the caller owns the stop decision,
    because the run's stages stop at their own boundary checks while a
    re-export simply unwinds. ``on_lost`` receives the operator-facing reason.
    A transient renew failure is logged and retried — a momentary SQLite
    hiccup must not kill a multi-hour render.
    """
    interval = max(1.0, lock.lease_seconds / 3.0)
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(lock.renew)
        except LedgerWriterLeaseLostError as exc:
            logger.error("Writer lease lost for job %s: %s", lock.job_id, exc)
            if on_lost is not None:
                on_lost(str(exc))
            return
        except Exception as exc:  # a transient renew failure must not kill the run
            logger.debug("Writer-lock renewal failed: %s", exc)
