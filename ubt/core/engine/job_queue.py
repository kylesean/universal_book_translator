"""Durable SQLite job queue for the service front door.

Enterprise deployments run air-gapped on a single host, so the queue must add
zero external infrastructure. This is a single-table SQLite queue in WAL mode:
submission persists a row, ``ubt worker`` processes claim rows atomically,
and a restart resumes whatever was queued or stranded. It deliberately reuses
the ledger's storage idiom (persistent connection, ``BEGIN IMMEDIATE``,
``busy_timeout``) because multiple processes — the API and N workers — share
one queue file.

Leasing is per **job**, not per block. Block concurrency belongs to the
pipeline's own semaphores, while the queue only decides which job belongs
to which worker.

The API/worker speak this class directly. ``claim`` is safe under concurrent
processes: the candidate scan and the state transition run inside one
``BEGIN IMMEDIATE`` transaction, and the ``UPDATE ... WHERE status='queued'``
re-checks the row so two workers can never claim the same job.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ubt.core.exceptions import QueueDepthExceededError

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Service-level job lifecycle (distinct from block statuses).

    Unified vocabulary across all runtime surfaces: the in-memory manager
    accepts a job as ``SUBMITTED`` and executes immediately, while queue mode
    places it in ``QUEUED`` until an asynchronous worker claims it.
    """

    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})

#: Default per-tenant running cap; tenants absent from the map share this.
DEFAULT_TENANT_MAX_RUNNING = 4


@dataclass(frozen=True, slots=True)
class QueuedJob:
    """One persisted queue row."""

    job_id: str
    tenant_id: str
    priority: int
    status: JobStatus
    attempts: int
    max_attempts: int
    payload: dict[str, Any]
    error: str | None
    worker_id: str | None
    enqueued_at: float
    started_at: float | None
    finished_at: float | None
    lease_expires_at: float | None
    heartbeat_at: float | None
    cancel_requested: bool
    progress: dict[str, Any]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_queue (
    job_id            TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL DEFAULT 'default',
    priority          INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL,
    attempts          INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL DEFAULT 3,
    payload_json      TEXT NOT NULL,
    error             TEXT,
    worker_id         TEXT,
    enqueued_at       REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL,
    lease_expires_at  REAL,
    heartbeat_at      REAL,
    cancel_requested  INTEGER NOT NULL DEFAULT 0,
    progress_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_job_queue_claim
    ON job_queue(status, priority DESC, enqueued_at ASC);
CREATE INDEX IF NOT EXISTS idx_job_queue_tenant
    ON job_queue(tenant_id, status);
"""


class JobQueue:
    """SQLite-backed durable queue with job-level leases and fairness caps."""

    def __init__(
        self,
        db_path: Path,
        *,
        timeout: float = 30.0,
        default_max_attempts: int = 3,
        global_max_running: int = 8,
        tenant_max_running: Mapping[str, int] | None = None,
        default_tenant_max_running: int = DEFAULT_TENANT_MAX_RUNNING,
        max_queued: int = 1000,
        terminal_retention: int = 1000,
    ) -> None:
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.default_max_attempts = default_max_attempts
        self.global_max_running = global_max_running
        self.tenant_max_running = dict(tenant_max_running or {})
        self.default_tenant_max_running = default_tenant_max_running
        # Depth cap on QUEUED rows. ``claim`` caps how many run at once but not
        # how many pile up, and workers drain at LLM speed, so an uncapped
        # intake (``POST /jobs/submit`` in a loop) fills the queue's SQLite file
        # long before anything retires it.
        self.max_queued = max(1, int(max_queued))
        # Terminal (completed/failed/cancelled) rows are kept for status queries
        # and submit idempotency, but nothing retired them, so the queue file
        # grew with every run. The newest ``terminal_retention`` are preserved;
        # older ones are pruned (see :meth:`prune_terminal`).
        self.terminal_retention = max(0, int(terminal_retention))
        # Amortize the O(rows) prune: run it once this many terminal rows have
        # accumulated since the last one, not on every completion.
        self._prune_batch = max(1, self.terminal_retention)
        self._terminal_since_prune = 0
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init_schema()

    # -- connection -----------------------------------------------------------
    def _init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA)

    def _init_connection(self) -> None:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; transactions are explicit
            timeout=self.timeout,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)};")
        self._conn = conn

    @contextmanager
    def _get_conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            # Initialize inside the lock: a concurrent first-use after close()
            # could otherwise build two connections, leak one, and leave the
            # other thread's queries on an orphaned handle.
            if self._conn is None:
                self._init_connection()
            if self._conn is None:  # explicit: -O strips asserts
                raise RuntimeError("job queue connection failed to initialize")
            try:
                yield self._conn
            except sqlite3.Error:
                self._safe_rollback()
                raise
            except BaseException:
                self._safe_rollback()
                raise

    def _safe_rollback(self) -> None:
        if self._conn is None:
            return
        with suppress(sqlite3.Error):
            self._conn.execute("ROLLBACK;")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # close must never raise
                    logger.warning("Error closing job queue %s", self.db_path)
                finally:
                    self._conn = None

    def __enter__(self) -> JobQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- retention ------------------------------------------------------------
    def prune_terminal(self, *, keep: int | None = None) -> int:
        """Delete terminal rows beyond the newest ``keep``; returns rows removed.

        The queue never retired finished jobs, so its SQLite file grew with
        every run. The newest ``keep`` (default :attr:`terminal_retention`) stay
        so recent ids remain queryable for status and submit idempotency; older
        terminal rows are dropped. Callers must treat a pruned id as unknown.
        """
        limit = self.terminal_retention if keep is None else max(0, int(keep))
        statuses = tuple(s.value for s in TERMINAL_JOB_STATUSES)
        placeholders = ",".join("?" for _ in statuses)
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.execute(
                    f"""
                    DELETE FROM job_queue
                     WHERE status IN ({placeholders})
                       AND job_id NOT IN (
                           SELECT job_id FROM job_queue
                            WHERE status IN ({placeholders})
                            ORDER BY COALESCE(finished_at, enqueued_at) DESC
                            LIMIT ?
                       )
                    """,
                    (*statuses, *statuses, limit),
                )
                removed = int(cursor.rowcount)
                conn.execute("COMMIT;")
            except BaseException:
                self._safe_rollback()
                raise
        return max(0, removed)

    def _note_terminal(self, count: int = 1) -> None:
        """Count terminal transitions; prune once the batch fills.

        Must be called *outside* a ``_get_conn`` block: :meth:`prune_terminal`
        takes the same non-reentrant lock.
        """
        if count <= 0:
            return
        self._terminal_since_prune += count
        if self._terminal_since_prune < self._prune_batch:
            return
        self._terminal_since_prune = 0
        with suppress(sqlite3.Error):
            self.prune_terminal()

    # -- row mapping ----------------------------------------------------------
    @staticmethod
    def _to_job(row: sqlite3.Row) -> QueuedJob:
        return QueuedJob(
            job_id=str(row["job_id"]),
            tenant_id=str(row["tenant_id"]),
            priority=int(row["priority"]),
            status=JobStatus(str(row["status"])),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            payload=json.loads(row["payload_json"]),
            error=row["error"],
            worker_id=row["worker_id"],
            enqueued_at=float(row["enqueued_at"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            cancel_requested=bool(row["cancel_requested"]),
            progress=json.loads(row["progress_json"]),
        )

    # -- submission -----------------------------------------------------------
    def enqueue(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        *,
        tenant_id: str = "default",
        priority: int = 0,
        max_attempts: int | None = None,
        now: float | None = None,
    ) -> QueuedJob:
        """Persist a queued job; idempotent on ``job_id``.

        A resubmitted id whose row is still live (queued/running/submitted) or
        already ``completed`` returns the existing row untouched, matching the
        API's submit idempotency contract. A ``failed``/``cancelled`` row is
        re-queued in place with the new payload: resubmitting a dead id is a
        request to run it again, and returning the dead status left the caller
        holding an id that would never execute.

        Raises :class:`QueueDepthExceededError` when this would be a *new* row
        and the queue already holds ``max_queued`` QUEUED jobs. The count and
        the insert share one ``BEGIN IMMEDIATE`` transaction, so two concurrent
        submitters cannot both pass the check.
        """
        ts = time.time() if now is None else now
        attempts_cap = self.default_max_attempts if max_attempts is None else max_attempts
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                already = conn.execute(
                    "SELECT status, tenant_id FROM job_queue WHERE job_id = ?", (job_id,)
                ).fetchone()
                if already is None:
                    queued = conn.execute(
                        "SELECT COUNT(*) FROM job_queue WHERE status = ?",
                        (JobStatus.QUEUED.value,),
                    ).fetchone()[0]
                    if int(queued) >= self.max_queued:
                        raise QueueDepthExceededError(
                            f"Job queue is full: {queued} jobs are already waiting "
                            f"(max_queued={self.max_queued}). Let the workers drain "
                            "before submitting more, or raise UBT_JOB_MAX_QUEUED."
                        )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO job_queue
                            (job_id, tenant_id, priority, status, attempts, max_attempts,
                             payload_json, enqueued_at)
                        VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                        """,
                        (
                            job_id,
                            tenant_id,
                            int(priority),
                            JobStatus.QUEUED.value,
                            int(attempts_cap),
                            json.dumps(dict(payload)),
                            ts,
                        ),
                    )
                elif (
                    str(already["status"])
                    in (
                        JobStatus.FAILED.value,
                        JobStatus.CANCELLED.value,
                    )
                    and str(already["tenant_id"]) == tenant_id
                ):
                    # Re-run a dead job in place: the ledger holds its prior
                    # progress, so the next claim resumes rather than restarts.
                    conn.execute(
                        """
                        UPDATE job_queue
                        SET status = ?, attempts = 0, max_attempts = ?,
                            payload_json = ?, tenant_id = ?, priority = ?,
                            error = NULL, worker_id = NULL, started_at = NULL,
                            finished_at = NULL, lease_expires_at = NULL,
                            heartbeat_at = NULL, cancel_requested = 0,
                            enqueued_at = ?, progress_json = '{}'
                        WHERE job_id = ?
                        """,
                        (
                            JobStatus.QUEUED.value,
                            int(attempts_cap),
                            json.dumps(dict(payload)),
                            tenant_id,
                            int(priority),
                            ts,
                            job_id,
                        ),
                    )
                # else: live or completed -> idempotent no-op.
                conn.execute("COMMIT;")
            except BaseException:
                self._safe_rollback()
                raise
        existing = self.get(job_id)
        if existing is None:  # pragma: no cover - insert just succeeded
            raise RuntimeError(f"enqueue({job_id!r}) did not persist a row")
        return existing

    # -- worker protocol ------------------------------------------------------
    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        global_limit: int | None = None,
        now: float | None = None,
    ) -> QueuedJob | None:
        """Atomically claim the highest-priority queued job, or ``None``.

        Expired leases are reclaimed first. Global and per-tenant running caps
        are enforced inside the same transaction, and candidates whose tenant is
        at its cap are skipped (so one tenant cannot occupy every worker).
        """
        ts = time.time() if now is None else now
        limit = self.global_max_running if global_limit is None else global_limit
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                self._reclaim_stale_locked(conn, ts)
                running_total = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM job_queue WHERE status = ?",
                        (JobStatus.RUNNING.value,),
                    ).fetchone()[0]
                )
                if running_total >= limit:
                    conn.execute("COMMIT;")
                    return None
                per_tenant = {
                    str(r[0]): int(r[1])
                    for r in conn.execute(
                        "SELECT tenant_id, COUNT(*) FROM job_queue WHERE status = ? "
                        "GROUP BY tenant_id",
                        (JobStatus.RUNNING.value,),
                    ).fetchall()
                }
                candidates = conn.execute(
                    "SELECT job_id, tenant_id FROM job_queue WHERE status = ? "
                    "ORDER BY priority DESC, enqueued_at ASC",
                    (JobStatus.QUEUED.value,),
                ).fetchall()
                chosen: str | None = None
                for cand in candidates:
                    tenant = str(cand["tenant_id"])
                    cap = self.tenant_max_running.get(tenant, self.default_tenant_max_running)
                    if per_tenant.get(tenant, 0) >= cap:
                        continue
                    chosen = str(cand["job_id"])
                    break
                if chosen is None:
                    conn.execute("COMMIT;")
                    return None
                cursor = conn.execute(
                    """
                    UPDATE job_queue
                       SET status = ?, attempts = attempts + 1, worker_id = ?,
                           started_at = ?, heartbeat_at = ?, lease_expires_at = ?,
                           cancel_requested = 0
                     WHERE job_id = ? AND status = ?
                    """,
                    (
                        JobStatus.RUNNING.value,
                        worker_id,
                        ts,
                        ts,
                        ts + lease_seconds,
                        chosen,
                        JobStatus.QUEUED.value,
                    ),
                )
                if cursor.rowcount != 1:  # pragma: no cover - defensive race guard
                    conn.execute("COMMIT;")
                    return None
                conn.execute("COMMIT;")
            except BaseException:
                self._safe_rollback()
                raise
        return self.get(chosen)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        now: float | None = None,
    ) -> bool:
        """Extend the lease; ``False`` if the worker no longer owns the job."""
        ts = time.time() if now is None else now
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE job_queue SET heartbeat_at = ?, lease_expires_at = ?
                 WHERE job_id = ? AND worker_id = ? AND status = ?
                """,
                (ts, ts + lease_seconds, job_id, worker_id, JobStatus.RUNNING.value),
            )
            return cursor.rowcount == 1

    def update_progress(
        self,
        job_id: str,
        worker_id: str,
        progress: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> bool:
        """Merge the worker's progress snapshot into the row (cross-process SSE).

        The write is scoped to the current owner so a worker whose lease was
        reclaimed cannot overwrite the new owner's progress; ``False`` means
        the row no longer belongs to ``worker_id`` (or is no longer running).
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE job_queue SET progress_json = ?
                 WHERE job_id = ? AND worker_id = ? AND status = ?
                """,
                (
                    json.dumps(dict(progress)),
                    job_id,
                    worker_id,
                    JobStatus.RUNNING.value,
                ),
            )
            return cursor.rowcount == 1

    def is_cancel_requested(self, job_id: str, worker_id: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM job_queue "
                "WHERE job_id = ? AND worker_id = ? AND status = ?",
                (job_id, worker_id, JobStatus.RUNNING.value),
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def complete(
        self,
        job_id: str,
        worker_id: str,
        *,
        status: JobStatus,
        error: str | None = None,
        progress: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> bool:
        """Finish a claimed job; ``False`` if the lease was already lost."""
        if status not in TERMINAL_JOB_STATUSES:
            raise ValueError(f"complete() requires a terminal status, got {status!r}")
        ts = time.time() if now is None else now
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE job_queue
                   SET status = ?, error = ?, finished_at = ?, lease_expires_at = NULL,
                       heartbeat_at = NULL, worker_id = NULL,
                       progress_json = COALESCE(?, progress_json)
                 WHERE job_id = ? AND worker_id = ? AND status = ?
                """,
                (
                    status.value,
                    error,
                    ts,
                    None if progress is None else json.dumps(dict(progress)),
                    job_id,
                    worker_id,
                    JobStatus.RUNNING.value,
                ),
            )
            finished = cursor.rowcount == 1
        if finished:
            self._note_terminal()
        return finished

    def release_claim(
        self,
        job_id: str,
        worker_id: str,
        *,
        error: str | None = None,
        decrement_attempt: bool = True,
        now: float | None = None,
    ) -> bool:
        """Release a claimed job back to QUEUED (e.g. on writer lock contention).

        Decrements attempt count by default so transient lock contention does not
        exhaust the job's max_attempts.
        """
        attempt_clause = ", attempts = MAX(0, attempts - 1)" if decrement_attempt else ""
        terminal_cancelled = False
        result = False
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                row = conn.execute(
                    "SELECT cancel_requested FROM job_queue"
                    " WHERE job_id = ? AND worker_id = ? AND status = ?",
                    (job_id, worker_id, JobStatus.RUNNING.value),
                ).fetchone()
                if row is not None and int(row["cancel_requested"]):
                    # Same rule as reclaim_stale: a cancel the user already got a 200
                    # for must not be dropped. Requeueing here hands the row to the
                    # next claim(), which clears the flag and re-runs (and re-bills)
                    # the whole job against that request.
                    conn.execute(
                        "UPDATE job_queue SET status = ?, finished_at = ?, worker_id = NULL, "
                        "lease_expires_at = NULL, heartbeat_at = NULL WHERE job_id = ?",
                        (
                            JobStatus.CANCELLED.value,
                            now if now is not None else time.time(),
                            job_id,
                        ),
                    )
                    conn.execute("COMMIT;")
                    terminal_cancelled = True
                    result = True
                else:
                    cursor = conn.execute(
                        f"""
                        UPDATE job_queue
                           SET status = ?, worker_id = NULL, lease_expires_at = NULL,
                               heartbeat_at = NULL, error = COALESCE(?, error)
                               {attempt_clause}
                         WHERE job_id = ? AND worker_id = ? AND status = ?
                        """,
                        (
                            JobStatus.QUEUED.value,
                            error,
                            job_id,
                            worker_id,
                            JobStatus.RUNNING.value,
                        ),
                    )
                    conn.execute("COMMIT;")
                    result = cursor.rowcount == 1
            except BaseException:
                self._safe_rollback()
                raise
        if terminal_cancelled:
            self._note_terminal()
        return result

    def reclaim_stale(self, *, now: float | None = None) -> tuple[int, int]:
        """Requeue / fail jobs whose lease expired; returns ``(requeued, failed)``."""
        ts = time.time() if now is None else now
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                requeued, failed = self._reclaim_stale_locked(conn, ts)
                conn.execute("COMMIT;")
            except BaseException:
                self._safe_rollback()
                raise
        if failed:
            self._note_terminal(failed)
        return requeued, failed

    @staticmethod
    def _reclaim_stale_locked(conn: sqlite3.Connection, now: float) -> tuple[int, int]:
        rows = conn.execute(
            "SELECT job_id, attempts, max_attempts, cancel_requested FROM job_queue "
            "WHERE status = ? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
            (JobStatus.RUNNING.value, now),
        ).fetchall()
        requeued = 0
        failed = 0
        for row in rows:
            if int(row["cancel_requested"]):
                # The user cancelled while the job was RUNNING; the worker then
                # died before acting on the flag. Honour the cancel instead of
                # requeueing — otherwise claim's `cancel_requested = 0` below
                # silently drops the intent and the whole job is re-run and
                # re-billed against the user's wishes.
                conn.execute(
                    "UPDATE job_queue SET status = ?, finished_at = ?, worker_id = NULL, "
                    "lease_expires_at = NULL, heartbeat_at = NULL WHERE job_id = ?",
                    (JobStatus.CANCELLED.value, now, row["job_id"]),
                )
            elif int(row["attempts"]) >= int(row["max_attempts"]):
                conn.execute(
                    "UPDATE job_queue SET status = ?, error = ?, finished_at = ?, "
                    "worker_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL "
                    "WHERE job_id = ?",
                    (
                        JobStatus.FAILED.value,
                        "lease expired after max attempts",
                        now,
                        row["job_id"],
                    ),
                )
                failed += 1
            else:
                conn.execute(
                    "UPDATE job_queue SET status = ?, worker_id = NULL, started_at = NULL, "
                    "lease_expires_at = NULL, heartbeat_at = NULL WHERE job_id = ?",
                    (JobStatus.QUEUED.value, row["job_id"]),
                )
                requeued += 1
        return requeued, failed

    # -- cancellation ---------------------------------------------------------
    def request_cancel(self, job_id: str, *, now: float | None = None) -> QueuedJob | None:
        """Cancel a queued job outright; flag a running one for its worker."""
        ts = time.time() if now is None else now
        missing = False
        terminal_cancelled = False
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                row = conn.execute(
                    "SELECT status FROM job_queue WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT;")
                    missing = True
                else:
                    if row["status"] == JobStatus.QUEUED.value:
                        conn.execute(
                            "UPDATE job_queue SET status = ?, finished_at = ? WHERE job_id = ?",
                            (JobStatus.CANCELLED.value, ts, job_id),
                        )
                        terminal_cancelled = True
                    elif row["status"] == JobStatus.RUNNING.value:
                        conn.execute(
                            "UPDATE job_queue SET cancel_requested = 1 WHERE job_id = ?",
                            (job_id,),
                        )
                    conn.execute("COMMIT;")
            except BaseException:
                self._safe_rollback()
                raise
        if terminal_cancelled:
            self._note_terminal()
        return None if missing else self.get(job_id)

    # -- queries --------------------------------------------------------------
    def get(self, job_id: str) -> QueuedJob | None:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM job_queue WHERE job_id = ?", (job_id,)).fetchone()
        return self._to_job(row) if row is not None else None

    def queue_position(self, job_id: str) -> int | None:
        """1-based position among queued jobs, or ``None`` if not queued."""
        job = self.get(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            return None
        with self._get_conn() as conn:
            ahead = int(
                conn.execute(
                    "SELECT COUNT(*) FROM job_queue WHERE status = ? AND "
                    "(priority > ? OR (priority = ? AND enqueued_at < ?))",
                    (JobStatus.QUEUED.value, job.priority, job.priority, job.enqueued_at),
                ).fetchone()[0]
            )
        return ahead + 1

    def depth(self, *, tenant_id: str | None = None) -> dict[str, int]:
        """Map status → count, optionally scoped to one tenant."""
        with self._get_conn() as conn:
            if tenant_id is None:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS n FROM job_queue GROUP BY status"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS n FROM job_queue WHERE tenant_id = ? "
                    "GROUP BY status",
                    (tenant_id,),
                ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def list_jobs(
        self,
        *,
        tenant_id: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[QueuedJob]:
        clauses: list[str] = []
        params: list[Any] = []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM job_queue {where} ORDER BY enqueued_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._to_job(r) for r in rows]
