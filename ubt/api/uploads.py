"""Housekeeping for the upload staging area.

A browser cannot hand the server a filesystem path, so the console POSTs the
document's bytes to ``/jobs/upload`` and the API stages a copy under
``<db_dir>/uploads``; that copy is the job's input for the rest of its life.
Nothing ever removed the copies, so a console in use for a while accumulates
every document ever uploaded -- each one book-sized -- on the same disk the
ledger lives on. The sweep here runs once at startup and deletes staged files
past their retention window.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path

from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobQueue, JobStatus

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0

#: Cap for staged uploads. Books are large but not unbounded; this also keeps
#: a rogue multipart body from filling the disk the ledger lives on. Lived in
#: ``ubt.api.app`` until the route split; the upload route and the size-limit
#: middleware both read it, and re-export it (an operator override test
#: monkeypatches it through :mod:`ubt.api.app`).
UPLOAD_MAX_BYTES = 512 * 1024 * 1024  # 512 MB

#: Paths whose request bodies are size-capped before the body is read. Only the
#: upload route streams a client-controlled body to disk; the rest are small
#: JSON payloads FastAPI buffers anyway.
UPLOAD_PATH_SUFFIX = "/jobs/upload"

#: Queue rows to inspect when deciding what is still in use. Non-terminal rows
#: are bounded by ``job_max_queued`` plus the running jobs, so this is a scan
#: limit in name only; it exists because ``list_jobs`` applies one in SQL.
_QUEUE_SCAN_LIMIT = 100_000


def _resolved(path: object) -> str | None:
    """A path as an absolute string, or ``None`` when it cannot be resolved.

    Both sides of the in-use comparison go through here, so a symlinked
    ``db_dir`` (macOS ``/tmp`` -> ``/private/tmp``) cannot make one spelling of
    a path look like a different file.
    """
    if not path:
        return None
    try:
        return str(Path(str(path)).resolve())
    except (OSError, ValueError):
        return None


def staged_uploads_in_use(job_queue: JobQueue | None) -> set[str]:
    """Resolved input paths of queue jobs that have not reached a terminal state.

    A staged upload belongs to its job until that job is finished, and the age
    of the file says nothing about whether a worker is still about to read it:
    a job can sit QUEUED across a restart, so its input is spared however old
    the file is. Embedded mode has no queue, so nothing is protected by it.
    """
    if job_queue is None:
        return set()
    in_use: set[str] = set()
    for job in job_queue.list_jobs(limit=_QUEUE_SCAN_LIMIT):
        if job.status in TERMINAL_JOB_STATUSES:
            continue
        resolved = _resolved((job.payload or {}).get("input_path"))
        if resolved:
            in_use.add(resolved)
    return in_use


def resumable_uploads(db_dir: Path) -> set[str]:
    """Resolved source paths of failed/cancelled embedded ledgers.

    The startup upload sweep only knew about live queue rows. In embedded mode
    a failed job's staged source is exactly what ``/resume`` needs, but after
    the retention window the sweep deleted it, turning a resumable ledger into
    an unrestorable one. The ledger is the durable record of that dependency.
    """
    from ubt.api.job_catalog import is_job_ledger_stem, summarize_job

    in_use: set[str] = set()
    if not db_dir.is_dir():
        return in_use
    for db_path in db_dir.glob("*.sqlite"):
        if not is_job_ledger_stem(db_path.stem):
            continue
        job_id = db_path.stem
        # ``summarize_job`` verifies the file actually has a ``job_meta`` row
        # for this id before opening it as a ledger, so service databases such
        # as ``tm.sqlite``/``job_queue.sqlite`` are skipped safely.
        summary = summarize_job(db_path)
        if summary is None or summary.get("status") not in (
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            continue
        resolved = _resolved(_job_source_path(db_path, job_id))
        if resolved:
            in_use.add(resolved)
    return in_use


def _job_source_path(db_path: Path, job_id: str) -> str | None:
    """Read a verified ledger's source path without re-stamping a service DB."""
    from ubt.core.engine.ledger import SQLiteJobLedger

    try:
        with SQLiteJobLedger(db_path, read_only=True) as ledger:
            return ledger.get_job_source_path(job_id)
    except Exception as exc:
        logger.debug("upload sweep: cannot read source from %s: %s", db_path, exc)
        return None


def prune_stale_uploads(
    uploads_dir: Path,
    *,
    max_age_days: float,
    keep: Iterable[str] = (),
    now: float | None = None,
) -> int:
    """Delete staged uploads older than ``max_age_days``; returns how many went.

    ``max_age_days <= 0`` disables the sweep (an operator keeping every staged
    document, e.g. as a submission archive). ``keep`` holds resolved paths a
    live job may still read. Only regular files are considered, and a file that
    cannot be resolved is left alone: the sweep deletes what it can account for
    and nothing else. Missing directory, unreadable entry, failed unlink -- all
    leave the file in place and the sweep going.
    """
    if max_age_days <= 0:
        return 0
    if not uploads_dir.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - max_age_days * _SECONDS_PER_DAY
    protected = set(keep)
    removed = 0
    for entry in sorted(uploads_dir.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            continue
        resolved = _resolved(entry)
        if resolved is None or resolved in protected:
            continue
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            entry.unlink()
        except OSError as exc:
            logger.warning("upload sweep: could not remove %s: %s", entry, exc)
            continue
        removed += 1
    return removed


def reap_stale_uploads(
    uploads_dir: Path,
    *,
    max_age_days: float,
    job_queue: JobQueue | None = None,
    db_dir: Path | None = None,
    now: float | None = None,
) -> int:
    """Prune the staging area, sparing every upload a live queue job still reads."""
    keep = staged_uploads_in_use(job_queue)
    if db_dir is not None:
        keep |= resumable_uploads(db_dir)
    return prune_stale_uploads(
        uploads_dir,
        max_age_days=max_age_days,
        keep=keep,
        now=now,
    )
