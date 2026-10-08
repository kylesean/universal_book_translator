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

from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobQueue

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0

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
    now: float | None = None,
) -> int:
    """Prune the staging area, sparing every upload a live queue job still reads."""
    return prune_stale_uploads(
        uploads_dir,
        max_age_days=max_age_days,
        keep=staged_uploads_in_use(job_queue),
        now=now,
    )
