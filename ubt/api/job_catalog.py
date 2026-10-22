"""Job catalog: enumerate the on-disk ledgers for the console's queue table.

The API's ``JobManager`` only knows the jobs this *process* started; a restarted
console would show an empty queue even though every finished job's ledger is
still on disk. The queue table therefore reads the durable store: one
``{job_id}.sqlite`` per job under ``db_dir``. Live jobs (still held by the
manager) are overlaid with their in-memory progress, which is fresher than the
``job_meta`` row a run only finalizes at the end.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.job_options import JOB_ID_MAX_LEN, JOB_ID_RE

logger = logging.getLogger(__name__)


def is_job_ledger_stem(stem: str) -> bool:
    """Whether a ``*.sqlite`` stem could name a job (id charset + length).

    Excludes sidecar databases such as ``tm.sqlite`` by shape; a file that
    passes this is still verified against its ``job_meta`` table before use.
    """
    return bool(stem) and len(stem) <= JOB_ID_MAX_LEN and JOB_ID_RE.fullmatch(stem) is not None


def summarize_job(db_path: Path) -> dict[str, Any] | None:
    """One job's summary from its ledger, or ``None`` when the file is not a job.

    Returns ``None`` (never raises) for a non-ledger sqlite file, so a stray
    database in ``db_dir`` cannot break the whole listing.
    """
    job_id = db_path.stem
    try:
        ledger = SQLiteJobLedger(db_path, read_only=True)
    except Exception as exc:  # unreadable/corrupt file is a skip, not a failure
        logger.debug("job catalog: cannot open %s: %s", db_path, exc)
        return None
    try:
        snapshot = ledger.get_job_snapshot(job_id)
        if snapshot is None:
            return None
        cost = ledger.get_job_metadata_value(job_id, "estimated_cost_usd")
        output_file = ledger.get_job_metadata_value(job_id, "output_file")
        identity = ledger.get_job_metadata_value(job_id, "run_identity")
        selected_pages = ledger.get_job_metadata_value(job_id, "selected_pages")
    except Exception as exc:  # no job_meta table -> not a job ledger
        logger.debug("job catalog: %s is not a job ledger: %s", db_path, exc)
        return None
    finally:
        ledger.close()

    total = int(snapshot.get("total", 0) or 0)
    completed = int(snapshot.get("completed", 0) or 0)
    failed = int(snapshot.get("failed", 0) or 0)
    needs_human = int(snapshot.get("needs_human", 0) or 0)
    blocked_human = int(snapshot.get("blocked_human", 0) or 0)
    processed = completed + failed + needs_human + blocked_human
    # No ``source_path`` in the summary: the queue table only ever shows the
    # basename, and this route is unauthenticated by default. Echoing the
    # absolute path (home directory included) is the same leak ``public_report``
    # and ``public_artifact`` close on the other doors; here it never leaves
    # the database at all.
    return {
        "job_id": str(snapshot["job_id"]),
        "file_name": Path(str(snapshot["source_path"])).name,
        "target_lang": str(snapshot["target_lang"]),
        "status": _reportable_status(
            str(snapshot["status"]),
            has_live_writer=_has_live_writer(db_path, job_id),
            total=total,
            completed=completed,
        ),
        "total_blocks": total,
        "completed_blocks": completed,
        "failed_blocks": failed,
        "needs_human_blocks": needs_human,
        "progress_percent": round(processed / total * 100, 1) if total else 0.0,
        "estimated_cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
        "created_at": snapshot.get("created_at"),
        "updated_at": snapshot.get("updated_at"),
        "has_output": bool(output_file),
        # A queue row that says nothing about the run's nature reads exactly
        # like a real delivery: a rehearsal ships ``[模拟翻译]`` echo text and a
        # ``--pages`` run ships a fragment, and both used to look identical to a
        # full run in the console's list.
        "rehearsal": bool(isinstance(identity, dict) and identity.get("mock_run")),
        "page_scope": page_scope_label(selected_pages),
    }


#: ``job_meta.status`` values that end a job's life.
_TERMINAL_LEDGER_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _reportable_status(stored: str, *, has_live_writer: bool, total: int, completed: int) -> str:
    """The status a listing may show, normalizing a ledger row left mid-run.

    ``job_meta.status`` has no ``running`` value: the row is written
    ``initialized`` at ingest and stays that way for the whole run, so the column
    alone cannot tell a live job from one whose process died. Reporting it
    verbatim made ``/jobs`` show ``initialized`` (which the console renders as a
    live run) for a job that had crashed hours earlier, while ``/status`` for the
    same id — which does normalize — reported it failed.

    A terminal row is passed through. A row whose blocks are all accounted for is
    completed — that is exactly the condition ``finalize_job("completed")`` is
    guarded on, and the export stage finalizes *before* it records the rendered
    digest, so a process lost in that window leaves a fully translated book
    behind a row that still says ``initialized``. Any other row is ``running``
    while a live writer holds the lease and ``failed`` once nothing does — the
    same test ``/status`` applies
    (:meth:`ubt.core.engine.writer_lock.LedgerWriterLock.lease_holder`).

    The lease, not the file's mtime, is what makes this safe. A run spends
    minutes inside one model call without writing a checkpoint, so a "quiet for
    N minutes" rule calls a *working* job dead — measured: with a live lease held
    and the ledger untouched for 1000s, an mtime rule reported ``failed`` where
    ``/status`` correctly reported ``running``. Asking the lease covers the
    reverse case for free: an expired lease is exactly what a crashed process
    leaves behind.
    """
    if stored in _TERMINAL_LEDGER_STATUSES:
        return stored
    if total > 0 and completed >= total:
        return "completed"
    return "running" if has_live_writer else "failed"


def _has_live_writer(db_path: Path, job_id: str) -> bool:
    """Whether a live process holds this job's writer lease (never raises)."""
    from ubt.core.engine.writer_lock import LedgerWriterLock

    try:
        return LedgerWriterLock.lease_holder(db_path, job_id) is not None
    except Exception:  # noqa: BLE001 - a listing must not fail on a probe
        return False


def page_scope_label(pages: Any) -> str | None:
    """``[1, 2, 3, 5]`` -> ``"1-3,5"``; ``None`` when the run was the whole book."""
    if not isinstance(pages, list) or not pages:
        return None
    numbers = sorted({int(p) for p in pages if isinstance(p, (int, float))})
    if not numbers:
        return None
    groups: list[str] = []
    start = prev = numbers[0]
    for value in numbers[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append(f"{start}-{prev}" if prev > start else str(start))
        start = prev = value
    groups.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(groups)


def summarize_queued_job(job: Any) -> dict[str, Any]:
    """A ``/jobs`` row for a queue entry that has no ledger yet.

    ``/jobs`` used to list only ``db_dir/*.sqlite``, and a queued job has no
    ledger until a worker claims it (the ingest stage creates the database).
    Submitting ten jobs therefore left the console showing nothing — or only the
    one that had started — while ``/status`` for the same ids reported "queued,
    position 5". Operators read an empty list as "my submissions were lost" and
    re-submitted, producing duplicates and duplicate spend.

    Shape mirrors :func:`summarize_job` so the client needs one row type.
    """
    payload = getattr(job, "payload", None) or {}
    progress = getattr(job, "progress", None) or {}
    input_path = str(payload.get("input_path") or "")
    return {
        "job_id": str(job.job_id),
        "file_name": Path(input_path).name,
        "target_lang": str(payload.get("target_lang") or ""),
        "status": job.status.value,
        "total_blocks": 0,
        "completed_blocks": 0,
        "failed_blocks": 0,
        "needs_human_blocks": 0,
        "progress_percent": 0.0,
        "estimated_cost_usd": None,
        "created_at": _isoformat(getattr(job, "enqueued_at", None)),
        "updated_at": _isoformat(getattr(job, "enqueued_at", None)),
        "has_output": bool(progress.get("output_file")),
        "rehearsal": bool(payload.get("dry_run", False)),
        "page_scope": page_scope_label(payload.get("pages")),
    }


def _isoformat(timestamp: float | None) -> str | None:
    """Queue rows store epoch seconds; the catalog reports ISO-8601 strings."""
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp), UTC).isoformat()
    except (ValueError, OSError, TypeError):
        return None


def list_job_summaries(
    db_dir: Path,
    *,
    live: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int = 200,
    queued: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    """Summaries of every job ledger in ``db_dir``, newest first.

    ``live`` maps a job id to an override dict (status + progress fields read
    from the manager's in-memory record), which wins over the on-disk values for
    a job that is still running.

    ``queued`` adds queue rows that have no ledger yet (see
    :func:`summarize_queued_job`), so a job is visible from the moment it is
    accepted rather than from the moment a worker starts it.
    """
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    if db_dir.is_dir():
        for path in db_dir.glob("*.sqlite"):
            if not is_job_ledger_stem(path.stem):
                continue
            summary = summarize_job(path)
            if summary is None:
                continue
            seen.add(str(summary["job_id"]))
            override = (live or {}).get(summary["job_id"])
            if override:
                summary.update({k: v for k, v in override.items() if v is not None})
            summaries.append(summary)

    for job in queued:
        job_id = str(getattr(job, "job_id", "") or "")
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        summary = summarize_queued_job(job)
        override = (live or {}).get(job_id)
        if override:
            summary.update({k: v for k, v in override.items() if v is not None})
        summaries.append(summary)

    summaries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return summaries[: max(1, limit)]
