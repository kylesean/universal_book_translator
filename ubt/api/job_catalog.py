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
from collections.abc import Mapping
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
    # absolute path (home directory included) is the same leak ``_public_report``
    # and ``_public_artifact`` close on the other doors; here it never leaves
    # the database at all.
    return {
        "job_id": str(snapshot["job_id"]),
        "file_name": Path(str(snapshot["source_path"])).name,
        "target_lang": str(snapshot["target_lang"]),
        "status": str(snapshot["status"]),
        "total_blocks": total,
        "completed_blocks": completed,
        "failed_blocks": failed,
        "needs_human_blocks": needs_human,
        "progress_percent": round(processed / total * 100, 1) if total else 0.0,
        "estimated_cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
        "created_at": snapshot.get("created_at"),
        "updated_at": snapshot.get("updated_at"),
        "has_output": bool(output_file),
    }


def list_job_summaries(
    db_dir: Path,
    *,
    live: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Summaries of every job ledger in ``db_dir``, newest first.

    ``live`` maps a job id to an override dict (status + progress fields read
    from the manager's in-memory record), which wins over the on-disk values for
    a job that is still running.
    """
    if not db_dir.is_dir():
        return []

    summaries: list[dict[str, Any]] = []
    for path in db_dir.glob("*.sqlite"):
        if not is_job_ledger_stem(path.stem):
            continue
        summary = summarize_job(path)
        if summary is None:
            continue
        override = (live or {}).get(summary["job_id"])
        if override:
            summary.update({k: v for k, v in override.items() if v is not None})
        summaries.append(summary)

    summaries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return summaries[: max(1, limit)]
