"""The public-shape reducers and helpers every route registrar shares.

Split out of :mod:`ubt.api.app` with the registrars that use them (see
:mod:`ubt.api.routes`); :mod:`ubt.api.app` re-exports the ones tests and the
console contract import from there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobStatus
from ubt.core.engine.progress import ARTIFACT_KEYS
from ubt.core.job_options import (
    companion_path,
    own_prior_deliverable,
    prior_output_of,
    sidecar_path,
)


def cross_tenant_404(job_id: str) -> HTTPException:
    # Same body as "not found": a cross-tenant probe must not learn that the
    # job exists.
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")


#: 422 Unprocessable Entity. Starlette renamed the constant (the new name is
#: absent on the ``fastapi>=0.111`` floor, which bundles an older Starlette),
#: so resolve it by name instead of pinning to whichever spelling this
#: environment happens to ship. The literal is stable across both.
UNPROCESSABLE_ENTITY = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


def public_artifact(value: Any) -> Any:
    """Reduce an artifact path to its basename before it leaves the process.

    ``/status`` and the SSE stream echoed the pipeline's absolute
    ``output_file``/``report_file`` — handing every caller that can reach this
    (unauthenticated-by-default) port the host's directory layout, home
    directory included. Only the *echo* changes: the
    routes that serve the bytes (``/download``, ``/report``, the visual-report
    fallback) still resolve the stored absolute path server-side.
    """
    return Path(value).name if isinstance(value, str) and value else value


def public_progress(progress: dict[str, Any]) -> dict[str, Any]:
    """Copy of a progress dict with every artifact path reduced to its basename."""
    data = dict(progress)
    for key in ARTIFACT_KEYS:
        if data.get(key):
            data[key] = public_artifact(data[key])
    return data


#: QualityReport fields that carry an absolute host path. ``/report`` echoed
#: them verbatim, leaking the host layout to any caller that could reach it.
_REPORT_PATH_KEYS = frozenset({"source_path", "output_path"})


def public_report(value: Any) -> Any:
    """Recursively reduce host paths in a report payload to their basenames.

    ``/status`` and the SSE stream already ran ``public_artifact``; ``/report``
    returned the raw report JSON, so the same absolute ``source_path`` /
    ``output_path`` leaked through a different door.
    """
    if isinstance(value, dict):
        return {
            key: (public_artifact(item) if key in _REPORT_PATH_KEYS else public_report(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [public_report(item) for item in value]
    return value


#: Deliverable key -> (human label, media type). The *path* for a key is derived
#: from the job's primary artifact by :func:`deliverable_paths`; this table only
#: names the keys the API is willing to serve, so an unknown key is a 404 rather
#: than a path guess.
DELIVERABLE_LABELS: dict[str, tuple[str, str]] = {
    "primary": ("Translated document", "application/octet-stream"),
    "secondary": ("Complementary mono/dual document", "application/octet-stream"),
    "epub": ("Reflowable EPUB", "application/epub+zip"),
    # The two companions the export stage writes by default. They were produced
    # on every run but absent from this table, so the REST surface and the
    # console could not list or download them: the CAT handoff file and the
    # readable HTML view existed on disk and 404'd through the API.
    "html": ("Readable HTML view", "text/html; charset=utf-8"),
    "xliff": ("Translation-ready XLIFF 2.1", "application/xliff+xml"),
    "contract": ("Delivery contract (JSON)", "application/json"),
    "delivery_contract": ("Artifact delivery contract (JSON)", "application/json"),
    "quality_report": ("Quality report (JSON)", "application/json"),
    "visual_report": ("Visual gate report (JSON)", "application/json"),
    "metrics": ("Run metrics (JSON)", "application/json"),
    "attestation": ("Text realization attestation (JSON)", "application/json"),
}


def deliverable_paths(output_file: str | Path) -> dict[str, Path]:
    """Every deliverable that hangs off a job's primary artifact.

    Uses the export stage's own naming helpers (``sidecar_path`` /
    ``companion_path`` plus the ``_mono`` / ``_dual`` siblings)
    rather than a second copy of the rule, so the API and the writer cannot
    drift. Missing files are still returned as candidate paths; the caller
    filters by existence.
    """
    out = Path(output_file)
    stem, suffix = out.stem, out.suffix
    paths: dict[str, Path] = {
        "primary": out,
        "epub": companion_path(out, ".epub"),
        "html": companion_path(out, ".html"),
        "xliff": companion_path(out, ".xliff"),
        "contract": sidecar_path(out, "contract.json"),
        "delivery_contract": sidecar_path(out, "delivery_contract.json"),
        "quality_report": sidecar_path(out, "quality_report.json"),
        "visual_report": sidecar_path(out, "visual_report.json"),
        "metrics": sidecar_path(out, "metrics.json"),
        "attestation": companion_path(out, "_attestations.json"),
    }
    for tag in ("_mono", "_dual"):
        candidate = out.with_name(f"{stem}{tag}{suffix}")
        if candidate.exists():
            paths["secondary"] = candidate
            break
    return paths


def progress_frame(snapshot: dict[str, Any], status: str) -> str:
    """The one non-terminal ``/jobs/{job_id}/stream`` frame, shared by both modes.

    Queue mode polls a durable snapshot; embedded mode folds engine events into
    the same ``ProgressSnapshot``. Emitting that snapshot as ``event: progress``
    in both cases is what stops one endpoint answering two different schemas
    depending on how the server was started.
    """
    return (
        f"event: progress\ndata: {json.dumps(public_progress({**snapshot, 'status': status}))}\n\n"
    )


def terminal_frame(
    status: str,
    output_file: str | None,
    error: str | None,
    *,
    counters: dict[str, Any] | None = None,
) -> str:
    """The one terminal ``/jobs/{job_id}/stream`` frame, shared by both modes.

    ``counters`` carries the final block totals. Without them the console's log
    line fell back to ``0/0 blocks``: the terminal frame used to hold only
    ``status``/``output_file``/``error``, so a finished job's last line read
    "completed · 0/0 blocks" — the one line a user looks at to confirm the run,
    reporting that zero blocks were processed.
    """
    payload: dict[str, Any] = {"status": status, "output_file": output_file, "error": error}
    if counters:
        payload.update(
            {
                key: counters[key]
                for key in (
                    "total_blocks",
                    "drafted_blocks",
                    "repair_pending_blocks",
                    "completed_blocks",
                    "skipped_blocks",
                    "failed_blocks",
                    "needs_human_blocks",
                    "blocked_human_blocks",
                    "processed_blocks",
                    "progress_percent",
                    "estimated_cost_usd",
                    "current_avg_qe",
                )
                if key in counters
            }
        )
    return f"event: {status}\ndata: {json.dumps(payload)}\n\n"


def output_path_claimant(
    manager: Any,
    job_queue: Any,
    submit_id: str,
    resolved_out: Path,
    db_dir: Path | None = None,
) -> str | None:
    """The live job id that already owns ``resolved_out``, or ``None``.

    Covers all three homes of a live job: the in-memory map (embedded mode, and
    the job this process is running in queue mode), the durable queue rows, and
    — since the MCP server and a CLI run share ``db_dir`` with this process but
    not its memory — the per-job ledgers. The queue scan alone missed a job
    still waiting in the queue, so two submissions naming one output path both
    passed and the two workers wrote one deliverable, last writer winning; the
    ledger scan closes the cross-process gap the same way.
    """
    target = Path(resolved_out).resolve()

    def _claims(candidate: Any) -> bool:
        if not candidate:
            return False
        try:
            return Path(str(candidate)).resolve() == target
        except (OSError, ValueError):
            return False

    for rec in manager.jobs.values():
        if rec.job_id == submit_id or rec.status in (
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            continue
        if _claims(getattr(rec.request, "output_path", None)):
            return str(rec.job_id)
    if job_queue is not None:
        for queued in job_queue.list_jobs(limit=1000):
            if queued.job_id == submit_id or queued.status in TERMINAL_JOB_STATUSES:
                continue
            if _claims((queued.payload or {}).get("output_path")):
                return str(queued.job_id)
    if db_dir is not None:
        return ledger_output_path_claimant(db_dir, submit_id, target)
    return None


def ledger_output_path_claimant(
    db_dir: Path,
    exclude_job_id: str | None,
    resolved_out: Path,
) -> str | None:
    """The job whose live ledger run already owns ``resolved_out``, or ``None``.

    The in-memory claimant cannot see another process's embedded jobs: the MCP
    server and the REST server each hold their own manager over the same
    ``db_dir``, and a CLI ``ubt translate`` writes ledgers there too, so a run
    another process started was invisible and both pipelines wrote (and billed)
    one deliverable. The per-job ledger is the durable record every writer
    shares. A non-terminal row counts as a claimant only while something is
    actually writing it — a live writer lease, or an agent session parked
    between tool calls (agent jobs are intentionally lock-free) — so a process
    that died mid-run does not block the path forever.
    """
    from ubt.api.job_catalog import is_job_ledger_stem, summarize_job
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.writer_lock import LedgerWriterLock

    if not db_dir.is_dir():
        return None
    for db_path in sorted(db_dir.glob("*.sqlite")):
        if not is_job_ledger_stem(db_path.stem):
            continue
        job_id = db_path.stem
        if exclude_job_id and job_id == exclude_job_id:
            continue
        summary = summarize_job(db_path)
        if summary is None:
            continue
        if str(summary.get("status", "")) in TERMINAL_JOB_STATUSES:
            continue
        try:
            live = LedgerWriterLock.lease_holder(db_path, job_id) is not None
        except Exception:  # noqa: BLE001 - a probe must not fail the scan
            live = False
        if not live:
            with SQLiteJobLedger(db_path, read_only=True) as ledger:
                agent_session = ledger.get_job_metadata_value(job_id, "agent_session")
            parked = isinstance(agent_session, dict) and agent_session.get("state") in (
                "awaiting_agent",
                "needs_attention",
            )
            if not parked:
                continue
        with SQLiteJobLedger(db_path, read_only=True) as ledger:
            request = ledger.get_job_metadata_value(job_id, "submitted_request")
        claimed = (request or {}).get("output_path") if isinstance(request, dict) else None
        if not claimed:
            continue
        try:
            if Path(str(claimed)).resolve() == Path(resolved_out).resolve():
                return job_id
        except (OSError, ValueError):
            continue
    return None


def is_own_prior_output(
    requested_id: str | None,
    target_candidate: Path,
    job_queue: Any,
    manager: Any,
    *,
    input_path: Path | str | None = None,
) -> bool:
    """Whether ``target_candidate`` is the output recorded for ``requested_id``.

    Thin wrapper over the shared authority in :mod:`ubt.core.job_options`: it
    looks the prior run up (queue first, then the in-memory manager) and asks
    :func:`~ubt.core.job_options.own_prior_deliverable` whether the target is
    that run's exact deliverable. Overwrite is a *re-run of one job*, not a
    licence to truncate any file inside the sandbox — see the shared helper for
    the full rule (a directory output licenses only the derived deliverable
    names, never every file in the tree).
    """
    if requested_id is None:
        return False
    prior: Any = None
    if job_queue is not None:
        prior = job_queue.get(requested_id)
    if prior is None:
        prior = manager.get_job(requested_id)
    return own_prior_deliverable(target_candidate, prior_output_of(prior), input_path=input_path)


def is_nonloopback_bind(host: str) -> bool:
    """True only for a literal, non-loopback IP bind (``0.0.0.0``, a LAN IP).

    Hostnames are skipped: uvicorn resolves them before ``accept``, so a real
    hostname bind reports its resolved IP in the socket scope anyway. This also
    keeps in-process test clients (whose scope host is ``testserver``) out of
    the guard — they never expose the service to the network.
    """
    import ipaddress

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return not addr.ipv4_mapped.is_loopback
    return not addr.is_loopback


__all__ = [
    "DELIVERABLE_LABELS",
    "UNPROCESSABLE_ENTITY",
    "cross_tenant_404",
    "deliverable_paths",
    "is_nonloopback_bind",
    "is_own_prior_output",
    "output_path_claimant",
    "progress_frame",
    "public_artifact",
    "public_progress",
    "public_report",
    "terminal_frame",
]
