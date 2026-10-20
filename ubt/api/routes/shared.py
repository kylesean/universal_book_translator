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
    "quality_report": ("Quality report (JSON)", "application/json"),
    "visual_report": ("Visual gate report (JSON)", "application/json"),
    "metrics": ("Run metrics (JSON)", "application/json"),
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
        "quality_report": sidecar_path(out, "quality_report.json"),
        "visual_report": sidecar_path(out, "visual_report.json"),
        "metrics": sidecar_path(out, "metrics.json"),
    }
    for tag in ("_mono", "_dual"):
        candidate = out.with_name(f"{stem}{tag}{suffix}")
        if candidate.exists():
            paths["secondary"] = candidate
            break
    return paths


def progress_frame(snapshot: dict[str, Any], status: str) -> str:
    """The one non-terminal ``/stream`` frame, shared by both deployment modes.

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
    """The one terminal ``/stream`` frame, shared by both deployment modes.

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
) -> str | None:
    """The live job id that already owns ``resolved_out``, or ``None``.

    Covers both homes of a live job: the in-memory map (embedded mode, and the
    job this process is running in queue mode) and the durable queue rows. The
    latter matters because ``manager.jobs`` only knows jobs *this* process
    started -- a job still waiting in the queue was invisible, so two
    submissions naming one output path both passed and the two workers wrote
    one deliverable, last writer winning.
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
    return None


def is_own_prior_output(
    requested_id: str | None,
    target_candidate: Path,
    job_queue: Any,
    manager: Any,
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
    return own_prior_deliverable(target_candidate, prior_output_of(prior))


#: Local-only bind addresses. Anything else exposes the service to the network
#: and therefore requires the isolation guards in ubt.api.app.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})


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
