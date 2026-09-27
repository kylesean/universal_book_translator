"""FastAPI microservice exposing REST and SSE streaming endpoints for UBT."""

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ubt import __version__
from ubt.api.manager import JobManager, JobRecord
from ubt.api.models import (
    JobAssessRequest,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
)
from ubt.api.security import (
    SENSITIVE_FILENAME_PARTS,
    SYSTEM_DISALLOWED_PREFIXES,
    _log_startup_auth_warning,
    _require_api_key_gate,
    _tenant_from_header,
    effective_allowed_bases,
    resolve_secure_path,
    validate_job_id,
    verify_api_key,
)
from ubt.core.config import MOCK_API_KEY, UBTConfig
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobQueue, JobStatus
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import ARTIFACT_KEYS, ProgressSnapshot
from ubt.core.exceptions import UBTError
from ubt.core.fs_perms import restrict_env_file
from ubt.core.job_options import (
    JOB_ID_RE,
    LANG_CODE_PATTERN,
    apply_config_overrides,
    default_output_path,
    overrides_from_request,
    run_kwargs_from_request,
)
from ubt.core.log_config import setup_logging
from ubt.core.qe import BaseQERunner
from ubt.core.router import ModelProfile, get_default_registry
from ubt.core.router.rate_limiter import build_rate_limiter
from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)

# Public symbol exports and test instrumentation hooks
__all__ = [
    "JobManager",
    "JobRecord",
    "JobSubmitRequest",
    "JobAssessRequest",
    "JobSubmitResponse",
    "JobStatusResponse",
    "SYSTEM_DISALLOWED_PREFIXES",
    "SENSITIVE_FILENAME_PARTS",
    "_log_startup_auth_warning",
    "_tenant_from_header",
    "verify_api_key",
    "validate_job_id",
    "resolve_secure_path",
    "SQLiteJobLedger",
    "create_app",
    "run_server",
    "_resolve_bind",
    "JOB_ID_RE",
    "LANG_CODE_PATTERN",
    "default_output_path",
    "overrides_from_request",
    "apply_config_overrides",
    "run_kwargs_from_request",
]


def _get_ledger_cls() -> type[Any]:
    """Resolve SQLiteJobLedger dynamically so test monkeypatching is respected."""
    app_mod = sys.modules.get("ubt.api.app")
    return getattr(app_mod, "SQLiteJobLedger", SQLiteJobLedger) if app_mod else SQLiteJobLedger


def _open_read_ledger(ledger_cls: type[Any], db_path: Path) -> Any:
    """Instantiate a ledger in read-only mode, tolerating test mocks without read_only."""
    try:
        return ledger_cls(db_path, read_only=True)
    except TypeError:
        return ledger_cls(db_path)


#: Local-only bind addresses. Anything else exposes the service to the network
#: and therefore requires the isolation guards below.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})


def _bind_isolation_gaps(cfg: UBTConfig) -> list[str]:
    """Isolation guards missing for a non-loopback bind (empty when complete)."""
    missing: list[str] = []
    if not cfg.service_api_key.get_secret_value().strip():
        missing.append("UBT_API_KEY (X-API-Key gate)")
    if not cfg.allowed_base_dirs():
        missing.append("UBT_ALLOWED_DIRS (path sandbox allowlist)")
    return missing


def _insecure_bind_allowed() -> bool:
    return os.environ.get("UBT_ALLOW_INSECURE_BIND", "").lower() in ("1", "true", "yes")


def _is_nonloopback_bind(host: str) -> bool:
    """True only for a literal, non-loopback IP bind (``0.0.0.0``, a LAN IP).

    Hostnames are skipped: uvicorn resolves them before ``accept``, so a real
    hostname bind reports its resolved IP in the socket scope anyway. This also
    keeps in-process test clients (whose scope host is ``testserver``) out of
    the guard — they never expose the service to the network.
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return not addr.ipv4_mapped.is_loopback
    return not addr.is_loopback


class QueueSubscriberCounter:
    """Per-job SSE subscriber accounting for queue mode.

    Embedded mode counts subscribers on the in-memory ``JobRecord``; queue mode
    has no such record but needs the same ceiling. Every open stream polls the
    queue from a worker thread once a second, so N subscribers occupy N threads
    of the shared default executor and stall every other endpoint — including
    this job's own status reads. Counting is per job id, not global.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(1, int(limit))
        self._counts: dict[str, int] = {}

    def acquire(self, job_id: str) -> bool:
        """Reserve a slot; ``False`` when the job already has ``limit`` of them."""
        current = self._counts.get(job_id, 0)
        if current >= self._limit:
            return False
        self._counts[job_id] = current + 1
        return True

    def release(self, job_id: str) -> None:
        """Give a slot back; never goes negative for an unbalanced release."""
        remaining = self._counts.get(job_id, 1) - 1
        if remaining > 0:
            self._counts[job_id] = remaining
        else:
            self._counts.pop(job_id, None)


#: Process-wide ceiling on simultaneously open SSE streams, across *all* jobs.
#:
#: The per-job ceilings above do not bound the process: with N jobs each at
#: ``_MAX_SUBSCRIBERS_PER_JOB`` the fan-out still grows linearly, and every
#: open stream polls from a worker thread once a second — enough subscribers
#: stall the shared default executor and every other endpoint with it
#: under high concurrency.
_MAX_GLOBAL_STREAM_SUBSCRIBERS = 64

#: Single key the global stream budget is booked under, so the per-job counter
#: doubles as a process-wide total (one shared slot bucket, one ceiling).
_GLOBAL_SUBSCRIBER_KEY = "__global__"


def _public_artifact(value: Any) -> Any:
    """Reduce an artifact path to its basename before it leaves the process.

    ``/status`` and the SSE stream echoed the pipeline's absolute
    ``output_file``/``report_file`` — handing every caller that can reach this
    (unauthenticated-by-default) port the host's directory layout, home
    directory included. Only the *echo* changes: the
    routes that serve the bytes (``/download``, ``/report``, the visual-report
    fallback) still resolve the stored absolute path server-side.
    """
    return Path(value).name if isinstance(value, str) and value else value


def _public_progress(progress: dict[str, Any]) -> dict[str, Any]:
    """Copy of a progress dict with every artifact path reduced to its basename."""
    data = dict(progress)
    for key in ARTIFACT_KEYS:
        if data.get(key):
            data[key] = _public_artifact(data[key])
    return data


#: QualityReport fields that carry an absolute host path. ``/report`` echoed
#: them verbatim, leaking the host layout to any caller that could reach it.
_REPORT_PATH_KEYS = frozenset({"source_path", "output_path"})


def _public_report(value: Any) -> Any:
    """Recursively reduce host paths in a report payload to their basenames.

    ``/status`` and the SSE stream already ran ``_public_artifact``; ``/report``
    returned the raw report JSON, so the same absolute ``source_path`` /
    ``output_path`` leaked through a different door.
    """
    if isinstance(value, dict):
        return {
            key: (_public_artifact(item) if key in _REPORT_PATH_KEYS else _public_report(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_public_report(item) for item in value]
    return value


def _progress_frame(snapshot: dict[str, Any], status: str) -> str:
    """The one non-terminal ``/stream`` frame, shared by both deployment modes.

    Queue mode polls a durable snapshot; embedded mode folds engine events into
    the same ``ProgressSnapshot``. Emitting that snapshot as ``event: progress``
    in both cases is what stops one endpoint answering two different schemas
    depending on how the server was started.
    """
    return (
        f"event: progress\ndata: {json.dumps(_public_progress({**snapshot, 'status': status}))}\n\n"
    )


def _terminal_frame(status: str, output_file: str | None, error: str | None) -> str:
    """The one terminal ``/stream`` frame, shared by both deployment modes."""
    return (
        f"event: {status}\n"
        f"data: {json.dumps({'status': status, 'output_file': output_file, 'error': error})}\n\n"
    )


def create_app(
    config: UBTConfig | None = None,
    router: ModelRouter | None = None,
    qe_runner: BaseQERunner | None = None,
    queue: JobQueue | None = None,
) -> FastAPI:
    """Factory creating FastAPI application instance.

    ``queue`` (or ``config.job_mode='queue'``) switches the Jobs endpoints to
    the durable queue: submit only enqueues, and separate ``ubt worker``
    processes drain it (restart-safe, multi-process). The default is embedded
    mode, where the API runs each job as an in-process task.

    Deliberately does NOT configure logging. This factory is imported as a
    library by tests and tooling, and calling ``setup_logging`` here silently
    reset the host process's root logger configuration. The
    process entry points configure it: ``run_server`` for the ``ubt-api``
    script, and the module-level bootstrap below for ``uvicorn
    ubt.api.app:app``.
    """
    app_config = config or UBTConfig.from_env()
    if queue is None and app_config.job_mode == "queue":
        queue = JobQueue(
            app_config.job_queue_path or (app_config.db_dir / "job_queue.sqlite"),
            global_max_running=app_config.job_max_running,
            default_tenant_max_running=app_config.job_tenant_max_running,
            max_queued=app_config.job_max_queued,
        )
    job_queue: JobQueue | None = queue
    # One token bucket for every job this process runs. The orchestrator still
    # builds its own provider per job so a request can override its models, but
    # `rate_limit_rpm` is a per-credential budget and must not be multiplied by
    # the number of concurrent jobs.
    # Reuse the one construction helper so the AIMD ``backoff_cooldown_sec``
    # (config default 3.0) cannot be silently dropped here again: this inline
    # ``AdaptiveTokenBucket(...)`` omitted it, so the documented oscillation
    # guard was absent on the default ``ubt-api`` deployment while CLI/worker
    # (which go through ``build_rate_limiter``/pipeline) kept it.
    shared_rate_limiter = None if router is not None else build_rate_limiter(app_config)
    app_mod = sys.modules.get("ubt.api.app")
    job_manager_cls = getattr(app_mod, "JobManager", JobManager) if app_mod else JobManager
    manager: JobManager = job_manager_cls(
        router=router,
        qe_runner=qe_runner,
        rate_limiter=shared_rate_limiter,
        # Honour UBT_JOB_MAX_RUNNING here too. Without it the embedded manager
        # kept its own default of 8, so an operator who set the ceiling to 1 to
        # serialise work (single GPU, cost control) still got 8 concurrent
        # pipelines — and the assess semaphore below inherited the same 8.
        max_running_jobs=app_config.job_max_running,
    )
    # ``/jobs/assess`` with ``deep=True`` runs full adapter ingest (docling
    # model load, minutes/GPU), yet unlike ``/jobs/submit`` it had no cap — the
    # one default-open endpoint that could pile unbounded heavy work onto the
    # process. It shares the job ceiling.
    assess_semaphore = asyncio.Semaphore(manager.max_running_jobs)

    def _tenant_allows(job_id: str, tenant: str) -> bool:
        """False when a queued job exists but belongs to another tenant.

        Tenant is a starvation/isolation boundary, not authentication — the API
        key still grants whole-service access. Scoping reads and cancels stops
        one tenant from reading or cancelling another tenant's job. Embedded
        (non-queue) mode has no tenant dimension, so it always allows.
        """
        if job_queue is None:
            return True
        job = job_queue.get(job_id)
        return job is None or job.tenant_id == tenant

    async def _tenant_allows_async(job_id: str, tenant: str) -> bool:
        if job_queue is None:
            return True
        job = await asyncio.to_thread(job_queue.get, job_id)
        return job is None or job.tenant_id == tenant

    def _cross_tenant_404(job_id: str) -> HTTPException:
        # Same body as "not found": a cross-tenant probe must not learn that the
        # job exists.
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
        )

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        _log_startup_auth_warning(app_config)
        yield

    def _artifact_path(valid_id: str, key: str) -> str | None:
        """Disk fallback for an artifact path a completed job persisted.

        The in-memory ``JobRecord`` is pruned/evicted across restarts, so the
        path is read back from whichever durable store actually holds it:
        ``output_file``/``report_file``/``visual_report_file`` live in the job
        ledger for an embedded run and in the queue row's payload for
        ``job_mode="queue"`` (the worker never writes ledger metadata). Before
        both were consulted a queued job 404'd its own download forever.
        """
        db_path = app_config.db_dir / f"{valid_id}.sqlite"
        if db_path.exists():
            try:
                ledger_cls = _get_ledger_cls()
                with _open_read_ledger(ledger_cls, db_path) as ledger:
                    value = ledger.get_job_metadata_value(valid_id, key)
            except Exception:
                value = None
            if value:
                return str(value)
        if job_queue is not None:
            job = job_queue.get(valid_id)
            if job is not None:
                queued = (job.progress or {}).get(key)
                if queued:
                    return str(queued)
        return None

    async def _artifact_path_async(valid_id: str, key: str) -> str | None:
        return await asyncio.to_thread(_artifact_path, valid_id, key)

    async def _verify_request_key(
        request: Request,
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if request.url.path.rstrip("/") == "/health":
            return
        verify_api_key(x_api_key, _config=app_config)

    auth_enabled = bool(app_config.service_api_key.get_secret_value().strip())
    api_app = FastAPI(
        title="Universal Book Translator API",
        version=__version__,
        description="Industrial asynchronous bilingual translation microservice with SSE streaming.",
        dependencies=[Depends(_verify_request_key)],
        lifespan=_lifespan,
        docs_url=None if auth_enabled else "/docs",
        redoc_url=None if auth_enabled else "/redoc",
        openapi_url=None if auth_enabled else "/openapi.json",
    )
    # Booked on ``app.state`` so a test can assert the process-wide limiter's
    # AIMD shape (e.g. the backoff cooldown) without reaching into the manager.
    api_app.state.shared_rate_limiter = shared_rate_limiter
    # Booked for tests that assert the embedded submit branch's idempotency.
    api_app.state.job_manager = manager

    # A bind guard that only lives in ``run_server`` is bypassed by
    # ``uvicorn ubt.api.app:app --host 0.0.0.0``, which imports the app object
    # directly. Enforce it per request from the socket's local address, so the
    # documented "loopback unless fully isolated" contract holds however the
    # process was started.
    @api_app.middleware("http")
    async def _enforce_bind_isolation(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        server = request.scope.get("server")
        host = str(server[0]) if server else ""
        if host and _is_nonloopback_bind(host):
            gaps = _bind_isolation_gaps(app_config)
            if gaps and not _insecure_bind_allowed():
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": (
                            f"Service is bound to non-loopback address {host!r} without full "
                            f"isolation: missing {', '.join(gaps)}. Set them, bind to "
                            "127.0.0.1, or set UBT_ALLOW_INSECURE_BIND=1 to override."
                        )
                    },
                )
        return await call_next(request)

    @api_app.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "universal-book-translator",
            "version": __version__,
        }

    @api_app.post(
        "/jobs/assess",
        response_model=dict[str, Any],
        tags=["Jobs"],
        summary="Assess a document quote, risks and route without spend",
    )
    async def assess_job(req: JobAssessRequest) -> dict[str, Any]:
        from ubt.core.assess import AssessmentError, assess_document_async

        cfg = app_config
        if req.preset is not None:
            from ubt.core.presets import apply_preset

            cfg = apply_preset(cfg, req.preset)

        resolved_in = resolve_secure_path(req.input_path, must_exist=True, config=cfg)

        async def _assess() -> dict[str, Any]:
            try:
                report = await assess_document_async(
                    resolved_in,
                    cfg,
                    deep=req.deep,
                    target_lang=req.target_lang,
                    source_lang=req.source_lang,
                )
                return report.to_dict()
            except AssessmentError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": exc.code, "message": str(exc)},
                ) from exc

        if req.deep:
            if assess_semaphore.locked():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server at capacity: too many deep assessments in flight",
                )
            async with assess_semaphore:
                return await _assess()
        if resolved_in.suffix.lower() == ".pdf":
            # A "shallow" assess is not a cheap manifest read: it still runs the
            # full pdfium page census + font-encoding witness (assess._pdf_facts).
            # Queue it on the same slots so a burst cannot pile unbounded
            # per-page work onto the process (the 503 guard above stays
            # deep-only; shallow requests wait rather than reject).
            async with assess_semaphore:
                return await _assess()
        return await _assess()

    @api_app.post(
        "/jobs/submit",
        response_model=JobSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Jobs"],
    )
    async def submit_job(
        req: JobSubmitRequest,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JobSubmitResponse:
        resolved_in = resolve_secure_path(req.input_path, must_exist=True, config=app_config)
        requested_id = validate_job_id(req.job_id) if req.job_id else None

        # Submit idempotency: if this job_id is already known and still live (or
        # completed), return it before checking output collisions (a completed
        # run's output file already exists). A failed/cancelled id falls through
        # so ``enqueue`` re-runs it instead of handing back a dead job.
        if job_queue is not None and requested_id is not None:
            existing_queued = await asyncio.to_thread(job_queue.get, requested_id)
            if existing_queued is not None:
                # Tenant isolation: a cross-tenant probe must not learn the job
                # exists (the read routes already return 404, but idempotency
                # here returned it outright — an existence/status oracle).
                if existing_queued.tenant_id != _tenant_from_header(x_ubt_tenant):
                    raise _cross_tenant_404(requested_id)
                if existing_queued.status not in (
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    return JobSubmitResponse(
                        job_id=existing_queued.job_id,
                        status=existing_queued.status.value,
                        stream_url=f"/jobs/{existing_queued.job_id}/stream",
                        status_url=f"/jobs/{existing_queued.job_id}/status",
                        rehearsal=bool(existing_queued.payload.get("dry_run", False)),
                    )
        if requested_id is not None:
            existing = manager.get_job(requested_id)
            # Embedded parity with the queue branch above: a failed/cancelled id
            # falls through so a fresh job replaces the dead record instead of
            # handing back a task that will never run.
            if existing is not None and existing.status not in (
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                return JobSubmitResponse(
                    job_id=existing.job_id,
                    status=existing.status,
                    stream_url=f"/jobs/{existing.job_id}/stream",
                    status_url=f"/jobs/{existing.job_id}/status",
                    rehearsal=existing.request.dry_run,
                )

        if req.output_path:
            resolved_out = resolve_secure_path(req.output_path, must_exist=False, config=app_config)
            if resolved_out is not None and resolved_out.exists():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="output_path already exists; refusing to overwrite it",
                )
        else:
            default_out = default_output_path(resolved_in)
            # Place the server-derived default inside the sandbox the validator
            # will enforce. With no UBT_ALLOWED_DIRS the default deliverable
            # (~/Documents/UBT/...) sits outside the implicit cwd+db_dir bases,
            # so an allowlist-less deployment 403'd on every submit that
            # omitted output_path.
            bases = effective_allowed_bases(app_config)
            try:
                default_resolved = default_out.resolve()
                if not any(default_resolved == b or b in default_resolved.parents for b in bases):
                    if any(
                        resolved_in.parent == b or b in resolved_in.parent.parents for b in bases
                    ):
                        default_out = resolved_in.parent / default_out.name
                    else:
                        default_out = bases[0] / default_out.name
            except Exception:
                pass
            resolved_out = resolve_secure_path(default_out, must_exist=False, config=app_config)
            if resolved_out is not None and resolved_out.exists():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="output_path already exists; refusing to overwrite it",
                )

        resolved_glossary: str | None = None
        if req.glossary:
            # Same sandbox as input_path: a glossary is a filesystem path, so it
            # must clear resolve_secure_path before the job may read it.
            resolved_glossary = str(
                resolve_secure_path(req.glossary, must_exist=True, config=app_config)
            )

        safe_req = req.model_copy(
            update={
                "input_path": str(resolved_in),
                "output_path": str(resolved_out) if resolved_out is not None else None,
                "glossary": resolved_glossary,
            }
        )
        # No provider key and no injected router → the run can only be a mock:
        # auto-set rehearsal so every downstream surface (submit response,
        # status, queue payload, worker) labels it instead of silently
        # reporting a mock delivery as a real one.
        app_key = app_config.api_key.get_secret_value()
        auto_rehearsal = False
        if not safe_req.dry_run and router is None and (not app_key or app_key == MOCK_API_KEY):
            safe_req = safe_req.model_copy(update={"dry_run": True})
            auto_rehearsal = True
            logger.warning(
                "No API key configured: auto-setting dry_run rehearsal for input %s",
                safe_req.input_path,
            )
        if job_queue is not None:
            queued_id = requested_id or f"job_{uuid.uuid4().hex[:12]}"
            queued_payload = safe_req.model_dump()
            if auto_rehearsal:
                # The worker recomputes rehearsal from its OWN key; this
                # marker says the dry_run was our auto-decision, not a user
                # request, so a keyed worker runs for real.
                queued_payload["rehearsal_auto"] = True
            try:
                # ``enqueue`` runs ``BEGIN IMMEDIATE`` with a 30s busy_timeout;
                # calling it inline on the event loop stalls every other request
                # under write contention.
                job = await asyncio.to_thread(
                    job_queue.enqueue,
                    queued_id,
                    queued_payload,
                    tenant_id=_tenant_from_header(x_ubt_tenant),
                    priority=int(req.priority),
                )
            except UBTError as exc:
                # QueueDepthExceededError: the intake is full. 429 with the
                # reason, matching the embedded path's at-capacity response.
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=str(exc),
                ) from exc
            return JobSubmitResponse(
                job_id=job.job_id,
                status=job.status.value,
                stream_url=f"/jobs/{job.job_id}/stream",
                status_url=f"/jobs/{job.job_id}/status",
                rehearsal=safe_req.dry_run,
            )
        try:
            record = manager.create_job(safe_req, job_id=requested_id)
        except (RuntimeError, UBTError) as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc

        record.task = asyncio.create_task(manager.execute_job(record, app_config))

        return JobSubmitResponse(
            job_id=record.job_id,
            status=JobStatus.SUBMITTED,
            stream_url=f"/jobs/{record.job_id}/stream",
            status_url=f"/jobs/{record.job_id}/status",
            rehearsal=safe_req.dry_run,
        )

    @api_app.post(
        "/jobs/{job_id}/cancel",
        tags=["Jobs"],
    )
    async def cancel_job(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> dict[str, str]:
        """Cancel an in-flight job (idempotent: terminal jobs return as-is)."""
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if job_queue is not None:
            job = await asyncio.to_thread(job_queue.request_cancel, valid_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job not found: {valid_id}")
            return {"job_id": valid_id, "status": job.status.value}
        record = manager.get_job(valid_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {valid_id}")
        if record.status not in (JobStatus.SUBMITTED, JobStatus.RUNNING):
            return {"job_id": valid_id, "status": record.status}
        record.status = JobStatus.CANCELLED
        task = record.task
        if task is not None and not task.done():
            task.cancel()

        def _finalize_cancelled_ledger() -> None:
            try:
                db_path = app_config.db_dir / f"{valid_id}.sqlite"
                if db_path.exists():
                    import time

                    from ubt.core.engine.writer_lock import LedgerWriterLock
                    from ubt.core.exceptions import LedgerWriterLockConflictError

                    # The pipeline holds the job-level writer lock for the whole
                    # run; taking it here too is what stops a cancellation from
                    # clobbering a still-writing pipeline's checkpoints (and vice
                    # versa). Wait briefly for the cancelled task to release it,
                    # then finalize; if it is still held, the pipeline itself
                    # will land the terminal status.
                    lock = LedgerWriterLock(db_path, valid_id)
                    deadline = time.monotonic() + 10.0
                    while True:
                        try:
                            lock.acquire()
                            break
                        except LedgerWriterLockConflictError:
                            if time.monotonic() >= deadline:
                                logger.debug(
                                    "Cancellation for %s: writer lock still held; "
                                    "the running pipeline will finalize it.",
                                    valid_id,
                                )
                                return
                            time.sleep(0.1)
                    try:
                        ledger_cls = _get_ledger_cls()
                        with ledger_cls(db_path) as ledger:
                            # Prevent rewriting a finished job to CANCELLED if it
                            # already reached a terminal status.
                            if ledger.get_job_status(valid_id) in TERMINAL_JOB_STATUSES:
                                return
                            ledger.finalize_job(valid_id, status=JobStatus.CANCELLED)
                    finally:
                        lock.release()
            except Exception as exc:
                logger.debug("Could not persist cancellation for %s: %s", valid_id, exc)

        await asyncio.to_thread(_finalize_cancelled_ledger)
        return {"job_id": valid_id, "status": JobStatus.CANCELLED}

    @api_app.get(
        "/jobs/{job_id}/status",
        response_model=JobStatusResponse,
        tags=["Jobs"],
    )
    def get_status(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JobStatusResponse:
        valid_id = validate_job_id(job_id)
        if not _tenant_allows(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if job_queue is not None:
            job = job_queue.get(valid_id)
            if job is not None:
                return JobStatusResponse(
                    **_public_progress(ProgressSnapshot.model_validate(job.progress).model_dump()),
                    job_id=job.job_id,
                    status=job.status.value,
                    created_at=datetime.fromtimestamp(job.enqueued_at, UTC),
                    error=job.error,
                    queue_position=job_queue.queue_position(valid_id),
                    rehearsal=bool(job.payload.get("dry_run", False)),
                )
        record = manager.get_job(valid_id)
        if record:
            return JobStatusResponse(
                **_public_progress(record.progress.model_dump()),
                job_id=record.job_id,
                status=record.status,
                created_at=record.created_at,
                error=record.error,
                rehearsal=record.request.dry_run,
            )

        # Fallback: inspect persisted SQLite ledger on disk
        db_path = app_config.db_dir / f"{valid_id}.sqlite"
        if db_path.exists():
            try:
                ledger_cls = _get_ledger_cls()
                with _open_read_ledger(ledger_cls, db_path) as ledger:
                    snap = ledger.get_job_snapshot(valid_id)
                    if snap:
                        created_str = str(snap.get("created_at", ""))
                        try:
                            created_dt = datetime.fromisoformat(created_str)
                        except Exception:
                            created_dt = datetime.now(UTC)

                        progress = ProgressSnapshot.from_ledger(
                            snap, lambda key: ledger.get_job_metadata_value(valid_id, key)
                        )
                        status_value = str(snap["status"])
                        error_note: str | None = None
                        if job_queue is None and status_value not in TERMINAL_JOB_STATUSES:
                            # Embedded mode has no durable worker and no startup
                            # reconcile, so a row a crash left mid-run will never
                            # move again. Reporting "initialized" forever keeps
                            # every polling client waiting on a job that no
                            # longer exists.
                            status_value = JobStatus.FAILED
                            error_note = (
                                "Job was interrupted before it reached a terminal "
                                "state (the server restarted); re-run it to resume."
                            )
                        return JobStatusResponse(
                            **_public_progress(progress.model_dump()),
                            job_id=str(snap["job_id"]),
                            status=status_value,
                            created_at=created_dt,
                            error=error_note,
                        )
            except Exception as exc:
                logger.exception("Failed to inspect ledger on disk for job %s: %s", valid_id, exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Database error inspecting job ledger; see server logs "
                        f"(job_id={valid_id})."
                    ),
                ) from exc

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {valid_id}",
        )

    # Queue mode has no in-memory ``JobRecord`` to count subscribers on, but it
    # needs the same ceiling (see QueueSubscriberCounter).
    queue_subscribers = QueueSubscriberCounter(JobManager._MAX_SUBSCRIBERS_PER_JOB)
    # ...and neither counter bounds the *process*: the global budget below is
    # what caps the fan-out across every job (see _MAX_GLOBAL_STREAM_SUBSCRIBERS).
    # Booked on ``app.state`` so a caller (or a test) can exhaust it deliberately.
    global_subscribers = QueueSubscriberCounter(_MAX_GLOBAL_STREAM_SUBSCRIBERS)
    api_app.state.global_stream_subscribers = global_subscribers

    def _acquire_global_stream_slot(per_job_key: str | None = None) -> None:
        """Reserve the process-wide stream slot or fail with 429.

        ``per_job_key`` is released back when the global budget is exhausted,
        so a refused subscriber never leaks the per-job slot it took first.
        """
        if global_subscribers.acquire(_GLOBAL_SUBSCRIBER_KEY):
            return
        if per_job_key is not None:
            queue_subscribers.release(per_job_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many concurrent stream subscribers: the global limit of "
                f"{_MAX_GLOBAL_STREAM_SUBSCRIBERS} is in use. Close a stream or "
                "retry after a job finishes."
            ),
        )

    @api_app.get(
        "/jobs/{job_id}/stream",
        response_class=StreamingResponse,
        tags=["Jobs"],
    )
    async def stream_progress(
        job_id: str, request: Request, x_ubt_tenant: str | None = Header(default=None)
    ) -> StreamingResponse:
        """SSE: ``event: progress`` snapshots until one ``event: <status>`` terminal frame.

        Both deployment modes emit that one contract (see ``_progress_frame`` /
        ``_terminal_frame``); the payload never depends on how the server was
        started.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if job_queue is not None:
            initial_job = await asyncio.to_thread(job_queue.get, valid_id)
            if initial_job is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job not found: {valid_id}",
                )
            if not queue_subscribers.acquire(valid_id):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many concurrent stream subscribers for this job",
                )
            _acquire_global_stream_slot(per_job_key=valid_id)
            queue_slot_released = False

            def _release_queue_stream_slots() -> None:
                nonlocal queue_slot_released
                if queue_slot_released:
                    return
                queue_slot_released = True
                queue_subscribers.release(valid_id)
                global_subscribers.release(_GLOBAL_SUBSCRIBER_KEY)

            async def _queue_events() -> AsyncIterator[str]:
                last: dict[str, Any] | None = None
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        job = await asyncio.to_thread(job_queue.get, valid_id)
                        if job is None:
                            break
                        progress_payload = {**ProgressSnapshot().to_payload(), **job.progress}
                        snapshot = {**progress_payload, "status": job.status.value}
                        if snapshot != last:
                            yield _progress_frame(progress_payload, job.status.value)
                            last = snapshot
                        if job.status in TERMINAL_JOB_STATUSES:
                            yield _terminal_frame(
                                job.status.value,
                                _public_artifact(job.progress.get("output_file")),
                                job.error,
                            )
                            break
                        await asyncio.sleep(1.0)
                finally:
                    _release_queue_stream_slots()

            from starlette.background import BackgroundTask

            return StreamingResponse(
                _queue_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
                background=BackgroundTask(_release_queue_stream_slots),
            )
        record = manager.get_job(valid_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {valid_id}",
            )

        if len(record.subscribers) >= JobManager._MAX_SUBSCRIBERS_PER_JOB:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many concurrent stream subscribers for this job",
            )
        _acquire_global_stream_slot()
        queue: asyncio.Queue[TranslationProgressEvent | None] = asyncio.Queue(
            maxsize=JobManager._SUBSCRIBER_QUEUE_MAXSIZE
        )
        record.subscribers.append(queue)
        mem_slot_released = False

        def _release_mem_stream_slots() -> None:
            nonlocal mem_slot_released
            if mem_slot_released:
                return
            mem_slot_released = True
            global_subscribers.release(_GLOBAL_SUBSCRIBER_KEY)
            if queue in record.subscribers:
                record.subscribers.remove(queue)

        async def event_generator() -> AsyncIterator[str]:
            try:
                # A late subscriber gets one current snapshot, not a replay of
                # the raw event log: the queue-mode branch emits this exact
                # shape, and one endpoint must not answer two schemas depending
                # on the deployment mode.
                yield _progress_frame(record.progress.model_dump(), record.status)

                if record.status in TERMINAL_JOB_STATUSES and queue.empty():
                    yield _terminal_frame(
                        record.status, _public_artifact(record.progress.output_file), record.error
                    )
                    return

                while True:
                    if await request.is_disconnected():
                        break

                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        if record.status in TERMINAL_JOB_STATUSES:
                            yield _terminal_frame(
                                record.status,
                                _public_artifact(record.progress.output_file),
                                record.error,
                            )
                            break
                        continue

                    if event is None:
                        yield _terminal_frame(
                            record.status,
                            _public_artifact(record.progress.output_file),
                            record.error,
                        )
                        break

                    yield _progress_frame(record.progress.model_dump(), record.status)
            finally:
                _release_mem_stream_slots()

        from starlette.background import BackgroundTask

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(_release_mem_stream_slots),
        )

    @api_app.get(
        "/jobs/{job_id}/report",
        tags=["Jobs"],
    )
    async def get_report(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        record = manager.get_job(valid_id)
        report_file = record.progress.report_file if record else None
        if not report_file:
            report_file = await _artifact_path_async(valid_id, "report_file")
        if not report_file or not Path(report_file).exists():
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            known = (
                record is not None
                or in_queue
                or (await _artifact_path_async(valid_id, "output_file")) is not None
                or (app_config.db_dir / f"{valid_id}.sqlite").exists()
            )
            raise HTTPException(
                status_code=404 if not known else 400,
                detail=(
                    f"Job not found: {valid_id}"
                    if not known
                    else "Quality report is not yet generated or available."
                ),
            )

        try:
            report_path = resolve_secure_path(report_file, must_exist=True, config=app_config)
            content = await asyncio.to_thread(report_path.read_text, encoding="utf-8")
            report_data: dict[str, Any] = json.loads(content)
            return JSONResponse(content=_public_report(report_data))
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as err:
            raise HTTPException(
                status_code=400,
                detail="Quality report is not yet generated or available.",
            ) from err
        except Exception as exc:
            logger.exception("Failed to read quality report for job %s: %s", valid_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(f"Failed to read quality report; see server logs (job_id={valid_id})."),
            ) from exc

    @api_app.get(
        "/jobs/{job_id}/visual-report",
        tags=["Jobs"],
    )
    async def get_visual_report(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        record = manager.get_job(valid_id)
        vis_file = record.progress.visual_report_file if record else None
        if not vis_file:
            vis_file = await _artifact_path_async(valid_id, "visual_report_file")
        if vis_file:
            try:
                vis_path = resolve_secure_path(vis_file, must_exist=True, config=app_config)
            except HTTPException:
                # ``resolve_secure_path`` signals a rejected or missing path with
                # HTTPException, not ValueError/FileNotFoundError — catching the
                # latter let it escape, so the ledger fallback below (which holds
                # the report that is actually persisted) was unreachable.
                pass
            else:
                if vis_path.exists():
                    try:
                        content = await asyncio.to_thread(vis_path.read_text, encoding="utf-8")
                        visual_data: dict[str, Any] = json.loads(content)
                        return JSONResponse(content=visual_data)
                    except Exception as exc:
                        logger.exception(
                            "Failed to parse visual report JSON for job %s: %s", valid_id, exc
                        )
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=(
                                f"Error reading visual report; see server logs (job_id={valid_id})."
                            ),
                        ) from exc

        db_path = app_config.db_dir / f"{valid_id}.sqlite"
        if db_path.exists():
            try:

                def _fetch_visual_report() -> Any:
                    ledger_cls = _get_ledger_cls()
                    with _open_read_ledger(ledger_cls, db_path) as ledger:
                        return ledger.get_visual_report(valid_id)

                snap_report = await asyncio.to_thread(_fetch_visual_report)
                if snap_report is not None:
                    return JSONResponse(content=snap_report)
            except Exception as exc:
                logger.exception(
                    "Failed to query visual report from ledger for job %s: %s", valid_id, exc
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Database error reading visual report; see server logs "
                        f"(job_id={valid_id})."
                    ),
                ) from exc

        if not record and not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Job not found: {valid_id}")
        raise HTTPException(
            status_code=400,
            detail="Visual report is not yet generated or available.",
        )

    @api_app.get(
        "/jobs/{job_id}/download",
        tags=["Jobs"],
    )
    async def download_file(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> FileResponse:
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        record = manager.get_job(valid_id)
        output_file = record.progress.output_file if record else None
        if not output_file:
            output_file = await _artifact_path_async(valid_id, "output_file")
        if not output_file:
            db_path = app_config.db_dir / f"{valid_id}.sqlite"
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            job_exists = (record is not None) or in_queue or db_path.exists()
            raise HTTPException(
                status_code=400 if job_exists else 404,
                detail=(
                    "Translated file is not ready for download."
                    if job_exists
                    else f"Job not found: {valid_id}"
                ),
            )

        output_path = resolve_secure_path(output_file, must_exist=False, config=app_config)
        if not output_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Translated file not found for job: {valid_id}",
            )
        return FileResponse(
            path=output_path,
            filename=output_path.name,
            media_type="application/octet-stream",
        )

    @api_app.get(
        "/api/v1/model-profiles",
        response_model=list[ModelProfile],
        tags=["Model Profiles"],
    )
    @api_app.get(
        "/model-profiles",
        response_model=list[ModelProfile],
        tags=["Model Profiles"],
        include_in_schema=False,
    )
    async def list_model_profiles() -> list[ModelProfile]:
        """List all currently registered model capability profiles."""
        return get_default_registry().list_profiles()

    @api_app.post(
        "/api/v1/model-profiles",
        response_model=ModelProfile,
        tags=["Model Profiles"],
    )
    @api_app.post(
        "/model-profiles",
        response_model=ModelProfile,
        tags=["Model Profiles"],
        include_in_schema=False,
    )
    async def register_model_profile(
        profile: ModelProfile,
        x_api_key: str | None = Header(default=None),
    ) -> ModelProfile:
        """Register a custom model capability profile (built-in profiles cannot be overridden)."""
        verify_api_key(x_api_key, _config=app_config)
        try:
            get_default_registry().register(profile, override=False)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return profile

    return api_app


def _bootstrap_asgi_app() -> FastAPI:
    """Build the module-level app that ``uvicorn ubt.api.app:app`` imports.

    Configures logging only when this process has none yet. ``create_app`` is
    also imported as a library (tests, tooling) and must not reconfigure a
    host's root logger, so the call lives here — on the ASGI-target path — and
    is skipped once anything has already set up handlers.
    """
    if not logging.getLogger().handlers:
        setup_logging()
    # Converge ``.env`` to 0600 before the config layer reads it: dotenv files
    # are created 0644 by editors/cp and hold the API credentials. Warn-only inside, so a
    # read-only mount never blocks a boot; `run_server` reaches this too because
    # uvicorn reuses the already-imported module.
    restrict_env_file()
    return create_app()


# Module-level app is built lazily via PEP 562 ``__getattr__``. Importing this
# module (e.g. ``from ubt.api.app import create_app``) must not configure the
# host's logging, chmod ``.env`` or build a FastAPI app as a side effect; only an
# actual ``ubt.api.app:app`` access (uvicorn's string target, ``from ubt.api.app
# import app``) pays for it, and only once.
_APP: FastAPI | None = None


def _get_app() -> FastAPI:
    global _APP
    if _APP is None:
        _APP = _bootstrap_asgi_app()
    return _APP


def __getattr__(name: str) -> Any:
    if name == "app":
        return _get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_bind(host: str | None, port: int | None) -> tuple[str, int]:
    """Merge explicit arguments with the ``ubt-api --host/--port`` flags."""
    if host is not None and port is not None:
        return host, port
    parser = argparse.ArgumentParser(
        prog="ubt-api",
        description="Serve the UBT translation API (localhost by default).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    # ``parse_known_args``, not ``parse_args``: when this is reached from the
    # Typer ``ubt api`` command, ``sys.argv`` still carries the ``api``
    # subcommand, which argparse would reject (exit 2). Unknown extras are
    # ignored; the ``ubt-api --host/--port`` flags are still honoured.
    args, _ = parser.parse_known_args()
    if host is not None:
        return host, int(args.port)
    if port is not None:
        return str(args.host), port
    return str(args.host), int(args.port)


def run_server(host: str | None = None, port: int | None = None) -> None:
    """CLI helper to boot uvicorn server.

    Defaults to binding localhost only. The service now refuses to start without
    an API key gate: set ``UBT_API_KEY`` (or ``UBT_STRICT_AUTH=1``). For a local,
    throwaway server, ``UBT_ALLOW_NO_AUTH=1`` restores the old open behaviour.
    """
    host, port = _resolve_bind(host, port)
    setup_logging()
    server_config = UBTConfig.from_env()
    if host not in _LOOPBACK_HOSTS:
        missing = _bind_isolation_gaps(server_config)
        if missing and not _insecure_bind_allowed():
            raise SystemExit(
                f"refusing to bind non-loopback host {host!r} without full isolation: "
                f"missing {', '.join(missing)}. Set them, or set "
                "UBT_ALLOW_INSECURE_BIND=1 to override (not recommended)."
            )
    _require_api_key_gate(server_config)
    import uvicorn

    uvicorn.run("ubt.api.app:app", host=host, port=port, reload=False)
