"""FastAPI microservice exposing REST and SSE streaming endpoints for UBT."""

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ubt import __version__
from ubt.api.assets import (
    GlossaryFormatError,
    TMImportError,
    add_glossary_term,
    detect_glossary_conflicts,
    evict_tm_entries,
    import_tm_entries,
    list_tm_entries,
    parse_tm_payload,
    read_glossary_terms,
    remove_glossary_term,
)
from ubt.api.job_catalog import list_job_summaries
from ubt.api.manager import JobManager, JobRecord
from ubt.api.models import (
    GlossaryTermRequest,
    JobAssessRequest,
    JobAssessResponse,
    JobDeleteResponse,
    JobListResponse,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    JobUploadResponse,
    SegmentEditRequest,
    SystemInfoResponse,
    TermPropagationRequest,
    TMevictRequest,
    TMImportRequest,
)
from ubt.api.review import (
    ISSUE_KINDS,
    ReviewBlockNotFound,
    ReviewEditConflict,
    ReviewEditError,
    apply_human_edit,
    apply_term_propagation,
    segment_issue_kinds,
    segment_matches_filter,
    serialize_segment,
    term_cascade_report,
)
from ubt.api.scope import ApiScope
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
from ubt.api.sse_replay import SseReplayBuffer
from ubt.core.config import MOCK_API_KEY, UBTConfig
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobQueue, JobStatus
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import ARTIFACT_KEYS, ProgressSnapshot
from ubt.core.exceptions import (
    BudgetExceededError,
    DocumentParseError,
    QueueDepthExceededError,
    ServerCapacityError,
    UBTError,
    UnsupportedDocumentFormatError,
)
from ubt.core.fs_perms import ensure_private_dir
from ubt.core.job_options import (
    JOB_ID_RE,
    LANG_CODE_PATTERN,
    apply_config_overrides,
    clean_source_stem,
    companion_path,
    default_output_path,
    overrides_from_request,
    resolve_target_output,
    run_kwargs_from_request,
    sidecar_path,
)
from ubt.core.log_config import setup_logging
from ubt.core.qe import BaseQERunner
from ubt.core.router import ModelProfile, get_default_registry
from ubt.core.router.rate_limiter import build_rate_limiter
from ubt.core.router.router import ModelRouter
from ubt.render.page_preview import (
    PagePreviewUnavailable,
    render_page_preview,
    render_source_page_png,
)

logger = logging.getLogger(__name__)

# Strong references to fire-and-forget tasks: the event loop holds only weak
# references, so an unreferenced task can be garbage-collected mid-flight and
# its work silently never happen.
_background_tasks: set[asyncio.Task[None]] = set()

# Public symbol exports and test instrumentation hooks
__all__ = [
    "JobManager",
    "JobRecord",
    "JobSubmitRequest",
    "JobAssessRequest",
    "JobSubmitResponse",
    "JobUploadResponse",
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
    "clean_source_stem",
    "overrides_from_request",
    "apply_config_overrides",
    "run_kwargs_from_request",
]


#: Local-only bind addresses. Anything else exposes the service to the network
#: and therefore requires the isolation guards below.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})

#: Cap for staged uploads. Books are large but not unbounded; this also keeps
#: a rogue multipart body from filling the disk the ledger lives on.
UPLOAD_MAX_BYTES = 512 * 1024 * 1024  # 512 MB


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


_SSE_POLL_EXECUTOR: ThreadPoolExecutor | None = None


def _get_sse_poll_executor() -> ThreadPoolExecutor:
    """Isolated, bounded thread pool for SSE queue polling to prevent default thread pool exhaustion."""
    global _SSE_POLL_EXECUTOR
    if _SSE_POLL_EXECUTOR is None:
        _SSE_POLL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ubt-sse-poll")
    return _SSE_POLL_EXECUTOR


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


#: Deliverable key -> (human label, media type). The *path* for a key is derived
#: from the job's primary artifact by :func:`_deliverable_paths`; this table only
#: names the keys the API is willing to serve, so an unknown key is a 404 rather
#: than a path guess.
DELIVERABLE_LABELS: dict[str, tuple[str, str]] = {
    "primary": ("Translated document", "application/octet-stream"),
    "secondary": ("Complementary mono/dual document", "application/octet-stream"),
    "epub": ("Reflowable EPUB", "application/epub+zip"),
    "contract": ("Delivery contract (JSON)", "application/json"),
    "quality_report": ("Quality report (JSON)", "application/json"),
    "visual_report": ("Visual gate report (JSON)", "application/json"),
    "metrics": ("Run metrics (JSON)", "application/json"),
}


def _deliverable_paths(output_file: str | Path) -> dict[str, Path]:
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


def _is_own_prior_output(
    requested_id: str | None,
    target_candidate: Path,
    job_queue: Any,
    manager: Any,
) -> bool:
    """Whether ``target_candidate`` is the output recorded for ``requested_id``.

    Overwrite is a *re-run of the same job*, not a licence to truncate any file
    inside the sandbox. ``fresh`` resumes the ledger; it does not authorize
    clobbering ``ubt.toml`` / ``job_queue.sqlite`` / another job's ledger. A
    submit naming an existing path may overwrite it only when that path is what
    this very job id already resolved to.
    """
    if requested_id is None:
        return False
    prior: Any = None
    if job_queue is not None:
        prior = job_queue.get(requested_id)
    if prior is None:
        prior = manager.get_job(requested_id)
    if prior is None:
        return False
    payload = getattr(prior, "payload", None)
    if not isinstance(payload, dict):
        request = getattr(prior, "request", None)
        payload = request.model_dump() if request is not None else None
    if not isinstance(payload, dict):
        return False
    prior_out = payload.get("output_path")
    prior_in = payload.get("input_path")
    if not prior_out or not prior_in:
        return False
    try:
        prior_out_path = Path(prior_out)
        if prior_out_path.is_dir():
            return target_candidate.resolve().parent == prior_out_path.resolve()
        prior_target = resolve_target_output(prior_out_path, prior_in)
        return prior_target.resolve() == target_candidate.resolve()
    except (OSError, ValueError):
        return False


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
    manager: JobManager = JobManager(
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

    scope = ApiScope(
        config=app_config,
        manager=manager,
        job_queue=job_queue,
        assess_semaphore=assess_semaphore,
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        _require_api_key_gate(app_config)
        _log_startup_auth_warning(app_config)
        yield

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

    _register_system_routes(api_app, scope)
    _register_job_routes(api_app, scope, router)
    _register_stream_routes(api_app, scope)
    _register_artifact_routes(api_app, scope)
    _register_review_routes(api_app, scope)
    _register_asset_routes(api_app, scope)
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists() and (static_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        class _SPAStaticFiles(StaticFiles):
            """StaticFiles that falls back to index.html for browser navigations.

            The console is a single-page app with real URLs (``/wizard``,
            ``/jobs/:id/quality``); a browser refresh or a pasted link must land
            on ``index.html`` and let the client router resolve the path, not a
            404. API routes are registered before this mount, so they still win.

            The fallback is gated on ``Accept: text/html``: an API client probing
            a closed surface (``/docs`` on a keyed server, a typo'd endpoint)
            must still get a JSON 404 rather than the SPA shell.
            """

            async def get_response(self, path: str, scope: Any) -> Response:
                from starlette.exceptions import HTTPException as StarletteHTTPException

                try:
                    return await super().get_response(path, scope)
                except StarletteHTTPException as exc:
                    accepts_html = any(
                        key == b"accept" and b"text/html" in value.lower()
                        for key, value in scope.get("headers", [])
                    )
                    if exc.status_code != 404 or not accepts_html:
                        raise
                    return await super().get_response("index.html", scope)

        api_app.mount("/", _SPAStaticFiles(directory=str(static_dir), html=True), name="console")

    return api_app


def _register_system_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the system surface."""
    app_config = scope.config
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _uploads_dir = scope.uploads_dir

    def _cross_tenant_404(job_id: str) -> HTTPException:
        # Same body as "not found": a cross-tenant probe must not learn that the
        # job exists.
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
        )

    @api_app.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "universal-book-translator",
            "version": __version__,
        }

    @api_app.get("/system/doctor", tags=["System"])
    async def system_doctor(probe: bool = False) -> JSONResponse:
        """Engine Doctor self-check (the ``ubt doctor`` checklist as JSON).

        Reuses the CLI's ``collect_checks`` so the console and the command line
        report the same verdicts. ``probe=true`` additionally contacts the
        provider endpoint (a live network call), so it is off by default.
        """
        from ubt.cli.commands.doctor import _overall_status, _summary, collect_checks

        checks = await asyncio.to_thread(collect_checks, app_config, probe=probe)
        summary = _summary(checks)
        return JSONResponse(
            {
                "status": _overall_status(summary),
                "summary": summary,
                "checks": [
                    {
                        "group": check.group,
                        "name": check.name,
                        "status": check.status,
                        "detail": check.detail,
                        "fix": check.fix,
                    }
                    for check in checks
                ],
            }
        )

    @api_app.get("/system/info", response_model=SystemInfoResponse, tags=["System"])
    async def system_info(request: Request) -> dict[str, Any]:
        """The console's security-boundary panel (real host + allowed roots).

        Reports the host this request reached (loopback vs exposed), whether the
        API-key gate is on, and the filesystem roots ``resolve_secure_path``
        actually enforces — so the panel shows the live policy, not a hardcoded
        sample.
        """
        from ubt import __version__

        host = request.url.hostname or ""
        disk_free_gb: float | None = None
        try:
            usage = shutil.disk_usage(app_config.db_dir)
            disk_free_gb = round(usage.free / (1024**3), 1)
        except OSError:
            pass

        return {
            "version": __version__,
            "host": host,
            "is_loopback": host in _LOOPBACK_HOSTS,
            "auth_enabled": bool(app_config.service_api_key.get_secret_value().strip()),
            "allowed_bases": [str(path) for path in effective_allowed_bases(app_config)],
            "db_dir": str(app_config.db_dir),
            "job_mode": app_config.job_mode,
            "disk_free_gb": disk_free_gb,
            "wal_status": "ONLINE (WAL Mode Active)",
        }


def _register_job_routes(
    api_app: FastAPI, scope: ApiScope, router: ModelRouter | None = None
) -> None:
    """Routes for the job surface."""
    app_config = scope.config
    manager = scope.manager
    job_queue = scope.job_queue
    assess_semaphore = scope.assess_semaphore
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _uploads_dir = scope.uploads_dir

    def _cross_tenant_404(job_id: str) -> HTTPException:
        # Same body as "not found": a cross-tenant probe must not learn that the
        # job exists.
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
        )

    @api_app.post(
        "/jobs/assess",
        response_model=JobAssessResponse,
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
                    pages=req.pages,
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
        "/jobs/upload",
        response_model=JobUploadResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Jobs"],
        summary="Stage a source document server-side; returns the input_path to submit",
    )
    async def upload_source_document(
        file: Annotated[UploadFile, File()],
    ) -> JobUploadResponse:
        # Browsers cannot send a usable filesystem path (File.path is an
        # Electron-only property), so the wizard uploads the bytes here and
        # submits the returned server-side path instead of a client guess.
        from ubt.adapters.factory import supported_suffixes

        original_name = Path(file.filename or "").name.strip()
        if not original_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing file name",
            )
        suffix = Path(original_name).suffix.lower()
        known = supported_suffixes()
        if suffix not in known:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported document format '{suffix or '(none)'}'. Supported: {', '.join(known)}.",
            )

        # Collisions cannot clobber a prior upload: every staged file gets a
        # timestamped, uuid-prefixed name. The readable stem is kept so the
        # operator can still tell the staged copies apart on disk.
        stem = (
            re.sub(r"[^\w.\- ]+", "_", Path(original_name).stem, flags=re.UNICODE).strip()
            or "upload"
        )
        uploads_dir = ensure_private_dir(_uploads_dir())
        dest = (
            uploads_dir / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}-{stem}{suffix}"
        )

        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > UPLOAD_MAX_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=(
                                f"Uploaded file exceeds the {UPLOAD_MAX_BYTES // (1024 * 1024)} MB limit"
                            ),
                        )
                    out.write(chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        except OSError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store upload: {exc}",
            ) from exc
        finally:
            await file.close()

        return JobUploadResponse(file_path=str(dest), file_name=original_name, size_bytes=size)

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
        # Resolve the id once (client-supplied or server-generated) so the
        # default deliverable directory can be derived per-job before enqueue.
        submit_id = requested_id or f"job_{uuid.uuid4().hex[:12]}"

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
            target_candidate = (
                resolve_target_output(resolved_out, resolved_in)
                if resolved_out is not None
                else None
            )
            if (
                target_candidate is not None
                and target_candidate.exists()
                and not _is_own_prior_output(requested_id, target_candidate, job_queue, manager)
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="output_path already exists; refusing to overwrite it",
                )
        else:
            # Server-derived default: deliverables live in their own per-job
            # directory under the managed ``outputs`` tree — never beside the
            # staged upload, which would mix sources with translations. Passing
            # the directory allows the export stage to derive the honest
            # deliverable name (_mono or _bilingual) dynamically based on the
            # resolved profile and bilingual_mode, rather than pre-allocating
            # a hardcoded filename.
            default_out_dir = ensure_private_dir(_managed_dir("outputs") / submit_id)
            resolved_out = resolve_secure_path(default_out_dir, must_exist=False, config=app_config)
            stem = clean_source_stem(resolved_in)
            for candidate in (
                default_out_dir / f"{stem}_mono{resolved_in.suffix}",
                default_out_dir / f"{stem}_bilingual{resolved_in.suffix}",
            ):
                if candidate.exists() and not _is_own_prior_output(
                    requested_id, candidate, job_queue, manager
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="output_path already exists; refusing to overwrite it",
                    )

        # Reject when another live job already claimed this output path. The
        # filesystem exists() check above cannot see an un-written target, so
        # the in-memory job map is the only guard against two concurrent
        # pipelines last-writer-wins-ing the same deliverable.
        if resolved_out is not None:
            for rec in manager.jobs.values():
                if rec.job_id == submit_id or rec.status in (
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    continue
                if (
                    rec.request.output_path
                    and Path(rec.request.output_path).resolve() == Path(resolved_out).resolve()
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"output_path already claimed by live job {rec.job_id}",
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
            queued_id = submit_id
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
            except QueueDepthExceededError as exc:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=str(exc),
                ) from exc
            except UBTError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
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
            record = manager.create_job(safe_req, job_id=submit_id)
        except (ServerCapacityError, QueueDepthExceededError) as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc
        except UnsupportedDocumentFormatError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
        except DocumentParseError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except BudgetExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=str(exc),
            ) from exc
        except (RuntimeError, UBTError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
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

    @api_app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
    async def list_jobs(
        request: Request,
        limit: int = 200,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> Any:
        """The job queue: every ledger in ``db_dir``, newest first.

        Reads the durable store (one ``{job_id}.sqlite`` per job) so a restarted
        console still lists finished jobs, and overlays live jobs with the
        manager's in-memory progress (fresher than the ``job_meta`` row a run
        only finalizes at the end).

        ``GET /jobs`` is also the console's Mission Control URL, so a browser
        navigation (``Accept: text/html``) is answered with the SPA shell and
        the client router resolves the screen; API clients get the JSON queue.
        """
        if "text/html" in request.headers.get("accept", ""):
            index = Path(__file__).resolve().parent / "static" / "index.html"
            if index.exists():
                return FileResponse(index, media_type="text/html")

        def _live_overrides() -> dict[str, dict[str, Any]]:
            overrides: dict[str, dict[str, Any]] = {}
            for job_id, record in manager.jobs.items():
                progress = record.progress
                overrides[job_id] = {
                    "status": record.status,
                    "total_blocks": progress.total_blocks,
                    "completed_blocks": progress.completed_blocks,
                    "failed_blocks": progress.failed_blocks,
                    "needs_human_blocks": progress.needs_human_blocks,
                    "progress_percent": progress.progress_percent,
                    "estimated_cost_usd": progress.estimated_cost_usd,
                }
            return overrides

        def _catalog() -> list[dict[str, Any]]:
            return list_job_summaries(
                app_config.db_dir, live=_live_overrides(), limit=min(max(limit, 1), 1000)
            )

        summaries = await asyncio.to_thread(_catalog)
        tenant = _tenant_from_header(x_ubt_tenant)
        if tenant is not None and job_queue is not None:
            # Match the per-job endpoints' rule exactly: a job still in the queue
            # is visible only to its tenant; a job the queue no longer tracks
            # (finished and pruned) is visible to any authenticated caller.
            def _visible() -> list[dict[str, Any]]:
                kept: list[dict[str, Any]] = []
                for item in summaries:
                    queued = job_queue.get(item["job_id"])
                    if queued is None or queued.tenant_id == tenant:
                        kept.append(item)
                return kept

            summaries = await asyncio.to_thread(_visible)
        return {"jobs": summaries}

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
                    deadline = time.monotonic() + 2.0
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
                        with SQLiteJobLedger(db_path) as ledger:
                            # Prevent rewriting a finished job to CANCELLED if it
                            # already reached a terminal status.
                            if ledger.get_job_status(valid_id) in TERMINAL_JOB_STATUSES:
                                return
                            ledger.finalize_job(valid_id, status=JobStatus.CANCELLED)
                    finally:
                        lock.release()
            except Exception as exc:
                logger.debug("Could not persist cancellation for %s: %s", valid_id, exc)

        # Pin the task: without a strong reference the event loop may collect
        # it before the thread pool finishes, losing the terminal ledger write.
        task = asyncio.create_task(asyncio.to_thread(_finalize_cancelled_ledger))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        return {"job_id": valid_id, "status": JobStatus.CANCELLED}

    @api_app.post("/jobs/{job_id}/resume", tags=["Jobs"])
    async def resume_job(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """Re-run a failed/cancelled job from its ledger checkpoints (PRD §4.2.4).

        The pipeline resumes from the blocks still pending in the ledger, so no
        finished segment is re-translated. Resume reuses the *original* request
        (kept on the in-memory record), which keeps the run-identity guard happy
        — a reconstructed request would silently drop knobs and could be refused
        as a different profile/engine. That makes this a same-session recovery
        (the network-drop case); a job whose record is gone after a restart must
        be resubmitted instead.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if job_queue is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Resume is available in embedded mode; in queue mode resubmit "
                    "the job to have a worker drain its pending blocks."
                ),
            )
        record = manager.get_job(valid_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Job {valid_id} is not tracked by this server session; "
                    "resubmit it to resume from its ledger."
                ),
            )
        if record.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is {record.status}; only a failed or cancelled job can resume.",
            )
        # The resume path obeys the same concurrency cap as a fresh submit: a
        # batch of failed jobs restarted together would otherwise open N
        # pipelines on a server configured for one. Checked before the record
        # flips to SUBMITTED, so a refusal leaves it resumable.
        try:
            manager.ensure_capacity()
        except ServerCapacityError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc

        # Never re-fresh (that would discard the checkpoints) and clear the
        # previous failure so the record restarts clean.
        record.request = record.request.model_copy(update={"fresh": False})
        record.error = None
        record.progress = ProgressSnapshot()
        record.status = JobStatus.SUBMITTED
        record.task = asyncio.create_task(manager.execute_job(record, app_config))
        return JSONResponse(
            {
                "job_id": valid_id,
                "status": record.status,
                "stream_url": f"/jobs/{valid_id}/stream",
                "status_url": f"/jobs/{valid_id}/status",
            }
        )

    @api_app.delete(
        "/jobs/{job_id}",
        response_model=JobDeleteResponse,
        tags=["Jobs"],
        summary="Remove a finished job's ledger and deliverables from the console",
    )
    async def delete_job(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JobDeleteResponse:
        """Delete one job's history: the db_dir ledger plus its deliverable dir.

        A running job must reach a terminal status first — deletion is history
        management, not a stop button. A non-terminal queue-mode record is
        likewise refused: the durable queue owns its rows (``prune_terminal``
        is their lifecycle) and removing the ledger underneath a worker would
        desynchronize its view.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)

        record = manager.get_job(valid_id)
        if record is not None and record.status not in TERMINAL_JOB_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Job is {record.status}; cancel it and wait for a "
                    "terminal status before deleting."
                ),
            )
        if job_queue is not None:
            queued = await asyncio.to_thread(job_queue.get, valid_id)
            if queued is not None and queued.status not in TERMINAL_JOB_STATUSES:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"Queue record is {queued.status.value}; only a "
                        "terminal job can be deleted."
                    ),
                )

        def _remove() -> tuple[bool, bool]:
            # job ids are [A-Za-z0-9_-]+ (validate_job_id), so the glob is
            # literal — exactly the ledger plus its -shm/-wal/.writer.lock.
            ledger_gone = False
            for path in sorted(app_config.db_dir.glob(f"{valid_id}.sqlite*")):
                path.unlink(missing_ok=True)
                ledger_gone = True
            outputs_dir = _managed_dir("outputs") / valid_id
            outputs_gone = outputs_dir.is_dir()
            if outputs_gone:
                shutil.rmtree(outputs_dir, ignore_errors=True)
            return ledger_gone, outputs_gone

        ledger_gone, outputs_gone = await asyncio.to_thread(_remove)
        manager.forget_job(valid_id)
        if not ledger_gone and not outputs_gone and record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {valid_id}",
            )
        return JobDeleteResponse(
            job_id=valid_id, removed_ledger=ledger_gone, removed_outputs=outputs_gone
        )

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
                with SQLiteJobLedger(db_path, read_only=True) as ledger:
                    snap = ledger.get_job_snapshot(valid_id)
                    if snap:
                        created_str = str(snap.get("created_at", ""))
                        try:
                            created_dt = datetime.fromisoformat(created_str)
                        except Exception:
                            # Malformed created_at in a corrupt/legacy ledger row:
                            # report a present-time placeholder rather than a
                            # fabricated earlier date — clients act on status,
                            # not on this field.
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


def _register_stream_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the stream surface."""
    manager = scope.manager
    job_queue = scope.job_queue
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _uploads_dir = scope.uploads_dir

    def _cross_tenant_404(job_id: str) -> HTTPException:
        # Same body as "not found": a cross-tenant probe must not learn that the
        # job exists.
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
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

    # Bounded per-job frame history for Last-Event-ID replay (PRD §9.2).
    sse_replay = SseReplayBuffer()
    api_app.state.sse_replay = sse_replay

    def _last_event_id(request: Request) -> int | None:
        """The client's replay cursor, from the SSE header or a query fallback.

        A fresh ``EventSource`` cannot set ``Last-Event-ID`` by hand, so a
        manually-reconnecting client passes it as ``?last_event_id=``.
        """
        raw = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

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
        started. Every frame carries an ``id:`` cursor, so a client reconnecting
        with ``Last-Event-ID`` (header or ``?last_event_id=``) receives the
        frames it missed before the live stream resumes (PRD §9.2).
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        last_event_id = _last_event_id(request)
        if job_queue is not None:
            initial_job = await asyncio.get_running_loop().run_in_executor(
                _get_sse_poll_executor(), job_queue.get, valid_id
            )
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
                    for frame in sse_replay.replay(valid_id, last_event_id):
                        yield frame
                    while True:
                        if await request.is_disconnected():
                            break
                        job = await asyncio.get_running_loop().run_in_executor(
                            _get_sse_poll_executor(), job_queue.get, valid_id
                        )
                        if job is None:
                            break
                        progress_payload = {**ProgressSnapshot().to_payload(), **job.progress}
                        snapshot = {**progress_payload, "status": job.status.value}
                        if snapshot != last:
                            yield sse_replay.emit(
                                valid_id, _progress_frame(progress_payload, job.status.value)
                            )
                            last = snapshot
                        if job.status in TERMINAL_JOB_STATUSES:
                            yield sse_replay.emit(
                                valid_id,
                                _terminal_frame(
                                    job.status.value,
                                    _public_artifact(job.progress.get("output_file")),
                                    job.error,
                                ),
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
                for frame in sse_replay.replay(valid_id, last_event_id):
                    yield frame
                # A late subscriber gets one current snapshot, not a replay of
                # the raw event log: the queue-mode branch emits this exact
                # shape, and one endpoint must not answer two schemas depending
                # on the deployment mode.
                yield sse_replay.emit(
                    valid_id, _progress_frame(record.progress.model_dump(), record.status)
                )

                if record.status in TERMINAL_JOB_STATUSES and queue.empty():
                    yield sse_replay.emit(
                        valid_id,
                        _terminal_frame(
                            record.status,
                            _public_artifact(record.progress.output_file),
                            record.error,
                        ),
                    )
                    return

                while True:
                    if await request.is_disconnected():
                        break

                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        if record.status in TERMINAL_JOB_STATUSES:
                            yield sse_replay.emit(
                                valid_id,
                                _terminal_frame(
                                    record.status,
                                    _public_artifact(record.progress.output_file),
                                    record.error,
                                ),
                            )
                            break
                        continue

                    if event is None:
                        yield sse_replay.emit(
                            valid_id,
                            _terminal_frame(
                                record.status,
                                _public_artifact(record.progress.output_file),
                                record.error,
                            ),
                        )
                        break

                    yield sse_replay.emit(
                        valid_id, _progress_frame(record.progress.model_dump(), record.status)
                    )
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


def _register_artifact_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the artifact surface."""
    app_config = scope.config
    manager = scope.manager
    job_queue = scope.job_queue
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _uploads_dir = scope.uploads_dir

    def _cross_tenant_404(job_id: str) -> HTTPException:
        # Same body as "not found": a cross-tenant probe must not learn that the
        # job exists.
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
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
                status_code=404 if not known else 409,
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
                    with SQLiteJobLedger(db_path, read_only=True) as ledger:
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
                status_code=409 if job_exists else 404,
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
        "/jobs/{job_id}/deliverables",
        tags=["Jobs"],
    )
    async def list_deliverables(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """List the deliverables a finished job actually left on disk.

        A run may emit a complementary dual/mono render, an EPUB and the JSON
        sidecars depending on its flags; the UI must render only what exists
        rather than offer four fixed buttons. Keys match
        :data:`DELIVERABLE_LABELS`.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        output_file = await _primary_output_file(valid_id)
        if not output_file:
            record = manager.get_job(valid_id)
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            known = (
                record is not None
                or in_queue
                or (app_config.db_dir / f"{valid_id}.sqlite").exists()
            )
            raise HTTPException(
                status_code=409 if known else 404,
                detail=(
                    "Deliverables are not ready yet." if known else f"Job not found: {valid_id}"
                ),
            )

        items: list[dict[str, Any]] = []
        for key, candidate in _deliverable_paths(output_file).items():
            try:
                resolved = resolve_secure_path(candidate, must_exist=False, config=app_config)
            except HTTPException:
                continue
            if resolved.is_file():
                label, media_type = DELIVERABLE_LABELS[key]
                items.append(
                    {
                        "key": key,
                        "label": label,
                        "filename": resolved.name,
                        "size_bytes": resolved.stat().st_size,
                        "media_type": media_type,
                    }
                )
        return JSONResponse({"job_id": valid_id, "deliverables": items})

    @api_app.get(
        "/jobs/{job_id}/download/{key}",
        tags=["Jobs"],
    )
    async def download_deliverable(
        job_id: str, key: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> FileResponse:
        """Serve one deliverable by key (see :data:`DELIVERABLE_LABELS`)."""
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if key not in DELIVERABLE_LABELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown deliverable: {key}",
            )
        output_file = await _primary_output_file(valid_id)
        if not output_file:
            record = manager.get_job(valid_id)
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            known = (
                record is not None
                or in_queue
                or (app_config.db_dir / f"{valid_id}.sqlite").exists()
            )
            raise HTTPException(
                status_code=409 if known else 404,
                detail=(
                    "Deliverables are not ready yet." if known else f"Job not found: {valid_id}"
                ),
            )

        candidate = _deliverable_paths(output_file).get(key)
        if candidate is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deliverable not available: {key}",
            )
        resolved = resolve_secure_path(candidate, must_exist=True, config=app_config)
        if not resolved.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deliverable not available: {key}",
            )
        _, media_type = DELIVERABLE_LABELS[key]
        return FileResponse(
            path=resolved,
            filename=resolved.name,
            media_type=media_type,
        )


def _register_review_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the review surface."""
    app_config = scope.config
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _uploads_dir = scope.uploads_dir

    def _cross_tenant_404(job_id: str) -> HTTPException:
        # Same body as "not found": a cross-tenant probe must not learn that the
        # job exists.
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
        )

    # -- L3 Review Workbench (segments + fault ribbon + human edit) -----------
    @api_app.get("/jobs/{job_id}/segments", tags=["Jobs"])
    async def list_segments(
        job_id: str,
        status_filter: str | None = Query(default=None, alias="status"),
        block_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """A page of a job's finalized blocks for the workbench grid.

        ``status=issues`` returns everything the fault ribbon counts plus the
        PE-queue members; a concrete ``BlockStatus`` value filters by lifecycle
        status. Blocks are projected with their source/target, QE score, flags
        and grouped issue kinds.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        blocks = await _read_job_blocks(valid_id)
        if blocks is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        filtered = [b for b in blocks if segment_matches_filter(b, status_filter)]
        if block_type:
            filtered = [b for b in filtered if b.block_type.value == block_type]
        total = len(filtered)
        start = max(offset, 0)
        window = filtered[start : start + min(max(limit, 1), 500)]
        return JSONResponse(
            {
                "job_id": valid_id,
                "total": total,
                "segments": [serialize_segment(b) for b in window],
            }
        )

    @api_app.get("/jobs/{job_id}/issues", tags=["Jobs"])
    async def list_issues(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """Fault-ribbon counts by issue kind, plus terminology drift detail."""
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        blocks = await _read_job_blocks(valid_id)
        if blocks is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        counts: dict[str, int] = dict.fromkeys(ISSUE_KINDS, 0)
        status_counts = {"needs_human": 0, "blocked_human": 0, "failed": 0}
        for block in blocks:
            for kind in segment_issue_kinds(block):
                counts[kind] += 1
            if block.status.value in status_counts:
                status_counts[block.status.value] += 1

        term_drift: list[Any] = []
        report_file = await _artifact_path_async(valid_id, "report_file")
        if report_file:
            try:
                report_path = resolve_secure_path(report_file, must_exist=True, config=app_config)
                report_data = json.loads(
                    await asyncio.to_thread(report_path.read_text, encoding="utf-8")
                )
                term_drift = report_data.get("entity_consistency", {}).get("top_drifted", []) or []
            except (HTTPException, ValueError, OSError):
                term_drift = []

        return JSONResponse(
            {
                "job_id": valid_id,
                "counts": counts,
                "status": status_counts,
                "total_issues": sum(counts.values()),
                "term_drift": term_drift,
            }
        )

    @api_app.post("/jobs/{job_id}/segments/{block_id}", tags=["Jobs"])
    async def edit_segment(
        job_id: str,
        block_id: str,
        req: SegmentEditRequest,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """Apply one human revision (ledger + shared TM), the L3 feedback path."""
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )
        if await _job_is_running(valid_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job is running; stop it before editing segments.",
            )

        try:
            result = await asyncio.to_thread(
                apply_human_edit,
                _job_db_path(valid_id),
                valid_id,
                block_id,
                req.target_text,
                tm_path=app_config.db_dir / "tm.sqlite",
            )
        except ReviewBlockNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ReviewEditConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ReviewEditError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

        updated = None
        blocks = await _read_job_blocks(valid_id)
        if blocks is not None:
            match = next((b for b in blocks if b.id == block_id), None)
            updated = serialize_segment(match) if match is not None else None
        return JSONResponse({**result, "job_id": valid_id, "segment": updated})

    @api_app.get("/jobs/{job_id}/segments/{block_id}/terms", tags=["Jobs"])
    async def segment_terms(
        job_id: str,
        block_id: str,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """Terminology findings for one block, with the cascade size each implies.

        Each finding carries the offending ``surface``, the ``expected``
        rendering, and how many *other* blocks in the job carry the same
        error — ``cascade_all`` for the whole book, ``cascade_subsequent`` for
        blocks at or after this one. This is what the "fix all N" checkbox
        counts.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        report = await asyncio.to_thread(
            term_cascade_report, _job_db_path(valid_id), valid_id, block_id
        )
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown block: {block_id}"
            )
        return JSONResponse({"job_id": valid_id, **report})

    @api_app.post("/jobs/{job_id}/term-propagation", tags=["Jobs"])
    async def propagate_term(
        job_id: str,
        req: TermPropagationRequest,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """Replace one offending term surface across the job (PRD §5.2.2).

        Rewrites the selected block and, per ``scope``, every matching block;
        each rewritten block is promoted to a human revision and its pair fed
        back to the shared TM, exactly like a manual edit.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )
        if await _job_is_running(valid_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job is running; stop it before editing segments.",
            )

        try:
            result = await asyncio.to_thread(
                apply_term_propagation,
                _job_db_path(valid_id),
                valid_id,
                req.block_id,
                req.surface,
                req.expected,
                scope=req.scope,
                tm_path=app_config.db_dir / "tm.sqlite",
            )
        except ReviewBlockNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ReviewEditConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ReviewEditError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

        return JSONResponse({"job_id": valid_id, **result})

    @api_app.get("/jobs/{job_id}/pages/{page}/preview", tags=["Jobs"])
    async def preview_page(
        job_id: str,
        page: int,
        dpi: int = 110,
        bilingual: bool = False,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> Response:
        """Re-compose one source page with the current ledger text, as PNG.

        Reuses the delivery compositor (source page as canvas) for a single page,
        so a human edit can be previewed without re-rendering the book. 503 when
        the page has nothing to compose or the rasterizer is unavailable.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="page must be >= 1"
            )
        blocks = await _read_job_blocks(valid_id)
        if blocks is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        def _meta() -> tuple[str | None, str | None]:
            with SQLiteJobLedger(_job_db_path(valid_id), read_only=True) as ledger:
                return (
                    ledger.get_job_source_path(valid_id),
                    ledger.get_job_target_lang(valid_id),
                )

        source_path, target_lang = await asyncio.to_thread(_meta)
        if not source_path:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job has no source document to preview against.",
            )
        resolved_source = resolve_secure_path(source_path, must_exist=True, config=app_config)

        with tempfile.TemporaryDirectory(prefix="ubt_page_preview_") as tmp:
            try:
                png = await asyncio.to_thread(
                    render_page_preview,
                    source_pdf=resolved_source,
                    blocks=blocks,
                    target_lang=target_lang or "zh",
                    page=page,
                    workdir=Path(tmp),
                    bilingual=bilingual,
                    dpi=min(max(dpi, 40), 200),
                )
            except PagePreviewUnavailable as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
                ) from exc
        return Response(content=png, media_type="image/png")

    @api_app.get("/jobs/{job_id}/pages/{page}/source", tags=["Jobs"])
    async def source_page(
        job_id: str,
        page: int,
        dpi: int = 110,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> Response:
        """Rasterize the *source* PDF's page ``page`` as PNG.

        The "before" half of the L3 pixel-witness view; pairs with
        ``/pages/{page}/preview`` (the composed "after"). 503 when the source is
        missing or the rasterizer is unavailable.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, _tenant_from_header(x_ubt_tenant)):
            raise _cross_tenant_404(valid_id)
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="page must be >= 1"
            )
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        def _meta() -> str | None:
            with SQLiteJobLedger(_job_db_path(valid_id), read_only=True) as ledger:
                return ledger.get_job_source_path(valid_id)

        source_path = await asyncio.to_thread(_meta)
        if not source_path:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job has no source document to preview against.",
            )
        resolved_source = resolve_secure_path(source_path, must_exist=True, config=app_config)
        try:
            png = await asyncio.to_thread(
                render_source_page_png,
                resolved_source,
                page,
                min(max(dpi, 40), 200),
            )
        except PagePreviewUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        return Response(content=png, media_type="image/png")


def _register_asset_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the asset surface."""
    app_config = scope.config
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _uploads_dir = scope.uploads_dir

    def _cross_tenant_404(job_id: str) -> HTTPException:
        # Same body as "not found": a cross-tenant probe must not learn that the
        # job exists.
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
        )

    def _resolved_glossary_path() -> Path:
        configured = app_config.glossary_path
        if configured is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "No glossary file is configured. Set glossary_path in ubt.toml "
                    "or pass --glossary."
                ),
            )
        return resolve_secure_path(configured, must_exist=True, config=app_config)

    def _tm_path() -> Path:
        return app_config.db_dir / "tm.sqlite"

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

    @api_app.get("/assets/glossary", tags=["Assets"])
    async def get_glossary() -> JSONResponse:
        path = _resolved_glossary_path()
        try:
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"path": path.name, "terms": terms})

    @api_app.post("/assets/glossary", tags=["Assets"])
    async def upsert_glossary_term(req: GlossaryTermRequest) -> JSONResponse:
        if not req.target.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="target is required when adding a term",
            )
        path = _resolved_glossary_path()
        try:
            await asyncio.to_thread(add_glossary_term, path, req.source, req.target)
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"path": path.name, "terms": terms})

    @api_app.delete("/assets/glossary", tags=["Assets"])
    async def delete_glossary_term(source: str) -> JSONResponse:
        path = _resolved_glossary_path()
        try:
            removed = await asyncio.to_thread(remove_glossary_term, path, source)
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"path": path.name, "removed": removed, "terms": terms})

    @api_app.get("/assets/glossary/conflicts", tags=["Assets"])
    async def get_glossary_conflicts() -> JSONResponse:
        """Sources configured with more than one target rendering (PRD §4.4.2)."""
        path = _resolved_glossary_path()
        try:
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"conflicts": detect_glossary_conflicts(terms)})

    @api_app.get("/assets/tm", tags=["Assets"])
    async def get_translation_memory(
        limit: int = 200,
        offset: int = 0,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> JSONResponse:
        data = await asyncio.to_thread(
            list_tm_entries,
            _tm_path(),
            limit=min(max(limit, 1), 1000),
            offset=max(offset, 0),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return JSONResponse(data)

    @api_app.post("/assets/tm/evict", tags=["Assets"])
    async def evict_translation_memory(req: TMevictRequest) -> JSONResponse:
        removed = await asyncio.to_thread(evict_tm_entries, _tm_path(), req.ids)
        return JSONResponse({"removed": removed})

    @api_app.post("/assets/tm/import", tags=["Assets"])
    async def import_translation_memory(req: TMImportRequest) -> JSONResponse:
        """Import TMX/JSON pairs into the shared TM (machine provenance by default).

        An import is unverified, so the default ``machine`` provenance cannot
        downgrade an existing ``human_pe`` row (the store's writeback enforces
        that). Parsing is strict: a malformed payload is a 422, not a silent
        empty import.
        """
        try:
            rows = await asyncio.to_thread(parse_tm_payload, req.content, req.format)
            imported = await asyncio.to_thread(
                import_tm_entries,
                _tm_path(),
                rows,
                default_src_lang=req.src_lang,
                default_tgt_lang=req.tgt_lang,
                provenance=req.provenance,
            )
        except TMImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return JSONResponse({"parsed": len(rows), "imported": imported})


def _bootstrap_asgi_app() -> FastAPI:
    """Build the module-level app that ``uvicorn ubt.api.app:app`` imports.

    Configures logging only when this process has none yet. ``create_app`` is
    also imported as a library (tests, tooling) and must not reconfigure a
    host's root logger, so the call lives here — on the ASGI-target path — and
    is skipped once anything has already set up handlers.
    """
    if not logging.getLogger().handlers:
        setup_logging()
    return create_app()


# Module-level app is built lazily via PEP 562 ``__getattr__``. Importing this
# module (e.g. ``from ubt.api.app import create_app``) must not configure the
# host's logging or build a FastAPI app as a side effect; only an actual
# ``ubt.api.app:app`` access (uvicorn's string target, ``from ubt.api.app
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
    an API key gate: set ``UBT_API_KEY`` (``UBT_STRICT_AUTH=1`` makes the
    refusal fatal instead of a warning-and-continue). For a local,
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
