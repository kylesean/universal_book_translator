"""FastAPI microservice exposing REST and SSE streaming endpoints for UBT."""

import argparse
import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Header,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse

from ubt import __version__
from ubt.api.config_source import ConfigSource
from ubt.api.manager import JobManager, JobRecord
from ubt.api.models import (
    JobAssessRequest,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    JobUploadResponse,
)
from ubt.api.routes import (
    register_artifact_routes,
    register_asset_routes,
    register_job_routes,
    register_review_routes,
    register_stream_routes,
    register_system_routes,
)
from ubt.api.routes.shared import (
    DELIVERABLE_LABELS,
    deliverable_paths,
    is_nonloopback_bind,
    is_own_prior_output,
    output_path_claimant,
)
from ubt.api.scope import ApiScope
from ubt.api.security import (
    SENSITIVE_FILENAME_PARTS,
    SYSTEM_DISALLOWED_PREFIXES,
    _log_startup_auth_warning,
    _require_api_key_gate,
    resolve_secure_path,
    validate_job_id,
    verify_api_key,
)
from ubt.api.sse_broadcaster import QueueSseBroadcaster
from ubt.api.uploads import UPLOAD_MAX_BYTES, UPLOAD_PATH_SUFFIX, reap_stale_uploads
from ubt.core.config import UBTConfig
from ubt.core.engine.job_queue import JobQueue
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.job_options import (
    JOB_ID_RE,
    LANG_CODE_PATTERN,
    apply_config_overrides,
    clean_source_stem,
    default_output_path,
    overrides_from_request,
    run_kwargs_from_request,
)
from ubt.core.log_config import setup_logging
from ubt.core.qe import BaseQERunner
from ubt.core.router.rate_limiter import build_rate_limiter
from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)


# Backwards-compatible names: the split moved the shared helpers to
# ubt.api.routes.shared, and tests (plus the console contract) import them from
# here. Kept as explicit aliases, not a star re-export.
_deliverable_paths = deliverable_paths
_is_own_prior_output = is_own_prior_output
_output_path_claimant = output_path_claimant

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
    "UPLOAD_MAX_BYTES",
    "UploadSizeLimitMiddleware",
    "DELIVERABLE_LABELS",
    "_deliverable_paths",
    "_is_own_prior_output",
    "_output_path_claimant",
]


#: Local-only bind addresses. Anything else exposes the service to the network
#: and therefore requires the isolation guards below.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})

#: Host-header names accepted without an explicit ``UBT_TRUSTED_HOSTS`` entry.
#: ``testserver`` is what Starlette's ``TestClient`` sends; loopback names are
#: the local console. An operator on a real domain adds it to the config.
_DEFAULT_TRUSTED_HOSTS = _LOOPBACK_HOSTS | {"testserver"}


def _trusted_host_names(cfg: UBTConfig) -> frozenset[str]:
    """The Host-header names this server answers for (lowercased).

    ``UBT_TRUSTED_HOSTS`` (comma- or os.pathsep-separated) extends the loopback
    defaults, so a self-hosted console on a LAN name or a real domain is reachable
    without disabling the DNS-rebinding guard.
    """
    extra = {
        token.strip().lower()
        for token in re.split(r"[,\s]+", cfg.trusted_hosts or "")
        if token.strip()
    }
    return frozenset(_DEFAULT_TRUSTED_HOSTS | extra)


class UploadSizeLimitMiddleware:
    """Reject an oversized upload *before* the body is spooled to disk.

    The handler's own byte counter only sees the body after Starlette has
    already parsed the multipart form and written every part to a SpooledTemporaryFile
    — so an unauthenticated 32 MiB POST wrote 33,554,432 bytes to disk before the
    401, and a 10 GiB body would fill the disk the ledger lives on. This ASGI
    middleware checks the declared ``Content-Length`` up front and refuses with
    413 without reading a byte; a chunked body (no Content-Length) is counted as
    it streams and refused the moment the cap is crossed.

    Pure ASGI, not ``BaseHTTPMiddleware``: the latter buffers the body through an
    anyio stream, which is exactly the disk write this exists to prevent.
    """

    def __init__(self, app: Any, *, max_bytes: int, path_suffix: str) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_suffix = path_suffix

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope.get("path", "").endswith(self.path_suffix):
            await self.app(scope, receive, send)
            return

        declared = 0
        for key, value in scope.get("headers", []):
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                break
        if declared > self.max_bytes:
            await _send_upload_too_large(send, self.max_bytes)
            return

        # No (or a lying) Content-Length: count the streamed bytes and refuse the
        # moment the cap is crossed.
        #
        # Refusing means *raising out of receive*, not flagging the read. The
        # previous version set a flag and kept handing chunks to the app, so a
        # chunked 200 MB body was read to the end and only then answered with a
        # 413 — the guard cost nothing and prevented nothing. Raising propagates
        # out of the app's body iteration (Starlette's form parser does not catch
        # arbitrary exceptions), so reading actually stops.
        received = 0
        forwarded_start = False

        class _TooLarge(Exception):
            """Internal: unwinds the app so the 413 can be sent."""

        async def _counting_receive() -> Any:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge
            return message

        async def _guarded_send(message: Any) -> None:
            # The 413 is the answer once the cap was crossed; the handler's own
            # (now meaningless) response is suppressed. ``forwarded_start`` tracks
            # what was actually sent, not what the handler tried to send: a
            # handler that swallows the exception and then answers would
            # otherwise leave the client with no response at all.
            nonlocal forwarded_start
            if received > self.max_bytes:
                return
            if message["type"] == "http.response.start":
                forwarded_start = True
            await send(message)

        with suppress(_TooLarge):
            await self.app(scope, _counting_receive, _guarded_send)
        if received > self.max_bytes and not forwarded_start:
            await _send_upload_too_large(send, self.max_bytes)


async def _send_upload_too_large(send: Any, max_bytes: int) -> None:
    body = json.dumps(
        {"detail": f"Uploaded file exceeds the {max_bytes // (1024 * 1024)} MB limit"}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status.HTTP_413_CONTENT_TOO_LARGE,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


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


def create_app(
    config: UBTConfig | None = None,
    router: ModelRouter | None = None,
    qe_runner: BaseQERunner | None = None,
    queue: JobQueue | None = None,
    config_overrides: dict[str, Any] | None = None,
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

    # Queue mode: one background poller fans snapshots out to every subscriber,
    # so N open streams no longer occupy N polling threads. None in embedded
    # mode, where the manager pushes events directly.
    sse_broadcaster = QueueSseBroadcaster(job_queue) if job_queue is not None else None

    # Deployment identity is fixed at boot (ledger, auth gate, job mode); model
    # configuration is not. The source re-reads the file when it changes, so a
    # console save is visible to the next request instead of only to the next
    # process. Without it, a save wrote the file successfully and the panel kept
    # showing the pre-save value forever.
    config_source = ConfigSource(app_config, overrides=dict(config_overrides or {}))
    scope = ApiScope(
        config=app_config,
        manager=manager,
        job_queue=job_queue,
        assess_semaphore=assess_semaphore,
        sse_broadcaster=sse_broadcaster,
        source=config_source,
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        _require_api_key_gate(app_config)
        _log_startup_auth_warning(app_config)
        # Startup is the one moment the staging area can be swept without
        # racing the uploads of this process: no request has been served yet.
        # (A file a still-unfinished queue job lists as its input survives --
        # see ubt.api.uploads.)
        removed = await asyncio.to_thread(
            reap_stale_uploads,
            scope.uploads_dir(),
            max_age_days=app_config.upload_retention_days,
            job_queue=job_queue,
        )
        if removed:
            logger.info(
                "Removed %d staged upload(s) older than %g days",
                removed,
                app_config.upload_retention_days,
            )
        yield
        # Stop the SSE fan-out poller so it does not outlive the app.
        if sse_broadcaster is not None:
            await sse_broadcaster.close()

    async def _verify_request_key(
        request: Request,
        x_api_key: str | None = Header(default=None),
        ubt_session: str | None = Cookie(default=None),
    ) -> None:
        if request.url.path.rstrip("/") == "/health":
            return
        verify_api_key(x_api_key, ubt_session=ubt_session, _config=app_config)

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
        if host and is_nonloopback_bind(host):
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

    # Reject an oversized upload before Starlette spools the body to disk (and
    # before the auth gate runs), so an unauthenticated flood cannot write
    # hundreds of megabytes — see UploadSizeLimitMiddleware.
    api_app.add_middleware(
        UploadSizeLimitMiddleware, max_bytes=UPLOAD_MAX_BYTES, path_suffix=UPLOAD_PATH_SUFFIX
    )

    # DNS-rebinding / Host-header defense: a browser tricked into resolving an
    # attacker domain to 127.0.0.1 reaches the loopback console with
    # ``Host: evil.example.com``. Refuse any Host outside the trusted set.
    @api_app.middleware("http")
    async def _enforce_trusted_host(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        host_header = request.headers.get("host", "")
        # Strip a port; a bare IPv6 literal arrives as ``[::1]:8000``.
        hostname = host_header.rsplit(":", 1)[0].strip("[]").lower() if host_header else ""
        if hostname and hostname not in _trusted_host_names(app_config):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": (
                        f"Host {host_header!r} is not trusted. Set UBT_TRUSTED_HOSTS to "
                        "the hostname(s) this server serves, or reach it via loopback."
                    )
                },
            )
        return await call_next(request)

    register_system_routes(api_app, scope)
    register_job_routes(api_app, scope, router)
    register_stream_routes(api_app, scope)
    register_artifact_routes(api_app, scope)
    register_review_routes(api_app, scope)
    register_asset_routes(api_app, scope)
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
    throwaway server, ``UBT_ALLOW_NO_AUTH=1`` opts into running open.
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
