"""The system surface: ``/health``, ``/system`` (info + doctor checks)."""

from __future__ import annotations

import asyncio
import shutil
from typing import Any

from fastapi import Cookie, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from ubt import __version__
from ubt.api.models import (
    ProviderDetailResponse,
    ProviderEditRequest,
    RuntimeConfigResponse,
    SystemInfoResponse,
)
from ubt.api.routes.shared import is_nonloopback_bind
from ubt.api.scope import ApiScope
from ubt.api.security import (
    SESSION_COOKIE_NAME,
    effective_allowed_bases,
    session_cookie_value,
    verify_api_key,
)
from ubt.core.fs_perms import sanitize_display_path


def register_system_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the system surface."""
    app_config = scope.config

    @api_app.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "universal-book-translator",
            "version": __version__,
        }

    @api_app.post("/system/session", tags=["System"])
    async def create_session(
        response: Response,
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, bool]:
        """Exchange the API key for a same-origin session cookie.

        The console's ``EventSource`` stream, page-preview ``<img>`` elements
        and download links cannot attach an ``X-API-Key`` header, so on a keyed
        server the UI would 401 on every one of them even though its fetches
        pass. The cookie carries a keyed *digest* of the service key (never the
        key itself), is HttpOnly + SameSite=Lax so it rides only same-origin
        browser requests, and is re-minted on every login. An open (keyless)
        server has nothing to exchange and returns ``authenticated: true``
        without setting a cookie.
        """
        verify_api_key(x_api_key, ubt_session=None, _config=app_config)
        expected = app_config.service_api_key.get_secret_value().strip()
        if not expected:
            return {"authenticated": True}
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_cookie_value(expected),
            httponly=True,
            samesite="lax",
            secure=False,
        )
        return {"authenticated": True}

    @api_app.delete("/system/session", tags=["System"])
    async def delete_session(response: Response) -> dict[str, bool]:
        """Log the console out (clears the session cookie)."""
        response.delete_cookie(SESSION_COOKIE_NAME)
        return {"authenticated": False}

    @api_app.get(
        "/system/runtime-config",
        response_model=RuntimeConfigResponse,
        tags=["System"],
    )
    async def system_runtime_config() -> RuntimeConfigResponse:
        """Model/provider configuration actually in force (read-only).

        The console used to render ``/model-profiles`` here, which is the
        engine's capability *preset* table — fourteen hard-coded name patterns,
        none of which is a configured model. This endpoint reports what the
        engine will really call, which preset (if any) that model matches, and
        the declared providers with credentials filtered out.
        """
        from ubt.api.runtime_config import describe_runtime_config

        return RuntimeConfigResponse(**describe_runtime_config(app_config))

    @api_app.get(
        "/system/providers/{name}",
        response_model=ProviderDetailResponse,
        tags=["System"],
    )
    async def get_provider(name: str) -> ProviderDetailResponse:
        """One provider block, with the credential reduced to a boolean."""
        from ubt.api.provider_editor import ProviderEditError, read_provider

        try:
            detail = read_provider(name)
        except ProviderEditError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return ProviderDetailResponse(**detail)

    @api_app.put(
        "/system/providers/{name}",
        response_model=ProviderDetailResponse,
        tags=["System"],
    )
    async def put_provider(
        name: str,
        request: ProviderEditRequest,
        x_api_key: str | None = Header(default=None),
        ubt_session: str | None = Cookie(default=None),
    ) -> ProviderDetailResponse:
        """Create or update a ``[providers.<name>]`` block.

        Writes ``ubt.toml``, which the engine re-reads on every job start, so the
        change applies to the next submitted job without restarting the server.

        **Credentials are not writable here, and that is deliberate.** The request
        model forbids unknown fields, so an attempt to set one is a 422 that names
        the field rather than a silently dropped key. A credential arrives through
        the provider block in the file, ``${VAR}`` expansion, or
        ``UBT_LLM_API_KEY`` — never over HTTP. See
        :mod:`ubt.api.provider_editor` for why that boundary is not negotiable:
        this server is routinely started on loopback without an API key.
        """
        from ubt.api.provider_editor import ProviderEditError, upsert_provider

        # An explicit gate even though the global middleware already ran: this is
        # the only route that writes a file the engine then trusts, so it must not
        # inherit a future relaxation of the middleware.
        verify_api_key(x_api_key, ubt_session, _config=app_config)
        # ``exclude_unset``, not ``exclude_none``: an omitted field means "leave
        # it alone", while an explicit ``null`` on a tri-state capability means
        # "remove the declaration and go back to the safe default". Dropping both
        # would make a mis-declared flag impossible to clear from the console.
        changes = request.model_dump(exclude_unset=True)
        try:
            detail = await asyncio.to_thread(upsert_provider, name, changes)
        except ProviderEditError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return ProviderDetailResponse(**detail)

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

        # The *bind* address, not the client-supplied ``Host`` header: a
        # ``Host: evil.example.com`` (DNS rebinding) used to read as
        # ``is_loopback=False``, and the reverse — ``Host: localhost`` against an
        # exposed bind — read as loopback. ``request.url`` is built from the
        # header, so it can never answer "is this service loopback-only"; the
        # socket's server address can. A hostname bind resolves before accept, so
        # the scope carries the resolved IP.
        server = request.scope.get("server")
        bind_host = str(server[0]) if server else ""
        display_host = request.url.hostname or bind_host
        disk_free_gb: float | None = None
        try:
            usage = shutil.disk_usage(app_config.db_dir)
            disk_free_gb = round(usage.free / (1024**3), 1)
        except OSError:
            pass

        return {
            "version": __version__,
            "host": display_host,
            # The client-visible port, so the sidebar can show where the console
            # is actually reachable instead of a hardcoded ``:8000``.
            "port": request.url.port,
            "is_loopback": bool(bind_host) and not is_nonloopback_bind(bind_host),
            "auth_enabled": bool(app_config.service_api_key.get_secret_value().strip()),
            "allowed_bases": [
                sanitize_display_path(path) for path in effective_allowed_bases(app_config)
            ],
            "db_dir": sanitize_display_path(app_config.db_dir),
            "job_mode": app_config.job_mode,
            "disk_free_gb": disk_free_gb,
            "wal_status": "ONLINE (WAL Mode Active)",
        }
