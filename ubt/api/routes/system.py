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
    QESettingsRequest,
    QESettingsResponse,
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
    """Routes for the system surface.

    Two configuration views, deliberately distinct:

    * ``app_config`` — the boot-time snapshot. Every deployment-bound field
      (ledger path, auth gate, job mode, allowed bases) was decided from it, so it
      is the only correct source for those.
    * ``live_config()`` — the configuration in force right now, re-read when the
      file changes. Anything that reports *model* configuration must use it, or a
      console save would be invisible until a restart.

    Using the snapshot for model configuration is what made a successful save
    leave the panel showing the pre-save value.
    """
    app_config = scope.config

    def live_config() -> Any:
        """Current model configuration (re-reads the file when it changed)."""
        return scope.config

    def _bind_is_loopback(request: Request) -> bool:
        """Whether this server is bound to a loopback address.

        Read from the socket's own server address, not from a header: a client can
        forge ``Host`` and ``X-Forwarded-*``, and this answer decides whether a
        credential may be written. ``testserver`` (the in-process test client) has
        no server address; it is treated as loopback because it never touches a
        network, which is what the console tests exercise.
        """
        server = request.scope.get("server")
        if not server:
            return True
        return not is_nonloopback_bind(str(server[0]))

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
    async def system_runtime_config(http_request: Request) -> RuntimeConfigResponse:
        """Model/provider configuration actually in force (read-only).

        The console used to render ``/model-profiles`` here, which is the
        engine's capability *preset* table — fourteen hard-coded name patterns,
        none of which is a configured model. This endpoint reports what the
        engine will really call, which preset (if any) that model matches, and
        the declared providers with credentials filtered out.
        """
        from ubt.api.provider_editor import credential_write_allowed
        from ubt.api.runtime_config import describe_runtime_config

        live = live_config()
        return RuntimeConfigResponse(
            **describe_runtime_config(
                live,
                can_write_credential=credential_write_allowed(
                    is_loopback=_bind_is_loopback(http_request),
                    is_strict=app_config.is_strict_auth(),
                ),
                frozen_fields_changed=(
                    scope.source.frozen_fields_changed() if scope.source is not None else []
                ),
            )
        )

    @api_app.get(
        "/system/qe-config",
        response_model=QESettingsResponse,
        tags=["System"],
    )
    async def system_qe_config() -> QESettingsResponse:
        """The quality-estimation gate actually in force.

        ``qe_engine`` and ``qe_judge_enabled`` are two switches, and only the
        second one scores anything — ``qe_engine = "tiered"`` on its own is a
        no-op the pipeline merely warns about. Reporting them separately is what
        lets the console say which one is off instead of a bare "QE not measured".
        """
        from ubt.api.qe_settings import describe_qe_settings

        return QESettingsResponse(**describe_qe_settings(live_config()))

    @api_app.put(
        "/system/qe-config",
        response_model=QESettingsResponse,
        tags=["System"],
    )
    async def put_qe_config(
        request: QESettingsRequest,
        x_api_key: str | None = Header(default=None),
        ubt_session: str | None = Cookie(default=None),
    ) -> QESettingsResponse:
        """Write the QE gate into ``ubt.toml [defaults]``.

        The engine re-reads the file per job, so the change applies to the next
        submitted job without a restart — which is the only useful behaviour here,
        since a finished job is never rescored.

        **The pair is written together by the console.** Selecting "tiered" sends
        both ``qe_engine = "tiered"`` and ``qe_judge_enabled = true``; selecting
        "heuristic" sends both back. Exposing the keys independently would let an
        operator save the documented-but-useless ``tiered`` + judge-off state.

        An explicit gate check even though the middleware already ran: this route
        writes a file the engine then trusts, so it must not inherit a future
        relaxation of the global filter (same rule as ``put_provider``).
        """
        from ubt.api.provider_editor import ProviderEditError, update_qe_settings
        from ubt.api.qe_settings import describe_qe_settings

        verify_api_key(x_api_key, ubt_session, _config=app_config)
        # ``exclude_unset``, not ``exclude_none``: an omitted key means "leave it
        # alone", while an explicit ``null`` clears it back to "not declared".
        changes = request.model_dump(exclude_unset=True)
        try:
            await asyncio.to_thread(update_qe_settings, changes)
        except ProviderEditError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Make the save visible to the very next request: the mtime check in
        # ConfigSource would catch it, but not within the same clock tick.
        if scope.source is not None:
            scope.source.invalidate()
        return QESettingsResponse(**describe_qe_settings(live_config()))

    @api_app.get(
        "/system/providers/{name}",
        response_model=ProviderDetailResponse,
        tags=["System"],
    )
    async def get_provider(name: str) -> ProviderDetailResponse:
        """One provider block, with the credential reduced to a mask."""
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
        http_request: Request,
        x_api_key: str | None = Header(default=None),
        ubt_session: str | None = Cookie(default=None),
    ) -> ProviderDetailResponse:
        """Create or update a ``[providers.<name>]`` block.

        Writes ``ubt.toml``, which the engine re-reads per job, so the change
        applies to the next submitted job without restarting the server.

        **A credential is accepted only on a local, non-strict deployment.** See
        :func:`ubt.api.provider_editor.credential_write_allowed` for the reasoning:
        on loopback the caller could already read the file, so refusing the write
        protected nothing while leaving a fresh install with no way to configure a
        model. On a non-loopback bind or under strict auth the request is refused
        and the message says where the secret belongs instead.
        """
        from ubt.api.provider_editor import (
            ProviderEditError,
            credential_write_allowed,
            upsert_provider,
        )

        # An explicit gate even though the global middleware already ran: this is
        # the only route that writes a file the engine then trusts, so it must not
        # inherit a future relaxation of the middleware.
        verify_api_key(x_api_key, ubt_session, _config=app_config)
        allow_credential = credential_write_allowed(
            is_loopback=_bind_is_loopback(http_request),
            is_strict=app_config.is_strict_auth(),
        )
        # ``exclude_unset``, not ``exclude_none``: an omitted field means "leave
        # it alone", while an explicit ``null`` on a tri-state capability means
        # "remove the declaration and go back to the safe default". Dropping both
        # would make a mis-declared flag impossible to clear from the console.
        changes = request.model_dump(exclude_unset=True)
        # ``make_default`` is a control flag, not a field of the block: it selects
        # which provider runs. Stripped here rather than written into the TOML,
        # where the loader would reject it as an unknown key.
        make_default = bool(changes.pop("make_default", False))
        try:
            detail = await asyncio.to_thread(
                upsert_provider,
                name,
                changes,
                allow_credential=allow_credential,
                make_default=make_default,
            )
        except ProviderEditError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Make the save visible to the very next request. The mtime check in
        # ConfigSource would catch it anyway, but not within the same clock tick.
        if scope.source is not None:
            scope.source.invalidate()
        return ProviderDetailResponse(**detail)

    @api_app.put(
        "/system/providers/{name}/default",
        tags=["System"],
    )
    async def set_default_provider_route(
        name: str,
        http_request: Request,
        x_api_key: str | None = Header(default=None),
        ubt_session: str | None = Cookie(default=None),
    ) -> dict[str, str]:
        """Make ``name`` the provider a run uses when nothing more explicit does.

        Writes ``[defaults] provider``. Creating a block without this left a fresh
        install unable to run: the block existed, the console listed it, and
        ``can_translate`` stayed false because nothing named it.

        A *selection*, not an edit — the request carries no body, so it works for a
        block the operator did not otherwise change.
        """
        from ubt.api.provider_editor import ProviderEditError, set_default_provider

        verify_api_key(x_api_key, ubt_session, _config=app_config)
        try:
            result = await asyncio.to_thread(set_default_provider, name)
        except ProviderEditError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if scope.source is not None:
            scope.source.invalidate()
        return result

    @api_app.get("/system/doctor", tags=["System"])
    async def system_doctor(probe: bool = False) -> JSONResponse:
        """Engine Doctor self-check (the ``ubt doctor`` checklist as JSON).

        Reuses the CLI's ``collect_checks`` so the console and the command line
        report the same verdicts. ``probe=true`` additionally contacts the
        provider endpoint (a live network call), so it is off by default.

        Reads the **live** config, not the boot snapshot. ``app_config`` is built
        once at startup, so a provider saved from the console an instant earlier
        was invisible here: the form said "saved", the panel below it reported the
        new provider as active, and this checklist still said ``API key FAIL`` and
        ``no draft model configured`` — two contradictory statements on one screen.
        """
        from ubt.cli.commands.doctor import _overall_status, _summary, collect_checks

        checks = await asyncio.to_thread(collect_checks, live_config(), probe=probe)
        summary = _summary(checks)
        return JSONResponse(
            {
                "status": _overall_status(summary),
                "summary": summary,
                # Each check ships its English copy AND the stable codes plus
                # interpolation values a localized console needs. A client that
                # knows the code renders its own sentence; one that does not falls
                # back to the English text rather than printing a raw key. Same
                # contract as ``AssessmentWarning`` in ubt.core.assess.
                "checks": [
                    {
                        "group": check.group,
                        "group_code": check.group_code,
                        "name": check.name,
                        "code": check.code,
                        "status": check.status,
                        "detail": check.detail,
                        "detail_code": check.detail_code,
                        "fix": check.fix,
                        "fix_code": check.fix_code,
                        "params": check.params or {},
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
