"""Security guards, API-key verification, and path isolation for the UBT REST API."""

import logging
import os
import secrets
import sys
from pathlib import Path

from fastapi import Header, HTTPException, status

from ubt.core.config import UBTConfig
from ubt.core.fs_perms import SENSITIVE_FILENAME_PARTS as SENSITIVE_FILENAME_PARTS
from ubt.core.fs_perms import SYSTEM_DISALLOWED_PREFIXES as SYSTEM_DISALLOWED_PREFIXES
from ubt.core.fs_perms import is_sensitive_path_part
from ubt.core.job_options import JOB_ID_MAX_LEN, JOB_ID_RE, job_id_is_valid

logger = logging.getLogger(__name__)


def _log_startup_auth_warning(config: UBTConfig | None = None) -> None:
    """Warn loudly when the service boots without an API key.

    Authentication stays opt-in by design (development convenience), but an
    unauthenticated boot must never be silent: the operator gets a clear
    statement of the exposure and how to close it.
    """
    cfg = config or UBTConfig()
    has_key = bool(cfg.service_api_key.get_secret_value().strip())
    if has_key or cfg.is_strict_auth():
        return
    logger.warning(
        "UBT API is starting WITHOUT authentication (UBT_API_KEY is not set): "
        "anyone who can reach this port can submit translation jobs and read "
        "files under the configured data directory. Set UBT_API_KEY to enable "
        "the X-API-Key gate, or keep the service bound to localhost / behind "
        "a trusted reverse proxy."
    )


#: Explicit opt-out for a throwaway local server. The old "open by default"
#: behaviour is now a deliberate, named decision rather than a silent default.
_NO_AUTH_OVERRIDE_ENV = "UBT_ALLOW_NO_AUTH"


def _no_auth_allowed() -> bool:
    return os.getenv(_NO_AUTH_OVERRIDE_ENV, "").strip().lower() in ("1", "true", "yes")


def _require_api_key_gate(config: UBTConfig | None = None) -> None:
    """Refuse to boot the API without an auth gate unless explicitly overridden.

    Default-secure now: the REST surface can read any parseable text under its
    sandbox, so it must not start unauthenticated by accident. ``UBT_API_KEY``
    (or strict mode) enables the gate; ``UBT_ALLOW_NO_AUTH=1`` is the named
    escape hatch for a local, throwaway server.
    """
    cfg = config or UBTConfig.from_env()
    has_service_key = bool(cfg.service_api_key.get_secret_value().strip())
    if has_service_key:
        return
    if _no_auth_allowed():
        logger.warning(
            "%s is set: booting the UBT API WITHOUT an API key (local/development only).",
            _NO_AUTH_OVERRIDE_ENV,
        )
        return
    if cfg.is_strict_auth():
        # Strict mode with no key would boot a server that 500s every request
        # (and exposes /docs): fail fast instead of starting unusable.
        raise SystemExit(
            "UBT_STRICT_AUTH is set but UBT_API_KEY is empty: set a key, or "
            "UBT_ALLOW_NO_AUTH=1 for a local, throwaway server."
        )
    raise SystemExit(
        "refusing to start the UBT API without authentication: set UBT_API_KEY to "
        "enable the X-API-Key gate, or set UBT_ALLOW_NO_AUTH=1 to run open on "
        "localhost (development only)."
    )


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    _config: UBTConfig | None = None,
) -> None:
    """Lightweight API-key gate for the service.

    Authentication is opt-in: it is only enforced when ``UBT_API_KEY`` is set.
    When unset the server runs open (development mode). In strict mode
    (``UBT_STRICT_AUTH=1`` or ``UBT_ENV=production``), a configured
    ``UBT_API_KEY`` is mandatory. Pass the key via the ``X-API-Key`` header
    (never query string).
    """
    cfg = _config or UBTConfig()
    strict = cfg.is_strict_auth()
    raw_expected = cfg.service_api_key.get_secret_value()
    expected = raw_expected.strip()
    # A key that is set but blank/whitespace-only is a misconfiguration, not
    # "auth disabled": silently running open would drop the gate the operator
    # believes is on.
    if raw_expected and not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="UBT_API_KEY is set but blank/whitespace-only; refusing to run open.",
        )
    # Empty = open (development mode); strict mode requires an explicit key.
    if not expected:
        if strict:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="UBT_API_KEY must be configured when running in strict/production mode.",
            )
        return
    provided = x_api_key or ""
    # Compare over bytes — str.compare_digest raises TypeError on
    # non-ASCII (Latin-1 header bytes like "café"), turning a should-be 401
    # into an unhandled 500.
    if not secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# The sensitive-name deny list itself lives in ``ubt.core.fs_perms`` and is
# re-exported above: the MCP tools run the same check and must not import this
# package (``ubt.api``'s ``__init__`` builds the FastAPI app on import). The
# list is casefolded at comparison time — see ``resolve_secure_path``.


def validate_job_id(job_id: str) -> str:
    """Validate job_id format to prevent directory traversal or database path manipulation."""
    clean_id = str(job_id).strip()
    if not job_id_is_valid(clean_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid job_id format: '{job_id}'. Only alphanumeric characters, "
                f"hyphens, and underscores are permitted, up to {JOB_ID_MAX_LEN} characters."
            ),
        )
    return clean_id


def _tenant_from_header(raw: str | None) -> str:
    """Resolve the queue tenant from ``X-UBT-Tenant`` (single-tenant by default).

    Enterprise/on-prem deployments leave it unset (every job is ``default``);
    a SaaS gateway may forward an authenticated tenant id in the header. This is
    an isolation boundary against tenant starvation, not access control: the API
    key grants access to the whole service.
    """
    if raw is None:
        return "default"
    cleaned = raw.strip()
    if not cleaned:
        return "default"
    if not JOB_ID_RE.fullmatch(cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tenant header format: {raw!r}",
        )
    return cleaned


def effective_allowed_bases(config: UBTConfig | None = None) -> list[Path]:
    """The base directories :func:`resolve_secure_path` will actually enforce.

    Mirrors its resolution order: the operator allowlist when configured, else
    the secure default of the working directory plus the ledger dir. A caller
    that derives a *server-side* default path (the API's default deliverable)
    needs the same answer so it can place that path inside the sandbox before
    validation. Deriving the default from the user's Documents folder instead
    made every allowlist-less deployment 403 on a submit that omitted
    ``output_path``.
    """
    cfg = config or UBTConfig()
    configured = cfg.allowed_base_dirs()
    if configured:
        return configured
    return [Path.cwd().resolve(), cfg.db_dir.resolve()]


def resolve_secure_path(
    raw_path: str | Path,
    base_dir: Path | None = None,
    must_exist: bool = True,
    allowed_bases: list[Path] | None = None,
    config: UBTConfig | None = None,
) -> Path:
    """Validate and resolve path against path traversal attacks and sandbox restrictions.

    Check order is deliberate, and the two rules below read like a
    contradiction unless they are stated together:

    1. A literal ``..`` is always refused.
    2. The path must be contained in the effective allowlist
       (``UBT_ALLOWED_DIRS``, else the working directory + ``config.db_dir``).
    3. The hard-coded *system* deny list is skipped for a path an operator
       explicitly allowlisted — otherwise the conventional ``/var/lib/ubt``
       deployment could never be served.
    4. The *sensitive-name* deny list (:data:`SENSITIVE_FILENAME_PARTS`) still
       applies to every path, allowlist included. An allowlist is a scope
       decision ("these books may be read"), not a licence to serve the
       ``.ssh``/``.env``/``credentials.json`` that happen to live beside them —
       and these routes are reachable without credentials by default. Refusing
       a whitelisted secret file on purpose, not by bug.

    Component matching is casefolded: ``.SSH`` and ``.ssh`` are the same
    directory on a case-insensitive filesystem.
    """
    path_str = str(raw_path).strip()
    if not path_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty path provided",
        )

    # Reject obvious directory traversal attempts
    parts = Path(path_str).parts
    if ".." in parts:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Directory traversal ('..') is prohibited in path: '{raw_path}'",
        )

    try:
        p = Path(path_str)
        effective_bases: list[Path] = []
        # An operator-configured base (UBT_ALLOWED_DIRS, an explicit caller
        # base) is authoritative for paths inside it — for the containment rule
        # and for the hard-coded system deny list below, which is the fallback
        # sandbox of the no-whitelist posture. The sensitive-name deny list is
        # the one rule it does NOT override (docstring point 4).
        explicit_whitelist = False
        if base_dir:
            effective_bases.append(base_dir.resolve())
            explicit_whitelist = True
        elif allowed_bases is not None:
            effective_bases.extend(allowed_bases)
            explicit_whitelist = True
        else:
            cfg = config or UBTConfig()
            if cfg.allowed_base_dirs():
                explicit_whitelist = True
            # Secure default: confine to the working directory and the
            # configured ledger dir. The previous empty default disabled
            # the containment check entirely, so a default deployment could
            # read ~/.gitconfig and write arbitrary user files. Operators
            # widen this with UBT_ALLOWED_DIRS.
            effective_bases.extend(effective_allowed_bases(cfg))

        if not p.is_absolute():
            anchor = effective_bases[0] if effective_bases else Path.cwd().resolve()
            resolved = (anchor / p).resolve()
        else:
            resolved = p.resolve()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid path format: {err}",
        ) from err

    contained = any(resolved == b or b in resolved.parents for b in effective_bases)

    # Reject attempts to read critical host system directories. A path the
    # operator explicitly whitelisted is exempt: otherwise the conventional
    # data location (/var/lib/ubt/...) could never be served, so a whitelist
    # that satisfies the non-loopback startup guardrail stayed unusable.
    app_mod = sys.modules.get("ubt.api.app")
    disallowed_prefixes = (
        getattr(app_mod, "SYSTEM_DISALLOWED_PREFIXES", SYSTEM_DISALLOWED_PREFIXES)
        if app_mod is not None
        else SYSTEM_DISALLOWED_PREFIXES
    )

    if not (explicit_whitelist and contained):
        for disallowed in disallowed_prefixes:
            if resolved == disallowed or disallowed in resolved.parents:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Path '{raw_path}' accesses restricted system directory '{disallowed}'",
                )

    # Reject attempts to read user secrets, shell configurations, or
    # credentials. Applied *after* the allowlist and to whitelisted paths too —
    # that precedence is intentional; see point 4 of the docstring above.
    for part in resolved.parts:
        if is_sensitive_path_part(part):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Accessing sensitive configuration directory or file '{part}' is prohibited",
            )

    if effective_bases and not contained:
        # Generic detail: the allowed bases are server-side paths and this route
        # is reachable without credentials by default (same sanitization as the
        # job-failure path), so they must not be echoed to the caller.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: path is outside the configured allowed directories",
        )

    if must_exist and not resolved.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source file not found: {raw_path}",
        )

    return resolved
