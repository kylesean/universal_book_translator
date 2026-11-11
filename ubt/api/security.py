"""Security guards, API-key verification, and path isolation for the UBT REST API.

The three deployment postures these guards implement (local / shared / strict)
are documented in the README's "部署姿势与信任模型" table; every behaviour
below is enforced there, not advisory.
"""

import hashlib
import hmac
import logging
import os
import secrets
from pathlib import Path

from fastapi import Cookie, Header, HTTPException, status

from ubt.core.config import UBTConfig
from ubt.core.fs_perms import SENSITIVE_FILENAME_PARTS as SENSITIVE_FILENAME_PARTS
from ubt.core.fs_perms import SYSTEM_DISALLOWED_PREFIXES as SYSTEM_DISALLOWED_PREFIXES
from ubt.core.fs_perms import ensure_private_dir, is_sensitive_path_part, restrict_file_to_owner
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


#: Explicit opt-out for a throwaway local server: a deliberate, named decision
#: rather than a silent open-by-default.
_NO_AUTH_OVERRIDE_ENV = "UBT_ALLOW_NO_AUTH"


def warn_broad_sandbox(config: UBTConfig | None = None) -> None:
    """Warn at startup when the path sandbox is wide enough to defeat the deny lists.

    Two postures look configured but leave the host exposed, and both must be
    named at boot rather than discovered after an incident:

    * a sandbox base that *is* the user's home directory (or contains it) puts
      every credential-bearing dotfile (``.ssh``, ``.aws``, ``.env``,
      ``.netrc``, …) inside the reach of the submit / MCP read surfaces — the
      sensitive-name deny list refuses the *known* names, but a home-wide
      sandbox is exactly the "read anything I can read" scope the sandbox was
      meant to shrink;
    * an ``UBT_ALLOWED_DIRS`` entry that equals or spans a system prefix
      (``/``, ``/etc``, ``/var``, …) switches off the system-directory deny
      list for everything under it — :func:`resolve_secure_path` deliberately
      lets an explicit allowlist win over that list, so a base that wide hands
      the exemption to the whole tree.

    The warning names the exposure and the knob; it never raises and never
    refuses the boot: the operator may legitimately want a wide sandbox, and a
    configuration that cannot start is a configuration that gets disabled.
    """
    cfg = config or UBTConfig()
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None

    for base in effective_allowed_bases(cfg):
        if home is not None and (base == home or home.is_relative_to(base)):
            logger.warning(
                "Path sandbox base %s contains the user home directory: the "
                "submit/MCP file surfaces can read every credential-bearing "
                "dotfile under it (.ssh, .aws, .env, .netrc, ...) that the "
                "sensitive-name deny list does not name. Narrow UBT_ALLOWED_DIRS "
                "to the directories that actually hold books.",
                base,
            )
        if base == Path("/"):
            logger.warning(
                "Path sandbox base %s is the filesystem root: the system-directory "
                "deny list is disabled for every path, so /etc, /var and every "
                "other host directory readable by this process is reachable via "
                "submit/MCP. Narrow UBT_ALLOWED_DIRS.",
                base,
            )
        for disallowed in SYSTEM_DISALLOWED_PREFIXES:
            if disallowed == base or disallowed.is_relative_to(base):
                logger.warning(
                    "Path sandbox base %s spans system directory %s: the "
                    "allowlist exempts everything under it from the "
                    "system-directory deny list, so it is readable via "
                    "submit/MCP. Narrow UBT_ALLOWED_DIRS to exclude it.",
                    base,
                    disallowed,
                )


#: Forwarded-header trust for credential-sensitive routes. Mirrors
#: ``UBT_TRUST_TENANT_HEADER``: proxy headers are client-forgeable, so they
#: only ever widen a decision when the operator explicitly declares a trusted
#: front proxy (``UBT_TRUST_PROXY``). Default OFF.
_TRUST_PROXY_ENV = "UBT_TRUST_PROXY"


def trust_proxy() -> bool:
    """Whether the operator declares a trusted reverse proxy in front of this server.

    The README's shared-hosting posture terminates TLS (and client identity) at
    a proxy bound to the same host, so the socket-level loopback answer is
    about the *proxy*, not the caller. When ``UBT_TRUST_PROXY`` is on,
    forwarded headers decide where a request really came from; when it is off,
    the mere presence of forwarded headers is treated as "arrived through a
    proxy we do not trust" and credential-sensitive routes answer non-local.
    """
    return os.getenv(_TRUST_PROXY_ENV, "").strip().lower() in ("1", "true", "yes")


def forwarded_client_ip(x_forwarded_for: str | None, x_real_ip: str | None) -> str | None:
    """The client IP a trusted proxy chain vouches for, or ``None``.

    ``X-Forwarded-For`` is a comma list the proxy appends to; the *leftmost*
    entry is the original client as recorded by the first trusted hop — an
    operator who trusts the chain to have sanitized the list trusts exactly
    that entry. ``X-Real-IP`` is the single-value fallback nginx sets.
    """
    for raw in (x_forwarded_for, x_real_ip):
        if not raw:
            continue
        first = raw.split(",")[0].strip()
        if first:
            return first
    return None


def _client_ip_is_loopback(ip: str) -> bool:
    """Whether a forwarded client IP is a loopback address.

    Fails closed: a forwarded value that is not a literal IP (a hostname, or
    garbage from a misbehaving chain) is not provably local, and the decision
    this feeds gates credential writes.
    """
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped.is_loopback
    return addr.is_loopback


def credential_route_is_local(
    *,
    server_bind: str | None,
    x_forwarded_for: str | None = None,
    x_real_ip: str | None = None,
) -> bool:
    """The loopback decision credential/base_url writes are gated on, proxy-aware.

    ``server_bind`` is the socket address the server is bound to — behind the
    README's reverse-proxy posture that is always loopback while the *callers*
    are not, which is exactly the gap a reverse proxy opened in the old
    bind-only answer. The rules, in order:

    * forwarded headers present + ``UBT_TRUST_PROXY`` off → **not local**. A
      request that claims a proxy chain arrived through a proxy; without the
      trust declaration the claim is client-forgeable and must never widen the
      decision (a spoofed ``X-Forwarded-For: 127.0.0.1`` included).
    * forwarded headers present + trust on → judge the forwarded client IP
      (leftmost trusted hop); a non-IP value fails closed to not-local.
    * no forwarded headers → the historical bind answer, unchanged for direct
      (unproxied) deployments.
    """
    has_forwarded = bool((x_forwarded_for or "").strip() or (x_real_ip or "").strip())
    if has_forwarded:
        if not trust_proxy():
            return False
        client_ip = forwarded_client_ip(x_forwarded_for, x_real_ip)
        if client_ip is None:
            return False
        return _client_ip_is_loopback(client_ip)
    if server_bind:
        from ubt.api.routes.shared import is_nonloopback_bind

        return not is_nonloopback_bind(server_bind)
    # No bind address at all: the in-process test client, which never touches
    # a network. Kept from the historical behaviour the console tests rely on.
    return True


def _no_auth_allowed() -> bool:
    return os.getenv(_NO_AUTH_OVERRIDE_ENV, "").strip().lower() in ("1", "true", "yes")


def _strict_auth_from_env() -> bool:
    """Strict-auth posture from the environment alone (no config file layer).

    The console's implicit override is decided before the server builds its
    config, so the value that will gate the boot is the one in the process
    environment — the layer ``UBTConfig.from_env`` reads last.
    """
    return (
        os.getenv("UBT_STRICT_AUTH", "").strip().lower() in ("1", "true", "yes")
        or os.getenv("UBT_ENV", "").strip().lower() == "production"
    )


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
    if cfg.is_strict_auth():
        # Strict mode is checked before the override: it is a *deployment*
        # promise ("this server authenticates"), and the override is a
        # *process* convenience that tools set implicitly (``ubt console``
        # exports it for loopback). Letting the override win meant a strict
        # production server booted open whenever anything set the variable —
        # and with no key the server would 500 every request while exposing
        # /docs. Fail fast instead; the message says how to do it deliberately.
        if _no_auth_allowed():
            logger.warning(
                "%s is set but strict auth is on: ignoring the override and "
                "refusing to boot without an API key.",
                _NO_AUTH_OVERRIDE_ENV,
            )
        raise SystemExit(
            "UBT_STRICT_AUTH is set but UBT_API_KEY is empty: set a key, or "
            "run a non-strict local server for a throwaway deployment."
        )
    if _no_auth_allowed():
        logger.warning(
            "%s is set: booting the UBT API WITHOUT an API key (local/development only).",
            _NO_AUTH_OVERRIDE_ENV,
        )
        return
    raise SystemExit(
        "refusing to start the UBT API without authentication: set UBT_API_KEY to "
        "enable the X-API-Key gate, or set UBT_ALLOW_NO_AUTH=1 to run open on "
        "localhost (development only)."
    )


#: Cookie the console sets so same-origin browser requests that *cannot* carry
#: the ``X-API-Key`` header — an ``EventSource`` stream, an ``<img>`` preview,
#: a download link — still authenticate. The value is a digest keyed over a
#: random *per-install* secret (never the service key, and never sent to the
#: browser): a leaked cookie must not hand out the credential for other
#: services, and the digest cannot be replayed as the ``X-API-Key`` header.
SESSION_COOKIE_NAME = "ubt_session"

#: Session lifetime. The cookie is re-minted on every console login, so a week
#: bounds how long a browser that was left signed in keeps authenticating
#: while never badgering an operator who uses the console weekly.
SESSION_COOKIE_MAX_AGE_S = 7 * 24 * 3600

#: Versioned domain string the cookie digest is computed over. Bumping it
#: rotates every outstanding cookie at once (old values stop verifying), which
#: is the second rotation lever beside regenerating the per-install secret.
_SESSION_COOKIE_DIGEST_DOMAIN = b"ubt-session-v2"

#: Minimum entropy (bytes) a persisted session secret must carry. A shorter or
#: corrupt file is treated as absent, not as a key.
_SESSION_SECRET_MIN_BYTES = 32

#: Process-local fallback for hosts that refuse the state-dir write. Cached so
#: cookies verify within the process; they simply do not survive a restart.
_EPHEMERAL_SESSION_SECRET: bytes | None = None


def _session_secret_path(config: UBTConfig) -> Path:
    """Where the per-install session secret lives (the ledger/state directory)."""
    return config.db_dir / "session_secret.key"


def load_or_create_session_secret(config: UBTConfig) -> bytes:
    """The random per-install secret the session cookie is keyed over.

    Created once, in the operator-owned state directory, at ``0600`` (via the
    same perms helpers the ledger uses), and reused by every subsequent boot so
    outstanding cookies survive a restart. Deleting the file is the documented
    rotation: the next call mints a fresh secret and every old cookie value
    stops verifying. All the failure modes degrade to a process-local secret
    rather than breaking cookie auth: a read-only state dir, a race with
    another worker, a corrupt short file — the cookie then dies with the
    process instead of the server refusing to authenticate.
    """
    global _EPHEMERAL_SESSION_SECRET
    path = _session_secret_path(config)
    ensure_private_dir(path.parent)
    for _attempt in range(2):
        try:
            data = path.read_bytes()
            if len(data) >= _SESSION_SECRET_MIN_BYTES:
                return data
        except FileNotFoundError:
            pass
        except OSError:
            break  # unreadable state dir: use the process-local fallback
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Another request or worker minted it between the read and the
            # create; read that one instead of clobbering it (clobbering would
            # silently invalidate cookies minted moments earlier).
            continue
        except OSError:
            break
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(secrets.token_bytes(_SESSION_SECRET_MIN_BYTES))
            restrict_file_to_owner(path)
        except OSError:
            break
    if _EPHEMERAL_SESSION_SECRET is None:
        _EPHEMERAL_SESSION_SECRET = secrets.token_bytes(_SESSION_SECRET_MIN_BYTES)
        logger.warning(
            "session secret unavailable at %s: using a process-local secret, "
            "so session cookies will not survive a restart",
            path,
        )
    return _EPHEMERAL_SESSION_SECRET


def session_cookie_value(expected_key: str, *, session_secret: bytes) -> str:
    """The session cookie value for a non-empty service key.

    HMAC'd over the per-install secret with a versioned domain string *and*
    the service key in the message, so rotating the secret file, changing the
    API key, or bumping the domain version all invalidate every outstanding
    cookie — an old value never verifies again.
    """
    return hmac.new(
        session_secret,
        _SESSION_COOKIE_DIGEST_DOMAIN + expected_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _session_cookie_matches(cookie: str | None, expected: str, *, session_secret: bytes) -> bool:
    if not cookie:
        return False
    # Constant-time over bytes, like the header comparison: a str compare_digest
    # raises TypeError on non-ASCII cookie bytes.
    return secrets.compare_digest(
        cookie.encode("utf-8"),
        session_cookie_value(expected, session_secret=session_secret).encode("utf-8"),
    )


def cookie_secure_for_request(scheme: str | None) -> bool:
    """Whether the session cookie must carry the ``Secure`` flag for this request.

    An https request is the direct signal. ``UBT_COOKIE_SECURE`` overrides both
    ways — a proxy terminating TLS whose forwarded scheme the app cannot see
    is exactly the case an explicit override exists for — and otherwise a
    production environment defaults to on. Plain-http local consoles stay off:
    a ``Secure`` cookie over ``http://`` is silently dropped by the browser and
    the console would then never authenticate.
    """
    env = os.getenv("UBT_COOKIE_SECURE", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    if scheme and scheme.lower() == "https":
        return True
    return os.getenv("UBT_ENV", "").strip().lower() == "production"


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    ubt_session: str | None = Cookie(default=None),
    _config: UBTConfig | None = None,
) -> None:
    """Lightweight API-key gate for the service.

    Authentication is opt-in: it is only enforced when ``UBT_API_KEY`` is set.
    When unset the server runs open (development mode). In strict mode
    (``UBT_STRICT_AUTH=1`` or ``UBT_ENV=production``), a configured
    ``UBT_API_KEY`` is mandatory. Pass the key via the ``X-API-Key`` header
    (never query string), or via the ``ubt_session`` cookie the console sets —
    a browser cannot attach the header to its SSE stream, image previews or
    download links, so without the cookie path a keyed server's own UI is
    unusable.
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
    # Compare over bytes — str.compare_digest raises TypeError on
    # non-ASCII (Latin-1 header bytes like "café"), turning a should-be 401
    # into an unhandled 500.
    if secrets.compare_digest((x_api_key or "").encode("utf-8"), expected.encode("utf-8")):
        return
    # The cookie path is checked only when a cookie was actually presented, so
    # header-authenticated requests never touch the session-secret file.
    if ubt_session and _session_cookie_matches(
        ubt_session,
        expected,
        session_secret=load_or_create_session_secret(cfg),
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )


# The sensitive-name deny list itself lives in ``ubt.core.fs_perms`` and is
# re-exported above: the MCP tools run the same check and must not import this
# package (``ubt.api`` is a PEP 562 lazy-export package). The list is casefolded
# at comparison time — see ``resolve_secure_path``.


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


def _tenant_from_header(raw: str | None, *, trust: bool = False) -> str:
    """Resolve the queue tenant from ``X-UBT-Tenant`` (single-tenant by default).

    Enterprise/on-prem deployments leave it unset (every job is ``default``);
    a SaaS gateway may forward an authenticated tenant id in the header.

    The header is *client-forgeable*: any caller can send ``X-UBT-Tenant: other``
    and, when honored, read or cancel another tenant's job. So it is only read
    when the operator explicitly declares a trusted front proxy sets it
    (``trust=True``, from ``UBT_TRUST_TENANT_HEADER``); otherwise every request is
    the single ``default`` tenant and a forged header is ignored. This keeps the
    isolation boundary honest for the common (loopback, keyed, single-tenant)
    deployment while the multi-tenant SaaS path stays available behind the flag.
    """
    if raw is None or not trust:
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
    must_exist: bool = True,
    allowed_bases: list[Path] | None = None,
    config: UBTConfig | None = None,
) -> Path:
    """Validate and resolve path against path traversal attacks and sandbox restrictions.

    Check order is deliberate, and the allowlist and the deny lists read like a
    contradiction unless their precedence is stated together:

    * A literal ``..`` is always refused.
    * The path must be contained in the effective allowlist
      (``UBT_ALLOWED_DIRS``, else the working directory + ``config.db_dir``).
    * Allowlist wins over the *system* deny list
      (:data:`SYSTEM_DISALLOWED_PREFIXES`): a path an operator explicitly
      allowlisted skips it — otherwise the conventional ``/var/lib/ubt``
      deployment could never be served.
    * Allowlist loses to the *sensitive-name* deny list
      (:data:`SENSITIVE_FILENAME_PARTS`), which still applies to every path,
      allowlist included. An allowlist is a scope decision ("these books may be
      read"), not a licence to serve the ``.ssh``/``.env``/``credentials.json``
      that happen to live beside them — and these routes are reachable without
      credentials by default. Refusing a whitelisted secret file on purpose, not
      by bug.

    The two allowlist-precedence rules are named rather than numbered so the
    inline note at each check can cite the rule, not an ordinal that shifts when
    a check is added.

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
        # the one rule it does NOT override: see "Allowlist loses to the
        # sensitive-name deny list" in the docstring above.
        explicit_whitelist = False
        if allowed_bases is not None:
            # Resolve every base: containment is compared against the *resolved*
            # candidate, so an unresolved base (``sub/..``, a symlinked dir)
            # would never match and every path inside it would 403.
            effective_bases.extend(b.resolve() for b in allowed_bases)
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
    if not (explicit_whitelist and contained):
        for disallowed in SYSTEM_DISALLOWED_PREFIXES:
            if resolved == disallowed or disallowed in resolved.parents:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Path '{raw_path}' accesses restricted system directory '{disallowed}'",
                )

    # Reject attempts to read user secrets, shell configurations, or
    # credentials. Applied *after* the allowlist and to whitelisted paths too —
    # that precedence is intentional; see "Allowlist loses to the sensitive-name
    # deny list" in the docstring above.
    for part in resolved.parts:
        if is_sensitive_path_part(part):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Accessing sensitive configuration directory or file '{part}' is prohibited",
            )

    if (explicit_whitelist or effective_bases) and not contained:
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
