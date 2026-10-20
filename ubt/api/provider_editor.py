"""Editing a ``[providers.<name>]`` block from the console.

**What the console may change.** Endpoint, wire protocol, model ids, the declared
capability flags, a few run-policy knobs — and, on a local server, the credential.

**Why the credential rule is conditional.** An earlier version refused to write a
credential at all, on the argument that a secret must never cross HTTP. That
argument is correct for a *hosted* deployment and wrong for this tool's default
shape. ``ubt console`` binds ``127.0.0.1`` and, for a loopback bind with no
``UBT_API_KEY``, deliberately runs without an auth gate — so anyone who can reach
the port can already read ``ubt.toml`` off the disk. Refusing the write protected
nothing while making a fresh install unusable without a shell:

    declared providers: 0   edit buttons: 0   draft_model: ""

The rule is therefore deployment-aware (:func:`credential_write_allowed`):

* **loopback + non-strict** → a credential may be written. This is a local tool
  being configured from its own local UI.
* **non-loopback, or strict auth** → refused with a reason. There the server is a
  shared service, the request may come from anywhere, and the operator's shell is
  the right place for a secret.

Cross-origin reach is already closed independently: this API sets no CORS
headers, and a JSON ``PUT`` is not a CORS-simple request, so a browser will not
send it cross-origin without a preflight the server never approves. The
Host-header guard covers DNS rebinding.

**Why a whitelist, twice.** The fields a request may set are enumerated here
(:data:`WRITABLE_FIELDS`) *and* filtered against
:data:`ubt.core.providers.PROVIDER_ALLOWED_KEYS` before the file is touched, so a
key this module forgets cannot reach the TOML. The response is built from the same
whitelist, so a value already in the file cannot leak back out by being present —
a credential is reported only as :data:`CREDENTIAL_MASK`.

**Why the file, and why atomically.** ``ubt.toml`` is the one place an operator
can see, diff and version their configuration, and ``ubt.core.providers``
re-reads it on every ``load_layer`` call — so a write takes effect on the next job
without restarting the server. The write goes through
``ubt.core.atomic.atomic_write_text`` (temp file in the same directory, fsync,
``os.replace``) so a crash mid-write cannot leave a half-parsed config that would
fail every subsequent job at startup. The file is then tightened to 0600, because
it may now hold a literal secret.
"""

from __future__ import annotations

import stat
import tomllib
from pathlib import Path
from typing import Any

import tomlkit

from ubt.core.atomic import atomic_write_text
from ubt.core.exceptions import UBTError
from ubt.core.providers import (
    PROTOCOL_ALIASES,
    PROVIDER_ALLOWED_KEYS,
    PROVIDER_FORBIDDEN_KEYS,
    _read_shipped_providers,
    find_config_file,
)

#: What a credential is reported as. Never the value: the console needs to know
#: whether one is configured, and a boolean plus this mask is the most that may
#: be revealed. A literal echo would put the secret into a browser tab, the
#: response log and the DOM.
CREDENTIAL_MASK = "********"

#: Fields a console request may set on a provider block.
#:
#: ``api_key`` is here but gated separately (:func:`credential_write_allowed`) —
#: it is listed so the field reaches validation, and refused there when the
#: deployment is not local. Excluded on purpose: ``ocr_api_key`` (a different
#: channel's credential, set by the OCR driver, not the LLM path), and every key
#: that selects *another* provider (``repair_provider``, ``qe_judge_provider``) or
#: a global path (``db_dir``, ``prices_file``) — editing those from a provider
#: form would silently reconfigure a different part of the engine.
WRITABLE_FIELDS: tuple[str, ...] = (
    "api_key",
    "base_url",
    "api_mode",
    "draft_model",
    "repair_model",
    "draft_reasoning_effort",
    "repair_reasoning_effort",
    "reasoning_dialect",
    "supports_temperature",
    "supports_reasoning_effort",
    "supports_system_prompt",
    "supports_vision",
    "extraction_strategy",
    "deployment_backend",
    "is_free",
    "api_timeout",
    "fallback_models",
)

#: Fields an edit may *remove* rather than set, by submitting ``null``.
#:
#: Only the tri-state capability flags: ``supports_temperature = false`` is a
#: claim that the endpoint rejects the parameter, while an absent key means "not
#: declared, use the safe default". A form that could set but never clear would
#: turn a mis-click into a permanent (and wrong) assertion. Endpoints and model
#: ids are not clearable — a block without them cannot run.
CLEARABLE_FIELDS: frozenset[str] = frozenset(
    {
        "supports_temperature",
        "supports_reasoning_effort",
        "supports_system_prompt",
        "supports_vision",
        "deployment_backend",
        "extraction_strategy",
        "fallback_models",
    }
)

#: Fields reported back to the console. A superset-free subset of the writable
#: list plus the credential *presence* flag: whether a key is configured is
#: operationally important (a block with no key and no environment variable
#: cannot run) and reveals only a boolean.
READABLE_FIELDS: tuple[str, ...] = (*WRITABLE_FIELDS, "api_key")

_VALID_API_MODES = frozenset(
    {"openai-chat", "openai-responses", "anthropic-messages", "gemini-native"}
)
_VALID_EXTRACTION = frozenset({"auto", "xml_tag", "raw", "cp"})
_VALID_DIALECTS = frozenset({"nested", "flat"})

#: Provider names are TOML bare keys in the path ``[providers.<name>]``, so the
#: charset is the one tomlkit can write unquoted. Restricting it also keeps a
#: request from injecting a table separator or a comment into the document.
_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ProviderEditError(UBTError):
    """A provider edit was rejected: unknown field, bad value, or no config file."""


def _config_path() -> Path:
    """The file a console edit writes to.

    The same search order every reader uses (:func:`find_config_file`), so the
    console never edits a file the engine would not read. When none exists the
    project-local ``ubt.toml`` is created rather than the user-level one: an
    operator working in a checkout expects the change next to the code.
    """
    found = find_config_file()
    if found is not None:
        return found
    return Path("ubt.toml").resolve()


def _validate_name(name: str) -> str:
    import re

    if not re.match(_NAME_PATTERN, name or ""):
        raise ProviderEditError(
            f"invalid provider name {name!r}: expected 1-64 characters from [A-Za-z0-9_-]"
        )
    return name


def _coerce_value(field: str, value: Any) -> Any:
    """Type-check one incoming value against the field it names.

    An unvalidated value here would be written to the TOML and then rejected by
    ``UBTConfig`` at the next job start — the operator would see a successful save
    and a broken engine. Rejecting at write time keeps the two in step.
    """
    if field in {
        "supports_temperature",
        "supports_reasoning_effort",
        "supports_system_prompt",
        "supports_vision",
        "is_free",
    }:
        if not isinstance(value, bool):
            raise ProviderEditError(f"{field} must be a boolean")
        return value
    if field == "api_timeout":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ProviderEditError("api_timeout must be a positive number of seconds")
        return float(value)
    if field == "fallback_models":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ProviderEditError("fallback_models must be a list of model ids")
        return [v for v in value if v.strip()]
    # Everything left is a string.
    if not isinstance(value, str):
        raise ProviderEditError(f"{field} must be a string")
    cleaned = value.strip()
    if field == "api_mode" and cleaned not in _VALID_API_MODES:
        raise ProviderEditError(f"api_mode must be one of {', '.join(sorted(_VALID_API_MODES))}")
    if field == "extraction_strategy" and cleaned not in _VALID_EXTRACTION:
        raise ProviderEditError(
            f"extraction_strategy must be one of {', '.join(sorted(_VALID_EXTRACTION))}"
        )
    if field == "reasoning_dialect" and cleaned not in _VALID_DIALECTS:
        raise ProviderEditError(
            f"reasoning_dialect must be one of {', '.join(sorted(_VALID_DIALECTS))}"
        )
    if field == "base_url" and cleaned and not cleaned.startswith(("http://", "https://")):
        raise ProviderEditError("base_url must start with http:// or https://")
    return cleaned


def credential_write_allowed(*, is_loopback: bool, is_strict: bool) -> bool:
    """Whether this deployment may accept a credential over HTTP.

    Two independent conditions, both required:

    * **loopback** — the request cannot have arrived from another machine. On a
      non-loopback bind the server is a shared service and a secret belongs in the
      operator's shell, not in a request body.
    * **not strict** — ``UBT_STRICT_AUTH`` / ``UBT_ENV=production`` is a promise
      that this server authenticates. Writing a credential there would mean a
      bearer token is enough to mint another bearer token.

    Both are passed in rather than read from the environment here: the caller owns
    the request's actual bind and the boot-time auth posture, and a helper that
    sniffed globals would answer for the wrong deployment in a test.
    """
    return bool(is_loopback) and not bool(is_strict)


def validate_edit(
    name: str,
    changes: dict[str, Any],
    *,
    allow_credential: bool = False,
) -> dict[str, Any]:
    """Reject an edit that may not be applied, returning the coerced values.

    Every rejection names the offending field: a console that answers "invalid
    request" to a form makes the operator guess which of a dozen inputs is wrong.
    """
    _validate_name(name)
    if "api_key" in changes and not allow_credential:
        raise ProviderEditError(
            "a credential cannot be written from the console on this deployment: it is "
            "reachable beyond loopback, or strict auth is on. Set it in ubt.toml or "
            "export UBT_LLM_API_KEY instead."
        )
    unknown = sorted(set(changes) - set(WRITABLE_FIELDS))
    if unknown:
        raise ProviderEditError(
            f"field(s) {', '.join(unknown)} cannot be set from the console; "
            f"writable: {', '.join(sorted(WRITABLE_FIELDS))}"
        )
    forbidden = sorted(set(changes) & PROVIDER_FORBIDDEN_KEYS)
    if forbidden:
        # Belt and braces: the whitelist above already excludes these, but a
        # future addition to WRITABLE_FIELDS must not quietly reopen the hole.
        raise ProviderEditError(
            f"field(s) {', '.join(forbidden)} are inbound gate secrets and may never "
            "be written to a provider block"
        )
    # Second, independent filter against the loader's own allow-list: the file is
    # read back by ``load_provider_block``, which rejects unknown keys, so writing
    # one would produce a config that fails on the next job.
    not_loadable = sorted(set(changes) - PROVIDER_ALLOWED_KEYS)
    if not_loadable:
        raise ProviderEditError(
            f"field(s) {', '.join(not_loadable)} are not accepted in a [providers.*] block"
        )
    coerced: dict[str, Any] = {}
    for field, value in changes.items():
        if value is None:
            if field not in CLEARABLE_FIELDS:
                raise ProviderEditError(
                    f"{field} cannot be cleared; it is required for the block to be usable"
                )
            # ``None`` means "remove the key", signalled to the caller below.
            continue
        coerced[field] = _coerce_value(field, value)
    return coerced


def _load_document(path: Path) -> tomlkit.TOMLDocument:
    """Parse the config file, or an empty document when it does not exist yet."""
    if not path.is_file():
        return tomlkit.document()
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProviderEditError(f"configuration file at {path} is not valid TOML: {exc}") from exc


def _tighten_permissions(path: Path) -> None:
    """Keep a config that may hold a literal credential at 0600.

    A new file created by ``atomic_write_text`` gets the process umask, which is
    commonly 0644 — world-readable. The operator's own ``ubt.toml`` is already
    0600 (it is gitignored *because* it can hold a key), so a console-created one
    must not be looser than the one it replaces.
    """
    try:
        current = stat.S_IMODE(path.stat().st_mode)
        if current & (stat.S_IRWXG | stat.S_IRWXO):
            path.chmod(0o600)
    except OSError:
        # Permission tightening is best-effort: a filesystem that refuses chmod
        # must not fail the save the operator asked for.
        pass


def upsert_provider(
    name: str,
    changes: dict[str, Any],
    *,
    allow_credential: bool = False,
) -> dict[str, Any]:
    """Create or update one ``[providers.<name>]`` block; returns what was stored.

    Only the keys in ``changes`` are touched, so a form that submits one edited
    field cannot silently drop the block's other settings. A credential already in
    the file is preserved verbatim unless ``changes`` names it, and then only when
    ``allow_credential`` is set (see :func:`credential_write_allowed`).
    """
    coerced = validate_edit(name, changes, allow_credential=allow_credential)
    cleared = {field for field, value in changes.items() if value is None}
    if not coerced and not cleared:
        raise ProviderEditError("no writable field was supplied")
    path = _config_path()
    document = _load_document(path)

    providers = document.get("providers")
    if providers is None:
        providers = tomlkit.table()
        document["providers"] = providers
    block = providers.get(name)
    if block is None:
        block = tomlkit.table()
        providers[name] = block
    for field, value in coerced.items():
        # ``item()`` so tomlkit picks a representation that round-trips: a bare
        # string stays a string, a list stays a list.
        block[field] = tomlkit.item(value)
    for field in cleared:
        # Removing the key restores "not declared", which is a different state
        # from an explicit ``false`` — see CLEARABLE_FIELDS.
        block.pop(field, None)

    atomic_write_text(path, tomlkit.dumps(document))
    _tighten_permissions(path)
    return read_provider(name)


def read_provider(name: str) -> dict[str, Any]:
    """What the console may see about one block: values plus credential presence.

    The credential itself is never read into the result. ``has_credential`` is
    what the form needs — a block with no key and no environment variable cannot
    make a call — and it is the only thing a boolean can reveal.
    """
    _validate_name(name)
    path = _config_path()
    document = _load_document(path)
    raw_block = document.get("providers", {}).get(name)
    declared: dict[str, Any] = dict(raw_block) if raw_block is not None else {}
    # Shipped wire protocols supply defaults a user block may only override, so
    # the form shows what will actually be sent, not just what was typed.
    shipped = _read_shipped_providers().get(name) or _read_shipped_providers().get(
        PROTOCOL_ALIASES.get(name, name), {}
    )
    merged = {**dict(shipped or {}), **declared}
    result: dict[str, Any] = {"name": name, "declared": bool(declared)}
    for field in READABLE_FIELDS:
        if field == "api_key":
            # Presence only. The value — literal or ``${VAR}`` — is replaced by a
            # fixed mask, so the field can be rendered as "already set" without
            # the secret entering the response, the browser or the DOM.
            raw = merged.get("api_key")
            result["has_credential"] = bool(raw)
            result["credential_hint"] = CREDENTIAL_MASK if raw else None
            continue
        if field in merged and merged[field] is not None:
            result[field] = merged[field]
    # Whether the key comes from the file or the environment matters when the
    # operator is deciding where to change it: a value in the file is what this
    # endpoint can replace, one from ``UBT_LLM_API_KEY`` is not.
    result["credential_in_file"] = bool(declared.get("api_key"))
    return result
