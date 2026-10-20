"""Editing a ``[providers.<name>]`` block from the console, without touching secrets.

**Scope.** The console may change *where* a provider points and *what it accepts*:
endpoint, wire protocol, model ids, the declared capability flags, and a few
run-policy knobs. It may **not** read, write, or clear a credential. That is not
a missing feature — it is the same boundary every job payload already enforces
(``allow_provider_keys=False`` in ``ubt.core.job_options``): a credential arrives
through the operator's environment or the file, never over HTTP. An API that
could write one would turn any local process into a credential-writer, and this
server is routinely started on loopback with no API key at all.

**Why a whitelist, twice.** The fields a request may set are enumerated here
(:data:`WRITABLE_FIELDS`) *and* filtered against
:data:`ubt.core.providers.PROVIDER_ALLOWED_KEYS` before the file is touched, so a
key this module forgets cannot reach the TOML. The response is built from the same
whitelist, so a value already in the file cannot leak back out by being present.

**Why the file, and why atomically.** ``ubt.toml`` is the one place an operator
can see, diff and version their configuration, and ``ubt.core.providers``
re-reads it on every ``load_layer`` call — so a write takes effect on the next job
without restarting the server. The write goes through
``ubt.core.atomic.atomic_write_text`` (temp file in the same directory, fsync,
``os.replace``) so a crash mid-write cannot leave a half-parsed config that would
fail every subsequent job at startup.
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

#: Fields a console request may set on a provider block.
#:
#: Deliberately excludes ``api_key`` and ``ocr_api_key`` (credentials), and every
#: key that selects *another* provider (``repair_provider``, ``qe_judge_provider``)
#: or a global path (``db_dir``, ``prices_file``) — editing those from a provider
#: form would silently reconfigure a different part of the engine.
WRITABLE_FIELDS: tuple[str, ...] = (
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


def validate_edit(name: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Reject an edit that may not be applied, returning the coerced values.

    Every rejection names the offending field: a console that answers "invalid
    request" to a form makes the operator guess which of a dozen inputs is wrong.
    """
    _validate_name(name)
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


def upsert_provider(name: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Create or update one ``[providers.<name>]`` block; returns what was stored.

    Only the keys in ``changes`` are touched, so a form that submits one edited
    field cannot silently drop the block's other settings. A credential already in
    the file is preserved verbatim — this function never reads or writes one.
    """
    coerced = validate_edit(name, changes)
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
            result["has_credential"] = bool(merged.get("api_key"))
            continue
        if field in merged and merged[field] is not None:
            result[field] = merged[field]
    # Whether the key comes from the file or the environment matters when the
    # operator is deciding where to change it.
    result["credential_in_file"] = bool(declared.get("api_key"))
    return result
