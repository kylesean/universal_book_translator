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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ubt.core.atomic import atomic_write_text
from ubt.core.config import (
    VALID_API_MODES as _VALID_API_MODES,
)
from ubt.core.config import (
    VALID_EXTRACTION_STRATEGIES as _VALID_EXTRACTION,
)
from ubt.core.config import (
    VALID_REASONING_DIALECTS as _VALID_DIALECTS,
)
from ubt.core.exceptions import UBTError
from ubt.core.providers import (
    PROTOCOL_ALIASES,
    PROVIDER_ALLOWED_KEYS,
    PROVIDER_FORBIDDEN_KEYS,
    _read_shipped_providers,
    credential_resolution,
    find_config_file,
    load_defaults_config,
)
from ubt.core.router.pricing import endpoint_is_local


class ProviderEditModel(BaseModel):
    """Pydantic V2 model for validating provider block edits."""

    model_config = ConfigDict(extra="forbid")

    make_default: bool | None = Field(
        default=None,
        description=(
            "Also write [defaults] provider = name. The console sends this when "
            "creating a block, because a declared block that nothing points at is "
            "still a rehearsal — the state a fresh install was stuck in"
        ),
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Credential for this provider block. Accepted only on a loopback, "
            "non-strict deployment; elsewhere the request is refused and the "
            "message says to use the environment or the file."
        ),
    )
    base_url: str | None = Field(default=None, description="Endpoint base URL (http/https)")
    api_mode: str | None = Field(
        default=None, description="Wire protocol (openai-chat, openai-responses, …)"
    )
    draft_model: str | None = Field(default=None, description="Model id used for drafting")
    repair_model: str | None = Field(default=None, description="Model id used for repair")
    draft_reasoning_effort: str | None = Field(default=None, description="Draft reasoning effort")
    repair_reasoning_effort: str | None = Field(default=None, description="Repair reasoning effort")
    reasoning_dialect: str | None = Field(
        default=None, description="How the responses protocol spells a reasoning effort"
    )
    supports_temperature: bool | None = Field(
        default=None, description="Whether the endpoint accepts a temperature"
    )
    supports_reasoning_effort: bool | None = Field(
        default=None, description="Whether the endpoint accepts a reasoning effort"
    )
    supports_system_prompt: bool | None = Field(
        default=None, description="Whether the endpoint accepts a system turn"
    )
    supports_vision: bool | None = Field(
        default=None, description="Whether the endpoint accepts image input"
    )
    extraction_strategy: str | None = Field(
        default=None, description="How to recover the translation (auto, xml_tag, raw, cp)"
    )
    deployment_backend: str | None = Field(
        default=None, description="Self-hosted stack serving this model (ollama, vllm, …)"
    )
    is_free: bool | None = Field(default=None, description="Declared free (no billable spend)")
    api_timeout: float | None = Field(default=None, gt=0, description="Per-request timeout (s)")
    fallback_models: list[str] | None = Field(
        default=None, description="Ordered model fallback chain"
    )

    @field_validator(
        "supports_temperature",
        "supports_reasoning_effort",
        "supports_system_prompt",
        "supports_vision",
        "is_free",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, v: Any, info: Any) -> bool | None:
        if v is None:
            return None
        if not isinstance(v, bool):
            raise ValueError(f"{info.field_name} must be a boolean")
        return v

    @field_validator("api_timeout", mode="before")
    @classmethod
    def _validate_api_timeout(cls, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("api_timeout must be a positive number of seconds")
        return float(v)

    @field_validator("fallback_models", mode="before")
    @classmethod
    def _validate_fallback_models(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError("fallback_models must be a list of model ids")
        return [x for x in v if x.strip()]

    @field_validator(
        "api_key",
        "base_url",
        "api_mode",
        "draft_model",
        "repair_model",
        "draft_reasoning_effort",
        "repair_reasoning_effort",
        "reasoning_dialect",
        "extraction_strategy",
        "deployment_backend",
        mode="before",
    )
    @classmethod
    def _validate_string_fields(cls, v: Any, info: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} must be a string")
        cleaned = v.strip()
        field = info.field_name
        if field == "api_mode" and cleaned not in _VALID_API_MODES:
            raise ValueError(f"api_mode must be one of {', '.join(sorted(_VALID_API_MODES))}")
        if field == "extraction_strategy" and cleaned not in _VALID_EXTRACTION:
            raise ValueError(
                f"extraction_strategy must be one of {', '.join(sorted(_VALID_EXTRACTION))}"
            )
        if field == "reasoning_dialect" and cleaned not in _VALID_DIALECTS:
            raise ValueError(
                f"reasoning_dialect must be one of {', '.join(sorted(_VALID_DIALECTS))}"
            )
        if field == "base_url" and cleaned and not cleaned.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return cleaned


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
WRITABLE_FIELDS: tuple[str, ...] = tuple(
    name for name in ProviderEditModel.model_fields if name != "make_default"
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
    """Type-check one incoming value against the field it names via ProviderEditModel.

    An unvalidated value here would be written to the TOML and then rejected by
    ``UBTConfig`` at the next job start — the operator would see a successful save
    and a broken engine. Rejecting at write time keeps the two in step.
    """
    if field not in WRITABLE_FIELDS:
        raise ProviderEditError(f"unknown field {field!r}")
    try:
        model = ProviderEditModel.model_validate({field: value})
        return getattr(model, field)
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            err = exc.errors()[0]
            msg = err["msg"]
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            raise ProviderEditError(msg) from exc
        raise ProviderEditError(str(exc)) from exc


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
    make_default: bool = False,
) -> dict[str, Any]:
    """Create or update one ``[providers.<name>]`` block; returns what was stored.

    Only the keys in ``changes`` are touched, so a form that submits one edited
    field cannot silently drop the block's other settings. A credential already in
    the file is preserved verbatim unless ``changes`` names it, and then only when
    ``allow_credential`` is set (see :func:`credential_write_allowed`).

    ``make_default`` writes ``[defaults] provider = name`` — the step without which
    a fresh install could never run. Creating a block alone left ``provider`` unset,
    so the console answered ``can_translate: false`` and the first job became a
    rehearsal even though the operator had just typed a working endpoint and key.
    The form passes this on create, where "the block I just made" is unambiguously
    what the operator wants; :func:`set_default_provider` does the same for an
    existing one.
    """
    coerced = validate_edit(name, changes, allow_credential=allow_credential)
    cleared = {field for field, value in changes.items() if value is None}
    if not coerced and not cleared and not make_default:
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

    _reject_plaintext_credential_redirect(block, coerced)

    for field, value in coerced.items():
        # ``item()`` so tomlkit picks a representation that round-trips: a bare
        # string stays a string, a list stays a list.
        block[field] = tomlkit.item(value)
    for field in cleared:
        # Removing the key restores "not declared", which is a different state
        # from an explicit ``false`` — see CLEARABLE_FIELDS.
        block.pop(field, None)

    if make_default:
        # The block must exist before it can be named, so this runs after the
        # table is created above. Written to ``[defaults]`` because that is where
        # ``UBTConfig.from_env`` looks when neither ``--provider`` nor
        # ``UBT_PROVIDER`` names one — a complete configuration then lives in the
        # file rather than in somebody's shell history.
        _write_default_provider(document, name)

    atomic_write_text(path, tomlkit.dumps(document))
    _tighten_permissions(path)
    return read_provider(name)


#: Hosts that stay on this machine, so a plaintext endpoint is not a wire exposure.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _is_plaintext_nonlocal(base_url: str) -> bool:
    """True for an ``http://`` endpoint that leaves this machine."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(base_url)
    except ValueError:
        return False
    if parts.scheme.lower() != "http":
        return False
    return (parts.hostname or "").lower() not in _LOOPBACK_HOSTS


def _reject_plaintext_credential_redirect(block: Any, coerced: dict[str, Any]) -> None:
    """Refuse to point a block that holds a credential at a plaintext endpoint.

    ``base_url`` is writable on its own, and the credential already in the file
    is preserved verbatim unless ``changes`` names it — which is the documented
    contract. The combination is the hazard: rewriting only ``base_url`` to
    ``http://…`` sends the stored key to a remote host in cleartext, and the
    operator's form said nothing about the credential at all.

    This is not an authorisation check (an operator who can reach this route can
    edit ``ubt.toml`` directly); it is a guard against a silent downgrade. A
    loopback endpoint is exempt, because a local model server has no wire to
    expose.
    """
    new_base = coerced.get("base_url")
    if not isinstance(new_base, str) or not _is_plaintext_nonlocal(new_base):
        return
    stored_key = block.get("api_key")
    # An explicit credential in this same request is the operator's own decision
    # and is already gated by ``allow_credential``.
    if stored_key and "api_key" not in coerced:
        raise ProviderEditError(
            f"base_url {new_base!r} is plain HTTP and not loopback: saving it would "
            "send the provider's stored credential in cleartext. Use an https:// "
            "endpoint, or edit ubt.toml directly if the endpoint really is "
            "unauthenticated."
        )


def _write_default_provider(document: tomlkit.TOMLDocument, name: str) -> None:
    """Point ``[defaults] provider`` at ``name`` inside an already-parsed document."""
    defaults = document.get("defaults")
    if defaults is None:
        defaults = tomlkit.table()
        document["defaults"] = defaults
    defaults["provider"] = tomlkit.item(name)


def set_default_provider(name: str) -> dict[str, Any]:
    """Name ``name`` as the provider a run uses when nothing more explicit does.

    Separate from :func:`upsert_provider` because switching which block runs is not
    an edit to that block: a form submitting an empty diff would otherwise fail
    with "no writable field was supplied". Refuses a name that is not declared,
    since ``[defaults] provider`` pointing at nothing makes every run raise
    ``ProviderNotFoundError`` at startup — the worst failure to leave behind in a
    config file.
    """
    _validate_name(name)
    path = _config_path()
    document = _load_document(path)
    providers = document.get("providers")
    declared = set(providers.keys()) if isinstance(providers, dict) else set()
    if name not in declared:
        raise ProviderEditError(
            f"{name!r} is not a declared provider; declare it first "
            f"(declared: {', '.join(sorted(declared)) or 'none'})"
        )
    _write_default_provider(document, name)
    atomic_write_text(path, tomlkit.dumps(document))
    return {"provider": name}


def default_provider_name() -> str | None:
    """The provider ``[defaults]`` names, or None. Never raises.

    Deliberately does not fall back to ``UBT_PROVIDER`` or the CLI: the console is
    reporting what the *file* says, and an environment variable that outranks it is
    a separate fact the panel already surfaces as the active provider.
    """
    try:
        declared = load_defaults_config().provider
    except Exception:  # a malformed [defaults] is reported by the doctor, not here
        return None
    return str(declared) if declared else None


def read_provider(name: str) -> dict[str, Any]:
    """What the console may see about one block: values plus credential presence.

    The credential itself is never read into the result. ``has_credential`` is
    what the form needs — a block with no key and no environment variable cannot
    make a call — and it is the only thing a boolean can reveal.

    ``has_credential`` means *usable*, not *present*: a block carrying
    ``api_key = "${OPENCODE_API_KEY}"`` whose variable is unset reported "credential
    set" while the loader resolved it to ``""``, so the form promised an
    authentication the next job could not perform. The state is resolved by
    :func:`ubt.core.providers.credential_resolution`, which the loader itself
    agrees with, and ``credential_issue`` names what is wrong when it is.
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
            state = credential_resolution(name)
            # A loopback endpoint accepts an empty bearer, so "no key" is not a
            # fault there. The same predicate the provider list uses, so the form
            # and the row cannot disagree about whether the block can run.
            local_endpoint = endpoint_is_local(merged.get("base_url"))
            result["has_credential"] = bool(state["usable"])
            result["credential_hint"] = CREDENTIAL_MASK if state["usable"] else None
            # The two ways a credential can be configured-but-unusable, or
            # usable-but-inert. Reported as a code plus the variable names, never
            # as a value: the console renders its own copy from the code.
            result["credential_issue"] = _credential_issue(state, local_endpoint)
            result["unresolved_vars"] = list(state["unresolved_vars"])
            result["credential_shadowed_by_env"] = bool(state["shadowed_by_env"])
            # Distinct from ``has_credential``: a block may hold no key of its own
            # and still authenticate through ``UBT_LLM_API_KEY``. Collapsing the
            # two would show "no credential" for a provider that works.
            result["authenticates"] = bool(state["authenticates"])
            continue
        if field in merged and merged[field] is not None:
            result[field] = merged[field]
    # Whether the key comes from the file or the environment matters when the
    # operator is deciding where to change it: a value in the file is what this
    # endpoint can replace, one from ``UBT_LLM_API_KEY`` is not.
    result["credential_in_file"] = bool(declared.get("api_key"))
    return result


def _credential_issue(state: dict[str, Any], local_endpoint: bool = False) -> str | None:
    """A stable code for what is wrong with this block's credential, or ``None``.

    Codes rather than sentences: the console localizes its own copy, and a test
    can assert on the code without pinning the wording.

    ``missing`` is only reported when nothing else can authenticate the block —
    a keyless block on a deployment that exports ``UBT_LLM_API_KEY`` works, and
    calling that "missing" would send the operator to fix a non-problem. The same
    goes for a loopback endpoint, which accepts an empty bearer by design.
    """
    if state["unresolved_vars"]:
        return "unresolved_variable"
    if not state["authenticates"] and not local_endpoint:
        return "missing"
    return None
