"""Provider registry: built-in wire protocols plus user-declared ``[providers.*]``.

A *provider* bundles everything about **where** and **how** to call an LLM — the
endpoint, the wire protocol, and the default models. Credential configuration
is specified directly via ``api_key`` (supporting ``${VAR}`` expansion), or via
the generic ``UBT_LLM_API_KEY`` environment variable or CLI flags.

Search order for the config file (unchanged from the old profile loader):
1. explicit path passed by the caller
2. ``./ubt.toml``
3. ``~/.config/ubt/config.toml``
4. ``~/.ubt/config.toml``
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from ubt.core.exceptions import UBTError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_LOCATIONS: tuple[Path, ...] = (
    Path("ubt.toml"),
    Path.home() / ".config" / "ubt" / "config.toml",
    Path.home() / ".ubt" / "config.toml",
)


class ProviderNotFoundError(UBTError):
    """A requested provider is neither built-in nor declared in the config file."""


class ProviderConfigError(UBTError):
    """A ``[providers.*]`` / ``[defaults]`` block is malformed or carries a secret."""


#: Vendor presets that ship inside the package. Kept as data, not code, so this
#: module names no vendor: the registry is read and validated through the very
#: same path as a user's ``[providers.*]`` block.
_SHIPPED_REGISTRY = Path(__file__).resolve().parent.parent / "resources" / "providers.toml"

#: Fields a ``[providers.*]`` / ``[defaults]`` block may set. An explicit
#: allow-list (not ``UBTConfig.model_fields``) keeps this module free of a
#: circular import and, more importantly, keeps a block from reaching a secret
#: or an unrelated knob.
PROVIDER_ALLOWED_KEYS = frozenset(
    {
        "api_key",
        "base_url",
        "api_mode",
        "draft_model",
        "repair_model",
        "draft_reasoning_effort",
        "repair_reasoning_effort",
        "fallback_models",
        "chat_template_kwargs",
        "extra_headers",
        "api_timeout",
        "prompt_caching_enabled",
        "supports_batch_api",
        "is_free",
        "cost_per_mtok",
        "capability_profile",
        "supports_temperature",
        "supports_reasoning_effort",
        "reasoning_dialect",
        "repair_provider",
        "ocr_api_key",
    }
)

#: Forbidden keys in outbound provider blocks. Inbound service auth uses UBT_API_KEY.
PROVIDER_FORBIDDEN_KEYS = frozenset({"service_api_key"})


def find_config_file(custom_path: Path | str | None = None) -> Path | None:
    """First existing config file in search order, or ``None`` when none exists."""
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Configuration file not found: {custom_path}")

    for candidate in DEFAULT_CONFIG_LOCATIONS:
        try:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved
        except (OSError, RuntimeError):
            continue
    return None


def _expand_env_vars(val: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in a value, recursing into tables.

    A table value (``extra_headers``, ``chat_template_kwargs``) is expanded too,
    so a header can name its variable: ``{ "x-opencode-session" = "${SID}" }``.
    """
    if isinstance(val, str):

        def _repl(m: re.Match[str]) -> str:
            var = m.group(1)
            default = m.group(3) if m.group(2) else ""
            return os.environ.get(var, default)

        return re.sub(r"\$\{([A-Za-z0-9_]+)(:-([^}]*))?\}", _repl, val)
    if isinstance(val, Mapping):
        return {str(k): _expand_env_vars(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_expand_env_vars(v) for v in val]
    return val


def _read_toml(custom_path: Path | str | None = None) -> dict[str, Any]:
    """Read the user config TOML, or ``{}`` when no file exists.

    A *malformed* user config fails loudly, matching
    :func:`_read_shipped_providers`. Returning ``{}`` here (the old fail-open)
    dropped every declared provider and fell back to the shipped defaults on a
    single typo, so a run silently used the wrong endpoint/model.

    The parser message quotes the offending source line, and a user
    ``[providers.*]`` block may hold a literal ``api_key``; only the path is
    echoed, with the original error kept as the chained cause.
    """
    config_file = find_config_file(custom_path)
    if not config_file:
        return {}
    try:
        with config_file.open("rb") as f:
            data = tomllib.load(f)
    except OSError as exc:
        raise ProviderConfigError(f"configuration file is unreadable at {config_file}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProviderConfigError(f"configuration file at {config_file} is not valid TOML") from exc
    return data if isinstance(data, dict) else {}


def _validate_block(label: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Reject invalid fields in a TOML block, then expand ``${VAR}``."""
    leaked = sorted(PROVIDER_FORBIDDEN_KEYS & set(fields))
    if leaked:
        raise ProviderConfigError(
            f"{label} must not contain inbound gate secret(s) {', '.join(leaked)}: "
            "service_api_key is for the inbound REST API service gate (configure via UBT_API_KEY)."
        )
    unknown = sorted(set(fields) - PROVIDER_ALLOWED_KEYS)
    if unknown:
        raise ProviderConfigError(
            f"{label} has unknown field(s) {', '.join(unknown)}; allowed: "
            f"{', '.join(sorted(PROVIDER_ALLOWED_KEYS))}"
        )
    return {str(k): _expand_env_vars(v) for k, v in fields.items()}


@lru_cache(maxsize=1)
def _read_shipped_providers() -> dict[str, dict[str, Any]]:
    """The packaged vendor presets, keyed by name, each run through ``_validate_block``.

    Read once and cached: the file ships with the wheel and never changes at
    runtime. A malformed or unreadable registry is a packaging fault, so it
    fails loudly with the path rather than silently yielding no providers.
    """
    try:
        with _SHIPPED_REGISTRY.open("rb") as f:
            data = tomllib.load(f)
    except OSError as exc:
        raise ProviderConfigError(
            f"shipped provider registry is unreadable at {_SHIPPED_REGISTRY}: {exc}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProviderConfigError(
            f"shipped provider registry at {_SHIPPED_REGISTRY} is malformed: {exc}"
        ) from exc
    providers = data.get("providers", {})
    if not isinstance(providers, dict):
        raise ProviderConfigError(f"{_SHIPPED_REGISTRY}: [providers] must be a table")
    shipped: dict[str, dict[str, Any]] = {}
    for name, block in providers.items():
        if not isinstance(block, dict):
            raise ProviderConfigError(f"{_SHIPPED_REGISTRY}: [providers.{name}] must be a table")
        if "api_key" in block:
            raise ProviderConfigError(
                f"{_SHIPPED_REGISTRY}: shipped preset must not contain api_key"
            )
        shipped[str(name)] = _validate_block(f"{_SHIPPED_REGISTRY}: [providers.{name}]", block)
    return shipped


#: Ergonomic aliases mapping short names to canonical wire protocols.
PROTOCOL_ALIASES: dict[str, str] = {
    # Provider shorthand aliases
    "openai": "openai-chat",
    "anthropic": "anthropic-messages",
    "gemini": "gemini-native",
    # Protocol shorthand aliases
    "chat": "openai-chat",
    "responses": "openai-responses",
    "messages": "anthropic-messages",
}


def list_providers(custom_path: Path | str | None = None) -> list[str]:
    """Shipped wire protocols and aliases plus any declared in the config file, sorted."""
    declared = _read_toml(custom_path).get("providers", {})
    names = set(_read_shipped_providers()) | set(PROTOCOL_ALIASES)
    if isinstance(declared, dict):
        names |= {str(k) for k in declared}
    return sorted(names)


def load_provider_block(name: str, custom_path: Path | str | None = None) -> dict[str, Any]:
    """Resolve ``name`` to a dictionary of config fields.

    The shipped wire protocol is the base; a user ``[providers.<name>]`` block overrides
    or extends it. Both are validated identically.
    """
    shipped = _read_shipped_providers()
    data = _read_toml(custom_path)
    declared = data.get("providers", {})
    if not isinstance(declared, dict):
        declared = {}

    canonical_name = PROTOCOL_ALIASES.get(name, name)
    if name not in shipped and canonical_name not in shipped and name not in declared:
        available = list_providers(custom_path)
        raise ProviderNotFoundError(
            f"Provider '{name}' not found. Available providers: {available if available else 'none'}"
        )

    fields: dict[str, Any] = {}

    base_preset = shipped.get(name) or shipped.get(canonical_name)
    if base_preset is not None:
        fields.update(dict(base_preset))

    user_block = declared.get(name)
    if user_block is not None:
        if not isinstance(user_block, dict):
            raise ProviderConfigError(f"[providers.{name}] must be a table")
        validated = _validate_block(f"[providers.{name}]", user_block)
        fields.update(validated)
    return fields


def load_defaults_block(custom_path: Path | str | None = None) -> dict[str, Any]:
    """The ``[defaults]`` baseline, applied under every provider block."""
    defaults = _read_toml(custom_path).get("defaults", {})
    if not isinstance(defaults, dict) or not defaults:
        return {}
    return _validate_block("[defaults]", defaults)


def load_layer(provider_name: str | None, custom_path: Path | str | None = None) -> dict[str, Any]:
    """The ``[defaults]`` baseline with ``[providers.<name>]`` layered over it.

    Returns a dictionary of config fields. With no provider the defaults stand alone;
    a provider overrides or extends them. One loader, shared by ``from_env`` and
    ``apply_config_overrides`` so both paths see the same block.
    """
    fields: dict[str, Any] = dict(load_defaults_block(custom_path))
    if provider_name:
        provider_fields = load_provider_block(provider_name, custom_path)
        fields.update(provider_fields)
    return fields


def merge_provider_under(
    explicit: Mapping[str, Any],
    fields: Mapping[str, Any],
    *,
    env_supplied: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Layer a provider block *under* explicit values, with the environment above it.

    One precedence rule, shared by the fresh-construction path (``from_env``) and
    the request-override path (``apply_config_overrides``):

    ``explicit > environment > provider block > [defaults] > field default``.

    ``env_supplied`` names fields the operator pinned in the process environment;
    those outrank the block. ``repair_model`` follows the effective draft unless
    the block (or the caller) chose a repair distinct from its own draft — the
    same rule ``UBTConfig``'s validator enforces.
    """
    layer = dict(fields)
    if env_supplied:
        for key in list(layer):
            if key in env_supplied:
                layer.pop(key, None)
    merged = {**layer, **explicit}
    if (
        "draft_model" in explicit
        and "repair_model" not in explicit
        and not (
            "repair_model" in fields and fields.get("repair_model") != fields.get("draft_model")
        )
    ):
        merged.pop("repair_model", None)
    return merged
