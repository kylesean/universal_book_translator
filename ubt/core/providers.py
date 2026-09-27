"""Provider registry: built-in vendors plus user-declared ``[providers.*]``.

A *provider* bundles everything about **where** and **how** to call an LLM — the
endpoint, the wire protocol, and the default models — and names the environment
variable that holds its credential (``api_key_env``). The secret itself never
lives in the TOML: a key committed to a version-controlled file is a leaked key,
so a block carrying one is rejected outright.

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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProviderSpec:
    """A built-in vendor: endpoint, protocol, credential variable, default models."""

    base_url: str
    api_mode: str
    api_key_env: str
    draft_model: str
    repair_model: str
    supports_batch_api: bool = False
    is_free: bool = False
    cost_per_mtok: tuple[float, float] | None = None
    capability_profile: str | None = None
    supports_temperature: bool | None = None
    supports_reasoning_effort: bool | None = None
    reasoning_dialect: str | None = None
    repair_provider: str | None = None

    def as_fields(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "draft_model": self.draft_model,
            "repair_model": self.repair_model,
            "supports_batch_api": self.supports_batch_api,
            "is_free": self.is_free,
        }
        if self.cost_per_mtok is not None:
            res["cost_per_mtok"] = self.cost_per_mtok
        if self.capability_profile is not None:
            res["capability_profile"] = self.capability_profile
        if self.supports_temperature is not None:
            res["supports_temperature"] = self.supports_temperature
        if self.supports_reasoning_effort is not None:
            res["supports_reasoning_effort"] = self.supports_reasoning_effort
        if self.reasoning_dialect is not None:
            res["reasoning_dialect"] = self.reasoning_dialect
        if self.repair_provider is not None:
            res["repair_provider"] = self.repair_provider
        return res


#: Vendor presets. Selecting one (``UBT_PROVIDER=anthropic``) is enough to point
#: the run at the right endpoint with sane models; the credential is read from
#: the named variable. Model IDs are defaults — override with ``UBT_DRAFT_MODEL``
#: or a ``[providers.<name>]`` block when a vendor ships a newer tier.
BUILTIN_PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "https://api.openai.com/v1",
        "openai-chat",
        "OPENAI_API_KEY",
        "gpt-4o-mini",
        "o3-mini",
        supports_batch_api=True,
    ),
    "anthropic": ProviderSpec(
        "https://api.anthropic.com",
        "anthropic-messages",
        "ANTHROPIC_API_KEY",
        "claude-3-5-haiku",
        "claude-3-7-sonnet",
        supports_batch_api=True,
    ),
    "gemini": ProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini-native",
        "GEMINI_API_KEY",
        "gemini-3.8-flash",
        "gemini-3.1-pro",
    ),
    "deepseek": ProviderSpec(
        "https://api.deepseek.com/v1",
        "openai-chat",
        "DEEPSEEK_API_KEY",
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    "opencode": ProviderSpec(
        "https://opencode.ai/zen/go/v1",
        "openai-responses",
        "OPENCODE_API_KEY",
        "muse-spark-1.3-contributor",
        "muse-spark-1.3-contributor",
        cost_per_mtok=(0.0, 0.0),
        reasoning_dialect="flat",
    ),
}

#: Fields a ``[providers.*]`` / ``[defaults]`` block may set. An explicit
#: allow-list (not ``UBTConfig.model_fields``) keeps this module free of a
#: circular import and, more importantly, keeps a block from reaching a secret
#: or an unrelated knob.
PROVIDER_ALLOWED_KEYS = frozenset(
    {
        "api_key_env",
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
    }
)

#: Never allowed in a block that lands in a version-controlled TOML.
PROVIDER_FORBIDDEN_KEYS = frozenset({"api_key", "service_api_key", "ocr_api_key"})


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
    config_file = find_config_file(custom_path)
    if not config_file:
        return {}
    try:
        with config_file.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        logger.warning("Failed to parse configuration file %s: %s", config_file, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _validate_block(label: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Reject secrets and unknown keys in a TOML block, then expand ``${VAR}``."""
    leaked = sorted(PROVIDER_FORBIDDEN_KEYS & set(fields))
    if leaked:
        raise ProviderConfigError(
            f"{label} must not contain credential field(s) {', '.join(leaked)}: put the "
            "secret in an environment variable (UBT_LLM_API_KEY, or the variable this "
            "provider names in api_key_env). A key committed to a TOML file is a leaked key."
        )
    unknown = sorted(set(fields) - PROVIDER_ALLOWED_KEYS)
    if unknown:
        raise ProviderConfigError(
            f"{label} has unknown field(s) {', '.join(unknown)}; allowed: "
            f"{', '.join(sorted(PROVIDER_ALLOWED_KEYS))}"
        )
    return {str(k): _expand_env_vars(v) for k, v in fields.items()}


def list_providers(custom_path: Path | str | None = None) -> list[str]:
    """Built-in providers plus any declared in the config file, sorted."""
    declared = _read_toml(custom_path).get("providers", {})
    names = set(BUILTIN_PROVIDERS)
    if isinstance(declared, dict):
        names |= {str(k) for k in declared}
    return sorted(names)


def load_provider_block(
    name: str, custom_path: Path | str | None = None
) -> tuple[dict[str, Any], str | None]:
    """Resolve ``name`` to ``(config fields, api_key_env)``.

    A built-in provider is the base; a user ``[providers.<name>]`` block overrides
    or extends it. ``api_key_env`` is returned separately — it names an
    environment variable, it is not a ``UBTConfig`` field.
    """
    data = _read_toml(custom_path)
    declared = data.get("providers", {})
    if not isinstance(declared, dict):
        declared = {}
    if name not in BUILTIN_PROVIDERS and name not in declared:
        available = list_providers(custom_path)
        raise ProviderNotFoundError(
            f"Provider '{name}' not found. Available providers: {available if available else 'none'}"
        )

    fields: dict[str, Any] = {}
    api_key_env: str | None = None
    if name in BUILTIN_PROVIDERS:
        spec = BUILTIN_PROVIDERS[name]
        fields.update(spec.as_fields())
        api_key_env = spec.api_key_env

    user_block = declared.get(name)
    if user_block is not None:
        if not isinstance(user_block, dict):
            raise ProviderConfigError(f"[providers.{name}] must be a table")
        validated = _validate_block(f"[providers.{name}]", user_block)
        if "api_key_env" in validated:
            api_key_env = str(validated.pop("api_key_env"))
        fields.update(validated)
    return fields, api_key_env


def load_defaults_block(custom_path: Path | str | None = None) -> dict[str, Any]:
    """The ``[defaults]`` baseline, applied under every provider block."""
    defaults = _read_toml(custom_path).get("defaults", {})
    if not isinstance(defaults, dict) or not defaults:
        return {}
    validated = _validate_block("[defaults]", defaults)
    validated.pop("api_key_env", None)  # only a provider names a credential variable
    return validated


def load_layer(
    provider_name: str | None, custom_path: Path | str | None = None
) -> tuple[dict[str, Any], str | None]:
    """The ``[defaults]`` baseline with ``[providers.<name>]`` layered over it.

    Returns ``(fields, api_key_env)``. With no provider the defaults stand alone;
    a provider overrides or extends them. One loader, shared by ``from_env`` and
    ``apply_config_overrides`` so both paths see the same block.
    """
    fields: dict[str, Any] = dict(load_defaults_block(custom_path))
    api_key_env: str | None = None
    if provider_name:
        provider_fields, api_key_env = load_provider_block(provider_name, custom_path)
        fields.update(provider_fields)
    return fields, api_key_env


def merge_provider_under(
    explicit: Mapping[str, Any],
    fields: Mapping[str, Any],
    api_key_env: str | None,
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
    same rule ``UBTConfig``'s validator enforces. The credential is taken from
    ``api_key_env`` only when neither an explicit value nor ``UBT_LLM_API_KEY``
    already supplied one.
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
    if "api_key" not in explicit and not os.environ.get("UBT_LLM_API_KEY") and api_key_env:
        value = os.environ.get(api_key_env)
        if value:
            merged["api_key"] = value
    return merged
