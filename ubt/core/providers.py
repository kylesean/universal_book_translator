"""Provider registry: built-in wire protocols plus user-declared ``[providers.*]``.

A *provider* bundles everything about **where** and **how** to call an LLM — the
endpoint, the wire protocol, and the default models. Credential configuration
is specified directly via ``api_key`` (supporting ``${VAR}`` expansion), or via
the generic ``UBT_LLM_API_KEY`` environment variable or CLI flags.

Search order for the config file:
1. explicit path passed by the caller
2. ``ubt.toml`` in the working directory, then in each parent directory
   (so a run started from a subdirectory finds the project's own file)
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


class ProviderNotFoundError(UBTError):
    """A requested provider is neither built-in nor declared in the config file."""


class ProviderConfigError(UBTError):
    """A ``[providers.*]`` / ``[defaults]`` block is malformed or carries a secret."""


#: Wire-protocol presets that ship inside the package. Kept as data, not code,
#: so the registry is read and validated through the very same path as a user's
#: ``[providers.*]`` block.
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
        "is_free",
        "cost_per_mtok",
        # Declared model capability. These describe what the endpoint accepts on
        # the wire, so they belong next to the endpoint and model id that they
        # describe — not in a package-wide table keyed by model-name substrings.
        # ``supports_*`` are tri-state: an absent key means "not declared" and
        # resolves to the conservative fallback, while an explicit ``false`` is a
        # real statement that the endpoint rejects that parameter.
        "supports_temperature",
        "supports_reasoning_effort",
        "supports_system_prompt",
        "supports_vision",
        "extraction_strategy",
        "deployment_backend",
        "reasoning_dialect",
        "repair_provider",
        "qe_judge_provider",
        "ocr_api_key",
        # Run-level, not provider-level, but a real ``UBTConfig`` field: the CLI
        # (``--db-dir``) and ``UBT_DB_DIR`` both set it, so the ``[defaults]``
        # block (documented as the config baseline) must accept it too or the
        # same key is settable on every surface except the config file.
        "db_dir",
        # Which ``[providers.<name>]`` block is in force.
        #
        # Only meaningful in ``[defaults]`` (a provider block naming itself is
        # circular, and ``load_layer`` reads this key from the defaults before it
        # merges a block). Without it the file could describe ten providers and
        # still not say which one runs: selection lived only in ``UBT_PROVIDER``
        # or ``--provider``, so the console could edit a block but never make it
        # the active one — the single most important setting on the page.
        #
        # Precedence is unchanged and still explicit-over-config: an explicit
        # ``--provider`` or ``UBT_PROVIDER`` outranks this key, because
        # ``from_env`` reads the environment *before* the file layer.
        "provider",
        # Run-level quality-gate knobs. Same reasoning as ``db_dir``: the
        # ``[defaults]`` block is documented as the config baseline, so a user
        # who wants the semantic judge on must be able to say so in a file
        # rather than only through an exported environment variable. These are
        # plain ``UBTConfig`` fields, not secrets, and not provider endpoints.
        "qe_engine",
        "qe_judge_enabled",
        "qe_judge_pass_sample",
        "qe_judge_sentences",
        "qe_judge_model",
        # Threshold and repair budget, so a config file can describe a whole
        # quality policy instead of scattering it across the CLI and the env.
        "qe_threshold",
        "max_repair_rounds",
        # Document-level terminology consistency. Set through the config file
        # only — it is a policy decision ("re-translate drifted terms, up to N"),
        # not a per-run experiment. Without these two, a user could not turn the
        # feature on from a file at all: the values exist on ``UBTConfig`` and are
        # read by the stage, but no documented surface could set them.
        "consistency_enforce",
        "consistency_max_repairs",
    }
)

#: Forbidden keys in outbound provider blocks. Inbound service auth uses UBT_API_KEY.
PROVIDER_FORBIDDEN_KEYS = frozenset({"service_api_key"})


def _search_candidates() -> list[Path]:
    """Every path ``find_config_file`` will try, in order, without stat-ing them.

    Shared with :func:`config_search_summary` so the message shown when a lookup
    fails lists exactly what was searched. A diagnostic that has to re-derive the
    list will eventually disagree with the lookup.
    """
    candidates = [Path.cwd() / "ubt.toml"]
    # Walk up from the working directory. A run started from a subdirectory —
    # ``cd web && ubt console``, or any tool that invokes the CLI from inside the
    # tree — otherwise missed the project's own ``ubt.toml`` and fell back to the
    # built-in providers, so ``UBT_PROVIDER=my-local`` failed with "not found"
    # while the file naming it sat one directory up. The walk stops at the
    # filesystem root and never leaves the current tree.
    for parent in Path.cwd().parents:
        candidates.append(parent / "ubt.toml")
    candidates.extend(
        (Path.home() / ".config" / "ubt" / "config.toml", Path.home() / ".ubt" / "config.toml")
    )
    return candidates


def find_config_file(custom_path: Path | str | None = None) -> Path | None:
    """First existing config file in search order, or ``None`` when none exists.

    Order: the explicit path, then ``ubt.toml`` in the working directory, then
    ``ubt.toml`` in each parent directory, then the two user-level locations. The
    parent walk is what makes a run from a subdirectory find the project's own
    configuration instead of silently falling back to the built-ins.
    """
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Configuration file not found: {custom_path}")

    for candidate in _search_candidates():
        try:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved
        except (OSError, RuntimeError):
            continue
    return None


def config_search_summary() -> str:
    """The paths a failed lookup tried, for an error message.

    The old message named only the provider and the built-ins, which reads as "you
    never declared this" — misleading when the declaration exists in a file that
    was not read. Listing the search path turns that into a one-line diagnosis.
    """
    tried = [str(c) for c in _search_candidates()]
    return "searched: " + ", ".join(tried)


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
    :func:`_read_shipped_providers`. Returning ``{}`` here (a silent fail-open)
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
    """The packaged wire-protocol presets, keyed by name, each run through ``_validate_block``.

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
    "claude": "anthropic-messages",
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


def declared_provider_names(custom_path: Path | str | None = None) -> set[str]:
    """Names the operator declared in ``[providers.*]``, excluding shipped ones.

    ``list_providers`` merges three populations — the operator's blocks, the
    shipped wire protocols and the bare aliases — which is right for validation
    ("is this name usable?") but wrong for a configuration screen: it renders a
    seventeen-row list for a six-provider config. This returns just the first
    population.
    """
    declared = _read_toml(custom_path).get("providers", {})
    return {str(k) for k in declared} if isinstance(declared, dict) else set()


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
    user_block = declared.get(name) or declared.get(canonical_name)
    if user_block is None:
        for alias, canon in PROTOCOL_ALIASES.items():
            if canon == canonical_name and alias in declared:
                user_block = declared.get(alias)
                break

    if name not in shipped and canonical_name not in shipped and user_block is None:
        available = list_providers(custom_path)
        # Name the file that WAS read, or say none was found and list the search
        # path. "Available providers: [...]" alone reads as "you never declared
        # this", which is misleading when the declaration exists in a file the
        # process did not open — the failure that produced this message was a run
        # started from a subdirectory, before the search walked up.
        source = find_config_file(custom_path)
        if source is None:
            where = f"no configuration file was found ({config_search_summary()})"
        else:
            where = f"the configuration file read was {source}"
        raise ProviderNotFoundError(
            f"Provider '{name}' not found: {where}. "
            f"Declared providers: {available if available else 'none'}. "
            f"Add a [providers.{name}] block, or set UBT_PROVIDER to one of the above."
        )

    fields: dict[str, Any] = {}

    base_preset = shipped.get(name) or shipped.get(canonical_name)
    if base_preset is not None:
        fields.update(dict(base_preset))

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
