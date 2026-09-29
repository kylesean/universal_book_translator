"""Per-model token price table for token cost accounting.

Prices are USD per 1M tokens as ``(input, output)`` and are matched by longest
prefix so model families resolve without exhaustive enumeration. Unknown models
resolve to ``(0.0, 0.0)`` (price table miss); ``estimate_cost_usd`` reports a
run that touched any unpriced model as None (unknown), not as free.

Self-hosted endpoints are a separate axis from the table. A request that never
left this machine has no provider bill, so its models are *free* regardless of
their names — that is what ``endpoint_is_local`` decides and ``price_is_known``
combines with the table. Keying it on model names instead (the earlier
``ollama/`` / ``localhost`` entries) meant the very names self-hosted users
actually run were billed as if they were cloud models: ``qwen3:8b`` matched the
cloud Qwen rate while ``llama3.2`` matched nothing and reported "unknown",
which refuses a capped run.

Prompt-cache hits are billed at a separate cached input rate where providers
publish one. Models without a published cached rate pay the full input price.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_models_warned_unpriced: set[str] = set()
# Models that answered 200 without a usage block; warned once each for the same
# reason as above (this runs on every priced progress event).
_models_warned_unmeasured: set[str] = set()

#: Cloud Batch API mode is billed at half the interactive rate. Only tokens the
#: batch parser marked (``ModelProvider._record_usage(batch=True)``) are
#: discounted, so repair/QE/judge calls — which always run interactively — keep
#: the full rate. ``ubt assess`` quotes the same constant, keeping the pre-run
#: quote and the runtime cap on one formula.
BATCH_API_DISCOUNT = 0.5


@dataclass(frozen=True)
class PriceEntry:
    """Pricing entry for a model: input, output, cached_input, batch_discount, verified_at."""

    input: float
    output: float
    cached_input: float
    batch_discount: float = 0.5
    verified_at: str = ""


_SHIPPED_PRICES_REGISTRY = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "prices.toml"
)

DEFAULT_PRICES_LOCATIONS: tuple[Path, ...] = (
    Path("prices.toml"),
    Path.home() / ".config" / "ubt" / "prices.toml",
    Path.home() / ".ubt" / "prices.toml",
)


def find_prices_file(custom_path: Path | str | None = None) -> Path | None:
    """Locate the active prices file in search order."""
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Prices file not found: {custom_path}")

    env_path = os.environ.get("UBT_PRICES_FILE")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        logger.warning("UBT_PRICES_FILE points to non-existent file: %s", env_path)

    for candidate in DEFAULT_PRICES_LOCATIONS:
        try:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved
        except (OSError, RuntimeError):
            continue
    return None


def _flatten_prices(raw: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    """Flatten potentially nested tables from dotted TOML keys like [prices.gemini-3.8-flash]."""
    result: dict[str, dict[str, Any]] = {}
    for key, val in raw.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            if "input" in val or "output" in val:
                result[full_key] = val
            else:
                result.update(_flatten_prices(val, prefix=full_key))
    return result


@lru_cache(maxsize=1)
def _read_shipped_prices() -> dict[str, PriceEntry]:
    """Read the bundled ubt/resources/prices.toml once."""
    if not _SHIPPED_PRICES_REGISTRY.is_file():
        return {}
    try:
        with _SHIPPED_PRICES_REGISTRY.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        logger.warning("Failed to parse shipped prices.toml: %s", exc)
        return {}
    raw_prices = data.get("prices", {})
    if not isinstance(raw_prices, dict):
        return {}
    flat_prices = _flatten_prices(raw_prices)
    result: dict[str, PriceEntry] = {}
    for name, block in flat_prices.items():
        if isinstance(block, dict):
            inp = float(block.get("input", 0.0))
            outp = float(block.get("output", 0.0))
            cached = float(block.get("cached_input", inp))
            discount = float(block.get("batch_discount", 0.5))
            v_at = str(block.get("verified_at", ""))
            result[str(name)] = PriceEntry(
                input=inp,
                output=outp,
                cached_input=cached,
                batch_discount=discount,
                verified_at=v_at,
            )
    return result


def _prices_file_signature(custom_path: Path | str | None) -> tuple[str | None, int, int]:
    """Identity of the user prices file that would layer over the shipped table.

    ``(path, mtime_ns, size)`` (or ``(None, 0, 0)`` when there is none) is the
    cache key for :func:`_load_prices_cached`: a rewritten file changes mtime
    and/or size, so the stale overlay is never served.
    """
    custom_file = find_prices_file(custom_path)
    if custom_file is None:
        return (None, 0, 0)
    try:
        st = custom_file.stat()
    except OSError:
        return (None, 0, 0)
    return (str(custom_file), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=8)
def _load_prices_cached(signature: tuple[str | None, int, int]) -> dict[str, PriceEntry]:
    """Shipped table plus the user overlay for one file signature, parsed once."""
    table = dict(_read_shipped_prices())
    path_str = signature[0]
    if not path_str:
        return table
    try:
        with Path(path_str).open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        logger.warning("Failed to parse user prices file %s: %s", path_str, exc)
        return table
    raw = data.get("prices", {})
    if isinstance(raw, dict):
        for name, block in _flatten_prices(raw).items():
            if isinstance(block, dict):
                inp = float(block.get("input", 0.0))
                table[str(name)] = PriceEntry(
                    input=inp,
                    output=float(block.get("output", 0.0)),
                    cached_input=float(block.get("cached_input", inp)),
                    batch_discount=float(block.get("batch_discount", 0.5)),
                    verified_at=str(block.get("verified_at", "")),
                )
    return table


def load_prices_table(custom_path: Path | str | None = None) -> dict[str, PriceEntry]:
    """Load pricing table, layering any user-defined prices.toml over the shipped table.

    Memoised on the resolved file's ``(path, mtime_ns, size)``. ``estimate_cost_usd``
    reaches this several times per model on every priced progress event, and
    re-parsing an unchanged file on each call was pure disk IO; a rewritten file
    gets a new signature and is re-read.
    """
    return dict(_load_prices_cached(_prices_file_signature(custom_path)))


def _current_prices_table() -> dict[str, PriceEntry]:
    """Get the currently active pricing table (shipped + user file if present)."""
    return load_prices_table()


# USD per 1M tokens: (input, output).
#
# Fallback table, NOT the primary source: ``ubt/resources/prices.toml`` is read
# first (step 4 of ``resolve_model_prices``) and shadows any same-key entry
# here (step 5 uses ``len(key) > len(best_key)``, which a same-length key can
# never satisfy). A key that also exists in the TOML is therefore unreachable
# and has been removed. What remains are the models the TOML does not carry,
# kept so a missing/partial resource file degrades to a *priced* run instead
# of an unpriced one. Add new rates to ``ubt/resources/prices.toml``, not here.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # Gemini (the TOML carries the 1.5/2.5/3.1/3.8 families)
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3-flash": (0.10, 0.40),
    "gemini-3-pro": (1.25, 10.00),
    # OpenAI (the TOML carries gpt-4 / gpt-4o / o1 / o3-mini / o4)
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "o3": (2.00, 8.00),
    # Anthropic (the TOML carries the 3.5 / 3.7 / sonnet-4 / opus-4 families)
    "claude-3-haiku": (0.80, 4.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
    "claude-opus": (15.00, 75.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    # Qwen
    "qwen3-mt": (0.50, 1.50),
    "qwen3": (0.50, 1.50),
}

#: Hosts that mean "this machine", or "this machine as seen from a container".
#: ``0.0.0.0`` is included because it is what a bind-all server prints and users
#: paste as a client target; the ``*.internal`` names are the Docker/Podman
#: aliases for the host. Anything else non-loopback — including a private LAN
#: address — is NOT assumed free: see ``UBT_LOCAL_ENDPOINTS``.
_LOCAL_HOSTNAMES = frozenset(
    {"localhost", "0.0.0.0", "host.docker.internal", "host.containers.internal"}
)

#: Comma/``os.pathsep`` separated extra hosts the operator declares self-hosted
#: (a LAN inference box). Read from the environment because this decision is
#: made deep inside pricing, where no config object is in scope.
_LOCAL_ENDPOINTS_ENV = "UBT_LOCAL_ENDPOINTS"
#: The reverse case: a PAID gateway behind loopback (LiteLLM, an OpenCode/Qoder
#: proxy). Setting it makes local endpoints bill against the price table again.
_BILL_LOCAL_ENV = "UBT_BILL_LOCAL_ENDPOINT"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _declared_local_hostnames() -> set[str]:
    """Extra hosts declared self-hosted via ``UBT_LOCAL_ENDPOINTS``."""
    raw = os.environ.get(_LOCAL_ENDPOINTS_ENV, "")
    parts = raw.replace(os.pathsep, ",").split(",")
    return {part.strip().lower() for part in parts if part.strip()}


_CUSTOM_PRICES: dict[str, tuple[float, float]] = {}
_CUSTOM_CACHED_PRICES: dict[str, float] = {}
_CUSTOM_FREE_ENDPOINTS: set[str] = set()


def register_custom_model_pricing(
    model_prefix: str,
    prices: tuple[float, float],
    cached_input: float | None = None,
) -> None:
    """Register or override a custom model pricing entry."""
    norm = model_prefix.strip().lower()
    _CUSTOM_PRICES[norm] = prices
    if cached_input is not None:
        _CUSTOM_CACHED_PRICES[norm] = cached_input


def declare_custom_free_endpoint(base_url: str) -> None:
    """Declare a base_url as an internal free endpoint."""
    if base_url:
        _CUSTOM_FREE_ENDPOINTS.add(base_url.strip().rstrip("/").lower())


def reset_custom_pricing() -> None:
    """Clear all registered custom prices and free endpoints."""
    _CUSTOM_PRICES.clear()
    _CUSTOM_CACHED_PRICES.clear()
    _CUSTOM_FREE_ENDPOINTS.clear()


def endpoint_is_free(base_url: str | None) -> bool:
    """True when base_url has been declared free via declare_custom_free_endpoint."""
    if not base_url:
        return False
    norm = base_url.strip().rstrip("/").lower()
    return any(norm == ep or norm.startswith(ep + "/") for ep in _CUSTOM_FREE_ENDPOINTS)


def billing_enabled_for_local_endpoints() -> bool:
    """True when the operator opted loopback back INTO billing.

    ``UBT_BILL_LOCAL_ENDPOINT`` exists so the common assumption (local = free)
    can be corrected for a paid gateway that happens to listen on 127.0.0.1.
    """
    return os.environ.get(_BILL_LOCAL_ENV, "").strip().lower() in _TRUTHY


def endpoint_is_local(base_url: str | None) -> bool:
    """True when ``base_url`` names an inference endpoint on this machine.

    Loopback (any ``127.0.0.0/8`` or ``::1`` literal), the container-host
    aliases and anything under ``.localhost`` qualify by default. A self-hosted
    box elsewhere on the LAN qualifies only once declared in
    ``UBT_LOCAL_ENDPOINTS``: guessing "a private address is free" would silently
    zero the bill of a paid gateway that happens to live on the LAN, and a
    model's name is never evidence either way.
    """
    if not base_url:
        return False
    if endpoint_is_free(base_url):
        return True
    host = (urlsplit(base_url.strip()).hostname or "").lower()
    if not host:
        return False
    if host in _LOCAL_HOSTNAMES or host.endswith(".localhost"):
        return True
    if host in _declared_local_hostnames():
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Not an IP literal and not a name we were told to trust.
        return False


def model_endpoint(
    model: str, base_url: str | None, endpoint_map: Mapping[str, str] | None
) -> str | None:
    """The endpoint ``model`` was reached through (per-model map wins).

    A run can bill two channels: the LLM at ``base_url`` and, say, cloud OCR at
    ``UBT_OCR_ENDPOINT``. Zeroing the whole run because the LLM is local would
    hide the OCR bill, so the channel each model used decides.
    """
    if endpoint_map and model in endpoint_map:
        return endpoint_map[model]
    return base_url


def price_is_known(
    model: str,
    *,
    base_url: str | None = None,
    endpoint_map: Mapping[str, str] | None = None,
) -> bool:
    """True when this model's spend is a knowable number.

    Either the model has a price-table entry, or it was reached through a
    self-hosted endpoint (a $0 bill by construction). Callers deciding between
    "priced", "free" and "unknown" must use this rather than
    :func:`has_price_entry` alone — conflating free with unknown either refuses
    a self-hosted run or blanks a paid one.
    """
    local = endpoint_is_local(model_endpoint(model, base_url, endpoint_map))
    if local and not billing_enabled_for_local_endpoints():
        return True
    return has_price_entry(model)


# Cached input price (USD per 1M tokens) where the provider publishes
# one. Absent an entry, the full input price applies (no discount assumed).
#
# Same fallback rule as MODEL_PRICES_USD_PER_MTOK: a key that also exists in
# ``ubt/resources/prices.toml`` is shadowed by the TOML (step 4 precedes step 5)
# and was removed; what remains are the models the TOML does not carry.
CACHED_INPUT_PRICES_USD_PER_MTOK: dict[str, float] = {
    # Gemini prompt caching
    "gemini-3-flash": 0.025,
    "gemini-3-pro": 0.3125,
    "gemini-2.0-flash": 0.025,
    # Anthropic publishes a cache-*read* rate of 10% of the input price
    "claude-sonnet-4": 0.30,
    "claude-opus-4": 1.50,
    "claude-3-sonnet": 0.30,
    "claude-3-haiku": 0.08,
    "claude-haiku": 0.08,
    "claude-sonnet": 0.30,
    "claude-opus": 1.50,
    # OpenAI gpt-5
    "gpt-5-mini": 0.0625,
    "gpt-5": 0.3125,
}


def resolve_model_prices(model: str) -> tuple[float, float]:
    """Resolve ``(input, output)`` USD price per 1M tokens.

    Exact match takes precedence over family prefix; longest-prefix match is used
    as fallback so model families resolve without exhaustive enumeration.
    """
    normalized = (model or "").strip().lower()
    candidates = [normalized]
    if "/" in normalized:
        candidates.append(normalized.split("/", 1)[-1])
        last_segment = normalized.rsplit("/", 1)[-1]
        if last_segment not in candidates:
            candidates.append(last_segment)

    # 1. Exact match in runtime custom prices:
    for cand in candidates:
        if cand in _CUSTOM_PRICES:
            return _CUSTOM_PRICES[cand]

    table = _current_prices_table()

    # 2. Exact match in table:
    for cand in candidates:
        if cand in table:
            entry = table[cand]
            return (entry.input, entry.output)

    # 3. Longest-prefix match in custom prices:
    best_key = ""
    best_prices = (0.0, 0.0)
    for cand in candidates:
        for key, prices in _CUSTOM_PRICES.items():
            if cand.startswith(key) and len(key) > len(best_key):
                best_key = key
                best_prices = prices

    # 4. Longest-prefix match in table:
    for cand in candidates:
        for key, entry in table.items():
            if cand.startswith(key) and len(key) > len(best_key):
                best_key = key
                best_prices = (entry.input, entry.output)

    # 5. Longest-prefix fallback in legacy MODEL_PRICES_USD_PER_MTOK:
    for cand in candidates:
        for key, prices in MODEL_PRICES_USD_PER_MTOK.items():
            if cand.startswith(key) and len(key) > len(best_key):
                best_key = key
                best_prices = prices

    if not best_key:
        logger.debug("No price table entry for model %r; cost will report as 0", model)
    return best_prices


def resolve_cached_input_price(model: str) -> float:
    """Resolve the cached input rate (USD per 1M tokens).

    Exact match takes precedence over family prefix; falls back to the model's
    full input price when the provider does not publish a separate cached rate.
    The result is capped at the model's own input price: a cache read is never
    dearer than a fresh read, and without the cap a longest-prefix match can
    inherit a *parent* entry's cached rate (``gpt-4.1`` matching ``gpt-4``, whose
    ``cached_input`` defaults to its input 30.0) so a hit cost up to 300x a miss.
    """
    normalized = (model or "").strip().lower()
    candidates = [normalized]
    if "/" in normalized:
        candidates.append(normalized.split("/", 1)[-1])
        last_segment = normalized.rsplit("/", 1)[-1]
        if last_segment not in candidates:
            candidates.append(last_segment)

    # 1. Exact match in runtime custom cached prices:
    for cand in candidates:
        if cand in _CUSTOM_CACHED_PRICES:
            return _CUSTOM_CACHED_PRICES[cand]

    table = _current_prices_table()

    # 2. Exact match in table:
    for cand in candidates:
        if cand in table:
            return table[cand].cached_input

    # 3. Longest-prefix match in custom cached prices:
    best_key = ""
    best_cached = -1.0
    for cand in candidates:
        for key, cached_price in _CUSTOM_CACHED_PRICES.items():
            if cand.startswith(key) and len(key) > len(best_key):
                best_key = key
                best_cached = cached_price

    # 4. Longest-prefix match in table:
    for cand in candidates:
        for key, entry in table.items():
            if cand.startswith(key) and len(key) > len(best_key):
                best_key = key
                best_cached = entry.cached_input

    # 5. Longest-prefix fallback in legacy CACHED_INPUT_PRICES_USD_PER_MTOK:
    for cand in candidates:
        for key, cached_price in CACHED_INPUT_PRICES_USD_PER_MTOK.items():
            if cand.startswith(key) and len(key) > len(best_key):
                best_key = key
                best_cached = cached_price

    input_price, _ = resolve_model_prices(model)
    if best_cached >= 0.0:
        # Cap at the model's own input price: the longest-prefix match may have
        # picked a parent entry whose cached rate is the parent's (higher) input,
        # which made a cache hit dearer than a miss and tripped UBT_BUDGET_USD
        # early. ``min`` also guards an exact entry with a bad cached > input.
        return min(best_cached, input_price)
    return input_price


def has_price_entry(model: str) -> bool:
    """True when the model resolves to an explicit price-table entry.

    This answers a model-**name** question only. A self-hosted model is usually
    absent from the table and still free, so anything deciding whether a run's
    spend is knowable must use :func:`price_is_known` instead. An explicit 0.0
    entry (shipped in the table, or registered by a provider's ``cost_per_mtok``)
    does count as priced: only a prefix-match miss is unknown pricing.
    """
    normalized = (model or "").strip().lower()
    candidates = [normalized]
    if "/" in normalized:
        candidates.append(normalized.split("/", 1)[-1])
        last_segment = normalized.rsplit("/", 1)[-1]
        if last_segment not in candidates:
            candidates.append(last_segment)
    table = _current_prices_table()
    merged_keys = set(table) | set(MODEL_PRICES_USD_PER_MTOK) | set(_CUSTOM_PRICES)
    return any(cand.startswith(key) for cand in candidates for key in merged_keys)


def estimate_cost_usd(
    model_totals: dict[str, dict[str, int]],
    *,
    base_url: str | None = None,
    endpoint_map: Mapping[str, str] | None = None,
) -> float | None:
    """Estimate cumulative USD cost from per-model token totals.

    ``model_totals`` maps model name -> ``{"prompt_tokens": int,
    "completion_tokens": int, "cached_tokens": int, ...}`` (see
    ``usage_totals_by_model``). prompt tokens reported as cache hits
    are billed at the cached input rate instead of the full input rate.
    Cache *writes* (Anthropic ``cache_creation_input_tokens``) are counted in
    ``prompt_tokens`` and billed here at the full input rate; the published
    1.25x creation premium is not modelled, so the figure stays an estimate.

    ``base_url`` is the endpoint the run reached its models through, and
    ``endpoint_map`` overrides it per model (a local LLM plus cloud OCR bill two
    channels). Models reached through a self-hosted endpoint cost $0 — known,
    not unknown — unless ``UBT_BILL_LOCAL_ENDPOINT`` says otherwise.

    Returns None when any model that actually consumed tokens has no knowable
    price: silently summing those models at $0 makes the report's only money
    figure systematically false ($0.00000 for a paid default model). Unknown
    beats wrong.

    Returns None for the same reason when a model made calls whose response
    carried no ``usage`` block (``unmeasured_calls``): those calls were billed
    by the provider but cannot be sized here, so summing the measured ones
    would understate the job — and, worse, would let UBT_BUDGET_USD look
    satisfied on spend it never saw.
    """
    total_cost = 0.0
    unmeasured = sorted(
        model
        for model, totals in model_totals.items()
        if int(totals.get("unmeasured_calls", 0) or 0)
    )
    if unmeasured:
        novel = [model for model in unmeasured if model not in _models_warned_unmeasured]
        if novel:
            logger.warning(
                "Model(s) %s answered without a usage block — their spend is "
                "unmeasurable, so token cost reports as unknown rather than a "
                "fake $0.00 (and UBT_BUDGET_USD cannot bound them)",
                ", ".join(novel),
            )
            _models_warned_unmeasured.update(novel)
        return None
    unpriced = [
        model
        for model, totals in model_totals.items()
        if (
            int(totals.get("prompt_tokens", 0) or 0) or int(totals.get("completion_tokens", 0) or 0)
        )
        and not price_is_known(model, base_url=base_url, endpoint_map=endpoint_map)
    ]
    if unpriced:
        # One line per model, not per call: this runs on every priced progress
        # event, so an unbounded warning would add a line per block for a book
        # whose model has no price entry.
        novel = [model for model in unpriced if model not in _models_warned_unpriced]
        if novel:
            logger.warning(
                "No price-table entry for model(s) %s — token cost reports as "
                "unknown; add the rate to ubt/resources/prices.toml (legacy "
                "fallback: MODEL_PRICES_USD_PER_MTOK)",
                ", ".join(sorted(novel)),
            )
            _models_warned_unpriced.update(novel)
        return None
    for model, totals in model_totals.items():
        if (
            endpoint_is_local(model_endpoint(model, base_url, endpoint_map))
            and not billing_enabled_for_local_endpoints()
        ):
            # Self-hosted: no provider bill to estimate. Skipping (rather than
            # pricing at the table's (0, 0)) also keeps a local `qwen3:8b` off
            # the cloud Qwen rate its name prefix would otherwise match.
            continue
        input_price, output_price = resolve_model_prices(model)
        cached_input_price = resolve_cached_input_price(model)
        prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
        completion_tokens = int(totals.get("completion_tokens", 0) or 0)
        cached_tokens = min(int(totals.get("cached_tokens", 0) or 0), prompt_tokens)
        # Batch-served tokens are a *subset* of the totals above, not an
        # addition: they were recorded with ``batch=True`` and bill at half
        # rate. Splitting them out here (rather than discounting the whole
        # model) is what keeps a repair/QE call on the same model at full
        # price — without it the runtime overstated batch runs ~2x and tripped
        # UBT_BUDGET_USD at roughly half the book.
        batch_prompt = min(int(totals.get("batch_prompt_tokens", 0) or 0), prompt_tokens)
        batch_completion = min(
            int(totals.get("batch_completion_tokens", 0) or 0), completion_tokens
        )
        interactive_prompt = prompt_tokens - batch_prompt
        interactive_completion = completion_tokens - batch_completion
        # Cache hits are attributed to the interactive side first (the Batch
        # API rarely reports them); any excess spills onto the batch side.
        interactive_cached = min(cached_tokens, interactive_prompt)
        batch_cached = min(cached_tokens - interactive_cached, batch_prompt)
        interactive_cost = (
            (interactive_prompt - interactive_cached) * input_price
            + interactive_cached * cached_input_price
            + interactive_completion * output_price
        )
        batch_cost = (
            (batch_prompt - batch_cached) * input_price
            + batch_cached * cached_input_price
            + batch_completion * output_price
        )
        total_cost += (interactive_cost + batch_cost * BATCH_API_DISCOUNT) / 1_000_000
    return round(total_cost, 6)


def cache_hit_rate_from_usage(model_totals: dict[str, dict[str, int]]) -> float:
    """Prompt-cache hit rate for one usage snapshot (0.0 when unmeasured).

    Shared by the progress event and the quality report so both derive the
    headline TCO number from the same snapshot. Reporting a *run-scoped*
    delta matters whenever one router serves several jobs (the API keeps a
    single process-wide router): the provider counters are cumulative, so
    charging a job the router's lifetime totals over-reports it.
    """
    prompt_tokens = sum(int(t.get("prompt_tokens", 0) or 0) for t in model_totals.values())
    if prompt_tokens <= 0:
        return 0.0
    cached_tokens = sum(int(t.get("cached_tokens", 0) or 0) for t in model_totals.values())
    return round(min(cached_tokens / prompt_tokens, 1.0), 4)
