"""Base transport and shared HTTP/accounting abstractions for LLM providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import httpx
from pydantic import SecretStr

from ubt.core.config import MOCK_API_KEY
from ubt.core.exceptions import ModelProviderError
from ubt.core.router.extractor import TranslationOutputExtractor
from ubt.core.router.pricing import endpoint_is_local

logger = logging.getLogger(__name__)

#: Per-phase HTTP timeouts. Connect/write/pool fail fast (a dead endpoint or a
#: stalled upload should not pin a pool slot), while ``read`` carries the long
#: window a large LLM completion needs. These are the values the pipeline
#: forces on every provider client; ``read`` falls back to the configured
#: ``api_timeout`` only when that is *larger*, so an operator raising the read
#: budget is honoured while a bare float can no longer disable the split.
_CONNECT_TIMEOUT_S = 15.0
_WRITE_TIMEOUT_S = 15.0
_POOL_TIMEOUT_S = 30.0
#: Default read window; the configured ``api_timeout`` overrides it upward.
_READ_TIMEOUT_S = 120.0


def build_timeout(api_timeout: float | None = None) -> httpx.Timeout:
    """The forced per-phase timeout for every provider HTTP client.

    ``connect``/``write``/``pool`` are fixed (fail fast); ``read`` is the larger
    of the default and the configured API timeout, so a long generation is not
    cut off while a dead endpoint still fails in seconds.
    """
    read = _READ_TIMEOUT_S
    if api_timeout is not None and api_timeout > read:
        read = float(api_timeout)
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT_S,
        read=read,
        write=_WRITE_TIMEOUT_S,
        pool=_POOL_TIMEOUT_S,
    )


def retry_after_seconds(headers: httpx.Headers) -> str | None:
    """``Retry-After`` in seconds, normalizing a millisecond ``Retry-After-Ms``.

    RFC 7231 ``Retry-After`` is delta-seconds or an HTTP-date, which the router
    parses as seconds. Some gateways (Azure OpenAI, one-api) instead send
    ``Retry-After-Ms`` in milliseconds; the millisecond value must be divided
    here or a 20s pause becomes a 20000s sleep.
    """
    value = headers.get("retry-after")
    if value:
        return str(value)
    ms = headers.get("retry-after-ms")
    if not ms:
        return None
    try:
        return str(float(ms) / 1000.0)
    except (TypeError, ValueError):
        return None


def raise_rate_limited(response: httpx.Response) -> None:
    """Raise the 429 error every transport raises identically.

    Four transports carried byte-identical blocks for this (same message, same
    ``details`` shape); a fix to one -- a new detail key, a wording change --
    silently skipped the other three. The router's backoff reads only
    ``retry_after`` and ``status_code``, so one owner keeps them in step.
    """
    raise ModelProviderError(
        f"Rate limit exceeded (HTTP 429): {response.text[:500]}",
        details={
            "status_code": 429,
            "retry_after": retry_after_seconds(response.headers),
            "body": response.text[:2000],
        },
    )


def raise_api_error(response: httpx.Response) -> None:
    """Raise the non-429 error status every transport raises identically.

    Same duplication history as :func:`raise_rate_limited`: one block, four
    copies, drifting one fix at a time.
    """
    raise ModelProviderError(
        f"Model API error ({response.status_code}): {response.text[:500]}",
        details={"status_code": response.status_code, "body": response.text[:2000]},
    )


def raise_for_status(response: httpx.Response) -> None:
    """The shared status gate: 429 by its own owner, any other failure by the generic one."""
    if response.status_code == 429:
        raise_rate_limited(response)
    if response.status_code != 200:
        raise_api_error(response)


def hostname_of(url: str) -> str:
    """Lowercased hostname of a URL, tolerating a missing scheme.

    Tells a self-hosted endpoint from a remote one — never picks a wire
    protocol: a bare ``"zen" in base_url`` substring check would match
    ``api.frozen.example.com`` and send it another family's payload.
    """
    from urllib.parse import urlsplit

    candidate = (url or "").strip()
    if "://" not in candidate:
        candidate = "https://" + candidate
    try:
        return (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        return ""


# Usage attribution for concurrent jobs. A long-lived API server drives several
# PipelineOrchestrators over ONE provider instance, so process-wide counters
# cannot say which job spent which tokens; a run attaches a private dict here
# and every call made in that task's context accumulates into it.
_usage_sink: ContextVar[dict[str, dict[str, int]] | None] = ContextVar(
    "ubt_usage_sink", default=None
)


def new_usage_totals() -> dict[str, int]:
    """Zeroed per-model counter set — the shape every usage view shares."""
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}


def _accumulate(
    totals: dict[str, int],
    prompt: int,
    completion: int,
    cached: int,
    *,
    batch_prompt: int = 0,
    batch_completion: int = 0,
    unmeasured: bool = False,
) -> None:
    totals["calls"] += 1
    totals["prompt_tokens"] += prompt
    totals["completion_tokens"] += completion
    totals["cached_tokens"] += cached
    if batch_prompt or batch_completion:
        totals["batch_prompt_tokens"] = totals.get("batch_prompt_tokens", 0) + batch_prompt
        totals["batch_completion_tokens"] = (
            totals.get("batch_completion_tokens", 0) + batch_completion
        )
    if unmeasured:
        totals["unmeasured_calls"] = totals.get("unmeasured_calls", 0) + 1


def attach_usage_sink() -> dict[str, dict[str, int]]:
    """Attach a fresh usage sink to the current task context and return it."""
    sink: dict[str, dict[str, int]] = {}
    _usage_sink.set(sink)
    return sink


def record_external_usage(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    measured: bool = True,
) -> bool:
    """Attribute usage from a channel that bypasses ``_record_usage``."""
    sink = _usage_sink.get()
    if sink is None:
        return False
    _accumulate(
        sink.setdefault(model, new_usage_totals()),
        int(prompt_tokens or 0),
        int(completion_tokens or 0),
        int(cached_tokens or 0),
        unmeasured=not measured,
    )
    return True


def _extract_cached_tokens(usage: dict[str, Any]) -> int:
    """Cache-hit input tokens from any known provider usage shape."""
    if "cache_read_input_tokens" in usage and usage.get("cache_read_input_tokens") is not None:
        return int(usage["cache_read_input_tokens"] or 0)
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens"):
        return int(details["cached_tokens"] or 0)
    flat = usage.get("prompt_cache_hit_tokens")
    if flat:
        return int(flat or 0)
    in_details = usage.get("input_tokens_details")
    if isinstance(in_details, dict) and in_details.get("cached_tokens"):
        return int(in_details["cached_tokens"] or 0)
    return 0


def sanitize_thought_output(raw_text: str, model: str | None = None) -> str:
    """Strip reasoning traces, code fences and conversational lead-ins.

    Always ``AUTO``: the transport's job is to hand the router clean text, and the
    router then applies the *declared* extraction strategy from the resolved
    capability profile (``router.draft`` → ``TranslationOutputExtractor.extract``).
    Doing the authoritative extraction here as well would apply a second, weaker
    rule to text the profile had already shaped.

    This used to resolve a per-model strategy out of the global registry, which
    was dead weight in both directions: the built-in table carried ``auto`` for
    every one of its entries, so the lookup could only ever return the default,
    and once that table was removed the lookup could only ever return the
    fallback. ``model`` is kept in the signature because callers pass it and a
    future profile-aware transport will want it.
    """
    return TranslationOutputExtractor.strip_reasoning(raw_text)


_EFFORT_REJECTION_KEYS = (
    "reasoning_effort",
    "thinking",
    "does not support",
    "unrecognized",
    "extra_forbidden",
    "unknown",
)

_CHAT_TEMPERATURE_KEYS = ("temperature", "unsupported parameter")
_ANTHROPIC_TEMPERATURE_KEYS = ("temperature", "thinking", "unsupported parameter")

#: A 400 self-heal rule: given the lowercased error body and the payload that
#: provoked it, return a cleaned payload or None to leave the response alone.
_HealFix = Callable[[str, dict[str, Any]], "dict[str, Any] | None"]


def _heal_reasoning_effort(err_text: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Chat-completions transport: clamp or drop a rejected reasoning_effort."""
    if "must be one of" in err_text and ("reasoning_effort" in err_text or "effort" in err_text):
        fixed = dict(payload)
        fixed["reasoning_effort"] = "low"
        return fixed
    if "reasoning_effort" in payload and any(k in err_text for k in _EFFORT_REJECTION_KEYS):
        fixed = dict(payload)
        fixed.pop("reasoning_effort", None)
        return fixed
    return None


def _heal_drop_temperature(*keys: str) -> _HealFix:
    """Build the 'endpoint has no temperature' cleaner for one transport's key set."""

    def fix(err_text: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if "temperature" in payload and any(k in err_text for k in keys):
            fixed = dict(payload)
            fixed.pop("temperature", None)
            return fixed
        return None

    return fix


_ANTHROPIC_THINKING_KEYS = ("thinking", "budget_tokens")


def _heal_drop_thinking(*keys: str) -> _HealFix:
    """Build the 'endpoint has no thinking/reasoning' cleaner for Anthropic/Claude."""

    def fix(err_text: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if "thinking" in payload and any(k in err_text for k in keys):
            fixed = dict(payload)
            fixed.pop("thinking", None)
            return fixed
        return None

    return fix


#: Error phrases an endpoint uses when it wants ``max_completion_tokens``
#: instead of the legacy ``max_tokens``. OpenAI rejects ``max_tokens`` on the
#: gpt-5 family (and o-series) with a 400 that names the replacement.
_MAX_TOKENS_REPLACEMENT_PHRASES = (
    "not supported",
    "unsupported parameter",
    "use max_completion_tokens",
)


def _heal_max_tokens_rename(err_text: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Chat-completions transport: rename a rejected ``max_tokens``.

    The gpt-5 / o-series endpoints advertise their pricing in this project's
    table but reject ``max_tokens``, so a run against them failed every call with
    a permanent 400. The replacement parameter the error names is
    ``max_completion_tokens``; apply it so those models are usable.
    """
    if "max_tokens" not in payload:
        return None
    if "max_completion_tokens" in err_text and any(
        p in err_text for p in _MAX_TOKENS_REPLACEMENT_PHRASES
    ):
        fixed = dict(payload)
        value = fixed.pop("max_tokens", None)
        if value is not None:
            fixed["max_completion_tokens"] = value
        return fixed
    return None


class BaseTransport(ABC):
    """Base transport providing connection pooling, metrics, and auth headers."""

    is_mock: bool = False

    def __init__(
        self,
        api_key: str | SecretStr,
        base_url: str,
        default_model: str = "",
        timeout: float = 60.0,
        provider_name: str = "base_transport",
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
        limits: httpx.Limits | None = None,
        extra_headers: dict[str, str] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        sanitize_output: bool = True,
        prompt_caching: bool = True,
        usage_log: list[dict[str, Any]] | None = None,
        cumulative_totals: dict[str, int] | None = None,
        model_totals: dict[str, dict[str, int]] | None = None,
    ) -> None:
        if isinstance(api_key, SecretStr) or hasattr(api_key, "get_secret_value"):
            self._api_key = api_key.get_secret_value().strip()
        else:
            self._api_key = str(api_key).strip()
        self._base_url = self._clean_base_url(base_url)
        self._default_model = default_model
        self._timeout = timeout
        self._name = provider_name
        self._transport = transport
        self._client: httpx.AsyncClient | None = client
        self._owned_client = client is None
        self._limits = limits or httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        self._extra_headers = dict(extra_headers or {})
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._sanitize_output = sanitize_output
        # Parameters this endpoint rejected with a 400 and a retry proved
        # unnecessary, keyed by model id. Instance-scoped on purpose: the
        # evidence is about *this* endpoint, and a shared cache would let one
        # gateway's rejection strip a parameter from a different one's calls.
        self._learned_omissions: dict[str, frozenset[str]] = {}
        # NOTE: only the Anthropic transport reads this. The OpenAI
        # chat/responses transports leave it inert — real OpenAI caches a static
        # prefix automatically, and a compat server that does not is simply not
        # cache-optimised. Kept on the base so providers share one constructor.
        self._prompt_caching = prompt_caching

        # Per-call token accounting (can be shared with parent provider)
        self.usage_log: list[dict[str, Any]] = usage_log if usage_log is not None else []
        self._cumulative_totals: dict[str, int] = (
            cumulative_totals if cumulative_totals is not None else new_usage_totals()
        )
        self._model_totals: dict[str, dict[str, int]] = (
            model_totals if model_totals is not None else {}
        )

    @classmethod
    def _clean_base_url(cls, base_url: str) -> str:
        base = base_url.rstrip("/")
        for suffix in ("/responses", "/chat/completions", "/chat", "/messages"):
            if base.endswith(suffix):
                base = base[: -len(suffix)].rstrip("/")
        return base

    @property
    def provider_name(self) -> str:
        return self._name

    def _resolve_model(self, model: str | None) -> str:
        """The model to call: the explicit argument, else this transport's default.

        Fail-closed: a blank model is a configuration error, not a request to
        send ``"model": ""`` and collect a 400 from the vendor.
        """
        target = (model or self._default_model or "").strip()
        if not target:
            raise ModelProviderError(
                "No model configured: pass model= or set UBT_DRAFT_MODEL / the "
                "provider block's draft_model."
            )
        return target

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        return dict(self._chat_template_kwargs)

    @property
    def supports_batch_api(self) -> bool:
        return False

    def _auth_headers(self) -> dict[str, str]:
        # A loopback endpoint authenticates nothing: llama.cpp, Ollama and
        # llama-swap accept any bearer, and this project's own default config
        # points at one with no credential at all. Sending ``Bearer mock-key``
        # to it is not harmless — the operator's own config carries a comment
        # about exactly that failure ("空值会产生非法请求头 'Bearer '（尾随空格），
        # 请求在到达网关前就失败"). Omitting the header entirely is what a local
        # server expects, and it is what makes a keyless local run possible.
        #
        # The sentinel is checked here as well as the empty string: ``UBTConfig``
        # fills an unconfigured key with ``MOCK_API_KEY``, so "unconfigured"
        # arrives as the literal string ``mock-key`` rather than as ``""``. A
        # transport that only tested truthiness would put it on the wire.
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "universal-book-translator/1.0",
        }
        if self._api_key and self._api_key != MOCK_API_KEY:
            auth_val = self._api_key
            if not auth_val.lower().startswith("bearer "):
                auth_val = f"Bearer {auth_val}"
            headers["Authorization"] = auth_val
        if self._extra_headers:
            headers.update(self._extra_headers)
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                # Per-phase timeout: fail fast on connect/write/pool, long read.
                timeout=build_timeout(self._timeout),
                transport=self._transport,
                limits=self._limits,
                # Local inference servers are reached directly; inherited
                # SOCKS proxies are for remote providers only.
                trust_env=not endpoint_is_local(self._base_url),
            )
            self._owned_client = True
        return self._client

    async def aclose(self) -> None:
        if self._owned_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self) -> BaseTransport:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    @property
    def usage_totals(self) -> dict[str, int]:
        return dict(self._cumulative_totals)

    @property
    def cache_hit_rate(self) -> float:
        prompt = self._cumulative_totals.get("prompt_tokens", 0)
        if prompt <= 0:
            return 0.0
        cached = self._cumulative_totals.get("cached_tokens", 0)
        return round(min(cached / prompt, 1.0), 4)

    @property
    def usage_totals_by_model(self) -> dict[str, dict[str, int]]:
        return {model: dict(totals) for model, totals in list(self._model_totals.items())}

    def begin_usage_sink(self) -> dict[str, dict[str, int]]:
        return attach_usage_sink()

    def _record_usage(
        self,
        target_model: str,
        usage_entry: dict[str, Any],
        *,
        batch: bool = False,
        unmeasured: bool = False,
    ) -> None:
        prompt_toks = int(usage_entry.get("prompt_tokens", 0) or 0)
        comp_toks = int(usage_entry.get("completion_tokens", 0) or 0)
        cached_toks = int(usage_entry.get("prompt_cache_hit_tokens", 0) or 0)
        batch_prompt = prompt_toks if batch else 0
        batch_completion = comp_toks if batch else 0
        _accumulate(
            self._cumulative_totals,
            prompt_toks,
            comp_toks,
            cached_toks,
            batch_prompt=batch_prompt,
            batch_completion=batch_completion,
            unmeasured=unmeasured,
        )

        model_totals = self._model_totals.setdefault(target_model, new_usage_totals())
        _accumulate(
            model_totals,
            prompt_toks,
            comp_toks,
            cached_toks,
            batch_prompt=batch_prompt,
            batch_completion=batch_completion,
            unmeasured=unmeasured,
        )

        sink = _usage_sink.get()
        if sink is not None:
            _accumulate(
                sink.setdefault(target_model, new_usage_totals()),
                prompt_toks,
                comp_toks,
                cached_toks,
                batch_prompt=batch_prompt,
                batch_completion=batch_completion,
                unmeasured=unmeasured,
            )

        self.usage_log.append(usage_entry)
        if len(self.usage_log) > 1000:
            # Truncate IN PLACE: the list is shared with the provider and the
            # other transports (passed via ``shared_kw``), so rebinding it here
            # detached this transport and stopped the others from receiving
            # entries.
            del self.usage_log[: len(self.usage_log) - 500]

    def _finalize_output(self, text: str, model: str | None = None) -> str:
        return sanitize_thought_output(text, model) if self._sanitize_output else text.strip()

    async def _request_json(
        self, client: httpx.AsyncClient, url: str, payload: dict[str, Any]
    ) -> httpx.Response:
        try:
            return await client.post(url, headers=self._auth_headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise ModelProviderError(f"Request timed out after {self._timeout}s: {exc}") from exc
        except httpx.RequestError as exc:
            raise ModelProviderError(f"HTTP request error: {exc}") from exc

    def _learned_omissions_for(self, model: Any) -> frozenset[str]:
        """Parameter names this endpoint already rejected for ``model``."""
        if not isinstance(model, str) or not model:
            return frozenset()
        return self._learned_omissions.get(model, frozenset())

    def _apply_learned_omissions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop parameters a previous 400 proved this endpoint rejects.

        Without this, every block paid the same failed round-trip: the heal rules
        fixed the *retry* but nothing remembered the outcome, so a model that
        rejects ``temperature`` (the gpt-5 / o-series endpoints do) sent it again
        on the next block and ate another 400. Measured on a 137-block job that is
        137 wasted requests.
        """
        omitted = self._learned_omissions_for(payload.get("model"))
        if not omitted:
            return payload
        return {key: value for key, value in payload.items() if key not in omitted}

    def _remember_rejected_parameters(self, before: dict[str, Any], after: dict[str, Any]) -> None:
        """Record parameters a successful heal had to remove, keyed by model.

        Only keys present in ``before`` and absent from ``after`` count, so the
        clamp path (``reasoning_effort: high → low``) is not recorded — the
        parameter is still accepted, just narrowed.
        """
        model = before.get("model")
        if not isinstance(model, str) or not model:
            return
        removed = {key for key in before if key not in after}
        # ``max_tokens`` is renamed rather than removed, and the nested
        # ``reasoning`` object is dialect-specific: neither is a plain omission.
        removed -= {"model", "max_tokens", "reasoning"}
        if not removed:
            return
        self._learned_omissions[model] = frozenset(
            set(self._learned_omissions.get(model, frozenset())) | removed
        )

    async def _self_heal_400(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        response: httpx.Response,
        fixes: tuple[_HealFix, ...],
    ) -> httpx.Response:
        if response.status_code != 400:
            return response
        err_text = response.text.lower()
        retry_payload = payload
        for fix in fixes:
            fixed = fix(err_text, retry_payload)
            if fixed is not None:
                retry_payload = fixed
        if retry_payload is payload:
            return response
        healed = await self._request_json(client, url, retry_payload)
        # Learn only on a proven outcome: an unrelated 400 fails every retry
        # identically, and recording its removals would silently disable a
        # parameter the endpoint actually accepts.
        if healed.status_code == 200:
            self._remember_rejected_parameters(payload, retry_payload)
        return healed

    async def _post_json(
        self, client: httpx.AsyncClient, url: str, payload: dict[str, Any]
    ) -> httpx.Response:
        try:
            response = await client.post(url, headers=self._auth_headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise ModelProviderError(
                f"Vision request timed out after {self._timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise ModelProviderError(f"HTTP request error: {exc}") from exc
        if response.status_code != 200:
            raise ModelProviderError(
                f"Model API error ({response.status_code}): {response.text[:500]}",
                details={"status_code": response.status_code, "body": response.text[:2000]},
            )
        return response

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Text only; the finish reason is dropped (see the sibling method)."""
        text, _ = await self.generate_with_finish_reason(
            prompt, system_prompt, model, temperature, max_tokens, reasoning_effort
        )
        return text

    @abstractmethod
    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        text = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return text, None

    async def generate_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError(f"{self.provider_name} does not support vision input")

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        raise NotImplementedError(f"{self.provider_name} does not support the Batch API")

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        raise NotImplementedError(f"{self.provider_name} does not support the Batch API")

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        raise NotImplementedError(f"{self.provider_name} does not support the Batch API")

    async def cleanup_batch_files(self, batch_id: str) -> None:
        return None

    async def cancel_batch_job(self, batch_id: str) -> None:
        return None
