"""Base transport and shared HTTP/accounting abstractions for LLM providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import httpx
from pydantic import SecretStr

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.capabilities import ExtractionStrategy
from ubt.core.router.extractor import TranslationOutputExtractor
from ubt.core.router.registry import get_default_registry

logger = logging.getLogger(__name__)


def hostname_of(url: str) -> str:
    """Lowercased hostname of a URL, tolerating a missing scheme.

    Used to tell a self-hosted endpoint from a remote one — never to pick a
    wire protocol: a bare ``"zen" in base_url`` substring check once matched
    ``api.frozen.example.com`` and sent it another family's payload.
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
    """De-noise by extraction strategy."""
    strategy = (
        ExtractionStrategy.AUTO
        if model is None
        else get_default_registry().resolve(model).extraction_strategy
    )
    if strategy == ExtractionStrategy.RAW:
        return raw_text.strip()
    if strategy == ExtractionStrategy.XML_TAG:
        return TranslationOutputExtractor.strip_reasoning(raw_text)
    return TranslationOutputExtractor.extract(raw_text, strategy=strategy)


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
        auth_val = self._api_key
        if not auth_val.lower().startswith("bearer "):
            auth_val = f"Bearer {auth_val}"
        headers = {
            "Authorization": auth_val,
            "Content-Type": "application/json",
            "User-Agent": "universal-book-translator/1.0",
        }
        if self._extra_headers:
            headers.update(self._extra_headers)
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
                limits=self._limits,
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
        return {model: dict(totals) for model, totals in self._model_totals.items()}

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
        return await self._request_json(client, url, retry_payload)

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

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str: ...

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
