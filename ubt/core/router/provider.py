"""Model provider interfaces and standard implementations.

The transports live in the ubt.core.router.transports package; this module
keeps the provider surface callers and test doubles depend on.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TypedDict

import httpx
from pydantic import SecretStr

from ubt.core.router.transports.anthropic import AnthropicMessagesTransport
from ubt.core.router.transports.base import (
    _CONNECT_TIMEOUT_S,
    BaseTransport,
    _extract_cached_tokens,
    _usage_sink,
    attach_usage_sink,
    new_usage_totals,
    record_external_usage,
    sanitize_thought_output,
)
from ubt.core.router.transports.gemini import GeminiTransport
from ubt.core.router.transports.openai_chat import OpenAIChatTransport
from ubt.core.router.transports.openai_responses import OpenAIResponsesTransport

logger = logging.getLogger(__name__)

# Re-export key transport functions so imports from ubt.core.router.provider remain valid
__all__ = [
    "BaseModelProvider",
    "MockModelProvider",
    "OpenAICompatibleProvider",
    "create_model_provider",
    "attach_usage_sink",
    "new_usage_totals",
    "record_external_usage",
    "sanitize_thought_output",
    "_extract_cached_tokens",
    "_usage_sink",
]


class BaseModelProvider(ABC):
    """Abstract interface for LLM inference providers."""

    is_mock: bool = False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for the model provider."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate translated text response from the model."""
        ...

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        """Generate text plus the provider finish reason."""
        text = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return text, None

    @property
    def supports_batch_api(self) -> bool:
        """Whether this provider can execute OpenAI-compatible batch jobs."""
        return False

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        """Chat-template flags merged into every chat/completions payload."""
        return {}

    @property
    def base_url(self) -> str:
        """Endpoint this provider talks to ("" when it has no URL).

        Read by cost accounting to tell a self-hosted endpoint (free by
        construction) from a paid one. A provider that does not know its URL
        answers "" rather than guessing.
        """
        return ""

    async def generate_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Vision completion over base64 PNGs (default: unsupported)."""
        raise NotImplementedError(f"{self.provider_name} does not support vision input")

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        """Upload batch JSONL requests and create a batch job (returns its ID)."""
        raise NotImplementedError(f"{self.provider_name} does not support the Batch API")

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        """Fetch batch job metadata (status and output/error file ids)."""
        raise NotImplementedError(f"{self.provider_name} does not support the Batch API")

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        """Download batch output mapped by ``custom_id``."""
        raise NotImplementedError(f"{self.provider_name} does not support the Batch API")

    async def cleanup_batch_files(self, batch_id: str) -> None:
        """Delete the Files-API objects a batch left on the provider."""
        return None

    async def cancel_batch_job(self, batch_id: str) -> None:
        """Best-effort cancel of a still-running batch so it stops accruing cost."""
        return None


class MockModelProvider(BaseModelProvider):
    """Deterministic mock provider for offline tests and simulations.

    A production class with production jobs — ``--dry-run`` rehearsal, the API's
    mock mode and the credential-free cost assessment all assemble a router
    around it.
    """

    is_mock = True

    def __init__(
        self,
        default_response: str = "[TRANSLATED]",
        custom_responses: dict[str, str] | None = None,
        prefix: str | None = None,
        sanitize_output: bool = False,
    ) -> None:
        self._default = default_response
        self._custom = custom_responses or {}
        self._prefix = prefix
        self._sanitize_output = sanitize_output
        self.call_history: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        self.call_history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
        )
        res = self._default
        for key, val in self._custom.items():
            if key in prompt:
                res = val
                break
        else:
            if "<blocks>" in prompt and "<block id=" in prompt:
                import re

                matches = re.findall(
                    r'<\s*block\s+id=["\'](.*?)["\']\s*>([\s\S]*?)<\s*/\s*block\s*>', prompt
                )
                if matches:
                    items = []
                    for bid, bsrc in matches:
                        btext = f"{self._prefix}{bsrc.strip()}" if self._prefix else self._default
                        items.append(f'<block id="{bid}">{btext}</block>')
                    res = "<blocks>\n" + "\n".join(items) + "\n</blocks>"
            elif self._prefix is not None:
                res = f"{self._prefix}{prompt.strip()}"

        if self._sanitize_output:
            return sanitize_thought_output(res, model)
        return res

    async def generate_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Mock vision: deterministic canned verdict without network."""
        self.call_history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
                "image_count": len(images_b64_png),
            }
        )
        return self._default


class _SharedTransportKwargs(TypedDict):
    """Constructor kwargs handed to every transport the provider builds.

    A plain ``dict`` literal infers as ``dict[str, <union of all value types>]``,
    which mypy 2.x can no longer unpack into the transports' named parameters —
    the single shared dict produced arg-type errors on all three transports.
    A TypedDict keeps each key's own type, so the dict stays DRY and still
    type-checks against ``BaseTransport.__init__``.
    """

    timeout: float
    transport: httpx.AsyncBaseTransport | None
    client: httpx.AsyncClient | None
    limits: httpx.Limits | None
    extra_headers: dict[str, str]
    sanitize_output: bool
    prompt_caching: bool
    usage_log: list[dict[str, Any]]
    cumulative_totals: dict[str, int]
    model_totals: dict[str, dict[str, int]]


class OpenAICompatibleProvider(BaseModelProvider):
    """Production provider that speaks one of the four wire protocols.

    Coordinates modular protocol transports (chat, responses, anthropic
    messages, gemini native) over a shared HTTP connection pool and unified
    usage accounting ledger.
    """

    def __init__(
        self,
        api_key: str | SecretStr,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "",
        timeout: float = 60.0,
        provider_name: str = "openai_compatible",
        transport: httpx.AsyncBaseTransport | None = None,
        api_mode: str = "openai-chat",
        sanitize_output: bool = True,
        client: httpx.AsyncClient | None = None,
        limits: httpx.Limits | None = None,
        extra_headers: dict[str, str] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        reasoning_dialect: str = "nested",
        prompt_caching: bool = True,
    ) -> None:
        if isinstance(api_key, SecretStr) or hasattr(api_key, "get_secret_value"):
            self._api_key = api_key.get_secret_value()
        else:
            self._api_key = str(api_key)
        base = base_url.rstrip("/")
        for suffix in ("/responses", "/chat/completions", "/chat", "/messages"):
            if base.endswith(suffix):
                base = base[: -len(suffix)].rstrip("/")
        self._base_url = base
        self._default_model = default_model
        self._timeout = timeout
        self._name = provider_name
        self._transport = transport
        # The protocol is whatever the caller selected — never inferred from the
        # endpoint or the model name. See ubt.core.config.ApiMode.
        self._api_mode = api_mode
        self._sanitize_output = sanitize_output
        self._prompt_caching = prompt_caching
        self._limits = limits or httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        self._extra_headers = dict(extra_headers or {})
        self._chat_template_kwargs = chat_template_kwargs or {}
        self._reasoning_dialect = reasoning_dialect

        # Shared token accounting
        self.usage_log: list[dict[str, Any]] = []
        self._cumulative_totals: dict[str, int] = new_usage_totals()
        self._model_totals: dict[str, dict[str, int]] = {}

        # Initialize modular transports sharing client, metrics, and configuration.
        # If client is not explicitly provided, create one shared httpx.AsyncClient
        # so all transports share the HTTP keep-alive connection pool.
        self._owned_client = client is None
        self._shared_client = client or httpx.AsyncClient(
            # Connect/pool fail fast; only read/write deserve the full api_timeout.
            # A bare float here would silently override the transports' own
            # connect=10s and pin a pool slot for the whole api_timeout on a dead
            # endpoint (see BaseTransport._get_client).
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT_S,
                read=self._timeout,
                write=self._timeout,
                pool=self._timeout,
            ),
            transport=self._transport,
            limits=self._limits,
        )

        shared_kw: _SharedTransportKwargs = {
            "timeout": self._timeout,
            "transport": self._transport,
            "client": self._shared_client,
            "limits": self._limits,
            "extra_headers": self._extra_headers,
            "sanitize_output": self._sanitize_output,
            "prompt_caching": self._prompt_caching,
            "usage_log": self.usage_log,
            "cumulative_totals": self._cumulative_totals,
            "model_totals": self._model_totals,
        }
        self._chat_transport = OpenAIChatTransport(
            api_key=self._api_key,
            base_url=self._base_url,
            default_model=self._default_model,
            chat_template_kwargs=self._chat_template_kwargs,
            **shared_kw,
        )
        self._anthropic_transport = AnthropicMessagesTransport(
            api_key=self._api_key,
            base_url=self._base_url,
            default_model=self._default_model,
            **shared_kw,
        )
        self._responses_transport = OpenAIResponsesTransport(
            api_key=self._api_key,
            base_url=self._base_url,
            default_model=self._default_model,
            reasoning_dialect=self._reasoning_dialect,
            **shared_kw,
        )
        self._gemini_transport = GeminiTransport(
            api_key=self._api_key,
            base_url=self._base_url,
            default_model=self._default_model,
            **shared_kw,
        )

    def _select_transport(self) -> BaseTransport:
        # The protocol is fixed for the provider's lifetime. It takes no model:
        # a fallback chain that lands on another model family must not silently
        # change the wire mid-run.
        if self._api_mode == "openai-responses":
            return self._responses_transport
        if self._api_mode == "anthropic-messages":
            return self._anthropic_transport
        if self._api_mode == "gemini-native":
            return self._gemini_transport
        return self._chat_transport

    def begin_usage_sink(self) -> dict[str, dict[str, int]]:
        """Start per-run usage attribution for this task."""
        return attach_usage_sink()

    def _auth_headers(self) -> dict[str, str]:
        """Auth headers for chat/responses/anthropic calls, incl. optional custom/Zen routing."""
        transport = self._select_transport()
        return transport._auth_headers()

    async def aclose(self) -> None:
        """Close connection pool and release underlying socket resources."""
        await self._chat_transport.aclose()
        await self._anthropic_transport.aclose()
        await self._responses_transport.aclose()
        await self._gemini_transport.aclose()
        if (
            self._owned_client
            and self._shared_client is not None
            and not self._shared_client.is_closed
        ):
            await self._shared_client.aclose()

    async def close(self) -> None:
        """Alias for aclose()."""
        await self.aclose()

    async def __aenter__(self) -> OpenAICompatibleProvider:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    @property
    def usage_totals(self) -> dict[str, int]:
        """Aggregate token usage across all calls."""
        return dict(self._cumulative_totals)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from the provider cache."""
        prompt = self._cumulative_totals.get("prompt_tokens", 0)
        if prompt <= 0:
            return 0.0
        cached = self._cumulative_totals.get("cached_tokens", 0)
        return round(min(cached / prompt, 1.0), 4)

    @property
    def usage_totals_by_model(self) -> dict[str, dict[str, int]]:
        """Aggregate token usage grouped by target model."""
        return {model: dict(totals) for model, totals in self._model_totals.items()}

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def base_url(self) -> str:
        """Endpoint this provider posts to (see ``BaseModelProvider.base_url``)."""
        return self._base_url

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        """Chat-template flags forwarded verbatim into chat payloads."""
        return dict(self._chat_template_kwargs)

    def _record_usage(
        self,
        target_model: str,
        usage_entry: dict[str, Any],
        *,
        batch: bool = False,
        unmeasured: bool = False,
    ) -> None:
        """Direct usage recording passthrough (delegates to chat transport)."""
        self._chat_transport._record_usage(
            target_model, usage_entry, batch=batch, unmeasured=unmeasured
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        transport = self._select_transport()
        return await transport.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        transport = self._select_transport()
        return await transport.generate_with_finish_reason(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    async def generate_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        transport = self._select_transport()
        return await transport.generate_with_images(
            prompt=prompt,
            images_b64_png=images_b64_png,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Batch API (delegated to OpenAIChatTransport)
    # ------------------------------------------------------------------
    @property
    def supports_batch_api(self) -> bool:
        return self._chat_transport.supports_batch_api if self._api_mode == "openai-chat" else False

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        return await self._chat_transport.create_batch_job(requests)

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        return await self._chat_transport.get_batch_job(batch_id)

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        return await self._chat_transport.fetch_batch_results(batch_id)

    async def cleanup_batch_files(self, batch_id: str) -> None:
        await self._chat_transport.cleanup_batch_files(batch_id)

    async def cancel_batch_job(self, batch_id: str) -> None:
        await self._chat_transport.cancel_batch_job(batch_id)


def create_model_provider(
    api_key: str | SecretStr,
    base_url: str = "https://api.openai.com/v1",
    default_model: str = "",
    api_mode: str = "openai-chat",
    timeout: float = 60.0,
    transport: httpx.AsyncBaseTransport | None = None,
    extra_headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    prompt_caching: bool = True,
    **kwargs: Any,
) -> OpenAICompatibleProvider:
    """Create an authenticated wire-protocol model provider from credentials.

    ``api_mode`` selects one of the four wire protocols; nothing about the
    endpoint or the model name influences the choice.
    """
    mode_str = api_mode.value if hasattr(api_mode, "value") else str(api_mode)

    return OpenAICompatibleProvider(
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
        timeout=timeout,
        transport=transport,
        api_mode=mode_str,
        extra_headers=dict(extra_headers or {}),
        client=client,
        prompt_caching=prompt_caching,
        **kwargs,
    )
