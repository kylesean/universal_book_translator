"""Anthropic Messages API wire protocol transport (/v1/messages)."""

from __future__ import annotations

import logging
from typing import Any

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.base import (
    _ANTHROPIC_TEMPERATURE_KEYS,
    _ANTHROPIC_THINKING_KEYS,
    BaseTransport,
    _heal_drop_temperature,
    _heal_drop_thinking,
)

logger = logging.getLogger(__name__)

#: ``reasoning_effort`` -> Anthropic extended-thinking token budget.
_REASONING_EFFORT_BUDGETS: dict[str, int] = {"low": 1024, "medium": 4096, "high": 8192}


class AnthropicMessagesTransport(BaseTransport):
    """Transport for Anthropic Claude native Messages API (/v1/messages)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        default_model: str = "",
        timeout: float = 60.0,
        provider_name: str = "anthropic_messages",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout=timeout,
            provider_name=provider_name,
            **kwargs,
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self._prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        if self._extra_headers:
            headers.update(self._extra_headers)
        return headers

    def _messages_url(self) -> str:
        return (
            f"{self._base_url}/messages"
            if self._base_url.endswith("/v1")
            else f"{self._base_url}/v1/messages"
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
        text, _ = await self.generate_with_finish_reason(
            prompt, system_prompt, model, temperature, max_tokens, reasoning_effort
        )
        return text

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        target_model = self._resolve_model(model)
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            if self._prompt_caching:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning_effort:
            budget = _REASONING_EFFORT_BUDGETS.get(reasoning_effort.lower())
            if budget is not None:
                # Anthropic expresses reasoning as an extended-thinking token
                # budget, not an "effort" enum, so the parameter used to be
                # silently dropped. Anthropic requires
                # ``max_tokens > budget_tokens`` and forbids a custom temperature
                # alongside thinking; the response parser reads only ``text``
                # blocks, so the extra thinking block is ignored.
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                payload["max_tokens"] = max(int(payload["max_tokens"]), budget + 1024)
                payload.pop("temperature", None)

        client = self._get_client()
        url = self._messages_url()
        response = await self._request_json(client, url, payload)

        if response.status_code == 400:
            response = await self._self_heal_400(
                client,
                url,
                payload,
                response,
                (
                    _heal_drop_temperature(*_ANTHROPIC_TEMPERATURE_KEYS),
                    _heal_drop_thinking(*_ANTHROPIC_THINKING_KEYS),
                ),
            )

        if response.status_code == 429:
            raise ModelProviderError(
                f"Rate limit exceeded (HTTP 429): {response.text[:500]}",
                details={
                    "status_code": 429,
                    "retry_after": response.headers.get("retry-after")
                    or response.headers.get("retry-after-ms"),
                    "body": response.text[:2000],
                },
            )
        if response.status_code != 200:
            raise ModelProviderError(
                f"Model API error ({response.status_code}): {response.text[:500]}",
                details={"status_code": response.status_code, "body": response.text[:2000]},
            )

        data = response.json()
        if data.get("error"):
            raise ModelProviderError(f"Anthropic API error: {data['error']}")

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        self._record_usage(
            target_model,
            {
                "model": target_model,
                "prompt_tokens": input_tokens + cache_read + cache_creation,
                "completion_tokens": output_tokens,
                "prompt_cache_hit_tokens": cache_read,
                "prompt_cache_miss_tokens": input_tokens + cache_creation,
            },
            unmeasured=not usage,
        )

        text = ""
        for item in data.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text += str(item.get("text", ""))

        result = text.strip()
        stop_reason = str(data.get("stop_reason") or "")
        finish_reason = (
            "length"
            if stop_reason == "max_tokens"
            else ("stop" if stop_reason in ("end_turn", "stop_sequence") else None)
        )
        return self._finalize_output(result, target_model), finish_reason

    async def generate_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        if not images_b64_png:
            raise ModelProviderError("Vision input requires at least one image")
        target_model = self._resolve_model(model)
        content: list[dict[str, Any]] = []
        for b64 in images_b64_png:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        client = self._get_client()
        url = self._messages_url()
        response = await self._post_json(client, url, payload)
        data = response.json()
        usage = data.get("usage") or {}
        v_input = int(usage.get("input_tokens", 0) or 0)
        v_cached = int(usage.get("cache_read_input_tokens", 0) or 0)
        v_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        self._record_usage(
            target_model,
            {
                "model": target_model,
                "prompt_tokens": v_input + v_cached + v_creation,
                "completion_tokens": usage.get("output_tokens", 0) or 0,
                "prompt_cache_hit_tokens": v_cached,
                "prompt_cache_miss_tokens": v_input + v_creation,
            },
            unmeasured=not usage,
        )
        text = ""
        for item in data.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text += str(item.get("text", ""))
        return text.strip()
