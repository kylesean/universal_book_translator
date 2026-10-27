"""OpenAI Chat Completions wire protocol transport (/chat/completions)."""

from __future__ import annotations

import logging
from typing import Any

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.base import (
    _CHAT_TEMPERATURE_KEYS,
    BaseTransport,
    _extract_cached_tokens,
    _heal_drop_temperature,
    _heal_max_tokens_rename,
    _heal_reasoning_effort,
    raise_for_status,
)

logger = logging.getLogger(__name__)


class OpenAIChatTransport(BaseTransport):
    """Transport for OpenAI Chat Completions API and compatible servers (DeepSeek, Ollama, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "",
        timeout: float = 60.0,
        provider_name: str = "openai_chat",
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
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if reasoning_effort is not None and reasoning_effort.strip():
            eff = reasoning_effort.strip().lower()
            if eff != "none":
                payload["reasoning_effort"] = eff
        if self._chat_template_kwargs:
            payload["chat_template_kwargs"] = self._chat_template_kwargs

        # Apply what an earlier 400 already proved this endpoint rejects, so the
        # fix is paid once per model instead of once per block.
        payload = self._apply_learned_omissions(payload)

        client = self._get_client()
        chat_url = f"{self._base_url}/chat/completions"
        response = await self._request_json(client, chat_url, payload)

        response = await self._self_heal_400(
            client,
            chat_url,
            payload,
            response,
            (
                _heal_reasoning_effort,
                _heal_drop_temperature(*_CHAT_TEMPERATURE_KEYS),
                _heal_max_tokens_rename,
            ),
        )

        raise_for_status(response)

        data = response.json()
        usage = data.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
        prompt_toks = int(usage.get("prompt_tokens", 0) or 0)
        cached = _extract_cached_tokens(usage)
        self._record_usage(
            target_model,
            {
                "model": target_model,
                "prompt_tokens": prompt_toks,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "reasoning_tokens": reasoning_tokens,
                "prompt_cache_hit_tokens": cached,
                "prompt_cache_miss_tokens": max(prompt_toks - cached, 0),
            },
            unmeasured=not usage,
        )
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content")
            refusal = msg.get("refusal")
            finish_reason = choice.get("finish_reason")
            if refusal or finish_reason == "content_filter":
                raise ModelProviderError(
                    f"LLM content refusal: {refusal or finish_reason}",
                    details={"fail_fast": True, "reason": "content_refusal"},
                )
            if content is None:
                # A null content is a content-filter refusal or a malformed
                # response, never a translation; returning reasoning_content
                # here shipped the chain-of-thought as the draft.
                if finish_reason == "length" and reasoning_tokens:
                    # Some reasoning models (e.g. hy3) spend the whole output
                    # budget on hidden reasoning and then return null content.
                    # This is a retryable budget failure, not a refusal: the
                    # router can reissue the call with a larger budget instead
                    # of failing every block in the book.
                    raise ModelProviderError(
                        "LLM response exhausted its output budget on reasoning "
                        f"(finish_reason=length, reasoning_tokens={reasoning_tokens})",
                        details={
                            "reasoning_budget_exhausted": True,
                            "reasoning_tokens": reasoning_tokens,
                        },
                    )
                raise ModelProviderError(
                    "LLM response has no message content "
                    f"(finish_reason={finish_reason or 'unknown'})",
                    details=(
                        {"fail_fast": True, "reason": "content_refusal"}
                        if finish_reason == "content_filter"
                        else None
                    ),
                )
            content_str = str(content)
            finish_reason = choice.get("finish_reason")
            return self._finalize_output(content_str, target_model), finish_reason
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ModelProviderError(f"Malformed LLM response JSON: {data}") from exc

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
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64 in images_b64_png:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens:
            payload["max_tokens"] = max_tokens
        client = self._get_client()
        response = await self._post_json(client, f"{self._base_url}/chat/completions", payload)
        data = response.json()
        usage = data.get("usage") or {}
        v_prompt_toks = int(usage.get("prompt_tokens", 0) or 0)
        v_cached = _extract_cached_tokens(usage)
        self._record_usage(
            target_model,
            {
                "model": target_model,
                "prompt_tokens": v_prompt_toks,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "prompt_cache_hit_tokens": v_cached,
                "prompt_cache_miss_tokens": max(v_prompt_toks - v_cached, 0),
            },
            unmeasured=not usage,
        )
        try:
            content_str = str(data["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError) as exc:
            raise ModelProviderError(f"Malformed vision response JSON: {data}") from exc
        return content_str.strip()
