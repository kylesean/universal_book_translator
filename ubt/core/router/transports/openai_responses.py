"""OpenAI Responses API wire protocol transport (/responses)."""

from __future__ import annotations

import logging
from typing import Any

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.base import (
    BaseTransport,
    _extract_cached_tokens,
    host_is,
)

logger = logging.getLogger(__name__)


def _is_opencode_zen_endpoint(base_url: str) -> bool:
    """Whether ``base_url`` points at the OpenCode Zen gateway (host-based)."""
    return host_is(base_url, "opencode.ai")


class OpenAIResponsesTransport(BaseTransport):
    """Transport for OpenAI Responses API (/responses)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "muse-spark-1.3-contributor",
        timeout: float = 60.0,
        provider_name: str = "openai_responses",
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
        self._model_reasoning_mode: dict[str, str] = {}

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
        return await self._generate_responses_meta(
            prompt,
            system_prompt,
            model or self._default_model,
            temperature,
            max_tokens,
            reasoning_effort,
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
        if not images_b64_png:
            raise ModelProviderError("Vision input requires at least one image")
        text, _ = await self._generate_responses_meta(
            prompt,
            system_prompt,
            model or self._default_model,
            temperature,
            max_tokens,
            images_b64_png=images_b64_png,
        )
        return text

    async def _generate_responses_meta(
        self,
        prompt: str,
        system_prompt: str | None,
        target_model: str,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None = None,
        images_b64_png: list[str] | None = None,
    ) -> tuple[str, str | None]:
        if images_b64_png:
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            for b64 in images_b64_png:
                content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
            input_body: Any = [{"role": "user", "content": content}]
        else:
            input_body = prompt
        payload: dict[str, Any] = {
            "model": target_model,
            "input": input_body,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if system_prompt:
            payload["instructions"] = system_prompt
        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        if reasoning_effort is not None and reasoning_effort.strip():
            eff = reasoning_effort.strip().lower()
            cached_mode = self._model_reasoning_mode.get(
                target_model
            ) or self._model_reasoning_mode.get("*")
            if cached_mode == "none":
                pass
            elif (
                cached_mode == "nested_minimal"
                or eff == "minimal"
                or (eff == "none" and _is_opencode_zen_endpoint(self._base_url))
            ):
                payload["reasoning"] = {"effort": "minimal"}
            elif _is_opencode_zen_endpoint(self._base_url):
                payload["reasoning_effort"] = eff
            else:
                payload["reasoning"] = {"effort": eff}

        client = self._get_client()
        responses_url = f"{self._base_url}/responses"
        response = await self._request_json(client, responses_url, payload)

        if response.status_code == 400 and (
            "reasoning_effort" in payload or "reasoning" in payload
        ):
            err_text = response.text.lower()
            if any(
                k in err_text
                for k in (
                    "reasoning_effort",
                    "reasoning",
                    "unrecognized",
                    "extra_forbidden",
                    "unknown",
                )
            ):

                def _without_reasoning() -> dict[str, Any]:
                    dropped = dict(payload)
                    dropped.pop("reasoning_effort", None)
                    dropped.pop("reasoning", None)
                    return dropped

                if _is_opencode_zen_endpoint(self._base_url):
                    retry_payload = dict(payload)
                    retry_payload.pop("reasoning_effort", None)
                    retry_payload["reasoning"] = {"effort": "minimal"}
                    retry_resp = await self._request_json(client, responses_url, retry_payload)
                    if retry_resp.status_code == 200:
                        self._model_reasoning_mode[target_model] = "nested_minimal"
                        response = retry_resp
                    else:
                        self._model_reasoning_mode[target_model] = "none"
                        response = await self._request_json(
                            client, responses_url, _without_reasoning()
                        )
                else:
                    self._model_reasoning_mode[target_model] = "none"
                    response = await self._request_json(client, responses_url, _without_reasoning())

        if response.status_code == 429:
            raise ModelProviderError(
                "Rate limit exceeded (HTTP 429)",
                details={
                    "status_code": 429,
                    "retry_after": response.headers.get("retry-after"),
                },
            )
        if response.status_code != 200:
            raise ModelProviderError(
                f"Model API error ({response.status_code}): {response.text[:500]}",
                details={"status_code": response.status_code, "body": response.text[:2000]},
            )

        data = response.json()
        if data.get("error"):
            raise ModelProviderError(f"Responses API error: {data['error']}")

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cached = _extract_cached_tokens(usage)
        self._record_usage(
            target_model,
            {
                "model": target_model,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "prompt_cache_hit_tokens": cached,
                "prompt_cache_miss_tokens": max(input_tokens - cached, 0),
            },
            unmeasured=not usage,
        )
        has_message = False
        text = ""
        refusal = ""
        for item in data.get("output") or []:
            if item.get("type") != "message":
                continue
            has_message = True
            for part in item.get("content") or []:
                part_type = part.get("type")
                if part_type == "output_text":
                    text += str(part.get("text", ""))
                elif part_type == "refusal":
                    refusal += str(part.get("refusal", ""))
        status = str(data.get("status") or "")
        incomplete_reason = str((data.get("incomplete_details") or {}).get("reason") or "")
        # A token-limit truncation may legitimately have no text yet.
        truncated = status == "incomplete" and incomplete_reason == "max_output_tokens"
        if not has_message:
            if truncated:
                return "", "length"
            raise ModelProviderError(f"Malformed Responses API output (no message items): {data}")
        if not text.strip() and not truncated:
            # A refusal-only (or otherwise empty) message is not a successful
            # empty translation: surface it so the fallback chain can try
            # instead of shipping "" as a finished block.
            detail = f": {refusal.strip()}" if refusal.strip() else ""
            raise ModelProviderError(f"Responses API returned an empty message{detail}")
        result = text.strip()
        finish_reason = "length" if truncated else ("stop" if status == "completed" else None)
        return self._finalize_output(result, target_model), finish_reason
