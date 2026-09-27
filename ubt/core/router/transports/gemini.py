"""Gemini native wire protocol transport (``models/{model}:generateContent``)."""

from __future__ import annotations

import logging
from typing import Any

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.base import BaseTransport

logger = logging.getLogger(__name__)

#: ``reasoning_effort`` -> Gemini ``thinkingConfig.thinkingBudget`` token budget.
_REASONING_EFFORT_BUDGETS: dict[str, int] = {"low": 1024, "medium": 4096, "high": 8192}

#: ``finishReason`` values that mean the model was cut off by the token limit.
_LENGTH_REASONS = frozenset({"MAX_TOKENS"})


def _heal_drop_thinking(err_text: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Drop a rejected ``thinkingConfig`` and retry without it."""
    generation_config = payload.get("generationConfig")
    if isinstance(generation_config, dict) and "thinkingConfig" in generation_config:
        fixed = dict(payload)
        fixed["generationConfig"] = {
            key: value for key, value in generation_config.items() if key != "thinkingConfig"
        }
        return fixed
    return None


class GeminiTransport(BaseTransport):
    """Transport for the Gemini native generateContent API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        default_model: str = "",
        timeout: float = 60.0,
        provider_name: str = "gemini_native",
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
        # The key travels in a header, not a ``?key=`` query string, so it
        # cannot leak through URL logging.
        headers = {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
        }
        if self._extra_headers:
            headers.update(self._extra_headers)
        return headers

    def _generate_url(self, model: str) -> str:
        return f"{self._base_url}/models/{model}:generateContent"

    def _build_payload(
        self,
        prompt: str,
        images_b64_png: list[str] | None,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [
            {"inlineData": {"mimeType": "image/png", "data": b64}} for b64 in (images_b64_png or [])
        ]
        parts.append({"text": prompt})
        payload: dict[str, Any] = {"contents": [{"role": "user", "parts": parts}]}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens
        if reasoning_effort:
            budget = _REASONING_EFFORT_BUDGETS.get(reasoning_effort.strip().lower())
            if budget is not None:
                generation_config["thinkingConfig"] = {"thinkingBudget": budget}
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(
            str(part.get("text", "")) for part in parts if isinstance(part, dict) and "text" in part
        )

    @staticmethod
    def _finish_reason(data: dict[str, Any]) -> str | None:
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        reason = str(candidates[0].get("finishReason") or "")
        if reason in _LENGTH_REASONS:
            return "length"
        if reason == "STOP":
            return "stop"
        return None

    def _record_gemini_usage(self, target_model: str, data: dict[str, Any]) -> None:
        usage = data.get("usageMetadata") or {}
        self._record_usage(
            target_model,
            {
                "model": target_model,
                "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "completion_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
                "prompt_cache_hit_tokens": int(usage.get("cachedContentTokenCount", 0) or 0),
            },
            unmeasured=not usage,
        )

    @staticmethod
    def _raise_for_error(data: dict[str, Any]) -> None:
        error = data.get("error")
        if isinstance(error, dict):
            raise ModelProviderError(f"Gemini API error: {error.get('message', error)}")

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
        payload = self._build_payload(
            prompt, None, system_prompt, temperature, max_tokens, reasoning_effort
        )
        client = self._get_client()
        url = self._generate_url(target_model)
        response = await self._request_json(client, url, payload)

        if response.status_code == 400:
            response = await self._self_heal_400(
                client, url, payload, response, (_heal_drop_thinking,)
            )

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
        self._raise_for_error(data)
        self._record_gemini_usage(target_model, data)
        return self._finalize_output(self._extract_text(data).strip(), target_model), (
            self._finish_reason(data)
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
        target_model = self._resolve_model(model)
        payload = self._build_payload(
            prompt, images_b64_png, system_prompt, temperature, max_tokens, None
        )
        client = self._get_client()
        response = await self._post_json(client, self._generate_url(target_model), payload)
        data = response.json()
        self._raise_for_error(data)
        self._record_gemini_usage(target_model, data)
        return self._extract_text(data).strip()
