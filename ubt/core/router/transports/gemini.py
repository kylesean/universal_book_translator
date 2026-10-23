"""Gemini native wire protocol transport (``models/{model}:generateContent``)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ubt.core.config import MOCK_API_KEY
from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.base import BaseTransport, raise_for_status

logger = logging.getLogger(__name__)

#: ``reasoning_effort`` -> Gemini ``thinkingConfig.thinkingBudget`` token budget.
_REASONING_EFFORT_BUDGETS: dict[str, int] = {
    "none": 0,
    "off": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}

#: ``finishReason`` values that mean the model was cut off by the token limit.
_LENGTH_REASONS = frozenset({"MAX_TOKENS"})

#: ``finishReason`` values that mean safety/recitation policy removed the
#: candidate content entirely.
_BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
)

#: Substrings of a lowercased 400 body that mean the request was rejected
#: *because of* ``thinkingConfig`` — the only case where dropping it and
#: retrying is a real fix. Without this gate any 400 (a bad model name, say)
#: would be retried once without the field, billing a second doomed call.
_THINKING_REJECTION_KEYS = ("thinkingconfig", "thinkingbudget", "thinking")


def _heal_drop_thinking(err_text: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Drop a rejected ``thinkingConfig`` and retry without it."""
    generation_config = payload.get("generationConfig")
    if not isinstance(generation_config, dict) or "thinkingConfig" not in generation_config:
        return None
    if not any(key in err_text for key in _THINKING_REJECTION_KEYS):
        return None
    fixed = dict(payload)
    fixed["generationConfig"] = {
        key: value for key, value in generation_config.items() if key != "thinkingConfig"
    }
    return fixed


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
        self._thinking_rejected_models: set[str] = set()

    def _auth_headers(self) -> dict[str, str]:
        # The key travels in a header, not a ``?key=`` query string, so it
        # cannot leak through URL logging. An unconfigured key (empty or the
        # ``mock-key`` sentinel) is omitted rather than sent as a literal,
        # matching ``BaseTransport._auth_headers``.
        headers = {"content-type": "application/json"}
        if self._api_key and self._api_key != MOCK_API_KEY:
            headers["x-goog-api-key"] = self._api_key
        if self._extra_headers:
            headers.update(self._extra_headers)
        return headers

    def _generate_url(self, model: str) -> str:
        clean_model = model.removeprefix("models/")
        return f"{self._base_url}/models/{clean_model}:generateContent"

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

    def _record_gemini_usage(
        self, target_model: str, data: dict[str, Any], *, batch: bool = False
    ) -> None:
        usage = data.get("usageMetadata") or {}
        self._record_usage(
            target_model,
            {
                "model": target_model,
                "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
                # candidatesTokenCount excludes thinking; thoughtsTokenCount is
                # billed as output, so both count toward completion tokens.
                "completion_tokens": int(usage.get("candidatesTokenCount", 0) or 0)
                + int(usage.get("thoughtsTokenCount", 0) or 0),
                "prompt_cache_hit_tokens": int(usage.get("cachedContentTokenCount", 0) or 0),
            },
            batch=batch,
            unmeasured=not usage,
        )

    @staticmethod
    def _raise_for_error(data: dict[str, Any]) -> None:
        error = data.get("error")
        if isinstance(error, dict):
            raise ModelProviderError(f"Gemini API error: {error.get('message', error)}")
        # Safety blocks arrive as a 200 with promptFeedback/candidate finish
        # reasons instead of an error object; reading them as success would
        # ship an empty draft and burn futile repair rounds.
        block_reason = str((data.get("promptFeedback") or {}).get("blockReason") or "")
        if block_reason:
            raise ModelProviderError(
                f"Gemini blocked the prompt (blockReason={block_reason})",
                details={"fail_fast": True, "reason": "content_refusal"},
            )
        candidates = data.get("candidates") or []
        if not candidates:
            raise ModelProviderError("Gemini returned no candidates")
        candidate_reason = str((candidates[0] or {}).get("finishReason") or "")
        if candidate_reason in _BLOCKED_FINISH_REASONS:
            raise ModelProviderError(
                f"Gemini blocked the response (finishReason={candidate_reason})",
                details={"fail_fast": True, "reason": "content_refusal"},
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
        effective_effort = (
            reasoning_effort if target_model not in self._thinking_rejected_models else None
        )
        payload = self._build_payload(
            prompt, None, system_prompt, temperature, max_tokens, effective_effort
        )
        client = self._get_client()
        url = self._generate_url(target_model)
        response = await self._request_json(client, url, payload)

        if response.status_code == 400:
            healed = await self._self_heal_400(
                client, url, payload, response, (_heal_drop_thinking,)
            )
            if healed.status_code == 200:
                self._thinking_rejected_models.add(target_model)
            response = healed

        raise_for_status(response)

        data = response.json()
        # Record usage before judging the response: a safety block arrives as a
        # 200 that still consumed prompt tokens, and skipping the write on the
        # raise let the budget cap undercount real spend.
        self._record_gemini_usage(target_model, data)
        self._raise_for_error(data)
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
        # Record usage before judging the response: a safety block arrives as a
        # 200 that still consumed prompt tokens, and skipping the write on the
        # raise let the budget cap undercount real spend.
        self._record_gemini_usage(target_model, data)
        self._raise_for_error(data)
        return self._extract_text(data).strip()

    # ------------------------------------------------------------------
    # Batch API (``:batchGenerateContent``, 50% discount for book runs)
    # ------------------------------------------------------------------
    @property
    def supports_batch_api(self) -> bool:
        return True

    def _batch_generate_url(self, model: str) -> str:
        clean_model = model.removeprefix("models/")
        return f"{self._base_url}/models/{clean_model}:batchGenerateContent"

    def _batch_url(self, name: str) -> str:
        """Resolve a batch operation name to a pollable URL.

        ``batchGenerateContent`` returns an operation whose ``name`` is
        ``batches/<id>`` (or a full resource path). The poll endpoint is
        ``{base}/{name}``.
        """
        clean = name.removeprefix("/").removeprefix("v1beta/")
        if not clean.startswith("batches/"):
            clean = f"batches/{clean}"
        return f"{self._base_url}/{clean}"

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        """Submit a batch and return its operation/batch id.

        Gemini takes the requests inline (``batch.inputConfig.requests``), each
        keyed by ``metadata.key`` (Gemini's per-request custom id) with the
        ``generateContent`` body under ``request``.
        """
        if not requests:
            raise ModelProviderError("Cannot create an empty batch job")
        inline_requests: list[dict[str, Any]] = []
        for req in requests:
            key = str(req.get("custom_id") or "")
            if not key:
                raise ModelProviderError("Batch request is missing custom_id")
            body = req.get("params")
            if not isinstance(body, dict):
                body = _gemini_body_from_request(req)
            inline_requests.append({"request": body, "metadata": {"key": key}})

        payload = {
            "batch": {
                "displayName": "ubt-book-batch",
                "inputConfig": {"requests": {"requests": inline_requests}},
            }
        }
        client = self._get_client()
        try:
            response = await client.post(
                self._batch_generate_url(self._default_model),
                headers=self._auth_headers(),
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise ModelProviderError(f"Batch creation timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise ModelProviderError(f"Batch creation HTTP request error: {exc}") from exc
        if response.status_code not in (200, 201):
            raise ModelProviderError(
                f"Batch creation failed ({response.status_code}): {response.text[:500]}",
                details={"status_code": response.status_code},
            )
        data = response.json()
        # The LRO name is the batch id (``batches/<id>``); some responses carry
        # the batch resource under ``metadata.name``.
        name = str(data.get("name") or (data.get("metadata") or {}).get("name") or "")
        if not name:
            raise ModelProviderError(f"Batch creation returned no name: {response.text[:500]}")
        return name

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = await client.get(self._batch_url(batch_id), headers=self._auth_headers())
        except httpx.TimeoutException as exc:
            raise ModelProviderError(f"Batch poll timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise ModelProviderError(f"Batch poll HTTP request error: {exc}") from exc
        if response.status_code != 200:
            raise ModelProviderError(
                f"Batch poll failed ({response.status_code}): {response.text[:500]}",
                details={"status_code": response.status_code},
            )
        data: dict[str, Any] = response.json()
        # Normalize the LRO shape onto the router's ``status`` vocabulary: an
        # operation with ``done=true`` is terminal; failures live in ``error``.
        state = str((data.get("metadata") or {}).get("state") or "")
        if data.get("done") is True:
            data["status"] = "failed" if data.get("error") else "completed"
        else:
            data["status"] = {
                "JOB_STATE_SUCCEEDED": "completed",
                "JOB_STATE_FAILED": "failed",
                "JOB_STATE_CANCELLED": "cancelled",
                "JOB_STATE_EXPIRED": "failed",
            }.get(state, "in_progress")
        data.setdefault("state", state)
        return data

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        job = await self.get_batch_job(batch_id)
        status = str(job.get("status") or "")
        if status == "failed":
            raise ModelProviderError(
                f"Batch job {batch_id} failed: {json.dumps(job.get('error') or {})[:500]}"
            )
        if status != "completed":
            raise ModelProviderError(f"Batch job {batch_id} is not completed (status={status!r})")

        responses = ((job.get("response") or {}).get("responses")) or (
            (job.get("metadata") or {}).get("output") or {}
        ).get("responses")
        results: dict[str, dict[str, Any]] = {}
        for entry in responses or []:
            if not isinstance(entry, dict):
                continue
            custom_id = str((entry.get("metadata") or {}).get("key") or "")
            if not custom_id:
                continue
            err = entry.get("error")
            if isinstance(err, dict) and err.get("message"):
                results[custom_id] = {"content": None, "error": str(err["message"])}
                continue
            body = entry.get("response") or {}
            self._record_gemini_usage(
                str(body.get("modelVersion") or self._default_model), body, batch=True
            )
            text = self._extract_text(body).strip()
            line_error: str | None = None
            if not text:
                line_error = "Batch response has no text content"
            # ``_finish_reason`` normalizes the wire's ``MAX_TOKENS`` to the
            # protocol-neutral ``"length"``, so comparing against
            # ``_LENGTH_REASONS`` (the raw wire vocabulary) was constant False:
            # a response cut off by ``maxOutputTokens`` that still carried a
            # partial sentence was finalized as a success, putting half a
            # translation into the book with no error and no continuation. The
            # batch path has no continuation loop, so the truncation must be
            # reported as a line error for the caller to retry.
            elif self._finish_reason(body) == "length":
                line_error = "Batch response truncated at maxOutputTokens"
            results[custom_id] = {"content": text or None, "error": line_error}
        return results

    async def cancel_batch_job(self, batch_id: str) -> None:
        client = self._get_client()
        try:
            await client.post(f"{self._batch_url(batch_id)}:cancel", headers=self._auth_headers())
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.debug("Batch cancel request failed for %s: %s", batch_id, exc)

    async def cleanup_batch_files(self, batch_id: str) -> None:
        # Gemini holds inline batch inputs/results server-side; nothing to delete.
        return None


def _gemini_body_from_request(req: dict[str, Any]) -> dict[str, Any]:
    """Build a Gemini ``generateContent`` body from a router batch request.

    Accepts the router's OpenAI-shaped request (``{custom_id, body: {messages}}``)
    or a plain ``{prompt, system_prompt, max_tokens}``, so one batch path serves
    every protocol.
    """
    body = req.get("body")
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
        parts: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for msg in body["messages"]:
            role = str(msg.get("role") or "user")
            content = str(msg.get("content") or "")
            if role == "system":
                system_parts.append(content)
            else:
                parts.append({"text": content})
        payload: dict[str, Any] = {"contents": [{"role": "user", "parts": parts or [{"text": ""}]}]}
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(p for p in system_parts if p)}]
            }
        generation_config: dict[str, Any] = {}
        if body.get("temperature") is not None:
            generation_config["temperature"] = body["temperature"]
        if body.get("max_tokens"):
            generation_config["maxOutputTokens"] = body["max_tokens"]
        effort = body.get("reasoning_effort") or req.get("reasoning_effort")
        if effort and isinstance(effort, str):
            budget = _REASONING_EFFORT_BUDGETS.get(effort.strip().lower())
            if budget is not None:
                generation_config["thinkingConfig"] = {"thinkingBudget": budget}
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload
    payload = {"contents": [{"role": "user", "parts": [{"text": str(req.get("prompt") or "")}]}]}
    if req.get("system_prompt"):
        payload["systemInstruction"] = {"parts": [{"text": str(req["system_prompt"])}]}
    fallback_config: dict[str, Any] = {}
    if req.get("temperature") is not None:
        fallback_config["temperature"] = req["temperature"]
    if req.get("max_tokens"):
        fallback_config["maxOutputTokens"] = req["max_tokens"]
    fallback_effort = req.get("reasoning_effort")
    if fallback_effort and isinstance(fallback_effort, str):
        fallback_budget = _REASONING_EFFORT_BUDGETS.get(fallback_effort.strip().lower())
        if fallback_budget is not None:
            fallback_config["thinkingConfig"] = {"thinkingBudget": fallback_budget}
    if fallback_config:
        payload["generationConfig"] = fallback_config
    return payload
