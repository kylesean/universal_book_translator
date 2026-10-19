"""Anthropic Messages API wire protocol transport (/v1/messages)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.base import (
    _ANTHROPIC_TEMPERATURE_KEYS,
    _ANTHROPIC_THINKING_KEYS,
    BaseTransport,
    _heal_drop_temperature,
    _heal_drop_thinking,
    raise_for_status,
)

logger = logging.getLogger(__name__)

#: ``reasoning_effort`` -> Anthropic extended-thinking token budget.
_REASONING_EFFORT_BUDGETS: dict[str, int] = {"low": 1024, "medium": 4096, "high": 8192}

#: Default completion budget when the caller names none. The Messages API has no
#: server-side default, so every request must carry one.
_DEFAULT_MAX_TOKENS = 4096


def _apply_max_tokens(params: dict[str, Any], requested: Any) -> dict[str, Any]:
    """Set ``max_tokens`` on ``params``, defaulting exactly as the live path does."""
    params["max_tokens"] = int(requested) if requested else _DEFAULT_MAX_TOKENS
    return params


def _apply_thinking(params: dict[str, Any], reasoning_effort: Any) -> dict[str, Any]:
    """Translate ``reasoning_effort`` into an extended-thinking budget.

    Anthropic expresses reasoning as a token budget, not an "effort" enum, and
    requires ``max_tokens > budget_tokens`` while forbidding a custom temperature
    alongside thinking. Shared by the interactive and batch paths so a
    reasoning-tier batch item cannot ship without a budget big enough to hold it.
    """
    budget = _REASONING_EFFORT_BUDGETS.get(str(reasoning_effort or "").lower())
    if budget is None:
        return params
    params["thinking"] = {"type": "enabled", "budget_tokens": budget}
    params["max_tokens"] = max(int(params.get("max_tokens") or _DEFAULT_MAX_TOKENS), budget + 1024)
    params.pop("temperature", None)
    return params


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
            **_apply_max_tokens({}, max_tokens),
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
        # Anthropic requires ``max_tokens > budget_tokens`` and forbids a custom
        # temperature alongside thinking; the response parser reads only ``text``
        # blocks, so the extra thinking block is ignored.
        _apply_thinking(payload, reasoning_effort)

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

        raise_for_status(response)

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
        # A 200 with no text (a refusal stop_reason, or thinking-only content)
        # is a provider refusal, not a translation; reading it as success would
        # ship an empty draft and burn futile repair rounds.
        if not result:
            if stop_reason == "refusal":
                raise ModelProviderError("Anthropic refused the request (stop_reason=refusal)")
            if not (data.get("content") or []):
                raise ModelProviderError("Anthropic returned no content blocks")
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

    # ------------------------------------------------------------------
    # Message Batches API (50% discount for large book runs)
    # ------------------------------------------------------------------
    #: ``supports_batch_api`` gates the pipeline's batch leg (see
    #: ``ubt.core.router.router.ModelRouter.draft_batch``). Anthropic's Message
    #: Batches API takes the requests inline (no upload step) and returns results
    #: as a JSONL stream from ``/results`` — different wire shape, same lifecycle.
    @property
    def supports_batch_api(self) -> bool:
        return True

    def _batches_url(self, *parts: str) -> str:
        base = f"{self._base_url}" if self._base_url.endswith("/v1") else f"{self._base_url}/v1"
        return "/".join([base, "messages", "batches", *parts])

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        """Submit a Message Batch; return its id.

        Each request carries ``custom_id`` and either a prebuilt ``params`` body
        or the raw ``prompt``/``system_prompt``/``max_tokens`` the interactive
        path would send. Anthropic caps a batch at 100k requests / 256 MB; the
        caller (``submit_batch``) bounds the batch size, so this only enforces
        the non-empty precondition.
        """
        if not requests:
            raise ModelProviderError("Cannot create an empty batch job")
        payload_requests: list[dict[str, Any]] = []
        for req in requests:
            custom_id = str(req.get("custom_id") or "")
            if not custom_id:
                raise ModelProviderError("Batch request is missing custom_id")
            params = req.get("params")
            if not isinstance(params, dict):
                params = _messages_params_from_request(req)
            payload_requests.append({"custom_id": custom_id, "params": params})

        client = self._get_client()
        try:
            response = await client.post(
                self._batches_url(),
                headers=self._auth_headers(),
                json={"requests": payload_requests},
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
        batch_id = str(response.json().get("id") or "")
        if not batch_id:
            raise ModelProviderError(f"Batch creation returned no id: {response.text[:500]}")
        return batch_id

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = await client.get(self._batches_url(batch_id), headers=self._auth_headers())
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
        # Normalize Anthropic's ``processing_status`` onto the OpenAI-style
        # ``status`` the router's poll loop reads, so the two providers share
        # one terminal-status check.
        raw_status = str(data.get("processing_status") or "")
        data["status"] = _anthropic_batch_status(raw_status)
        return data

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        job = await self.get_batch_job(batch_id)
        status = str(job.get("status") or "")
        if status == "failed":
            raise ModelProviderError(
                f"Batch job {batch_id} failed: {json.dumps(job.get('request_counts') or {})[:500]}"
            )
        if status != "completed":
            raise ModelProviderError(f"Batch job {batch_id} is not completed (status={status!r})")

        client = self._get_client()
        try:
            response = await client.get(
                self._batches_url(batch_id, "results"), headers=self._auth_headers()
            )
        except httpx.TimeoutException as exc:
            raise ModelProviderError(f"Batch results download timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise ModelProviderError(f"Batch results HTTP request error: {exc}") from exc
        if response.status_code != 200:
            raise ModelProviderError(
                f"Batch results fetch failed ({response.status_code}): {response.text[:500]}",
                details={"status_code": response.status_code},
            )

        results: dict[str, dict[str, Any]] = {}
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed batch result line: %.120s", line)
                continue
            custom_id = str(entry.get("custom_id") or "")
            if not custom_id:
                continue
            result = entry.get("result") or {}
            rtype = str(result.get("type") or "")
            if rtype != "succeeded":
                # errored / canceled / expired — surface why, never a fake score.
                err = result.get("error") or {}
                message = (
                    err.get("message") if isinstance(err, dict) else str(err)
                ) or f"Batch line {rtype or 'failed'}"
                results[custom_id] = {"content": None, "error": str(message)}
                continue
            message = result.get("message") or {}
            usage = message.get("usage") or {}
            v_input = int(usage.get("input_tokens", 0) or 0)
            v_cached = int(usage.get("cache_read_input_tokens", 0) or 0)
            v_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
            self._record_usage(
                str(message.get("model") or self._default_model or "unknown"),
                {
                    "model": str(message.get("model") or self._default_model or "unknown"),
                    "prompt_tokens": v_input + v_cached + v_creation,
                    "completion_tokens": usage.get("output_tokens", 0) or 0,
                    "prompt_cache_hit_tokens": v_cached,
                    "prompt_cache_miss_tokens": v_input + v_creation,
                },
                batch=True,
                unmeasured=not usage,
            )
            text = ""
            for item in message.get("content") or []:
                if isinstance(item, dict) and item.get("type") == "text":
                    text += str(item.get("text", ""))
            content = text.strip()
            line_error: str | None = None
            if not content:
                line_error = "Batch response has no text content"
            elif str(message.get("stop_reason") or "") == "max_tokens":
                line_error = "Batch response truncated at max_tokens (stop_reason=max_tokens)"
            results[custom_id] = {"content": content or None, "error": line_error}
        return results

    async def cancel_batch_job(self, batch_id: str) -> None:
        client = self._get_client()
        try:
            await client.post(self._batches_url(batch_id, "cancel"), headers=self._auth_headers())
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.debug("Batch cancel request failed for %s: %s", batch_id, exc)

    async def cleanup_batch_files(self, batch_id: str) -> None:
        # Anthropic holds batch inputs/results server-side; there is no uploaded
        # file object to delete, so cleanup is a no-op (unlike OpenAI's /files).
        return None


def _anthropic_batch_status(processing_status: str) -> str:
    """Map Anthropic's ``processing_status`` onto the router's status vocabulary."""
    return {
        "in_progress": "in_progress",
        "canceling": "cancelling",
        "ended": "completed",
    }.get(processing_status, processing_status or "unknown")


def _messages_params_from_request(req: dict[str, Any]) -> dict[str, Any]:
    """Build a ``/v1/messages`` body from a router batch request.

    The router builds OpenAI-shaped requests (``{custom_id, method, url, body}``
    with a ``messages`` array), and batch callers may instead pass a plain
    ``{prompt, system_prompt, max_tokens}``. Accept both so one router path
    drives every protocol's batch leg.
    """
    body = req.get("body")
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
        params: dict[str, Any] = {"model": body.get("model") or req.get("model") or ""}
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for msg in body["messages"]:
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if role == "system":
                system_parts.append(str(content or ""))
            else:
                messages.append({"role": role, "content": str(content or "")})
        params["messages"] = messages or [{"role": "user", "content": ""}]
        if system_parts:
            params["system"] = "\n\n".join(part for part in system_parts if part)
        # ``max_tokens`` is REQUIRED by the Messages API and has no server-side
        # default. Copying it only when truthy meant a reasoning-tier request --
        # which deliberately omits it and lets the thinking budget drive the
        # value -- produced a batch item with no ``max_tokens`` at all, and the
        # whole batch 400'd item by item. Apply the same default the interactive
        # path uses, and let thinking raise it the same way.
        _apply_max_tokens(params, body.get("max_tokens"))
        if body.get("temperature") is not None:
            params["temperature"] = body["temperature"]
        _apply_thinking(params, body.get("reasoning_effort"))
        return params
    return _apply_thinking(
        {
            "model": req.get("model") or "",
            "messages": [{"role": "user", "content": str(req.get("prompt") or "")}],
            **_apply_max_tokens({}, req.get("max_tokens")),
        },
        req.get("reasoning_effort"),
    )
