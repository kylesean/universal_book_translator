"""OpenAI Chat Completions wire protocol transport (/chat/completions)."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from typing import Any

import httpx

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.base import (
    _CHAT_TEMPERATURE_KEYS,
    BaseTransport,
    _extract_cached_tokens,
    _heal_drop_temperature,
    _heal_reasoning_effort,
)

logger = logging.getLogger(__name__)


class OpenAIChatTransport(BaseTransport):
    """Transport for OpenAI Chat Completions API and compatible servers (DeepSeek, Ollama, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "deepseek-v4-flash",
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

    @property
    def supports_batch_api(self) -> bool:
        return True

    def _batch_headers(self) -> dict[str, str]:
        # Match ``_auth_headers``: a key that already carries the ``Bearer``
        # prefix must not gain a second one, or every /files and /batches call
        # 401s and the run silently falls back to full-price interactive.
        api_key = self._api_key
        if not api_key.lower().startswith("bearer "):
            api_key = f"Bearer {api_key}"
        headers = {"Authorization": api_key}
        if self._extra_headers:
            headers.update(self._extra_headers)
        return headers

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
        target_model = model or self._default_model
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

        client = self._get_client()
        chat_url = f"{self._base_url}/chat/completions"
        response = await self._request_json(client, chat_url, payload)

        response = await self._self_heal_400(
            client,
            chat_url,
            payload,
            response,
            (_heal_reasoning_effort, _heal_drop_temperature(*_CHAT_TEMPERATURE_KEYS)),
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
            if content is None:
                content = msg.get("reasoning_content") or ""
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
        target_model = model or self._default_model
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

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------
    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        if not requests:
            raise ModelProviderError("Cannot create an empty batch job")
        client = self._get_client()
        jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests)
        file_id: str | None = None
        try:
            upload = await client.post(
                f"{self._base_url}/files",
                headers=self._batch_headers(),
                data={"purpose": "batch"},
                files={
                    "file": (
                        "ubt-batch.jsonl",
                        jsonl.encode("utf-8"),
                        "application/jsonl",
                    )
                },
            )
            if upload.status_code != 200:
                raise ModelProviderError(
                    f"Batch file upload failed ({upload.status_code}): {upload.text[:500]}",
                    details={"status_code": upload.status_code},
                )
            file_id = str(upload.json().get("id") or "")
            if not file_id:
                raise ModelProviderError(f"Batch file upload returned no id: {upload.text[:500]}")

            create = await client.post(
                f"{self._base_url}/batches",
                headers={**self._batch_headers(), "Content-Type": "application/json"},
                json={
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            if create.status_code not in (200, 201):
                with suppress(Exception):
                    await client.delete(
                        f"{self._base_url}/files/{file_id}",
                        headers=self._batch_headers(),
                    )
                raise ModelProviderError(
                    f"Batch creation failed ({create.status_code}): {create.text[:500]}",
                    details={"status_code": create.status_code},
                )
            batch_id = str(create.json().get("id") or "")
            if not batch_id:
                with suppress(Exception):
                    await client.delete(
                        f"{self._base_url}/files/{file_id}",
                        headers=self._batch_headers(),
                    )
                raise ModelProviderError(f"Batch creation returned no id: {create.text[:500]}")
            return batch_id
        except httpx.TimeoutException as exc:
            if file_id:
                with suppress(Exception):
                    await client.delete(
                        f"{self._base_url}/files/{file_id}",
                        headers=self._batch_headers(),
                    )
            raise ModelProviderError(f"Batch request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            if file_id:
                with suppress(Exception):
                    await client.delete(
                        f"{self._base_url}/files/{file_id}",
                        headers=self._batch_headers(),
                    )
            raise ModelProviderError(f"Batch HTTP request error: {exc}") from exc

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = await client.get(
                f"{self._base_url}/batches/{batch_id}",
                headers=self._batch_headers(),
            )
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
        return data

    async def cancel_batch_job(self, batch_id: str) -> None:
        client = self._get_client()
        try:
            await client.post(
                f"{self._base_url}/batches/{batch_id}/cancel",
                headers=self._batch_headers(),
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.debug("Batch cancel request failed for %s: %s", batch_id, exc)

    async def _download_batch_file(self, batch_id: str, file_id: str) -> str:
        client = self._get_client()
        try:
            content_resp = await client.get(
                f"{self._base_url}/files/{file_id}/content",
                headers=self._batch_headers(),
            )
        except httpx.TimeoutException as exc:
            raise ModelProviderError(f"Batch result download timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise ModelProviderError(f"Batch result download error: {exc}") from exc
        if content_resp.status_code != 200:
            raise ModelProviderError(
                f"Batch result download failed ({content_resp.status_code}): "
                f"{content_resp.text[:500]}",
                details={"status_code": content_resp.status_code},
            )
        return content_resp.text

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        job = await self.get_batch_job(batch_id)
        status = str(job.get("status") or "")
        output_file_id = str(job.get("output_file_id") or "")
        error_file_id = str(
            job.get("error_file_id") or (job.get("error") or {}).get("file_id") or ""
        )
        if status == "failed":
            raise ModelProviderError(
                f"Batch job {batch_id} failed: {json.dumps(job.get('errors') or {})[:500]}"
            )
        if status != "completed" or (not output_file_id and not error_file_id):
            raise ModelProviderError(f"Batch job {batch_id} is not completed (status={status!r})")

        results: dict[str, dict[str, Any]] = {}
        if output_file_id:
            content_text = await self._download_batch_file(batch_id, output_file_id)
            for line in content_text.splitlines():
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
                error = entry.get("error")
                resp = entry.get("response") or {}
                status_code = resp.get("status_code")
                body = resp.get("body") or {}
                line_error: str | None = None
                if isinstance(error, dict) and error.get("message"):
                    line_error = str(error.get("message"))
                elif (
                    isinstance(body, dict)
                    and isinstance(body.get("error"), dict)
                    and body["error"].get("message")
                ):
                    line_error = str(body["error"]["message"])
                elif status_code is not None and status_code >= 400:
                    line_error = f"HTTP {status_code}: {json.dumps(body)}"
                content: str | None = None
                if body and not line_error and status_code in (None, 200):
                    usage = body.get("usage") or {}
                    cached = _extract_cached_tokens(usage)
                    self._record_usage(
                        str(body.get("model") or self._default_model),
                        {
                            "model": str(body.get("model") or self._default_model),
                            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                            "completion_tokens": usage.get("completion_tokens", 0) or 0,
                            "prompt_cache_hit_tokens": cached,
                            "prompt_cache_miss_tokens": max(
                                int(usage.get("prompt_tokens", 0) or 0) - cached, 0
                            ),
                        },
                        batch=True,
                        unmeasured=not usage,
                    )
                    choices = body.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        raw = msg.get("content")
                        if raw is None:
                            raw = msg.get("reasoning_content") or ""
                        content = str(raw)
                        content = (
                            self._finalize_output(
                                content, str(body.get("model") or self._default_model)
                            )
                            if self._sanitize_output
                            else content.strip()
                        )
                        # A length-capped answer is a truncation, not a
                        # translation: surface it as an error so the caller
                        # retries/falls back instead of shipping half a sentence.
                        if str(choices[0].get("finish_reason") or "") == "length":
                            line_error = line_error or (
                                "Batch response truncated at max_tokens (finish_reason=length)"
                            )
                    else:
                        line_error = line_error or "Batch response body has no choices"
                results[custom_id] = {"content": content, "error": line_error}
        if error_file_id:
            try:
                error_text = await self._download_batch_file(batch_id, error_file_id)
            except ModelProviderError as exc:
                logger.warning(
                    "Batch job %s error file %s could not be downloaded: %s",
                    batch_id,
                    error_file_id,
                    exc,
                )
                return results
            parsed_errors = 0
            for line in error_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed batch error line: %.120s", line)
                    continue
                custom_id = str(entry.get("custom_id") or "")
                if not custom_id:
                    continue
                if custom_id in results and results[custom_id].get("content") is not None:
                    continue
                err = entry.get("error") or {}
                message = (
                    err.get("message") if isinstance(err, dict) else str(err)
                ) or "Batch line failed (provider error file)"
                results[custom_id] = {"content": None, "error": str(message)}
                parsed_errors += 1
            if parsed_errors:
                logger.warning(
                    "Batch job %s parsed %d errors from error file %s",
                    batch_id,
                    parsed_errors,
                    error_file_id,
                )
        return results

    async def _delete_remote_file(self, client: httpx.AsyncClient, file_id: str) -> None:
        try:
            resp = await client.delete(
                f"{self._base_url}/files/{file_id}",
                headers=self._batch_headers(),
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.debug("Batch file %s delete failed (ignored): %s", file_id, exc)
            return
        if resp.status_code not in (200, 202, 204):
            logger.debug(
                "Batch file %s delete returned %s (ignored): %.120s",
                file_id,
                resp.status_code,
                resp.text,
            )

    async def cleanup_batch_files(self, batch_id: str) -> None:
        try:
            job = await self.get_batch_job(batch_id)
        except ModelProviderError as exc:
            logger.debug("Batch %s metadata unreadable, skipping file cleanup: %s", batch_id, exc)
            return
        error_file = job.get("error")
        nested_error_file_id = (
            str(error_file.get("file_id") or "") if isinstance(error_file, dict) else ""
        )
        file_ids = [
            fid
            for fid in (
                str(job.get("input_file_id") or ""),
                str(job.get("output_file_id") or ""),
                str(job.get("error_file_id") or ""),
                nested_error_file_id,
            )
            if fid
        ]
        if not file_ids:
            return
        client = self._get_client()
        for file_id in file_ids:
            await self._delete_remote_file(client, file_id)
