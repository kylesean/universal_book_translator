"""Regression tests for the batch-2 billing / translation-correctness fixes.

Five defects, each a silent loss of money or of a protected span:

- the OpenAI batch parser fell back to ``reasoning_content`` when ``content``
  was null, shipping the chain-of-thought as the translation;
- the Gemini transport raised on a blocked/candidate-less 200 *before* recording
  usage, so those tokens never counted against the budget;
- the translate cache was written inside ``draft`` before the caller judged the
  draft, so a defective draft replayed as a hit on every later run;
- ``draft_batch`` ran the caller's cancel/budget callback only inside the poll
  loop, after the whole book had already been submitted;
- a post-submission failure other than a ``ModelProviderError`` escaped without
  the ``batch_id``, so the interactive fallback could not cancel the still-live
  batch and billed the same blocks twice.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ubt.core.exceptions import BudgetExceededError, ModelProviderError
from ubt.core.router.router import (
    BatchDraftRequest,
    BatchTranslationError,
    ModelRouter,
)
from ubt.core.router.transports.gemini import GeminiTransport
from ubt.core.router.transports.openai_chat import OpenAIChatTransport

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# OpenAI batch parser: a null content is a refusal, never a CoT draft.
# --------------------------------------------------------------------------- #


class _BatchOpenAI(OpenAIChatTransport):
    """An OpenAI transport whose batch results come from a canned JSONL blob."""

    def __init__(self, lines: str) -> None:
        super().__init__(api_key="k", default_model="m")
        self._lines = lines

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        return {"status": "completed", "output_file_id": "f1"}

    async def _download_batch_file(self, batch_id: str, file_id: str) -> str:
        return self._lines


def _batch_line(message: dict[str, Any], finish_reason: str = "stop") -> str:
    return json.dumps(
        {
            "custom_id": "c1",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "m",
                    "choices": [{"message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_openai_batch_null_content_is_a_refusal_not_a_cot_draft() -> None:
    # ``content: null`` with a populated ``reasoning_content`` is a content-filter
    # refusal; returning the CoT shipped untranslated model reasoning as the draft.
    transport = _BatchOpenAI(
        _batch_line({"content": None, "reasoning_content": "Let me think about this..."})
    )
    results = await transport.fetch_batch_results("b1")
    await transport.aclose()

    assert results["c1"]["content"] is None
    assert "no message content" in (results["c1"]["error"] or "")
    assert "reasoning" not in json.dumps(results)


@pytest.mark.asyncio
async def test_openai_batch_real_content_is_still_used() -> None:
    transport = _BatchOpenAI(_batch_line({"content": "译文", "reasoning_content": "ignored"}))
    results = await transport.fetch_batch_results("b1")
    await transport.aclose()

    assert results["c1"]["content"] == "译文"
    assert results["c1"]["error"] is None


# --------------------------------------------------------------------------- #
# Gemini: usage is recorded before the response is judged an error.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gemini_blocked_200_still_bills_its_prompt_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "promptFeedback": {"blockReason": "SAFETY"},
                "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 0},
            },
        )

    transport = GeminiTransport(
        api_key="k",
        default_model="gemini-x",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelProviderError, match="blocked the prompt"):
        await transport.generate_with_finish_reason("hello")
    await transport.aclose()

    # The blocked call consumed 42 prompt tokens; skipping the write let the
    # budget cap undercount real spend.
    assert transport.usage_log
    assert transport.usage_log[0]["prompt_tokens"] == 42


# --------------------------------------------------------------------------- #
# draft_batch: pre-submission gate + post-submission batch_id propagation.
# --------------------------------------------------------------------------- #


class _FakeBatchProvider:
    """A batch-capable provider double for the router's orchestration paths."""

    is_mock = True
    supports_batch_api = True

    def __init__(self) -> None:
        self.created = 0
        self.cancelled: list[str] = []
        self.raise_on_get: Exception | None = None

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        return {}

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        self.created += 1
        return "batch-1"

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return {"status": "completed"}

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        return {}

    async def cancel_batch_job(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)

    async def cleanup_batch_files(self, batch_id: str) -> None:
        return None


def _request() -> BatchDraftRequest:
    return BatchDraftRequest(custom_id="c1", source_text="hello world")


@pytest.mark.asyncio
async def test_draft_batch_gate_aborts_before_any_submission() -> None:
    provider = _FakeBatchProvider()
    router = ModelRouter(provider=provider, draft_model="m")  # type: ignore[arg-type]

    def gate(stage: str, payload: dict[str, Any]) -> None:
        if stage == "submitting":
            raise BudgetExceededError("over budget")

    with pytest.raises(BudgetExceededError):
        await router.draft_batch([_request()], status_callback=gate)

    # The cap governs spend, not just the poll loop: nothing may be submitted.
    assert provider.created == 0


@pytest.mark.asyncio
async def test_draft_batch_post_submission_failure_carries_batch_id() -> None:
    provider = _FakeBatchProvider()
    provider.raise_on_get = RuntimeError("ledger exploded")
    router = ModelRouter(provider=provider, draft_model="m")  # type: ignore[arg-type]

    with pytest.raises(BatchTranslationError) as excinfo:
        await router.draft_batch([_request()], poll_interval=0.5, poll_timeout=1.0)

    # A failure after create_batch_job must report the live batch, or the
    # caller's interactive fallback cannot cancel it and bills twice.
    assert provider.created == 1
    assert excinfo.value.batch_id == "batch-1"
