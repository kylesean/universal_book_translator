"""The Anthropic and Gemini Batch APIs (module B, item 4).

The spec requires both providers' official batch endpoints (50% discount) to be
wired, not just OpenAI-compatible chat. These tests drive each transport's batch
lifecycle through ``httpx.MockTransport`` and pin the wire shape plus the
terminal-status normalization the router's poll loop depends on.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.provider import OpenAICompatibleProvider
from ubt.core.router.transports.anthropic import AnthropicMessagesTransport
from ubt.core.router.transports.gemini import GeminiTransport
from ubt.core.router.transports.openai_chat import OpenAIChatTransport

pytestmark = pytest.mark.fast


def _router_request(custom_id: str, system: str, user: str) -> dict[str, Any]:
    """The OpenAI-shaped request the router actually builds for a batch."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "m",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        },
    }


# --------------------------------------------------------------------------- #
# Anthropic Message Batches
# --------------------------------------------------------------------------- #


def _anthropic(
    calls: list[httpx.Request], responses: dict[str, httpx.Response]
) -> AnthropicMessagesTransport:
    def _dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.get(request.url.path, httpx.Response(404, text="unexpected"))

    return AnthropicMessagesTransport(
        api_key="k", base_url="https://api.anthropic.com", transport=httpx.MockTransport(_dispatch)
    )


def test_anthropic_supports_batch_and_posts_inline_requests() -> None:
    calls: list[httpx.Request] = []
    created = httpx.Response(200, json={"id": "msgbatch_1"})
    t = _anthropic(calls, {"/v1/messages/batches": created})
    assert t.supports_batch_api is True

    import asyncio

    bid = asyncio.run(t.create_batch_job([_router_request("c1", "sys", "hello")]))
    assert bid == "msgbatch_1"
    posted = json.loads(calls[0].content)
    assert posted["requests"][0]["custom_id"] == "c1"
    params = posted["requests"][0]["params"]
    assert params["messages"][0]["content"] == "hello"
    assert params["system"] == "sys"
    assert params["max_tokens"] == 2048


def test_anthropic_status_is_normalized_for_the_poll_loop() -> None:
    calls: list[httpx.Request] = []
    job = httpx.Response(200, json={"id": "b1", "processing_status": "ended", "request_counts": {}})
    t = _anthropic(calls, {"/v1/messages/batches/b1": job})

    import asyncio

    data = asyncio.run(t.get_batch_job("b1"))
    assert data["status"] == "completed"


def test_anthropic_results_parse_text_and_errors() -> None:
    calls: list[httpx.Request] = []
    job = httpx.Response(200, json={"id": "b1", "processing_status": "ended"})
    lines = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "c1",
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "model": "claude",
                            "content": [{"type": "text", "text": "你好"}],
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                            "stop_reason": "end_turn",
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "c2",
                    "result": {"type": "errored", "error": {"message": "too long"}},
                }
            ),
        ]
    )
    results_resp = httpx.Response(200, text=lines)
    t = _anthropic(
        calls,
        {"/v1/messages/batches/b1": job, "/v1/messages/batches/b1/results": results_resp},
    )

    import asyncio

    results = asyncio.run(t.fetch_batch_results("b1"))
    assert results["c1"]["content"] == "你好"
    assert results["c1"]["error"] is None
    assert results["c2"]["content"] is None
    assert "too long" in results["c2"]["error"]


# --------------------------------------------------------------------------- #
# Gemini Batch
# --------------------------------------------------------------------------- #


def _gemini(calls: list[httpx.Request], responses: dict[str, httpx.Response]) -> GeminiTransport:
    def _dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.get(request.url.path, httpx.Response(404, text="unexpected"))

    return GeminiTransport(
        api_key="k",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-x",
        transport=httpx.MockTransport(_dispatch),
    )


def test_gemini_supports_batch_and_posts_inline_requests() -> None:
    calls: list[httpx.Request] = []
    created = httpx.Response(200, json={"name": "batches/123"})
    t = _gemini(calls, {"/v1beta/models/gemini-x:batchGenerateContent": created})
    assert t.supports_batch_api is True

    import asyncio

    bid = asyncio.run(t.create_batch_job([_router_request("c1", "sys", "hello")]))
    assert bid == "batches/123"
    posted = json.loads(calls[0].content)
    entry = posted["batch"]["inputConfig"]["requests"]["requests"][0]
    assert entry["metadata"]["key"] == "c1"
    assert entry["request"]["contents"][0]["parts"][0]["text"] == "hello"
    assert entry["request"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_done_operation_normalizes_to_completed() -> None:
    calls: list[httpx.Request] = []
    op = httpx.Response(
        200,
        json={
            "name": "batches/123",
            "done": True,
            "response": {
                "responses": [
                    {
                        "metadata": {"key": "c1"},
                        "response": {
                            "candidates": [
                                {"content": {"parts": [{"text": "你好"}]}, "finishReason": "STOP"}
                            ]
                        },
                    }
                ]
            },
        },
    )
    t = _gemini(calls, {"/v1beta/batches/123": op})

    import asyncio

    job = asyncio.run(t.get_batch_job("batches/123"))
    assert job["status"] == "completed"
    results = asyncio.run(t.fetch_batch_results("batches/123"))
    assert results["c1"]["content"] == "你好"


def test_gemini_empty_requests_rejected() -> None:
    t = _gemini([], {})

    import asyncio

    with pytest.raises(ModelProviderError):
        asyncio.run(t.create_batch_job([]))


# --------------------------------------------------------------------------- #
# Provider routes batch to the selected protocol.
# --------------------------------------------------------------------------- #


def test_provider_batch_routing_follows_api_mode() -> None:
    anthropic = OpenAICompatibleProvider(api_key="k", api_mode="anthropic")
    gemini = OpenAICompatibleProvider(api_key="k", api_mode="gemini")
    chat = OpenAICompatibleProvider(api_key="k", api_mode="openai-chat")
    assert isinstance(anthropic._batch_transport(), AnthropicMessagesTransport)
    assert isinstance(gemini._batch_transport(), GeminiTransport)
    assert isinstance(chat._batch_transport(), OpenAIChatTransport)
    # All three now advertise batch support.
    assert anthropic.supports_batch_api and gemini.supports_batch_api and chat.supports_batch_api
