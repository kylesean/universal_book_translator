"""The Gemini native wire protocol transport (``generateContent``)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.gemini import GeminiTransport

pytestmark = pytest.mark.fast

_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _transport(handler: Any) -> GeminiTransport:
    return GeminiTransport(
        api_key="gm-test-key",
        base_url=_BASE,
        default_model="gemini-3.8-flash",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_request_shape_auth_header_and_url() -> None:
    """The key travels in a header, and the model rides in the URL path."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "你好"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
            },
        )

    out = await _transport(handler).generate("hi", system_prompt="Be terse", temperature=0.2)

    assert out == "你好"
    assert seen["url"].endswith("/v1beta/models/gemini-3.8-flash:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "gm-test-key"
    assert "authorization" not in seen["headers"]
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert seen["body"]["systemInstruction"] == {"parts": [{"text": "Be terse"}]}
    assert seen["body"]["generationConfig"]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_usage_maps_cached_content_tokens() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 20,
                    "cachedContentTokenCount": 40,
                },
            },
        )

    transport = _transport(handler)
    await transport.generate("hi")

    totals = transport.usage_totals
    assert totals["prompt_tokens"] == 100
    assert totals["completion_tokens"] == 20
    assert totals["cached_tokens"] == 40


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("MAX_TOKENS", "length"), ("STOP", "stop"), ("SAFETY", None)],
)
async def test_finish_reason_mapping(finish_reason: str, expected: str | None) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "x"}]}, "finishReason": finish_reason}
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )

    _, finish = await _transport(handler).generate_with_finish_reason("hi")
    assert finish == expected


@pytest.mark.asyncio
async def test_vision_sends_inline_data_parts() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "seen"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )

    out = await _transport(handler).generate_with_images("describe", ["QUJD"])

    assert out == "seen"
    parts = seen["body"]["contents"][0]["parts"]
    assert parts[0] == {"inlineData": {"mimeType": "image/png", "data": "QUJD"}}
    assert parts[1] == {"text": "describe"}


@pytest.mark.asyncio
async def test_reasoning_effort_becomes_a_thinking_budget() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )

    await _transport(handler).generate("hi", reasoning_effort="high")
    assert seen["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 8192}


@pytest.mark.asyncio
async def test_a_rejected_thinking_config_is_dropped_and_retried() -> None:
    """A model that rejects thinkingConfig must self-heal instead of failing."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "thinkingConfig" in body.get("generationConfig", {}):
            return httpx.Response(
                400,
                json={"error": {"message": "thinkingConfig is not supported by this model"}},
            )
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "自愈"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )

    out = await _transport(handler).generate("hi", reasoning_effort="low")

    assert out == "自愈"
    assert len(bodies) == 2
    assert "thinkingConfig" in bodies[0]["generationConfig"]
    assert "thinkingConfig" not in bodies[1]["generationConfig"]


@pytest.mark.asyncio
async def test_an_api_error_payload_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "quota exceeded"}})

    with pytest.raises(ModelProviderError, match="quota exceeded"):
        await _transport(handler).generate("hi")
