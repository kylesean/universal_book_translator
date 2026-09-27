"""Unit tests for Anthropic native Messages API protocol in OpenAICompatibleProvider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.provider import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_anthropic_generate_success_and_headers() -> None:
    captured_request: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "msg_01X",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "Hello in Chinese is 你好"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 8,
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test-key",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
        default_model="claude-3-5-sonnet-20241022",
        transport=httpx.MockTransport(handler),
    )

    text, finish_reason = await provider.generate_with_finish_reason(
        prompt="How to say hello in Chinese?",
        system_prompt="You are a translation assistant.",
    )

    assert text == "Hello in Chinese is 你好"
    assert finish_reason == "stop"
    assert captured_request["url"] == "https://api.anthropic.com/v1/messages"
    assert captured_request["headers"]["x-api-key"] == "sk-ant-test-key"
    system_body = captured_request["body"]["system"]
    if isinstance(system_body, list):
        assert system_body[0]["text"] == "You are a translation assistant."
        assert system_body[0]["cache_control"] == {"type": "ephemeral"}
    else:
        assert system_body == "You are a translation assistant."
    assert captured_request["body"]["messages"] == [
        {"role": "user", "content": "How to say hello in Chinese?"}
    ]
    assert captured_request["body"]["max_tokens"] == 4096

    # Test token & prompt cache accounting. Anthropic's ``input_tokens``
    # EXCLUDES the cache fields, so the billable prompt is their sum (20 + 8);
    # counting only ``input_tokens`` dropped every cache read.
    assert provider.usage_totals["prompt_tokens"] == 28
    assert provider.usage_totals["completion_tokens"] == 10
    assert provider.usage_totals["cached_tokens"] == 8
    assert provider.cache_hit_rate == pytest.approx(8 / 28, abs=1e-4)


@pytest.mark.asyncio
async def test_anthropic_cache_creation_tokens_are_counted_in_the_prompt() -> None:
    """Cache *writes* are prompt tokens too; ignoring them under-reports cost."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_creation",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "你好"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 50,
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test-key",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
        default_model="claude-3-5-sonnet-20241022",
        transport=httpx.MockTransport(handler),
    )
    await provider.generate_with_finish_reason(prompt="hi")

    assert provider.usage_totals["prompt_tokens"] == 160
    assert provider.usage_totals["cached_tokens"] == 100


@pytest.mark.asyncio
async def test_anthropic_truncation_detection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_02Y",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "Incomplete transla"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test-key",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
        transport=httpx.MockTransport(handler),
    )

    text, finish_reason = await provider.generate_with_finish_reason("Long prompt")
    assert text == "Incomplete transla"
    assert finish_reason == "length"


@pytest.mark.asyncio
async def test_anthropic_rate_limit_429() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "5"}, text="Rate limit exceeded")

    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test-key",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        await provider.generate("Test prompt")
    assert (exc_info.value.details or {}).get("status_code") == 429


@pytest.mark.asyncio
async def test_anthropic_vision_call() -> None:
    captured_request: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "msg_03Z",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "Visual description"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 10},
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test-key",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
        transport=httpx.MockTransport(handler),
    )

    res = await provider.generate_with_images(
        prompt="Describe this diagram",
        images_b64_png=[
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        ],
    )

    assert res == "Visual description"
    msg_content = captured_request["body"]["messages"][0]["content"]
    assert len(msg_content) == 2
    assert msg_content[0]["type"] == "image"
    assert msg_content[0]["source"]["type"] == "base64"
    assert msg_content[0]["source"]["media_type"] == "image/png"
    assert msg_content[1]["type"] == "text"
    assert msg_content[1]["text"] == "Describe this diagram"


@pytest.mark.asyncio
async def test_responses_vision_call() -> None:
    """The Responses API carries images as ``input_image`` content parts, so a
    responses-only model (opencode-zen muse-spark) can serve the visual judge.
    The old code wrongly gated vision to chat/anthropic and refused it."""
    captured_request: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "verdict: PASS"}],
                    }
                ],
                "usage": {"input_tokens": 40, "output_tokens": 5},
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://opencode.ai/zen/go/v1",
        api_mode="responses",
        transport=httpx.MockTransport(handler),
    )
    res = await provider.generate_with_images(
        prompt="QA this page",
        images_b64_png=[
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        ],
    )
    assert res == "verdict: PASS"
    assert captured_request["url"].endswith("/responses")
    inp = captured_request["body"]["input"]
    assert isinstance(inp, list) and inp[0]["role"] == "user"
    parts = inp[0]["content"]
    assert parts[0]["type"] == "input_text" and parts[0]["text"] == "QA this page"
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_anthropic_maps_reasoning_effort_to_a_thinking_budget() -> None:
    """``reasoning_effort`` must reach Anthropic as a thinking budget."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test-key",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
        default_model="claude-3-5-sonnet-20241022",
        transport=httpx.MockTransport(handler),
    )
    await provider.generate("hi", reasoning_effort="high")
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert captured["max_tokens"] > 8192
    assert "temperature" not in captured
