"""Tests for the modularized LLM Transport architecture and extra_headers support."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ubt.core.router.provider import create_model_provider
from ubt.core.router.transports import (
    AnthropicMessagesTransport,
    OpenAIChatTransport,
    OpenAIResponsesTransport,
)


@pytest.mark.asyncio
async def test_extra_headers_passed_to_requests() -> None:
    """Verify arbitrary custom headers are forwarded to the wire protocol."""
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Header test passed"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    provider = create_model_provider(
        api_key="sk-test",
        base_url="https://api.custom-ai.org/v1",
        api_mode="chat",
        extra_headers={
            "x-custom-tenant": "tenant-42",
            "x-request-source": "ubt-engine",
        },
        transport=httpx.MockTransport(handler),
    )

    res = await provider.generate("hello")
    assert res == "Header test passed"
    assert captured_headers.get("x-custom-tenant") == "tenant-42"
    assert captured_headers.get("x-request-source") == "ubt-engine"


@pytest.mark.asyncio
async def test_opencode_session_id_maps_to_extra_headers_backward_compat() -> None:
    """Verify legacy opencode_session_id argument seamlessly maps to x-opencode-session header."""
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    provider = create_model_provider(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        opencode_session_id="ses_abc12345",
        transport=httpx.MockTransport(handler),
    )

    await provider.generate("hello")
    assert captured_headers.get("x-opencode-session") == "ses_abc12345"


@pytest.mark.asyncio
async def test_anthropic_transport_standalone_with_custom_proxy() -> None:
    """AnthropicMessagesTransport must work with custom proxy URL without needing 'api.anthropic.com'."""
    captured_req: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_req["url"] = str(request.url)
        captured_req["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "msg_proxy",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Proxy Anthropic response"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 8, "cache_read_input_tokens": 5},
            },
        )

    transport = AnthropicMessagesTransport(
        api_key="sk-ant-proxy-key",
        base_url="https://internal-gateway.corp.com/anthropic/v1",
        default_model="claude-3-7-sonnet",
        transport=httpx.MockTransport(handler),
    )

    text, finish = await transport.generate_with_finish_reason("translate this")
    assert text == "Proxy Anthropic response"
    assert finish == "stop"
    assert captured_req["url"] == "https://internal-gateway.corp.com/anthropic/v1/messages"
    assert captured_req["headers"]["x-api-key"] == "sk-ant-proxy-key"
    assert transport.usage_totals["prompt_tokens"] == 20  # 15 + 5
    assert transport.usage_totals["cached_tokens"] == 5


@pytest.mark.asyncio
async def test_openai_responses_transport_standalone_with_custom_model() -> None:
    """OpenAIResponsesTransport must work with any model name, not only 'muse-' prefix."""
    captured_req: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_req["url"] = str(request.url)
        captured_req["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "resp_001",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Responses API translated"}],
                    }
                ],
                "usage": {"input_tokens": 30, "output_tokens": 12},
            },
        )

    transport = OpenAIResponsesTransport(
        api_key="sk-resp-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5-responses-preview",
        transport=httpx.MockTransport(handler),
    )

    text, finish = await transport.generate_with_finish_reason("translate this")
    assert text == "Responses API translated"
    assert finish == "stop"
    assert captured_req["url"] == "https://api.openai.com/v1/responses"
    assert captured_req["body"]["model"] == "gpt-5-responses-preview"
    assert transport.usage_totals["prompt_tokens"] == 30
    assert transport.usage_totals["completion_tokens"] == 12


@pytest.mark.asyncio
async def test_openai_chat_transport_standalone_batch_support() -> None:
    """OpenAIChatTransport must declare batch support and handle batch jobs."""
    transport = OpenAIChatTransport(
        api_key="sk-chat-key",
        base_url="https://api.openai.com/v1",
    )
    assert transport.supports_batch_api is True
    assert transport.provider_name == "openai_chat"


def test_orchestrator_wires_config_extra_headers_to_provider(tmp_path: Any) -> None:
    """Verify that UBTConfig.extra_headers is forwarded into the orchestrator's provider."""
    from pydantic import SecretStr

    from ubt.core.config import UBTConfig
    from ubt.core.engine.pipeline import PipelineOrchestrator
    from ubt.core.router.provider import OpenAICompatibleProvider

    cfg = UBTConfig.from_env(
        api_key=SecretStr("test-key"),
        base_url="https://api.openai.com/v1",
        db_dir=tmp_path,
        extra_headers={"x-gateway-tenant": "tenant-99", "cf-access-id": "client-abc"},
    )
    orchestrator = PipelineOrchestrator(config=cfg)
    provider = orchestrator.router.provider
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._extra_headers.get("x-gateway-tenant") == "tenant-99"
    auth_headers = provider._auth_headers()
    assert auth_headers.get("x-gateway-tenant") == "tenant-99"
    assert auth_headers.get("cf-access-id") == "client-abc"


@pytest.mark.fast
def test_httpx_socks_proxy_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify httpx AsyncClient initializes without error when SOCKS proxy env vars are set."""
    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:10808")
    monkeypatch.setenv("all_proxy", "socks5h://127.0.0.1:10808")
    client = httpx.AsyncClient()
    assert client is not None
