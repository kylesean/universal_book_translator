"""Tests for the modularized LLM Transport architecture and extra_headers support."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ubt.core.exceptions import ModelProviderError
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
        default_model="mock-model",
        api_mode="openai-chat",
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
async def test_extra_headers_carry_an_endpoint_specific_session_id() -> None:
    """The OpenCode session id is just an ``extra_headers`` entry now.

    It used to be a first-class config field injected as ``x-opencode-session``;
    a provider block declares it generically instead, so the header travels the
    same way any other custom header does.
    """
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
        default_model="mock-model",
        extra_headers={"x-opencode-session": "ses_abc12345"},
        transport=httpx.MockTransport(handler),
    )

    await provider.generate("hello")
    assert captured_headers.get("x-opencode-session") == "ses_abc12345"


@pytest.mark.asyncio
async def test_a_call_without_a_model_fails_locally() -> None:
    """A blank model is a local configuration error, not a request that ships
    ``"model": ""`` and collects a 400 from the vendor."""
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = create_model_provider(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelProviderError, match="No model configured"):
        await provider.generate("hello")
    assert calls == []  # the blank model never reached the wire

    # An explicit model satisfies the call even when the provider has no default.
    assert await provider.generate("hello", model="some-model") == "ok"
    assert calls[0]["model"] == "some-model"


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
        draft_model="mock-draft",
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


@pytest.mark.fast
def test_responses_api_reasoning_nested_format() -> None:
    transport = OpenAIResponsesTransport(api_key="mock", base_url="https://api.openai.com/v1")
    captured_payloads: list[dict[str, Any]] = []

    mock_client = AsyncMock()

    async def fake_post(url: str, **kwargs: Any) -> MagicMock:
        json_payload = kwargs.get("json")
        if isinstance(json_payload, dict):
            captured_payloads.append(json_payload)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
        }
        return resp

    mock_client.post = fake_post
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    asyncio.run(
        transport.generate(
            prompt="test",
            model="o3-mini",
            reasoning_effort="high",
        )
    )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert "reasoning_effort" not in payload, (
        "Responses API must not have top-level reasoning_effort"
    )
    assert payload.get("reasoning") == {"effort": "high"}


@pytest.mark.fast
def test_responses_api_incomplete_reasoning_returns_length_finish() -> None:
    transport = OpenAIResponsesTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "still thinking..."}],
            }
        ],
    }
    mock_client.post.return_value = resp
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    result, finish_reason = asyncio.run(
        transport.generate_with_finish_reason(
            prompt="test",
            model="o3-mini",
        )
    )
    assert finish_reason == "length"
    assert result == ""


def test_responses_api_refusal_is_an_error_not_an_empty_success() -> None:
    """A refusal-only message must not be returned as a finished empty block."""
    transport = OpenAIResponsesTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "I can't help with that."}],
            }
        ],
    }
    mock_client.post.return_value = resp
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    with pytest.raises(ModelProviderError, match="empty message"):
        asyncio.run(
            transport.generate_with_finish_reason(
                prompt="test",
                model="gpt-5-responses-preview",
            )
        )


@pytest.mark.fast
def test_openai_chat_batch_input_file_cleanup_on_error() -> None:
    transport = OpenAIChatTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    # Upload succeeds
    upload_resp = MagicMock()
    upload_resp.status_code = 200
    upload_resp.json.return_value = {"id": "file-12345"}

    # Batch creation fails with 400
    create_resp = MagicMock()
    create_resp.status_code = 400
    create_resp.text = "Invalid batch"

    mock_client.post.side_effect = [upload_resp, create_resp]
    delete_called_with = []

    async def fake_delete(url: str, **kwargs: object) -> MagicMock:
        delete_called_with.append(url)
        del_resp = MagicMock()
        del_resp.status_code = 200
        return del_resp

    mock_client.delete = fake_delete
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    with pytest.raises(ModelProviderError):
        asyncio.run(
            transport.create_batch_job(
                [
                    {
                        "custom_id": "c1",
                        "body": {
                            "model": "gpt-4o",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    }
                ],
            )
        )

    assert any("file-12345" in url for url in delete_called_with), (
        "file_id must be cleaned up on batch creation failure"
    )


@pytest.mark.fast
def test_openai_chat_batch_cleanup_handles_error_file_id() -> None:
    transport = OpenAIChatTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    job_status_resp = MagicMock()
    job_status_resp.status_code = 200
    job_status_resp.json.return_value = {
        "id": "batch_abc",
        "input_file_id": "file-in",
        "output_file_id": "file-out",
        "error_file_id": "file-err",
        "error": None,
    }
    mock_client.get.return_value = job_status_resp

    deleted_urls = []

    async def fake_delete(url: str, **kwargs: object) -> MagicMock:
        deleted_urls.append(url)
        del_resp = MagicMock()
        del_resp.status_code = 200
        return del_resp

    mock_client.delete = fake_delete
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    asyncio.run(transport.cleanup_batch_files("batch_abc"))
    assert any("file-err" in url for url in deleted_urls), (
        "error_file_id must be deleted during batch cleanup"
    )


@pytest.mark.fast
def test_openai_chat_batch_result_preserves_http_error() -> None:
    transport = OpenAIChatTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    batch_status_resp = MagicMock()
    batch_status_resp.status_code = 200
    batch_status_resp.json.return_value = {
        "id": "batch_abc",
        "status": "completed",
        "output_file_id": "file-out",
    }

    # One line failed with HTTP 400
    line_json = {
        "custom_id": "req-1",
        "response": {
            "status_code": 400,
            "body": {
                "error": {
                    "message": "Context length exceeded",
                    "type": "invalid_request_error",
                }
            },
        },
        "error": None,
    }
    file_content_resp = MagicMock()
    file_content_resp.status_code = 200
    file_content_resp.text = json.dumps(line_json) + "\n"

    async def fake_get(url: str, **kwargs: object) -> MagicMock:
        if "/batches/" in url:
            return batch_status_resp
        return file_content_resp

    mock_client.get = fake_get
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    results = asyncio.run(transport.fetch_batch_results("batch_abc"))
    assert "req-1" in results
    err = results["req-1"].get("error")
    assert err is not None
    assert "Context length exceeded" in err, f"Expected actual error message, got {err}"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_openai_responses_transport_caches_reasoning_fallback_after_first_400() -> None:
    """Once a model on /responses returns 400 for reasoning_effort and succeeds on retry,
    subsequent calls for that model must use the working reasoning format on the first request."""
    from ubt.core.router.transports.openai_responses import OpenAIResponsesTransport

    transport = OpenAIResponsesTransport(
        api_key="test",
        base_url="https://opencode.ai/zen/go/v1",
        reasoning_dialect="flat",
    )
    sent_payloads: list[dict[str, object]] = []

    async def fake_request_json(_client: object, _url: str, payload: dict[str, object]) -> object:
        sent_payloads.append(dict(payload))
        resp = MagicMock()
        if "reasoning_effort" in payload:
            resp.status_code = 400
            resp.text = '{"error": "unrecognized field reasoning_effort"}'
        else:
            resp.status_code = 200
            resp.json.return_value = {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "translated text"}],
                    }
                ]
            }
        return resp

    transport._request_json = fake_request_json  # type: ignore[assignment]

    out1, _ = await transport._generate_responses_meta(
        prompt="hi",
        system_prompt=None,
        target_model="muse-spark-1.3-contributor",
        temperature=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )
    assert out1 == "translated text"
    assert len(sent_payloads) == 2

    out2, _ = await transport._generate_responses_meta(
        prompt="hello",
        system_prompt=None,
        target_model="muse-spark-1.3-contributor",
        temperature=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )
    assert out2 == "translated text"
    assert len(sent_payloads) == 3


@pytest.mark.fast
@pytest.mark.asyncio
async def test_the_reasoning_dialect_comes_from_data_not_the_host() -> None:
    """Two transports on the *same* host must spell an effort differently when
    their declared dialect differs.

    The Zen gateway used to be recognised by hostname and rewritten in the
    transport; the dialect is now endpoint data, so the host is irrelevant.
    """
    from ubt.core.router.transports.openai_responses import OpenAIResponsesTransport

    async def first_payload(dialect: str) -> dict[str, object]:
        transport = OpenAIResponsesTransport(
            api_key="test",
            base_url="https://opencode.ai/zen/go/v1",
            reasoning_dialect=dialect,
        )
        sent: list[dict[str, object]] = []

        async def fake_request_json(
            _client: object, _url: str, payload: dict[str, object]
        ) -> object:
            sent.append(dict(payload))
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]
            }
            return resp

        transport._request_json = fake_request_json  # type: ignore[assignment]
        await transport._generate_responses_meta(
            prompt="hi",
            system_prompt=None,
            target_model="some-model",
            temperature=0.1,
            max_tokens=10,
            reasoning_effort="low",
        )
        return sent[0]

    nested = await first_payload("nested")
    flat = await first_payload("flat")
    assert nested["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in nested
    assert flat["reasoning_effort"] == "low"
    assert "reasoning" not in flat


def test_the_protocol_comes_from_api_mode_alone() -> None:
    """Each of the four protocols maps to its own transport."""
    from ubt.core.router.provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(api_key="k", base_url="https://x.example/v1")
    assert provider._select_transport() is provider._chat_transport

    for mode, attr in (
        ("openai-responses", "_responses_transport"),
        ("anthropic-messages", "_anthropic_transport"),
        ("gemini-native", "_gemini_transport"),
    ):
        configured = OpenAICompatibleProvider(
            api_key="k", base_url="https://x.example/v1", api_mode=mode
        )
        assert configured._select_transport() is getattr(configured, attr)


def test_a_model_name_does_not_switch_the_transport() -> None:
    """``muse-`` used to upgrade a chat provider to the responses wire.

    A fallback chain that lands on another model family must not silently change
    protocol mid-run, so a provider whose default model is a ``muse-`` one still
    speaks the protocol its ``api_mode`` names.
    """
    from ubt.core.router.provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://x.example/v1",
        default_model="muse-spark-1.3-contributor",
        api_mode="openai-chat",
    )
    assert provider._select_transport() is provider._chat_transport
