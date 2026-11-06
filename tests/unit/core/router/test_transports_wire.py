from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.anthropic import AnthropicMessagesTransport
from ubt.core.router.transports.gemini import GeminiTransport
from ubt.core.router.transports.openai_chat import OpenAIChatTransport
from ubt.core.router.transports.openai_responses import OpenAIResponsesTransport

pytestmark = pytest.mark.fast


class _Recorder:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self._handler = handler
        self.transport = httpx.MockTransport(self._dispatch)

    async def _dispatch(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raw = request.content
        if raw:
            try:
                self.bodies.append(json.loads(raw))
            except ValueError:
                self.bodies.append({})
        else:
            self.bodies.append({})
        return self._handler(request)


def _chat_ok(content: str = "译文") -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    return handler


async def test_openai_chat_sends_the_expected_request() -> None:
    rec = _Recorder(_chat_ok())
    t = OpenAIChatTransport(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        default_model="m-1",
        transport=rec.transport,
    )
    text, finish = await t.generate_with_finish_reason("hello", system_prompt="sys")
    assert text == "译文"
    assert finish == "stop"

    req = rec.requests[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = rec.bodies[0]
    assert body["model"] == "m-1"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]


async def test_openai_chat_does_not_double_the_bearer_prefix() -> None:
    rec = _Recorder(_chat_ok())
    t = OpenAIChatTransport(
        api_key="Bearer sk-test",
        base_url="https://x/v1",
        default_model="m",
        transport=rec.transport,
    )
    await t.generate_with_finish_reason("hi")
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test"


async def test_openai_chat_marks_a_refused_connection_as_fail_fast() -> None:

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="http://127.0.0.1:2345/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError) as exc:
        await t.generate_with_finish_reason("hi")
    assert exc.value.details == {
        "fail_fast": True,
        "reason": "transport_unreachable",
    }
    assert "HTTP connection failed" in str(exc.value)


async def test_openai_chat_maps_429_to_a_retryable_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError) as exc:
        await t.generate_with_finish_reason("hi")
    assert exc.value.details["status_code"] == 429
    assert exc.value.details["retry_after"] == "7"


async def test_openai_chat_rejects_a_null_content_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": None, "reasoning_content": "hmm"}}],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError, match="no message content"):
        await t.generate_with_finish_reason("hi")


async def test_openai_chat_marks_a_reasoning_budget_exhaustion_retryable() -> None:

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": None, "reasoning_content": "thinking"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1024,
                    "completion_tokens_details": {"reasoning_tokens": 1024},
                },
            },
        )

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError) as exc:
        await t.generate_with_finish_reason("hi", max_tokens=1024)
    assert exc.value.details["reasoning_budget_exhausted"] is True
    assert exc.value.details["reasoning_tokens"] == 1024


async def test_openai_chat_usage_is_recorded() -> None:
    rec = _Recorder(_chat_ok())
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    await t.generate_with_finish_reason("hi")
    assert t.usage_log, "usage was not recorded"
    assert t.usage_log[-1]["prompt_tokens"] == 10


async def test_openai_chat_renames_max_tokens_for_a_gpt5_endpoint() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "Unsupported parameter: 'max_tokens' is not supported with this "
                            "model. Use 'max_completion_tokens' instead."
                        )
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5",
        transport=rec.transport,
    )
    text, _ = await t.generate_with_finish_reason("hi", max_tokens=64)
    assert text == "译文"
    assert calls[0]["max_tokens"] == 64
    assert "max_tokens" not in calls[1]
    assert calls[1]["max_completion_tokens"] == 64

    # The rename must be *remembered*, not rediscovered per call. It used to be
    # excluded from the learned set (a rename is not an omission: dropping the
    # key would lose the cap), so every later block paid the same 400 again —
    # measured, three calls produced six requests against this handler.
    before = len(calls)
    text, _ = await t.generate_with_finish_reason("hi", max_tokens=64)
    assert text == "译文"
    assert len(calls) - before == 1, "a learned rename must not cost a 400 on every call"
    assert calls[-1]["max_completion_tokens"] == 64
    assert "max_tokens" not in calls[-1]


async def test_openai_chat_surfaces_a_400_that_is_not_a_max_tokens_rename() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError) as exc:
        await t.generate_with_finish_reason("hi", max_tokens=64)
    assert exc.value.details["status_code"] == 400
    assert len(calls) == 1


async def test_openai_responses_reads_the_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "resp-translation"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk",
        base_url="https://api.example.com/v1",
        default_model="m",
        transport=rec.transport,
    )
    text = await t.generate("hello")
    assert text == "resp-translation"
    assert str(rec.requests[0].url).endswith("/responses")
    body = rec.bodies[0]
    assert body["model"] == "m"
    assert body["input"] == "hello"


async def test_openai_responses_refusal_only_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
                    }
                ],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk",
        base_url="https://api.example.com/v1",
        default_model="m",
        transport=rec.transport,
    )
    with pytest.raises(ModelProviderError, match="empty message"):
        await t.generate("hello")


async def test_anthropic_reads_the_content_blocks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "anthropic-translation"}],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )

    rec = _Recorder(handler)
    t = AnthropicMessagesTransport(
        api_key="sk-ant",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-x",
        transport=rec.transport,
    )
    text = await t.generate("hello", system_prompt="sys")
    assert text == "anthropic-translation"

    req = rec.requests[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert "anthropic-version" in {k.lower() for k in req.headers}
    assert req.headers["x-api-key"] == "sk-ant"
    body = rec.bodies[0]
    assert body["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_tokens"] >= 1


async def test_gemini_reads_the_candidate_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "gemini-translation"}]}}],
                "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 3},
            },
        )

    rec = _Recorder(handler)
    t = GeminiTransport(
        api_key="gk",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-x",
        transport=rec.transport,
    )
    text = await t.generate("hello")
    assert text == "gemini-translation"
    req = rec.requests[0]
    assert req.headers["x-goog-api-key"] == "gk"
    assert "key=" not in str(req.url)
    assert ":generateContent" in str(req.url)


@pytest.mark.parametrize(
    "factory",
    [
        lambda rec: OpenAIChatTransport(
            api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
        ),
        lambda rec: OpenAIResponsesTransport(
            api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
        ),
        lambda rec: AnthropicMessagesTransport(
            api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
        ),
        lambda rec: GeminiTransport(
            api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
        ),
    ],
)
async def test_every_transport_raises_a_provider_error_on_a_500(factory: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    rec = _Recorder(handler)
    t = factory(rec)
    with pytest.raises(ModelProviderError) as exc:
        await t.generate("hi")
    assert exc.value.details.get("status_code") == 500


async def test_openai_responses_effort_none_in_nested_dialect_sends_reasoning_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    await t.generate("hi", reasoning_effort="none")
    assert rec.bodies[0]["reasoning"] == {"effort": "none"}


async def test_openai_responses_flat_dialect_uses_the_flat_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk",
        base_url="https://x/v1",
        default_model="m",
        transport=rec.transport,
        reasoning_dialect="flat",
    )
    await t.generate("hi", reasoning_effort="high")
    assert rec.bodies[0]["reasoning_effort"] == "high"
    assert "reasoning" not in rec.bodies[0]


async def test_openai_responses_a_400_that_names_another_argument_does_not_latch() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw = json.loads(request.content)
        calls.append(raw)
        return httpx.Response(400, json={"error": {"message": "Unrecognized: seed"}})

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError):
        await t.generate("hi", reasoning_effort="high")
    assert t._model_reasoning_mode == {}
    assert len(calls) == 2
    assert "reasoning" in calls[0]
    assert "reasoning" not in calls[1]


async def test_openai_responses_nested_ladder_drops_reasoning_and_latches_none() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw = json.loads(request.content)
        calls.append(raw)
        if "reasoning" in raw:
            return httpx.Response(400, json={"error": {"message": "extra_forbidden: reasoning"}})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    text = await t.generate("hi", reasoning_effort="high")
    assert text == "ok"
    assert t._model_reasoning_mode["m"] == "none"
    await t.generate("again", reasoning_effort="high")
    assert "reasoning" not in rec.bodies[-1]


async def test_openai_responses_truncation_returns_length_not_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    text, finish = await t.generate_with_finish_reason("hi")
    assert text == ""
    assert finish == "length"


async def test_openai_responses_malformed_output_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "completed", "output": [], "usage": {}})

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError, match="Malformed"):
        await t.generate("hi")


async def test_openai_responses_vision_sends_input_image_parts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "seen"}]}
                ],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="sk", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    text = await t.generate_with_images("describe", ["QUJD"])
    assert text == "seen"
    content = rec.bodies[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "describe"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].endswith("QUJD")
    with pytest.raises(ModelProviderError, match="at least one image"):
        await t.generate_with_images("describe", [])


async def test_anthropic_reasoning_effort_maps_to_a_thinking_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "text", "text": "done"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )

    rec = _Recorder(handler)
    t = AnthropicMessagesTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    text, finish = await t.generate_with_finish_reason("hi", reasoning_effort="medium")
    assert text == "done"
    assert finish == "stop"
    body = rec.bodies[0]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert body["max_tokens"] >= 4096 + 1024
    assert "temperature" not in body


async def test_anthropic_refusal_stop_reason_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"content": [], "stop_reason": "refusal", "usage": {}},
        )

    rec = _Recorder(handler)
    t = AnthropicMessagesTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError, match="refused"):
        await t.generate("hi")


async def test_anthropic_empty_content_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn", "usage": {}})

    rec = _Recorder(handler)
    t = AnthropicMessagesTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError, match="no content blocks"):
        await t.generate("hi")


async def test_anthropic_vision_sends_base64_image_blocks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "seen"}],
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = AnthropicMessagesTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    text = await t.generate_with_images("describe", ["QUJD"])
    assert text == "seen"
    content = rec.bodies[0]["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    assert content[1] == {"type": "text", "text": "describe"}
    with pytest.raises(ModelProviderError, match="at least one image"):
        await t.generate_with_images("describe", [])


async def test_anthropic_length_stop_reason_maps_to_finish_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "cut off"}],
                "stop_reason": "max_tokens",
                "usage": {},
            },
        )

    rec = _Recorder(handler)
    t = AnthropicMessagesTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    _, finish = await t.generate_with_finish_reason("hi")
    assert finish == "length"


@pytest.mark.asyncio
async def test_transport_client_timeout_is_split() -> None:
    transport = OpenAIChatTransport(
        api_key="k", base_url="https://api.example.com/v1", timeout=180.0
    )
    client = transport._get_client()
    assert client.timeout.connect == 15.0
    assert client.timeout.read == 180.0
    assert client.timeout.write == 15.0
    assert client.timeout.pool == 30.0
    await transport.aclose()


@pytest.mark.asyncio
async def test_transport_client_read_defaults_when_api_timeout_is_small() -> None:
    transport = OpenAIChatTransport(
        api_key="k", base_url="https://api.example.com/v1", timeout=30.0
    )
    client = transport._get_client()
    assert client.timeout.read == 120.0
    await transport.aclose()


async def test_provider_shared_client_splits_connect_timeout() -> None:
    from ubt.core.router.provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(api_key="k", timeout=180.0)
    client = provider._shared_client
    assert client.timeout.connect == 15.0
    assert client.timeout.read == 180.0
    assert client.timeout.write == 15.0
    assert client.timeout.pool == 30.0
    await client.aclose()


async def test_local_transport_ignores_ambient_socks_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:10808")
    transport = OpenAIChatTransport(api_key="k", base_url="http://127.0.0.1:9090/v1")
    client = transport._get_client()
    assert client._trust_env is False
    await transport.aclose()


async def test_remote_transport_still_honours_ambient_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:10808")
    transport = OpenAIChatTransport(api_key="k", base_url="https://api.example.com/v1")
    client = transport._get_client()
    assert client._trust_env is True
    await transport.aclose()


async def test_local_provider_shared_client_ignores_ambient_socks_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.core.router.provider import OpenAICompatibleProvider

    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:10808")
    provider = OpenAICompatibleProvider(api_key="k", base_url="http://localhost:9090/v1")
    assert provider._shared_client._trust_env is False
    await provider._shared_client.aclose()


def test_retry_after_ms_is_converted_to_seconds() -> None:
    from ubt.core.router.transports.base import retry_after_seconds

    assert retry_after_seconds(httpx.Headers({"retry-after": "20"})) == "20"
    assert retry_after_seconds(httpx.Headers({"retry-after-ms": "20000"})) == "20.0"
    assert retry_after_seconds(httpx.Headers({})) is None
    assert retry_after_seconds(httpx.Headers({"retry-after-ms": "abc"})) is None


async def test_a_rejected_parameter_is_remembered_across_calls() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "temperature" in body:
            return httpx.Response(
                400,
                json={
                    "error": {"message": "Unsupported parameter: 'temperature' is not supported"}
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="o3", transport=rec.transport
    )

    await t.generate_with_finish_reason("hi", temperature=0.3)
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]

    calls.clear()
    await t.generate_with_finish_reason("hi", temperature=0.3)
    assert len(calls) == 1, "the second call paid the same 400 again"
    assert "temperature" not in calls[0]


async def test_the_learned_omission_is_per_model() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if body.get("model") == "strict-model" and "temperature" in body:
            return httpx.Response(
                400, json={"error": {"message": "'temperature' unsupported parameter"}}
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="strict-model", transport=rec.transport
    )
    await t.generate_with_finish_reason("hi", model="strict-model", temperature=0.3)

    calls.clear()
    await t.generate_with_finish_reason("hi", model="lenient-model", temperature=0.3)
    assert calls[0]["temperature"] == 0.3


async def test_an_unrelated_400_teaches_nothing() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k", base_url="https://x/v1", default_model="m", transport=rec.transport
    )
    with pytest.raises(ModelProviderError):
        await t.generate_with_finish_reason("hi", temperature=0.3)
    assert t._learned_omissions == {}


async def test_openai_responses_heals_a_rejected_temperature() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "temperature" in body:
            return httpx.Response(
                400, json={"error": {"message": "'temperature' unsupported parameter"}}
            )
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "译文"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIResponsesTransport(
        api_key="k", base_url="https://x/v1", default_model="gpt-5", transport=rec.transport
    )
    text, _ = await t.generate_with_finish_reason("hi", temperature=0.3)
    assert text == "译文"
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


async def test_temperature_costs_exactly_one_400_per_model() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "temperature" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported value: 'temperature' does not support 0.3 "
                        "with this model."
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k",
        base_url="https://x/v1",
        default_model="reasoning-model",
        transport=rec.transport,
    )
    for _ in range(4):
        await t.generate_with_finish_reason("hi", temperature=0.3)

    assert len(calls) == 5, f"expected 4 blocks + 1 rejected probe, got {len(calls)}"
    assert "temperature" in calls[0]
    assert not any("temperature" in body for body in calls[1:]), "the 400 was paid twice"
    assert t._learned_omissions["reasoning-model"] >= {"temperature"}


async def test_a_rejected_system_role_is_not_self_healed() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if any(m.get("role") == "system" for m in body.get("messages", [])):
            return httpx.Response(
                400,
                json={"error": {"message": "Invalid role: system messages are not supported"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    rec = _Recorder(handler)
    t = OpenAIChatTransport(
        api_key="k",
        base_url="https://x/v1",
        default_model="no-system-model",
        transport=rec.transport,
    )
    with pytest.raises(ModelProviderError):
        await t.generate_with_finish_reason("hi", system_prompt="You are a translator.")
    assert len(calls) == 1
