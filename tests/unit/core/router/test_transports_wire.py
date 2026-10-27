"""Wire-contract tests for the four provider transports.

The transports carry every billable byte the product sends, and they were the
largest untested surface in the tree (13–25% line coverage): a wrong URL, a
missing auth header, or a misread response field is invisible until a real run
bills real money and returns an empty draft.

``BaseTransport`` accepts an ``httpx.AsyncBaseTransport``, so each test drives
the transport over ``httpx.MockTransport`` — a real request is built, serialized
and dispatched, and the real response is parsed. No socket, no network (the
suite's autouse ``_block_network`` fixture would fail a real connect).
"""

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
    """Captures each request and answers from a scripted handler."""

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


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions
# --------------------------------------------------------------------------- #


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
    # A key that already carries the prefix must not gain a second one, or the
    # provider 401s.
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
    """A dead endpoint must not be retried once per block.

    ``httpx.ConnectError`` used to become a generic provider error with no
    structured status, so the router treated it as transient. A stopped local
    model then burned the retry ladder and the draft fail-fast breaker counted
    nothing, turning one configuration mistake into an entire-book failure.
    """

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
    # A null content is a content-filter refusal, never a translation; returning
    # reasoning_content here would ship the chain-of-thought as the draft.
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
    """Null content after spending the budget on reasoning is not a refusal.

    ``hy3`` returns ``content=None`` with ``finish_reason=length`` and all
    completion tokens attributed to reasoning. The router needs to widen the
    budget; treating this as a content refusal fails every block instead.
    """

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
    """P2-9: the gpt-5 family rejects ``max_tokens``; it must be healed.

    OpenAI answers a ``max_tokens`` on a gpt-5 / o-series model with a 400 that
    names ``max_completion_tokens`` as the replacement. Without a heal, every
    call to a model the pricing table advertises failed the whole run.
    """
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
    # First attempt used max_tokens (rejected); the healed retry used the
    # replacement name and carried the same value.
    assert calls[0]["max_tokens"] == 64
    assert "max_tokens" not in calls[1]
    assert calls[1]["max_completion_tokens"] == 64


async def test_openai_chat_surfaces_a_400_that_is_not_a_max_tokens_rename() -> None:
    """The heal must not fire on an unrelated 400."""
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
    assert len(calls) == 1  # no futile retry


# --------------------------------------------------------------------------- #
# OpenAI Responses
# --------------------------------------------------------------------------- #


async def test_openai_responses_reads_the_structured_output() -> None:
    # The raw Responses API returns ``output`` items, each with typed content
    # parts (``output_text`` / ``refusal``); there is no top-level
    # ``output_text`` field on the wire.
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
    # A refusal-only message must surface, not ship "" as a finished block.
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


# --------------------------------------------------------------------------- #
# Anthropic Messages
# --------------------------------------------------------------------------- #


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
    # Anthropic requires the version header on every call.
    assert "anthropic-version" in {k.lower() for k in req.headers}
    assert req.headers["x-api-key"] == "sk-ant"
    body = rec.bodies[0]
    # Prompt caching is on by default, so ``system`` is a typed block carrying
    # ``cache_control`` rather than a bare string.
    assert body["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_tokens"] >= 1


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


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
    # The key travels in a header (``x-goog-api-key``), not a ``?key=`` query
    # string, so it cannot leak into logs or proxies that record URLs.
    assert req.headers["x-goog-api-key"] == "gk"
    assert "key=" not in str(req.url)
    assert ":generateContent" in str(req.url)


# --------------------------------------------------------------------------- #
# Cross-transport invariants
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# OpenAI Responses: the reasoning fallback ladder (regression surface: a wrong
# latch silently disables reasoning for the model for the rest of the process)
# --------------------------------------------------------------------------- #


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
    """An unrelated 400 fails every retry identically and latches nothing.

    Latching "none" on it would silently disable reasoning for the model for
    the rest of the process (the bug the latch-on-200-only comment warns of).
    """
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
    # No mode was latched: the next call would still send reasoning.
    assert t._model_reasoning_mode == {}
    # The ladder tried: nested -> without-reasoning (2 requests, both 400).
    assert len(calls) == 2
    assert "reasoning" in calls[0]
    assert "reasoning" not in calls[1]


async def test_openai_responses_nested_ladder_drops_reasoning_and_latches_none() -> None:
    """A reasoning-rejecting endpoint: first 400, then success without reasoning."""
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
    # A second call skips reasoning entirely (the latch is honored) -- only
    # the first call's 400 carried a reasoning key.
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


# --------------------------------------------------------------------------- #
# Anthropic Messages: reasoning budget mapping, refusal, vision
# --------------------------------------------------------------------------- #


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
    # The effort enum became a token budget; max_tokens must exceed it and the
    # custom temperature had to go (Anthropic forbids the pair).
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
    """Connect/write/pool fail fast; read gets the long window (api_timeout wins).

    The forced split is connect=15, read=max(120, api_timeout), write=15,
    pool=30. A single-float timeout used to pin a pool slot for the full 180s on
    a dead endpoint before the connect phase gave up.
    """
    transport = OpenAIChatTransport(
        api_key="k", base_url="https://api.example.com/v1", timeout=180.0
    )
    client = transport._get_client()
    assert client.timeout.connect == 15.0
    assert client.timeout.read == 180.0  # api_timeout(180) > default read(120)
    assert client.timeout.write == 15.0
    assert client.timeout.pool == 30.0
    await transport.aclose()


@pytest.mark.asyncio
async def test_transport_client_read_defaults_when_api_timeout_is_small() -> None:
    transport = OpenAIChatTransport(
        api_key="k", base_url="https://api.example.com/v1", timeout=30.0
    )
    client = transport._get_client()
    assert client.timeout.read == 120.0  # default floor beats a small api_timeout
    await transport.aclose()


async def test_provider_shared_client_splits_connect_timeout() -> None:
    """The provider's shared client must keep the fast per-phase split.

    A bare ``timeout=self._timeout`` here silently overrode every transport's
    own split (``BaseTransport._get_client`` returns the injected client
    unchanged), pinning a pool slot for the full api_timeout on a dead endpoint.
    """
    from ubt.core.router.provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(api_key="k", timeout=180.0)
    client = provider._shared_client
    assert client.timeout.connect == 15.0
    assert client.timeout.read == 180.0  # api_timeout wins over the 120 default
    assert client.timeout.write == 15.0
    assert client.timeout.pool == 30.0
    await client.aclose()


async def test_local_transport_ignores_ambient_socks_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A localhost inference server must not inherit a shell's SOCKS proxy."""
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
    """A ``Retry-After-Ms`` header is milliseconds; the router sleeps seconds.

    Passing ``20000`` through verbatim made a 429 retry sleep ~5.5 hours.
    """
    from ubt.core.router.transports.base import retry_after_seconds

    assert retry_after_seconds(httpx.Headers({"retry-after": "20"})) == "20"
    assert retry_after_seconds(httpx.Headers({"retry-after-ms": "20000"})) == "20.0"
    assert retry_after_seconds(httpx.Headers({})) is None
    assert retry_after_seconds(httpx.Headers({"retry-after-ms": "abc"})) is None


async def test_a_rejected_parameter_is_remembered_across_calls() -> None:
    """A 400 heal must be paid once per model, not once per block.

    The heal rules fixed the *retry* but nothing remembered the outcome, so a
    model that rejects ``temperature`` (the gpt-5 / o-series endpoints do) sent it
    again on the next block and ate another 400. On a 137-block job that is 137
    wasted round-trips.
    """
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

    # First block: rejected, healed.
    await t.generate_with_finish_reason("hi", temperature=0.3)
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]

    # Second block: the parameter is never sent again.
    calls.clear()
    await t.generate_with_finish_reason("hi", temperature=0.3)
    assert len(calls) == 1, "the second call paid the same 400 again"
    assert "temperature" not in calls[0]


async def test_the_learned_omission_is_per_model() -> None:
    """Evidence about one endpoint must not strip a parameter from another."""
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
    """Only a *proven* outcome may be recorded.

    An unrelated 400 ("context length exceeded") fails every retry identically;
    recording its removals would silently disable a parameter the endpoint
    accepts for the rest of the process.
    """
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
    """The responses transport had no temperature rule at all.

    A model whose chat-completions sibling healed fine failed every call here
    with a permanent 400.
    """
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
    """The measurement that made ``supports_temperature`` an optional control.

    A rejected temperature is dropped by the self-heal and then remembered, so the
    entire cost of not declaring it is one failed call per model per process. Four
    blocks therefore take five requests, not eight, and the parameter never
    reappears. This is why asking an operator to answer the question was wrong: the
    engine answers it itself.
    """
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
    """The one capability the engine genuinely cannot discover.

    Dropping the parameter is mechanical; merging the system text into the user
    turn is a decision. With no heal rule the call fails outright — which is why
    ``supports_system_prompt`` is the only flag the console asks for plainly.
    """
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
    # Exactly one attempt: there is no rule that could retry it differently.
    assert len(calls) == 1
