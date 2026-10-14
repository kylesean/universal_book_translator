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
