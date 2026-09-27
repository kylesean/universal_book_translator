"""Unit tests for ModelRouter and prompt construction."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from ubt.core.exceptions import ModelProviderError
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.router.capabilities import (
    ExtractionStrategy,
    ModelProfile,
    PromptStrategy,
)
from ubt.core.router.pricing import cache_hit_rate_from_usage
from ubt.core.router.provider import BaseModelProvider, MockModelProvider, OpenAICompatibleProvider
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.registry import ModelCapabilityRegistry
from ubt.core.router.router import ModelRouter


@pytest.mark.asyncio
async def test_model_router_draft_and_repair_tier_dispatch() -> None:
    mock_provider = MockModelProvider(prefix="[TRANSLATED] ")
    router = ModelRouter(
        provider=mock_provider,
        draft_model="cheap-flash",
        repair_model="flagship-pro",
    )

    block = IRBlock(
        id="b01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="The sky was dark.",
        status=BlockStatus.PENDING,
    )

    # 1. Test draft routing
    draft_res = await router.draft(
        block=block,
        glossary_table="| Sky | 天空 |",
        neighbor_context="[READ-ONLY CONTEXT]",
        target_lang="zh",
    )
    assert "[TRANSLATED]" in draft_res
    assert len(mock_provider.call_history) == 1
    assert mock_provider.call_history[0]["model"] == "cheap-flash"
    assert mock_provider.call_history[0]["reasoning_effort"] == "low"
    assert "Translation Bible" in mock_provider.call_history[0]["prompt"]
    assert "[READ-ONLY CONTEXT]" in mock_provider.call_history[0]["prompt"]

    # 2. Test repair routing
    repair_res = await router.repair(
        block=block,
        draft_text="天空是黑的。",
        error_flags=["html_tag_mismatch", "unnatural_phrasing"],
        glossary_table="| Sky | 天空 |",
        target_lang="zh",
    )
    assert "[TRANSLATED]" in repair_res
    assert len(mock_provider.call_history) == 2
    assert mock_provider.call_history[1]["model"] == "flagship-pro"
    assert mock_provider.call_history[1]["reasoning_effort"] == "high"
    assert "Quality Critique Issues" in mock_provider.call_history[1]["prompt"]
    assert "html_tag_mismatch" in mock_provider.call_history[1]["prompt"]
    assert "<final_translation>" in mock_provider.call_history[1]["prompt"]


@pytest.mark.asyncio
async def test_model_router_heterogeneous_repair_provider() -> None:
    """Verify distinct draft and repair providers receive respective requests."""
    draft_provider = MockModelProvider(prefix="[DRAFT] ")
    repair_provider = MockModelProvider(prefix="[REPAIR] ")
    router = ModelRouter(
        provider=draft_provider,
        repair_provider=repair_provider,
        draft_model="local-vllm",
        repair_model="claude-3-7-sonnet",
    )
    block = IRBlock(
        id="b01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="The sky was dark.",
        status=BlockStatus.PENDING,
    )
    draft_res = await router.draft(block=block, glossary_table="", target_lang="zh")
    assert "[DRAFT]" in draft_res
    assert len(draft_provider.call_history) == 1
    assert len(repair_provider.call_history) == 0

    repair_res = await router.repair(
        block=block,
        draft_text=draft_res,
        error_flags=["grammar"],
        glossary_table="",
        target_lang="zh",
    )
    assert "[REPAIR]" in repair_res
    assert len(draft_provider.call_history) == 1
    assert len(repair_provider.call_history) == 1


class FailingThenSucceedingProvider(BaseModelProvider):
    """Simulates transient 429 then success."""

    def __init__(self) -> None:
        self.attempts = 0

    @property
    def provider_name(self) -> str:
        return "transient"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        self.attempts += 1
        if self.attempts == 1:
            raise ModelProviderError("Rate limited", details={"status_code": 429})
        return "Recovered after 429"


@pytest.mark.asyncio
async def test_model_router_recovers_from_429_with_rate_limiter() -> None:
    provider = FailingThenSucceedingProvider()
    rate_limiter = AdaptiveTokenBucket(initial_rpm=600)
    router = ModelRouter(provider=provider, rate_limiter=rate_limiter, max_retries=2)

    block = IRBlock(
        id="b02",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Hello world",
    )

    result = await router.draft(block)
    assert result == "Recovered after 429"
    assert provider.attempts == 2
    assert rate_limiter.consecutive_429 == 0


@pytest.mark.asyncio
async def test_openai_compatible_provider_records_usage() -> None:
    """Usage accounting: tokens from the API response land in usage_log/totals."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "  译文内容  "}}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 45,
                    "total_tokens": 165,
                    "prompt_cache_hit_tokens": 80,
                    "prompt_cache_miss_tokens": 40,
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        transport=httpx.MockTransport(handler),
    )

    out = await provider.generate("hello", system_prompt="sys")

    assert out == "译文内容"  # response content is stripped
    assert provider.usage_totals == {
        "calls": 1,
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "cached_tokens": 80,
    }
    assert provider.usage_log[0]["model"] == "deepseek-chat"
    assert provider.usage_log[0]["prompt_cache_hit_tokens"] == 80


@pytest.mark.asyncio
async def test_openai_compatible_provider_usage_without_usage_field() -> None:
    """APIs that omit `usage` must not break generation — and must not look free.

    The tokens of such a call cannot be sized, so it is booked as *unmeasured*
    instead of zero: ``estimate_cost_usd`` then reports the run's cost as
    unknown rather than a confident $0.00 that would leave UBT_BUDGET_USD
    unable to fire.
    """
    import httpx

    from ubt.core.router.pricing import estimate_cost_usd
    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        transport=httpx.MockTransport(handler),
    )
    out = await provider.generate("hello")
    assert out == "ok"
    assert provider.usage_totals == {
        "calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "unmeasured_calls": 1,
    }
    assert estimate_cost_usd(provider.usage_totals_by_model) is None
    # Same model, a response that does report usage: cost comes back numeric.
    assert (
        estimate_cost_usd(
            {
                "deepseek-chat": {
                    "calls": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                }
            }
        )
        is not None
    )


@pytest.mark.asyncio
async def test_chat_template_kwargs_forwarded_to_payload() -> None:
    """llama.cpp/vLLM chat_template_kwargs must be forwarded verbatim into the chat body."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        return httpx.Response(200, json={"choices": [{"message": {"content": "你好"}}]})

    provider = OpenAICompatibleProvider(
        api_key="test-key",
        # The local stack's one endpoint: llama-swap assigns the backend its own
        # ephemeral port, so the old :8081 is not an address any client uses.
        base_url="http://127.0.0.1:9090/v1",
        default_model="bonsai2-27b",
        chat_template_kwargs={"enable_thinking": False},
        transport=httpx.MockTransport(handler),
    )
    out = await provider.generate("hello")

    assert out == "你好"
    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_chat_template_kwargs_omitted_when_unset() -> None:
    """Default provider (no chat_template_kwargs) must not inject the key."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(api_key="test-key", transport=httpx.MockTransport(handler))
    await provider.generate("hello")
    assert "chat_template_kwargs" not in captured["body"]


@pytest.mark.asyncio
async def test_responses_api_mode_records_usage_and_text() -> None:
    """Responses API (/responses) mode: reasoning-only outputs are skipped,
    message text extracted, usage mapped from input/output tokens."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path.endswith("/responses")
        assert body["input"] == "hello"
        assert body["instructions"] == "sys"
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "status": "completed", "encrypted_content": "..."},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "  译文  "}],
                    },
                ],
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 50,
                    "input_tokens_details": {"cached_tokens": 120},
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://opencode.ai/zen/go/v1",
        default_model="muse-spark-1.3-contributor",
        api_mode="responses",
        transport=httpx.MockTransport(handler),
    )
    out = await provider.generate("hello", system_prompt="sys")
    assert out == "译文"
    assert provider.usage_totals == {
        "calls": 1,
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "cached_tokens": 120,
    }
    assert provider.usage_log[0]["prompt_cache_hit_tokens"] == 120
    assert provider.usage_log[0]["prompt_cache_miss_tokens"] == 80


@pytest.mark.asyncio
async def test_responses_api_mode_error_payload_raises() -> None:
    import httpx

    from ubt.core.exceptions import ModelProviderError
    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"type": "error", "error": {"message": "unsupported"}})

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://x/v1",
        api_mode="responses",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelProviderError):
        await provider.generate("hi")


@pytest.mark.asyncio
async def test_responses_api_mode_recovers_on_400_reasoning_effort() -> None:
    """Responses API mode gracefully drops reasoning_effort if provider rejects with 400."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_payloads.append(body)
        if len(request_payloads) == 1 and "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "unknown parameter: reasoning_effort"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_fallback",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "  降级成功  "}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://opencode.ai/zen/go/v1",
        default_model="muse-spark-1.3-contributor",
        api_mode="responses",
        transport=httpx.MockTransport(handler),
    )
    out = await provider.generate("hello", reasoning_effort="low")
    assert out == "降级成功"
    assert len(request_payloads) == 2
    assert request_payloads[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in request_payloads[1]


@pytest.mark.asyncio
async def test_responses_api_mode_passes_minimal_reasoning_directly() -> None:
    """Responses API mode formats reasoning_effort='minimal' as payload['reasoning']={'effort':'minimal'}."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_payloads.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_minimal",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "精简思考译文"}],
                    }
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 80,
                    "output_tokens_details": {"reasoning_tokens": 60},
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://opencode.ai/zen/v1",
        default_model="muse-spark-1.3-contributor-free",
        api_mode="responses",
        transport=httpx.MockTransport(handler),
    )
    out = await provider.generate("hello", reasoning_effort="minimal")
    assert out == "精简思考译文"
    assert len(request_payloads) == 1
    assert request_payloads[0].get("reasoning") == {"effort": "minimal"}
    assert "reasoning_effort" not in request_payloads[0]


def test_sanitize_thought_output_cleans_traces_and_prefixes() -> None:
    from ubt.core.router.provider import sanitize_thought_output

    # 1. Strip reasoning tags (<think>, <thought>, <thinking>, <reasoning>)
    raw_with_think = "<think>Let me ponder this translation carefully...</think>天空是黑色的。"
    assert sanitize_thought_output(raw_with_think) == "天空是黑色的。"

    raw_with_thought = "<thought>Gemini style thinking...</thought>知识就是力量。"
    assert sanitize_thought_output(raw_with_thought) == "知识就是力量。"

    raw_with_claude_thinking = "<thinking>Claude 3.7 hybrid deliberation</thinking>代码即诗歌。"
    assert sanitize_thought_output(raw_with_claude_thinking) == "代码即诗歌。"

    raw_with_reasoning = "<reasoning>Step 1... Step 2...</reasoning>万物皆有裂痕。"
    assert sanitize_thought_output(raw_with_reasoning) == "万物皆有裂痕。"

    # 2. Extract <final_translation>
    raw_with_tags = (
        "<think>Checking glossary terms...</think>\n"
        "Here is my thought process.\n"
        "<final_translation>这是最终精修译文。</final_translation>\n"
        "Hope this helps!"
    )
    assert sanitize_thought_output(raw_with_tags) == "这是最终精修译文。"

    # 3. Strip conversational prefixes
    raw_chatty = "Here is the final translation: 宇宙的尽头是代码。"
    assert sanitize_thought_output(raw_chatty) == "宇宙的尽头是代码。"

    raw_translation_header = "### Translation:\n量子计算是一门交叉学科。"
    assert sanitize_thought_output(raw_translation_header) == "量子计算是一门交叉学科。"

    # 4. Truncated thinking trace (unclosed <think> / <thought>)
    raw_truncated_think = "<think>Let me ponder this but token limit hits"
    assert sanitize_thought_output(raw_truncated_think) == ""

    raw_truncated_thought = "<thought>Pondering something"
    assert sanitize_thought_output(raw_truncated_thought) == ""

    # 5. Truncated final translation (unclosed <final_translation>)
    raw_truncated_trans = "<think>Internal deliberation</think><final_translation>未闭合的译文内容"
    assert sanitize_thought_output(raw_truncated_trans) == "未闭合的译文内容"


@pytest.mark.asyncio
async def test_openai_compatible_provider_passes_reasoning_effort_and_recovers_on_400() -> None:
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_payloads.append(body)
        # First call with reasoning_effort returns 400 (simulating server that rejects it)
        if len(request_payloads) == 1 and "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "unrecognized parameter: reasoning_effort"}},
            )
        # Second call without reasoning_effort succeeds
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<think>Deliberating...</think><final_translation>成功降级恢复</final_translation>"
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 15},
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )

    result = await provider.generate("hello", reasoning_effort="low")
    assert result == "成功降级恢复"
    assert len(request_payloads) == 2
    assert request_payloads[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in request_payloads[1]
    assert provider.usage_log[0]["reasoning_tokens"] == 15


@pytest.mark.asyncio
async def test_openai_compatible_provider_reuses_client_and_closes() -> None:
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )

    client1 = provider._get_client()
    await provider.generate("first")
    client2 = provider._get_client()
    assert client1 is client2
    assert not client1.is_closed

    await provider.generate("second")
    assert provider.usage_totals["calls"] == 2

    await provider.aclose()
    assert client1.is_closed
    assert provider._client is None


@pytest.mark.asyncio
async def test_openai_compatible_provider_recovers_when_reasoning_effort_must_be_low() -> None:
    """When an o-series model rejects an invalid effort with 'must be one of [low, medium, high]',
    provider retries with reasoning_effort='low'."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_payloads.append(body)
        if len(request_payloads) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Invalid value for reasoning_effort: effort must be one of ['low', 'medium', 'high']"
                    }
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "成功适配 o-series"}}]},
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate("test prompt", reasoning_effort="minimal")
    assert result == "成功适配 o-series"
    assert len(request_payloads) == 2
    assert request_payloads[0]["reasoning_effort"] == "minimal"
    assert request_payloads[1]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_openai_compatible_provider_recovers_on_temperature_rejection() -> None:
    """When a reasoning model (e.g. o1/o3-mini) rejects temperature with 400,
    provider retries without temperature."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_payloads.append(body)
        if len(request_payloads) == 1 and "temperature" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'temperature' is not supported with this model"
                    }
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "温度参数自动剥离成功"}}]},
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate("test prompt", temperature=0.3)
    assert result == "温度参数自动剥离成功"
    assert len(request_payloads) == 2
    assert "temperature" in request_payloads[0]
    assert "temperature" not in request_payloads[1]


@pytest.mark.asyncio
async def test_openai_compatible_provider_handles_null_content_with_reasoning_content() -> None:
    """When a reasoning model (e.g. DeepSeek-R1) returns content=None and reasoning_content,
    provider handles it safely without crashing."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "<think>Thinking trace...</think><final_translation>安全恢复翻译</final_translation>",
                        }
                    }
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate("test prompt")
    assert result == "安全恢复翻译"


class AlwaysFailingProvider(BaseModelProvider):
    """Raises a fixed ModelProviderError on every call, counting attempts."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.attempts = 0

    @property
    def provider_name(self) -> str:
        return "always-failing"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        self.attempts += 1
        raise self.error


def test_classify_provider_error_taxonomy() -> None:
    from ubt.core.router.router import classify_provider_error

    assert (
        classify_provider_error(ModelProviderError("x", details={"status_code": 400})).retryable
        is False
    )
    assert (
        classify_provider_error(ModelProviderError("x", details={"status_code": 401})).retryable
        is False
    )
    assert (
        classify_provider_error(ModelProviderError("x", details={"status_code": 403})).retryable
        is False
    )
    action_402 = classify_provider_error(ModelProviderError("x", details={"status_code": 402}))
    assert action_402.retryable is False and action_402.top_up_hint is True
    assert (
        classify_provider_error(ModelProviderError("x", details={"status_code": 429})).retryable
        is True
    )
    assert (
        classify_provider_error(ModelProviderError("x", details={"status_code": 503})).retryable
        is True
    )
    assert classify_provider_error(ModelProviderError("timed out after 60s")).retryable is True
    assert classify_provider_error(TimeoutError()).retryable is True
    assert classify_provider_error(ConnectionError("reset")).retryable is True
    # Unknown failures stay retryable.
    assert classify_provider_error(ModelProviderError("weird")).retryable is True


@pytest.mark.asyncio
async def test_router_fail_fast_on_401_single_attempt() -> None:
    """Auth errors raise on the first attempt (no quota burn)."""
    provider = AlwaysFailingProvider(
        ModelProviderError("Unauthorized", details={"status_code": 401})
    )
    router = ModelRouter(provider=provider, max_retries=3)
    block = IRBlock(
        id="b03",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Hello world",
    )
    with pytest.raises(ModelProviderError):
        await router.draft(block)
    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_router_402_raises_top_up_hint() -> None:
    provider = AlwaysFailingProvider(
        ModelProviderError("Insufficient balance", details={"status_code": 402})
    )
    router = ModelRouter(provider=provider, max_retries=3)
    block = IRBlock(
        id="b04",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Hello world",
    )
    with pytest.raises(ModelProviderError, match="top up"):
        await router.draft(block)
    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_fallback_chain_skips_on_billing_error() -> None:
    """402 billing cannot resolve by renaming the model — one call, no chain walk."""
    provider = AlwaysFailingProvider(
        ModelProviderError("Insufficient balance", details={"status_code": 402})
    )
    router = ModelRouter(
        provider=provider, draft_model="a", fallback_models=["b", "c"], max_retries=0
    )
    with pytest.raises(ModelProviderError, match="top up"):
        await router._execute_with_retry(
            system_prompt="sys", user_prompt="hi", model="a", temperature=0.3
        )
    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_fallback_chain_skips_on_credential_401() -> None:
    """A credential-side 401 (no model-identity phrase) fails fast."""
    provider = AlwaysFailingProvider(
        ModelProviderError("Incorrect API key provided", details={"status_code": 401})
    )
    router = ModelRouter(provider=provider, draft_model="a", fallback_models=["b"], max_retries=0)
    with pytest.raises(ModelProviderError):
        await router._execute_with_retry(
            system_prompt="sys", user_prompt="hi", model="a", temperature=0.3
        )
    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_router_still_retries_500() -> None:
    provider = AlwaysFailingProvider(ModelProviderError("x", details={"status_code": 500}))
    router = ModelRouter(provider=provider, max_retries=2)
    block = IRBlock(
        id="b05",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Hello world",
    )
    with pytest.raises(ModelProviderError):
        await router.draft(block)
    assert provider.attempts == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_router_429_obeys_same_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the 429 branch used to skip the budget check and add one
    extra backoff sleep before exiting; it must stop at max_retries like
    every other retryable error."""
    import asyncio

    class _NoopRateLimiter:
        async def acquire(self, estimated_tokens: int = 0) -> None:
            return None

        def report_429(self) -> None:
            return None

        def report_success(self) -> None:
            return None

    slept: list[float] = []

    async def _fast_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    provider = AlwaysFailingProvider(
        ModelProviderError("rate limit exceeded", details={"status_code": 429})
    )
    router = ModelRouter(
        provider=provider,
        max_retries=2,
        rate_limiter=_NoopRateLimiter(),  # type: ignore[arg-type]
    )
    block = IRBlock(
        id="b06",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Hello world",
    )
    with pytest.raises(ModelProviderError):
        await router.draft(block)
    assert provider.attempts == 3  # initial + 2 retries, same as 500s
    assert len(slept) == 2  # one backoff per retry, no extra farewell sleep


# ---------------------------------------------------------------------------
# Unified cache-hit token parsing across chat / responses / batch
# wire formats, plus the observable cache_hit_rate metric.
# ---------------------------------------------------------------------------


def test_extract_cached_tokens_probe_order() -> None:
    from ubt.core.router.provider import _extract_cached_tokens

    # OpenAI chat completions shape
    openai_chat = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}}
    assert _extract_cached_tokens(openai_chat) == 80
    # DeepSeek flat shape
    deepseek = {"prompt_tokens": 100, "prompt_cache_hit_tokens": 64}
    assert _extract_cached_tokens(deepseek) == 64
    # OpenAI responses shape
    responses = {"input_tokens": 50, "input_tokens_details": {"cached_tokens": 25}}
    assert _extract_cached_tokens(responses) == 25
    # No cache info at all
    assert _extract_cached_tokens({"prompt_tokens": 100}) == 0
    assert _extract_cached_tokens({}) == 0
    # Details present but zero -> falls through to the next probe
    zero_then_flat = {
        "prompt_tokens_details": {"cached_tokens": 0},
        "prompt_cache_hit_tokens": 30,
    }
    assert _extract_cached_tokens(zero_then_flat) == 30


@pytest.mark.asyncio
async def test_chat_path_records_openai_cached_tokens() -> None:
    """The chat path previously read only DeepSeek's flat field, so
    OpenAI cache hits were silently counted as zero."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "你好"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k", default_model="gpt-x", transport=httpx.MockTransport(handler)
    )
    await provider.generate("hello")
    assert provider.usage_log[0]["prompt_cache_hit_tokens"] == 80
    assert provider.usage_log[0]["prompt_cache_miss_tokens"] == 20
    assert provider.usage_totals["cached_tokens"] == 80
    assert provider.cache_hit_rate == 0.8


@pytest.mark.asyncio
async def test_chat_path_records_deepseek_cache_hit_rate() -> None:
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "你好"}}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 45,
                    "prompt_cache_hit_tokens": 96,
                    "prompt_cache_miss_tokens": 24,
                },
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        transport=httpx.MockTransport(handler),
    )
    await provider.generate("hello")
    assert provider.usage_totals["cached_tokens"] == 96
    assert provider.cache_hit_rate == 0.8  # 96 / 120


def test_router_cache_hit_rate_delegates_and_defaults() -> None:
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    mock = MockModelProvider(default_response="x")
    router = ModelRouter(provider=mock, draft_model="d", repair_model="d")
    # MockModelProvider does not track cache hits -> 0.0 (unmeasured), no raise.
    assert router.cache_hit_rate() == 0.0


# ---------------------------------------------------------------------------
# finish_reason truncation detection + tail-anchored continuation.
# ---------------------------------------------------------------------------


def test_merge_continuation_drops_repeated_overlap() -> None:
    from ubt.core.router.router import _merge_continuation

    # Model echoed the seam despite the instruction — overlap removed.
    assert (
        _merge_continuation("今晚的月色很美，", "月色很美，风也温柔。")
        == "今晚的月色很美，风也温柔。"
    )
    # Clean continuation joins verbatim.
    assert _merge_continuation("abc", "def") == "abcdef"
    # Single-character coincidental overlap is NOT removed (corruption guard).
    assert _merge_continuation("end", "daily") == "enddaily"
    # Empty continuation is a no-op.
    assert _merge_continuation("abc", "") == "abc"


@pytest.mark.asyncio
async def test_execute_with_retry_continues_truncated_output() -> None:
    """finish_reason == 'length' triggers tail-anchored continuation; the
    merged result contains both parts with the echoed seam removed."""

    class TruncatedThenContinuedProvider(MockModelProvider):
        def __init__(self) -> None:
            super().__init__(default_response="")
            self.calls = 0

        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            self.calls += 1
            if self.calls == 1:
                return "The reactor outputs 42 megawatts", "length"
            # Continuation call: echo the seam (models often do), then continue.
            assert "cut off" in prompt
            return "42 megawatts daily according to the report.", "stop"

    router = ModelRouter(provider=TruncatedThenContinuedProvider(), draft_model="d")
    result = await router._execute_with_retry(
        system_prompt="sys", user_prompt="translate this", model="d", temperature=0.3
    )
    assert result == "The reactor outputs 42 megawatts daily according to the report."
    assert router.provider.calls == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_continuation_failure_keeps_partial_output() -> None:
    from ubt.core.exceptions import ModelProviderError

    class ExplodingContinuationProvider(MockModelProvider):
        def __init__(self) -> None:
            super().__init__(default_response="partial translation")
            self.continuation_attempted = False

        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            if "cut off" in prompt:
                self.continuation_attempted = True
                raise ModelProviderError("continuation backend down")
            return "partial translation", "length"

    router = ModelRouter(provider=ExplodingContinuationProvider(), draft_model="d")
    result = await router._execute_with_retry(
        system_prompt="sys", user_prompt="x", model="d", temperature=0.3
    )
    assert result == "partial translation"
    assert router.provider.continuation_attempted is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_continuation_non_provider_error_keeps_partial_output() -> None:
    """A malformed continuation body (``json.JSONDecodeError``) must not throw
    away an already-billed partial.

    The recovery only caught ``ModelProviderError``; a continuation whose 200
    body failed ``response.json()`` raises ``ValueError``, escaped, and the outer
    parse handler failed the whole model — re-running (and re-billing) the
    primary generation on the next fallback candidate.
    """

    class MalformedContinuationProvider(MockModelProvider):
        def __init__(self) -> None:
            super().__init__(default_response="partial translation")
            self.continuation_attempted = False

        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            if "cut off" in prompt:
                self.continuation_attempted = True
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return "partial translation", "length"

    router = ModelRouter(provider=MalformedContinuationProvider(), draft_model="d")
    result = await router._execute_with_retry(
        system_prompt="sys", user_prompt="x", model="d", temperature=0.3
    )
    assert result == "partial translation"
    assert router.provider.continuation_attempted is True  # type: ignore[attr-defined]


def test_parse_retry_after_handles_both_rfc_forms() -> None:
    """``Retry-After`` may be delta-seconds or an HTTP-date (RFC 7231)."""
    import datetime as dt
    from email.utils import format_datetime

    from ubt.core.router.router import _parse_retry_after

    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after(" 0.25 ") == 0.25
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("not-a-date") is None
    # An HTTP-date in the future becomes a positive wait; the date form used to
    # be silently dropped, so the client retried after ~1s instead.
    future = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30)
    parsed = _parse_retry_after(format_datetime(future, usegmt=True))
    assert parsed is not None and 20.0 < parsed <= 31.0


@pytest.mark.asyncio
async def test_max_tokens_reaches_provider() -> None:
    """The caller's max_tokens budget is passed through the funnel."""

    class CapturingProvider(MockModelProvider):
        def __init__(self) -> None:
            super().__init__(default_response="ok")
            self.seen_max_tokens: int | None = None

        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            self.seen_max_tokens = max_tokens
            return "ok", "stop"

    router = ModelRouter(provider=CapturingProvider(), draft_model="d")
    await router._execute_with_retry(
        system_prompt="s", user_prompt="u", model="d", temperature=0.3, max_tokens=1234
    )
    assert router.provider.seen_max_tokens == 1234  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_openai_chat_path_surfaces_finish_reason() -> None:
    """The chat wire format's finish_reason reaches the router."""
    import httpx

    from ubt.core.router.provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "半句翻译"}, "finish_reason": "length"}]},
        )

    provider = OpenAICompatibleProvider(
        api_key="k", default_model="m", transport=httpx.MockTransport(handler)
    )
    text, finish = await provider.generate_with_finish_reason(prompt="hello")
    assert text == "半句翻译"
    assert finish == "length"
    # Plain generate() still returns just the text (back-compat).
    assert await provider.generate(prompt="hello") == "半句翻译"


# ---------------------------------------------------------------------------
# Model-level fallback chain
# ---------------------------------------------------------------------------


class _ChainProbeProvider(MockModelProvider):
    """Serves one model, fails every other with a fail-fast 400."""

    def __init__(self) -> None:
        super().__init__(default_response="fallback output")
        self.calls_by_model: list[str | None] = []

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        self.calls_by_model.append(model)
        if model != "rescue":
            raise ModelProviderError(
                f"model {model} not available for this account",
                details={"status_code": 400},
            )
        return "fallback output", "stop"


@pytest.mark.asyncio
async def test_fallback_chain_serves_from_next_model_on_primary_failure() -> None:
    """Primary-model failure degrades to the fallback, not block FAILED."""
    provider = _ChainProbeProvider()
    router = ModelRouter(
        provider=provider,
        draft_model="primary-broken",
        fallback_models=["rescue"],
    )
    result = await router._execute_with_retry(
        system_prompt="sys", user_prompt="translate", model="primary-broken", temperature=0.3
    )
    assert result == "fallback output"
    assert provider.calls_by_model == ["primary-broken", "rescue"]


@pytest.mark.asyncio
async def test_fallback_chain_exhausted_raises_last_error() -> None:
    """When every model in the chain fails, the last error surfaces."""
    provider = _ChainProbeProvider()
    router = ModelRouter(
        provider=provider,
        draft_model="a",
        fallback_models=["b", "c"],
    )
    with pytest.raises(ModelProviderError, match="model c not available"):
        await router._execute_with_retry(
            system_prompt="sys", user_prompt="translate", model="a", temperature=0.3
        )
    assert provider.calls_by_model == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_primary_success_never_touches_fallback_chain() -> None:
    """A healthy primary model keeps the fallbacks idle."""
    provider = _ChainProbeProvider()

    class _HealthyProvider(MockModelProvider):
        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            provider.calls_by_model.append(model)
            return "primary output", "stop"

    router = ModelRouter(
        provider=_HealthyProvider(),
        draft_model="healthy",
        fallback_models=["rescue"],
    )
    result = await router._execute_with_retry(
        system_prompt="sys", user_prompt="translate", model="healthy", temperature=0.3
    )
    assert result == "primary output"
    assert provider.calls_by_model == ["healthy"]


@pytest.mark.asyncio
async def test_fallback_skips_duplicates_of_primary() -> None:
    """A fallback identical to the primary is not retried as itself."""
    provider = _ChainProbeProvider()
    router = ModelRouter(
        provider=provider,
        draft_model="primary-broken",
        fallback_models=["primary-broken", "rescue"],
    )
    result = await router._execute_with_retry(
        system_prompt="sys", user_prompt="translate", model="primary-broken", temperature=0.3
    )
    assert result == "fallback output"
    assert provider.calls_by_model == ["primary-broken", "rescue"]


@pytest.mark.asyncio
async def test_fallback_chain_dedupes_repeated_entries() -> None:
    """A repeated fallback entry is tried once, not once per duplicate."""
    provider = _ChainProbeProvider()
    router = ModelRouter(
        provider=provider,
        draft_model="a",
        fallback_models=["b", "b", "c"],
    )
    with pytest.raises(ModelProviderError, match="model c not available"):
        await router._execute_with_retry(
            system_prompt="sys", user_prompt="translate", model="a", temperature=0.3
        )
    assert provider.calls_by_model == ["a", "b", "c"]


def test_pipeline_wires_config_fallback_models_into_router() -> None:
    """The orchestrator passes the configured chain to the router."""
    from ubt.core.config import UBTConfig
    from ubt.core.engine.pipeline import PipelineOrchestrator

    config = UBTConfig.from_env(api_key=SecretStr(""), fallback_models=["m-two", "m-three"])
    orchestrator = PipelineOrchestrator(config=config)
    assert orchestrator.router.fallback_models == ["m-two", "m-three"]


def test_config_fallback_models_default_empty() -> None:
    """No fallbacks configured -> empty chain (existing behavior)."""
    from ubt.core.config import UBTConfig

    assert UBTConfig.from_env(api_key=SecretStr("")).fallback_models == []


class _RejectingProvider(BaseModelProvider):
    """Primary gateway stub: rejects with a fixed ModelProviderError."""

    def __init__(self, status: int, message: str) -> None:
        self._status = status
        self._message = message
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "rejecting-gateway"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        raise AssertionError("unreachable: gateway stub only speaks finish-reason")

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        self.calls += 1
        raise ModelProviderError(self._message, details={"status_code": self._status})


def _self_hosted_registry(backend: str = "self_hosted") -> ModelCapabilityRegistry:
    reg = ModelCapabilityRegistry()
    reg.register(ModelProfile(model_pattern="translategemma", deployment_backend=backend))
    return reg


def _self_hosted_router(
    primary: BaseModelProvider,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "self_hosted",
) -> tuple[ModelRouter, MockModelProvider]:
    local = MockModelProvider(default_response="[LOCAL-SELF-HOSTED]")
    router = ModelRouter(
        provider=primary,
        draft_model="translategemma:4b",
        repair_model="translategemma:4b",
        max_retries=0,
        registry=_self_hosted_registry(backend),
    )
    monkeypatch.setattr(router, "_local_fallback_provider", lambda: local)
    return router, local


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend",
    ["self_hosted", "ollama", "llama.cpp", "llama-swap", "vllm", "lmstudio"],
)
async def test_gateway_rejection_retries_the_local_self_hosted_stack(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One mechanism for every self-hosted stack, not just Ollama.

    The trigger used to be spelled ``deployment_backend == "ollama"``, so
    llama.cpp / llama-swap / vLLM / LM Studio — the rest of the self-hosted
    world — had the mechanism dormant. ``ollama`` stays accepted so existing
    profiles keep working.
    """
    primary = _RejectingProvider(401, "Model translategemma:4b is not supported")
    router, local = _self_hosted_router(primary, monkeypatch, backend=backend)
    out = await router._execute_single_model(
        system_prompt="",
        user_prompt="Translate: CHAPTER 3",
        model="translategemma:4b",
        temperature=0.3,
    )
    assert out == "[LOCAL-SELF-HOSTED]"
    assert primary.calls == 1
    assert len(local.call_history) == 1
    assert local.call_history[0]["model"] == "translategemma:4b"


def test_local_fallback_endpoint_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local stack's port is not Ollama's.

    llama-swap serves on :9090, llama-server on :8081/:8082, vLLM on :8000 and
    LM Studio on :1234 — a hard-coded 11434 made the retry useless for all of
    them. The Ollama default is kept for compatibility.
    """
    router = ModelRouter(
        provider=_RejectingProvider(401, "Model x is not supported"),
        draft_model="translategemma:4b",
        max_retries=0,
        registry=_self_hosted_registry(),
    )
    assert router._local_fallback_provider().base_url == "http://localhost:11434/v1"

    monkeypatch.setenv("UBT_LOCAL_FALLBACK_BASE_URL", "http://127.0.0.1:9090/v1")
    swapped = ModelRouter(
        provider=_RejectingProvider(401, "Model x is not supported"),
        draft_model="translategemma:4b",
        max_retries=0,
        registry=_self_hosted_registry(),
    )
    assert swapped._local_fallback_provider().base_url == "http://127.0.0.1:9090/v1"


@pytest.mark.asyncio
async def test_non_self_hosted_model_never_touches_the_local_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _RejectingProvider(401, "Model deepseek-v4-flash is not supported")
    router, local = _self_hosted_router(primary, monkeypatch)
    with pytest.raises(ModelProviderError):
        await router._execute_single_model(
            system_prompt="", user_prompt="hi", model="deepseek-v4-flash", temperature=0.3
        )
    assert primary.calls == 1
    assert local.call_history == []


@pytest.mark.asyncio
async def test_self_hosted_model_rate_limit_skips_local_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _RejectingProvider(429, "rate limit exceeded, retry later")
    router, local = _self_hosted_router(primary, monkeypatch)
    with pytest.raises(ModelProviderError):
        await router._execute_single_model(
            system_prompt="", user_prompt="hi", model="translategemma:4b", temperature=0.3
        )
    assert local.call_history == []


@pytest.mark.asyncio
async def test_local_failure_chains_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DeadLocal(BaseModelProvider):
        @property
        def provider_name(self) -> str:
            return "dead-local"

        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            raise ConnectionError("localhost refused")

    primary = _RejectingProvider(401, "Model translategemma:4b is not supported")
    router = ModelRouter(
        provider=primary,
        draft_model="translategemma:4b",
        repair_model="translategemma:4b",
        max_retries=0,
        registry=_self_hosted_registry(),
    )
    monkeypatch.setattr(router, "_local_fallback_provider", _DeadLocal)
    with pytest.raises(ModelProviderError, match="Local self-hosted fallback failed"):
        await router._execute_single_model(
            system_prompt="", user_prompt="hi", model="translategemma:4b", temperature=0.3
        )


@pytest.mark.asyncio
async def test_draft_fallback_adapts_prompt_and_extraction_strategy() -> None:
    """Verify that when draft falls back to another model tier, prompt and extractor re-adapt."""
    from ubt.core.ir.models import BlockType, FlowID, IRBlock

    recorded_prompts: dict[str, tuple[str | None, str]] = {}

    class _FallbackProbeProvider(MockModelProvider):
        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            m = model or "default"
            recorded_prompts[m] = (system_prompt, prompt)
            if m == "primary-rich":
                # Identity-style rejection: the model name (not the
                # credential) is the problem, so the chain may try the next
                # model. A bare "account issue" 401 fails the chain fast.
                raise ModelProviderError("Model primary-rich is not supported (401)")
            # Fallback model returns plain text without XML tags
            return "这是备选模型的纯文本翻译", "stop"

    provider = _FallbackProbeProvider()
    reg = ModelCapabilityRegistry()
    reg.register(
        ModelProfile(
            model_pattern="custom-minimal-fallback",
            prompt_strategy=PromptStrategy.MINIMAL,
            extraction_strategy=ExtractionStrategy.RAW,
            supports_system_prompt=False,
        )
    )
    router = ModelRouter(
        provider=provider,
        draft_model="primary-rich",
        fallback_models=["custom-minimal-fallback"],
        max_retries=0,
        registry=reg,
    )

    block = IRBlock(
        id="b1",
        spine_index=0,
        flow_id=FlowID.MAIN_STORY,
        block_type=BlockType.NARRATIVE,
        source_text="Hello world.",
    )

    result = await router.draft(block=block, target_lang="zh", source_lang="en")

    # Fallback model was invoked
    assert "custom-minimal-fallback" in recorded_prompts
    fallback_sys, fallback_user = recorded_prompts["custom-minimal-fallback"]
    assert fallback_sys in (None, "")
    assert "<translation>" not in fallback_user
    # And extraction succeeded on plain text (not requiring <translation> tags)
    assert result == "这是备选模型的纯文本翻译"


def test_usage_accessors_read_property_style_provider() -> None:
    """Provider usage is a @property, so the accessors must not require callables.

    Regression: the accessors tested ``callable(...)`` on the value returned by
    ``getattr``. Because ``BaseModelProvider``/``OpenAICompatibleProvider``
    publish usage as a ``@property``, ``getattr`` yields the mapping itself,
    ``callable(dict)`` is False, and both accessors silently returned ``{}``.
    That pinned ``estimate_cost_usd()`` at 0.00 for the only production provider
    and zeroed the progress event's cache rate while the quality report — which
    reads ``cache_hit_rate`` through a different (correct) guard — showed the
    real 52-57%.
    """

    class PropertyStyleProvider(MockModelProvider):
        @property
        def usage_totals(self) -> dict[str, int]:
            return {"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 64}

        @property
        def usage_totals_by_model(self) -> dict[str, dict[str, int]]:
            return {
                "deepseek-chat": {"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 64}
            }

        # Distinct from the usage mappings above: ModelRouter.cache_hit_rate()
        # delegates to this provider-level (cumulative) property, while the
        # run-scoped rate comes from cache_hit_rate_from_usage(usage_delta).
        @property
        def cache_hit_rate(self) -> float:
            return 0.64

    router = ModelRouter(provider=PropertyStyleProvider())
    assert router.usage_totals()["prompt_tokens"] == 100
    assert router.usage_totals_by_model()["deepseek-chat"]["cached_tokens"] == 64
    assert router.cache_hit_rate() == 0.64
    assert cache_hit_rate_from_usage(router.usage_totals_by_model()) == 0.64
    # Real token usage must now produce a real cost, not the silent $0.00.
    priced = router.estimate_cost_usd()
    assert priced is not None and priced > 0.0


def test_usage_accessors_still_accept_method_style_and_absent_usage() -> None:
    """Method-style providers keep working, and absent usage stays {} (not a crash)."""

    class MethodStyleProvider(MockModelProvider):
        def usage_totals(self) -> dict[str, int]:
            return {"prompt_tokens": 7}

        def usage_totals_by_model(self) -> dict[str, dict[str, int]]:
            return {"m": {"prompt_tokens": 7}}

    class NoUsageProvider(MockModelProvider):
        pass

    method_router = ModelRouter(provider=MethodStyleProvider())
    assert method_router.usage_totals() == {"prompt_tokens": 7}
    assert method_router.usage_totals_by_model() == {"m": {"prompt_tokens": 7}}

    absent_router = ModelRouter(provider=NoUsageProvider())
    assert absent_router.usage_totals() == {}
    assert absent_router.usage_totals_by_model() == {}
    assert absent_router.estimate_cost_usd() == 0.0


def test_prompt_places_epoch_after_glossary_before_macro() -> None:
    """L3 rides between global glossary and the L2 macro snapshot."""
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    router = ModelRouter(provider=MockModelProvider(), draft_model="test-model")
    system_prompt, user_prompt = router.build_draft_prompt(
        source_text="Hello world.",
        glossary_table="cat -> 猫",
        rolling_summary="最近发生的事件摘要。",
        global_glossary="GLOBAL TERMS TABLE",
        epoch_summary="全书已推进至第三章的压缩历史。",
        model="deepseek-v4-flash",
    )
    assert "GLOBAL TERMS TABLE" in user_prompt or "GLOBAL TERMS TABLE" in system_prompt
    assert "### Book Continuity (Compressed History)" in user_prompt
    macro_labels = ("### Story Context", "### Document Continuation Context")
    macro_pos = min(pos for label in macro_labels if (pos := user_prompt.find(label)) != -1)
    epoch_pos = user_prompt.find("Book Continuity")
    # Global glossary rides the static prefix: inside the system prompt
    # (RICH) or ahead of the epoch in the user prompt (HYBRID).
    assert "GLOBAL TERMS TABLE" in system_prompt or (
        user_prompt.find("GLOBAL TERMS TABLE") < epoch_pos
    )
    assert epoch_pos < macro_pos


def test_router_draft_prompt_genre_profiles() -> None:
    """Verify draft prompt builders adapt system prompts according to genre_profile."""
    from ubt.core.router.prompts import build_hybrid_draft_prompt, build_rich_draft_prompt

    sys_fiction, _ = build_hybrid_draft_prompt(
        source_text="He gazed at the sea.",
        genre_profile="fiction",
    )
    assert "literary translator" in sys_fiction

    sys_acad, _ = build_hybrid_draft_prompt(
        source_text="The sample was heated to 300K.",
        genre_profile="academic",
    )
    assert "scholarly scientific translator" in sys_acad

    sys_rich, user_rich = build_rich_draft_prompt(
        source_text="The experiments were carried out in triplicate.",
        genre_profile="academic",
    )
    assert "Natural Register & Translationese Elimination" in sys_rich
    assert "scientific translator" in sys_rich


@pytest.mark.asyncio
async def test_continuation_reserves_prompt_sized_tpm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A continuation re-sends the full prompt; the fixed 500-token
    reservation under-counted long macro-chunks by an order of magnitude,
    so the bucket believed it was idle while the provider 429'd."""

    class TruncatingProvider(MockModelProvider):
        def __init__(self) -> None:
            super().__init__(default_response="")
            self.calls = 0

        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            self.calls += 1
            if self.calls == 1:
                return "x" * 4000, "length"
            return "", "stop"

    router = ModelRouter(provider=TruncatingProvider(), draft_model="d")
    acquires: list[int] = []

    async def _capture(*, estimated_tokens: int = 500) -> None:
        acquires.append(estimated_tokens)

    monkeypatch.setattr(router.rate_limiter, "acquire", _capture)
    long_prompt = "t" * 20000
    await router._execute_with_retry(
        system_prompt="sys", user_prompt=long_prompt, model="d", temperature=0.3
    )
    assert len(acquires) == 2
    from ubt.core.engine.cost_estimate import count_text_tokens

    primary_tokens = count_text_tokens("sys") + count_text_tokens(long_prompt)
    assert acquires[0] == primary_tokens * 2  # completion fallback mirrors prompt
    continuation_tokens = count_text_tokens("sys") + count_text_tokens(long_prompt)
    assert acquires[1] >= continuation_tokens * 2  # superset prompt, >= primary


@pytest.mark.fast
@pytest.mark.asyncio
async def test_cjk_prompt_reserves_script_aware_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CJK prompt must reserve ~0.85 tokens/char, not ``chars/4``.

    Regression: the router reserved ``(len+3)//4`` for every script, so a zh
    source under-reserved ~3.4x; the TPM bucket believed it had budget while
    the provider returned 429, and AIMD then halved capacity for a
    self-inflicted breach.
    """
    from ubt.core.engine.cost_estimate import count_text_tokens

    router = ModelRouter(provider=MockModelProvider(default_response="译文"), draft_model="d")
    acquires: list[int] = []

    async def _capture(*, estimated_tokens: int = 500) -> None:
        acquires.append(estimated_tokens)

    monkeypatch.setattr(router.rate_limiter, "acquire", _capture)
    zh = "这是一段中文测试文本，用于验证令牌计数。" * 3
    await router._execute_with_retry(system_prompt="", user_prompt=zh, model="d", temperature=0.0)

    naive = (len(zh) + 3) // 4
    assert acquires, "the router never reserved TPM for the call"
    assert acquires[0] >= count_text_tokens(zh)
    assert acquires[0] > naive * 2  # ~3.4x, comfortably above the old heuristic


@pytest.mark.fast
def test_provider_aclose_closes_all_transports() -> None:
    provider = OpenAICompatibleProvider(api_key="mock", base_url="https://api.openai.com/v1")
    t1 = provider._chat_transport
    t2 = provider._anthropic_transport
    t3 = provider._responses_transport

    m1 = AsyncMock()
    m2 = AsyncMock()
    m3 = AsyncMock()
    t1.__dict__["aclose"] = m1
    t2.__dict__["aclose"] = m2
    t3.__dict__["aclose"] = m3

    asyncio.run(provider.aclose())

    m1.assert_awaited_once()
    m2.assert_awaited_once()
    m3.assert_awaited_once()


def test_draft_source_is_recovered_from_every_builder() -> None:
    import inspect

    from ubt.core.router.prompts import (
        build_hybrid_draft_prompt,
        build_minimal_draft_prompt,
        build_rich_draft_prompt,
        draft_source_from_prompt,
    )

    source = "This block has 250万 pixels.\n\nSecond paragraph."
    for builder in (
        build_minimal_draft_prompt,
        build_hybrid_draft_prompt,
        build_rich_draft_prompt,
    ):
        params = inspect.signature(builder).parameters
        kwargs = {
            "source_text": source,
            "target_lang": "zh",
            "source_lang": "en",
            "glossary_table": "node | 节点",
            "global_glossary": "FET | 场效应管",
            "few_shot_reference": "EN: x\nZH: y",
            "genre_profile": "textbook",
        }
        _, user_prompt = builder(**{k: v for k, v in kwargs.items() if k in params})
        recovered = draft_source_from_prompt(user_prompt)
        assert recovered == source.strip(), f"{builder.__name__} leaked: {recovered[:60]!r}"


class _UsageProvider:
    """Duck-typed provider exposing only the usage surface ``estimate_cost`` reads."""

    is_mock = True

    def __init__(self, base_url: str, by_model: dict[str, dict[str, int]]) -> None:
        self.base_url = base_url
        self._by = by_model

    def usage_totals_by_model(self) -> dict[str, dict[str, int]]:
        return {model: dict(totals) for model, totals in self._by.items()}

    def usage_totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for totals in self._by.values():
            for key, value in totals.items():
                out[key] = out.get(key, 0) + value
        return out


def test_estimate_cost_prices_exclusive_fallback_models_at_their_endpoint() -> None:
    """A local-fallback model must be billed $0, not reported unknown.

    ``estimate_cost_usd`` merged the fallback provider's usage but priced it at
    the primary's cloud ``base_url``, so a local-only model name (absent from
    the table) made the whole run report None instead of the real cloud spend.
    """
    primary = _UsageProvider(
        "https://api.deepseek.com",
        {"deepseek-chat": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
    )
    router = ModelRouter(provider=primary, draft_model="deepseek-chat")  # type: ignore[arg-type]
    router._fallback_provider = _UsageProvider(  # type: ignore[assignment]
        "http://127.0.0.1:11434/v1",
        {"translategemma:4b": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
    )
    cost = router.estimate_cost_usd()
    assert cost is not None
    assert cost == pytest.approx(0.27)


def test_domain_descriptor_reaches_every_draft_prompt() -> None:
    """`--domain` was accepted on all surfaces but read by nothing (dead option)."""
    from ubt.core.router.prompts import (
        build_hybrid_draft_prompt,
        build_minimal_draft_prompt,
        build_rich_draft_prompt,
    )

    _, minimal_user = build_minimal_draft_prompt(
        source_text="Hello.",
        target_lang="zh",
        source_lang="en",
        genre_profile="general",
        domain="semiconductor physics",
    )
    assert "semiconductor physics" in minimal_user

    hybrid_sys, _ = build_hybrid_draft_prompt(
        source_text="Hello.",
        target_lang="zh",
        source_lang="en",
        genre_profile="general",
        domain="semiconductor physics",
    )
    assert "semiconductor physics" in hybrid_sys

    rich_sys, _ = build_rich_draft_prompt(
        source_text="Hello.",
        target_lang="zh",
        source_lang="en",
        genre_profile="general",
        domain="biomedicine",
    )
    assert "biomedicine" in rich_sys


def test_domain_falls_back_to_the_profile_hint() -> None:
    from ubt.core.router.prompts import build_minimal_draft_prompt

    _, user = build_minimal_draft_prompt(
        source_text="Hello.",
        target_lang="zh",
        source_lang="en",
        genre_profile="textbook",
    )
    assert "textbook" in user


def test_draft_prompt_neutralizes_reserved_wrapper_tag_mentions() -> None:
    """A source that mentions ``<translation>`` must not inject a wrapper token.

    The source span is interpolated into the prompt; a literal wrapper tag made
    the model emit a real one, which the extractor then treated as the envelope.
    """
    from ubt.core.router.prompts import build_minimal_draft_prompt

    _, user = build_minimal_draft_prompt("Use <translation> tags to wrap output.")
    assert "<translation>" not in user
    assert "&lt;translation>" in user


def test_billing_endpoint_map_attributes_fallback_only_models() -> None:
    """A model served only by the fallback bills at the fallback URL."""
    from typing import Any, cast

    from ubt.core.router.provider import BaseModelProvider

    class _FakeProvider:
        is_mock = False

        def __init__(self, base_url: str, models: list[str]) -> None:
            self.base_url = base_url
            self._models = models

        def usage_totals_by_model(self) -> dict[str, dict[str, int]]:
            return {m: {"calls": 1} for m in self._models}

    router = ModelRouter(
        provider=cast(
            BaseModelProvider, _FakeProvider("https://primary", ["shared", "primary-only"])
        ),
        draft_model="d",
    )
    router._fallback_provider = cast(
        Any, _FakeProvider("http://localhost:9090", ["shared", "local-only"])
    )
    # A name served by both channels keeps the primary endpoint; only the
    # fallback-exclusive model is attributed to the fallback.
    assert router.billing_endpoint_map() == {"local-only": "http://localhost:9090"}


@pytest.mark.asyncio
@pytest.mark.fast
async def test_cjk_output_token_budget_not_underestimated() -> None:
    """CJK source text token budget must be calculated using token counting, not len // 2."""
    from ubt.core.engine.cost_estimate import count_text_tokens
    from ubt.core.ir.models import IRBlock

    class CapturingProvider(MockModelProvider):
        def __init__(self) -> None:
            super().__init__(default_response="<translation>翻译结果</translation>")
            self.seen_max_tokens: int | None = None

        async def generate_with_finish_reason(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> tuple[str, str | None]:
            self.seen_max_tokens = max_tokens
            return "<translation>翻译结果</translation>", "stop"

    provider = CapturingProvider()
    router = ModelRouter(provider=provider, draft_model="d")
    cjk_text = "这是一段很长的中文文本，包含大量的汉字。" * 100  # 2300 chars
    src_tokens = count_text_tokens(cjk_text)
    block = IRBlock(id="b1", spine_index=0, source_text=cjk_text)

    await router.draft(block)

    assert provider.seen_max_tokens is not None

    assert provider.seen_max_tokens >= int(src_tokens * 1.8)
    assert provider.seen_max_tokens > 2300
