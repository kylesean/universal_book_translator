"""Unit tests for Protocol-First Architecture and 3-element model access (base_url, api_key, model)."""

import json
from typing import Any

import httpx
import pytest

from ubt.core.config import UBTConfig
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.router import ModelRouter, create_model_provider


@pytest.mark.asyncio
async def test_zero_friction_three_elements_arbitrary_future_model() -> None:
    """Verify that an arbitrary, unregistered future model works seamlessly via standard 3 credentials."""
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers.get("authorization") == "Bearer sk-test-key-2027"
        body = json.loads(request.content)
        request_bodies.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>Deliberating translation...</think><translation>未来通用大模型翻译成功</translation>",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 10},
                },
            },
        )

    # 1. User configures ONLY 3 elements: base_url, api_key, model
    provider = create_model_provider(
        api_key="sk-test-key-2027",
        base_url="https://api.future-frontier.ai/v1",
        default_model="future-frontier-super-v9",
        transport=httpx.MockTransport(handler),
    )

    router = ModelRouter(
        provider=provider,
        draft_model="future-frontier-super-v9",
    )

    block = IRBlock(
        id="blk_01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Welcome to the future of decentralized AI.",
        status=BlockStatus.PENDING,
    )

    result = await router.draft(block=block, target_lang="zh")
    assert result == "未来通用大模型翻译成功"

    # Verify wire payload
    assert len(request_bodies) == 1
    assert request_bodies[0]["model"] == "future-frontier-super-v9"
    assert any(m["role"] == "system" for m in request_bodies[0]["messages"])
    assert any(
        "Welcome to the future" in m["content"]
        for m in request_bodies[0]["messages"]
        if m["role"] == "user"
    )

    # Verify usage accounting
    totals = router.usage_totals()
    assert totals["prompt_tokens"] == 50
    assert totals["completion_tokens"] == 20
    assert totals["cached_tokens"] == 10


@pytest.mark.asyncio
async def test_an_explicit_anthropic_mode_speaks_the_messages_wire() -> None:
    """``api_mode="anthropic-messages"`` routes to the native Messages API.

    The host alone used to select the protocol; now the choice is explicit and
    the endpoint is merely where the request lands.
    """
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers.get("x-api-key") == "sk-ant-api03-secret"
        assert request.headers.get("anthropic-version") == "2023-06-01"
        body = json.loads(request.content)
        request_bodies.append(body)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "<translation>克劳德原生协议翻译完成</translation>"}
                ],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 60,
                    "output_tokens": 15,
                    "cache_read_input_tokens": 30,
                },
            },
        )

    provider = create_model_provider(
        api_key="sk-ant-api03-secret",
        base_url="https://api.anthropic.com",
        api_mode="anthropic-messages",
        default_model="claude-custom-unseen-model",
        transport=httpx.MockTransport(handler),
    )

    router = ModelRouter(
        provider=provider,
        draft_model="claude-custom-unseen-model",
    )

    block = IRBlock(
        id="blk_02",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="The harbor was peaceful at dawn.",
    )

    result = await router.draft(block=block, target_lang="zh")
    assert result == "克劳德原生协议翻译完成"

    assert len(request_bodies) == 1
    assert request_bodies[0]["model"] == "claude-custom-unseen-model"
    assert "system" in request_bodies[0]
    assert request_bodies[0]["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_anthropic_temperature_rejection_self_heals() -> None:
    """Verify that if Anthropic rejects temperature (e.g. extended thinking mode), provider self-heals."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        if call_count == 1 and "temperature" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "temperature is not supported when thinking is enabled",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "思考模式自愈成功"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 40, "output_tokens": 10},
            },
        )

    provider = create_model_provider(
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com",
        api_mode="anthropic-messages",
        default_model="claude-3-7-thinking",
        transport=httpx.MockTransport(handler),
    )

    res = await provider.generate("test prompt", temperature=0.5)
    assert res == "思考模式自愈成功"
    assert call_count == 2


def test_cli_three_element_configuration_wiring() -> None:
    """Verify UBTConfig merges base_url, api_key, and draft_model from overrides cleanly."""
    cfg = UBTConfig.from_env(
        api_key="sk-custom-direct-key",
        base_url="https://custom-gateway.local:8080/v1",
        draft_model="my-fintech-mt-2027",
    )
    assert cfg.api_key.get_secret_value() == "sk-custom-direct-key"
    assert cfg.base_url == "https://custom-gateway.local:8080/v1"
    assert cfg.draft_model == "my-fintech-mt-2027"
    assert cfg.repair_model == "my-fintech-mt-2027"
