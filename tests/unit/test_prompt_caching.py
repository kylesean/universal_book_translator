"""Unit tests for Phase 1: Prompt Caching Prefix Alignment and Anthropic cache_control."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import SecretStr

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.router.provider import (
    OpenAICompatibleProvider,
    _extract_cached_tokens,
    create_model_provider,
)


def test_extract_cached_tokens_supports_anthropic_and_openai() -> None:
    # Anthropic usage shape
    anthropic_usage = {
        "input_tokens": 1500,
        "output_tokens": 120,
        "cache_read_input_tokens": 1250,
        "cache_creation_input_tokens": 0,
    }
    assert _extract_cached_tokens(anthropic_usage) == 1250

    # OpenAI chat completions
    openai_usage = {
        "prompt_tokens": 1500,
        "completion_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 1100},
    }
    assert _extract_cached_tokens(openai_usage) == 1100

    # DeepSeek flat shape
    deepseek_usage = {
        "prompt_tokens": 1500,
        "completion_tokens": 120,
        "prompt_cache_hit_tokens": 950,
    }
    assert _extract_cached_tokens(deepseek_usage) == 950


def test_anthropic_auth_headers_include_prompt_caching_beta() -> None:
    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
    )
    headers = provider._auth_headers()
    assert headers.get("anthropic-beta") == "prompt-caching-2024-07-31"
    assert headers.get("x-api-key") == "sk-ant-test"


@pytest.mark.asyncio
async def test_anthropic_generate_wraps_system_prompt_with_ephemeral_cache() -> None:
    provider = OpenAICompatibleProvider(
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
    )

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "content": [{"type": "text", "text": "Claude 翻译"}],
        "usage": {
            "input_tokens": 2000,
            "output_tokens": 100,
            "cache_read_input_tokens": 1800,
        },
    }

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=fake_response)
    provider._client = mock_client

    text, finish_reason = await provider._generate_anthropic_meta(
        prompt="Translate this",
        system_prompt="You are a professional translator.",
        target_model="claude-3-5-sonnet-20241022",
        temperature=0.3,
        max_tokens=2048,
    )

    assert text == "Claude 翻译"
    # Inspect payload sent to client.post
    call_args = mock_client.post.call_args
    sent_payload = call_args[1]["json"]

    assert "system" in sent_payload
    sys_blocks = sent_payload["system"]
    assert isinstance(sys_blocks, list)
    assert sys_blocks[0]["type"] == "text"
    assert sys_blocks[0]["text"] == "You are a professional translator."
    assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}

    # Verify usage accounting. Anthropic's ``input_tokens`` counts only the
    # *uncached* prompt, so the full prompt is input + cache_read + cache_creation
    # (2000 + 1800 + 0) and the hit rate is 1800/3800 — not 1800/2000, which
    # under-counted the prompt and inflated the reported cache hit rate.
    assert provider.usage_totals["cached_tokens"] == 1800
    assert provider.usage_totals["prompt_tokens"] == 3800
    assert provider.cache_hit_rate == pytest.approx(1800 / 3800, abs=1e-4)


def test_create_model_provider_forwards_prompt_caching_disabled() -> None:
    """create_model_provider must expose the flag explicitly (not lost in kwargs)."""
    provider = create_model_provider(
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com",
        api_mode="anthropic",
        prompt_caching=False,
    )
    assert provider._prompt_caching is False
    assert "anthropic-beta" not in provider._auth_headers()


def test_orchestrator_wires_config_prompt_caching_to_provider(tmp_path: Path) -> None:
    """P1 regression: UBTConfig.prompt_caching_enabled was a dead switch — the
    orchestrator built its provider without it, so disabling it via env/config
    had no effect. The value must now reach the provider."""
    off = UBTConfig.from_env(
        api_key=SecretStr("test-key"),
        base_url="https://api.anthropic.com",
        db_dir=tmp_path,
        prompt_caching_enabled=False,
    )
    off_provider = PipelineOrchestrator(config=off).router.provider
    assert isinstance(off_provider, OpenAICompatibleProvider)
    assert off_provider._prompt_caching is False

    on = UBTConfig.from_env(
        api_key=SecretStr("test-key"),
        base_url="https://api.anthropic.com",
        db_dir=tmp_path,
    )
    on_provider = PipelineOrchestrator(config=on).router.provider
    assert isinstance(on_provider, OpenAICompatibleProvider)
    assert on_provider._prompt_caching is True
