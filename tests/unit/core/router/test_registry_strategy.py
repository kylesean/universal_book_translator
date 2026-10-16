"""Unit tests verifying ModelProfile prompt strategy defaulting to HYBRID and registry resolution."""

from __future__ import annotations

import pytest

from ubt.core.router.capabilities import ModelProfile, PromptStrategy
from ubt.core.router.registry import ModelCapabilityRegistry
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


def test_model_profile_default_strategy() -> None:
    """ModelProfile without explicit prompt_strategy defaults to HYBRID."""
    profile = ModelProfile(model_pattern="test-model")
    assert profile.prompt_strategy == PromptStrategy.HYBRID


def test_builtin_models_resolve_to_hybrid() -> None:
    """Builtin mainstream models (o1, claude, qwen, deepseek, flash, pro) use HYBRID."""
    registry = ModelCapabilityRegistry()
    test_models = [
        "o1",
        "o3-mini",
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "qwen-2.5-72b",
        "deepseek-chat",
        "deepseek-reasoner",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    for model in test_models:
        profile = registry.resolve(model)
        assert profile.prompt_strategy == PromptStrategy.HYBRID, (
            f"{model} did not resolve to HYBRID"
        )


def test_unknown_and_heuristic_models_resolve_to_hybrid() -> None:
    """Unregistered reasoning and arbitrary unknown models fallback to HYBRID."""
    registry = ModelCapabilityRegistry()
    assert registry.resolve("custom-r1-distill").prompt_strategy == PromptStrategy.HYBRID
    assert registry.resolve("my-totally-unknown-llm").prompt_strategy == PromptStrategy.HYBRID


class _DummyProvider:
    is_mock = True

    async def generate_with_finish_reason(self, **kwargs: object) -> tuple[str, str | None]:
        return "translation", "stop"

    async def generate(self, prompt: str, **kwargs: object) -> str:
        return "translation"


def test_router_build_draft_prompt_uses_hybrid() -> None:
    """ModelRouter builds hybrid prompt for mainstream model by default."""
    registry = ModelCapabilityRegistry()
    router = ModelRouter(provider=_DummyProvider(), draft_model="deepseek-chat", registry=registry)
    sys_prompt, user_prompt = router.build_draft_prompt(
        source_text="This is an introduction to quantum computing.",
        target_lang="zh",
        source_lang="en",
    )
    # Hybrid prompt produces concise instructions without rigid relative order negative rules
    assert "Typography & Terminology" in sys_prompt
    assert "verbatim in their original relative order" not in sys_prompt
    assert "### Source Paragraph to Translate" in user_prompt

    # Also test router with default/None registry fallback
    router_fallback = ModelRouter(provider=_DummyProvider(), draft_model="arbitrary-model")
    sys_prompt_fb, _ = router_fallback.build_draft_prompt(
        source_text="This is an introduction to quantum computing.",
        target_lang="zh",
        source_lang="en",
    )
    assert "Typography & Terminology" in sys_prompt_fb
