from __future__ import annotations

import json
from typing import Any

import pytest

from ubt.core.router.capabilities import ModelProfile, PromptStrategy
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.registry import (
    FALLBACK_PATTERN,
    ModelCapabilityRegistry,
    fallback_profile,
    registry_from_config,
)
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


def test_model_profile_default_strategy() -> None:
    profile = ModelProfile(model_pattern="test-model")
    assert profile.prompt_strategy == PromptStrategy.HYBRID


def test_an_undeclared_model_gets_the_generic_fallback() -> None:
    registry = ModelCapabilityRegistry()
    for model in ("gemma4-e4b", "translategemma:4b", "bonsai2-27b-fast", "gpt-5"):
        profile = registry.resolve(model)
        assert profile.model_pattern == FALLBACK_PATTERN, model
        assert profile.is_fallback, model
        assert profile.supports_reasoning_effort is False, model


def test_no_substring_or_tier_word_match() -> None:
    registry = ModelCapabilityRegistry()
    for model in (
        "llama-3.1-pro",
        "doubao-pro",
        "hunyuan-pro",
        "minimax-m2-pro",
        "glm-4.6-flash",
        "tar1ff",
        "mar1ne-embed",
        "thinkingcap",
        "claude-sonnet-4-5",
        "claude-opus-4-1",
    ):
        assert registry.resolve(model).is_fallback, model


def test_a_declared_profile_wins_and_is_exact() -> None:
    registry = ModelCapabilityRegistry(
        [
            ModelProfile(
                model_pattern="qwen3:32b",
                supports_reasoning_effort=True,
                supports_temperature=False,
                display_name="Qwen3 local",
            )
        ]
    )
    exact = registry.resolve("qwen3:32b")
    assert exact.model_pattern == "qwen3:32b"
    assert exact.supports_reasoning_effort is True
    assert exact.supports_temperature is False

    assert registry.resolve("QWEN3:32B").model_pattern == "qwen3:32b"
    assert registry.resolve("qwen3:32b-instruct").is_fallback
    assert registry.resolve("qwen").is_fallback


def test_declared_profile_requires_exact_keys() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelProfile(model_pattern="m", supports_temperatur=False)  # type: ignore[call-arg]


def test_router_falls_back_without_a_registry() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="arbitrary-model")
    profile = router._get_profile()
    assert profile.is_fallback
    assert profile.supports_reasoning_effort is False
    assert profile.supports_system_prompt is True


def test_router_build_draft_prompt_uses_hybrid() -> None:
    registry = ModelCapabilityRegistry()
    router = ModelRouter(
        provider=MockModelProvider(), draft_model="deepseek-chat", registry=registry
    )
    sys_prompt, user_prompt = router.build_draft_prompt(
        source_text="This is an introduction to quantum computing.",
        target_lang="zh",
        source_lang="en",
    )
    assert "Typography & Terminology" in sys_prompt
    assert "verbatim in their original relative order" not in sys_prompt
    assert "### Source Paragraph to Translate" in user_prompt

    router_fallback = ModelRouter(provider=MockModelProvider(), draft_model="arbitrary-model")
    sys_prompt_fb, _ = router_fallback.build_draft_prompt(
        source_text="This is an introduction to quantum computing.",
        target_lang="zh",
        source_lang="en",
    )
    assert "Typography & Terminology" in sys_prompt_fb


def _config(**overrides: Any) -> Any:
    from ubt.core.config import UBTConfig

    return UBTConfig(**overrides)


def test_capability_profiles_come_from_the_configuration() -> None:
    config = _config(
        draft_model="local-model",
        repair_model="local-model",
        supports_reasoning_effort=True,
        supports_temperature=False,
        deployment_backend="llama-swap",
    )
    profile = registry_from_config(config).resolve("local-model")
    assert profile.supports_reasoning_effort is True
    assert profile.supports_temperature is False
    assert profile.deployment_backend == "llama-swap"
    assert profile.supports_vision is False
    assert profile.supports_system_prompt is True


def test_a_config_override_does_not_leak_into_another_config() -> None:
    from ubt.core.router.registry import get_default_registry

    before = len(get_default_registry().list_profiles())

    job_a = _config(draft_model="shared-model", supports_system_prompt=False)
    job_a.bootstrap_runtime_environment()
    assert registry_from_config(job_a).resolve("shared-model").supports_system_prompt is False

    job_b = _config(draft_model="shared-model")
    job_b.bootstrap_runtime_environment()
    assert registry_from_config(job_b).resolve("shared-model").supports_system_prompt is True

    assert len(get_default_registry().list_profiles()) == before


def test_an_override_preserves_flags_it_does_not_mention() -> None:
    config = _config(
        draft_model="vision-model",
        supports_vision=True,
        deployment_backend="ollama",
        supports_temperature=False,
    )
    profile = registry_from_config(config).resolve("vision-model")
    assert profile.supports_vision is True
    assert profile.deployment_backend == "ollama"
    assert profile.supports_temperature is False


def test_prompt_strategy_auto_keeps_the_profile_default() -> None:
    config = _config(draft_model="m", prompt_strategy="auto")
    profile = registry_from_config(config).resolve("m")
    assert profile.prompt_strategy is PromptStrategy.HYBRID

    forced = _config(draft_model="m", prompt_strategy="rich")
    assert registry_from_config(forced).resolve("m").prompt_strategy is PromptStrategy.RICH


def test_one_malformed_env_profile_does_not_discard_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = [
        {"model_pattern": "good-one", "supports_temperature": False},
        {"model_pattern": "bad-one", "prompt_strategy": "nonsense"},
        {"model_pattern": "good-two", "supports_vision": True},
    ]
    monkeypatch.setenv("UBT_MODEL_PROFILES_JSON", json.dumps(raw))
    monkeypatch.delenv("UBT_MODEL_PROFILES_FILE", raising=False)

    from ubt.core.router.registry import _load_environment_profiles

    loaded = {p.model_pattern: p for p in _load_environment_profiles()}
    assert set(loaded) == {"good-one", "good-two"}
    assert loaded["good-one"].supports_temperature is False
    assert loaded["good-two"].supports_vision is True


def test_a_typo_in_an_env_profile_is_rejected_not_defaulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = [{"model_pattern": "typo-one", "supports_temperatur": False}]
    monkeypatch.setenv("UBT_MODEL_PROFILES_JSON", json.dumps(raw))
    monkeypatch.delenv("UBT_MODEL_PROFILES_FILE", raising=False)

    from ubt.core.router.registry import _load_environment_profiles

    assert _load_environment_profiles() == []


def test_the_fallback_profile_claims_nothing_unsafe() -> None:
    profile = fallback_profile("anything")
    assert profile.supports_temperature is True
    assert profile.supports_system_prompt is True
    assert profile.supports_reasoning_effort is False
    assert profile.supports_vision is False


def test_a_config_that_declares_nothing_produces_no_profile() -> None:
    undeclared = _config(draft_model="gemma4-e4b")
    assert undeclared.capability_profiles() == []
    assert registry_from_config(undeclared).resolve("gemma4-e4b").is_fallback

    for declared in (
        _config(draft_model="gemma4-e4b", supports_temperature=False),
        _config(draft_model="gemma4-e4b", supports_vision=True),
        _config(draft_model="gemma4-e4b", prompt_strategy="rich"),
        _config(draft_model="gemma4-e4b", deployment_backend="llama-swap"),
    ):
        profiles = declared.capability_profiles()
        assert len(profiles) == 1
        assert not registry_from_config(declared).resolve("gemma4-e4b").is_fallback
