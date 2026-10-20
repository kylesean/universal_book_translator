"""Capability resolution: declared per provider, never inferred from a model name.

The engine used to ship a fourteen-entry table of name patterns (``o1``, ``o3``,
``claude-3-7``, ``qwen``, ``muse-spark``, ``deepseek-r1``, ``reasoning``,
``reasoner``, ``deepseek``, ``flash``, ``pro``) and resolve a model by exact
match, then separator-bounded substring, then a heuristic on
``reasoner``/``reasoning``/``thinking``/``r1``. These tests pin the four measured
defects that made the table unfixable, so a future "just add the missing model
names" patch fails here first.
"""

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
    """ModelProfile without explicit prompt_strategy defaults to HYBRID."""
    profile = ModelProfile(model_pattern="test-model")
    assert profile.prompt_strategy == PromptStrategy.HYBRID


def test_an_undeclared_model_gets_the_generic_fallback() -> None:
    """No name table: an unregistered model resolves to the safe default.

    The old table returned a *named* profile for any model whose name contained
    ``pro`` or ``flash``, which set ``supports_reasoning_effort=True`` and thereby
    removed the output-token cap from models that had never asked for it.
    """
    registry = ModelCapabilityRegistry()
    for model in ("gemma4-e4b", "translategemma:4b", "bonsai2-27b-fast", "gpt-5"):
        profile = registry.resolve(model)
        assert profile.model_pattern == FALLBACK_PATTERN, model
        assert profile.is_fallback, model
        assert profile.supports_reasoning_effort is False, model


def test_no_substring_or_tier_word_match() -> None:
    """A name ending in ``-pro`` or containing ``r1`` is not a capability claim.

    Both were real: ``pro`` captured ``llama-3.1-pro``, ``doubao-pro``,
    ``hunyuan-pro`` and ``minimax-m2-pro``; ``r1`` captured ``tar1ff`` and
    ``mar1ne-embed``. The latter made the diagnostics screen report
    ``matched_pattern: "tar1ff"`` — an invented profile name presented as a real
    match.
    """
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
    """A declared profile applies to its model id and nothing else."""
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

    # Case-insensitive, as a model id is not case-significant on the wire.
    assert registry.resolve("QWEN3:32B").model_pattern == "qwen3:32b"
    # A different id must NOT inherit it — that is the whole point.
    assert registry.resolve("qwen3:32b-instruct").is_fallback
    assert registry.resolve("qwen").is_fallback


def test_declared_profile_requires_exact_keys() -> None:
    """A misspelled capability is an error, not a silent default.

    ``extra="ignore"`` made ``supports_temperatur`` fall back to the field
    default (``True``), so a declaration that the endpoint rejects temperature
    silently asserted the opposite.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelProfile(model_pattern="m", supports_temperatur=False)  # type: ignore[call-arg]


def test_router_falls_back_without_a_registry() -> None:
    """A router built with no registry uses the same fallback, not a second copy."""
    router = ModelRouter(provider=MockModelProvider(), draft_model="arbitrary-model")
    profile = router._get_profile()
    assert profile.is_fallback
    assert profile.supports_reasoning_effort is False
    assert profile.supports_system_prompt is True


def test_router_build_draft_prompt_uses_hybrid() -> None:
    """ModelRouter builds the hybrid prompt for an undeclared model."""
    registry = ModelCapabilityRegistry()
    router = ModelRouter(
        provider=MockModelProvider(), draft_model="deepseek-chat", registry=registry
    )
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
    router_fallback = ModelRouter(provider=MockModelProvider(), draft_model="arbitrary-model")
    sys_prompt_fb, _ = router_fallback.build_draft_prompt(
        source_text="This is an introduction to quantum computing.",
        target_lang="zh",
        source_lang="en",
    )
    assert "Typography & Terminology" in sys_prompt_fb


# ---------------------------------------------------------------------------
# Per-configuration profiles: the cross-job leak
# ---------------------------------------------------------------------------


def _config(**overrides: Any) -> Any:
    """A bare ``UBTConfig`` with only the fields a test cares about set."""
    from ubt.core.config import UBTConfig

    return UBTConfig(**overrides)


def test_capability_profiles_come_from_the_configuration() -> None:
    """Declared flags reach the profile; undeclared ones keep the safe default."""
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
    # Not declared: the safe default, not an assertion.
    assert profile.supports_vision is False
    assert profile.supports_system_prompt is True


def test_a_config_override_does_not_leak_into_another_config() -> None:
    """The measured cross-job leak: job A's flags became job B's capability.

    ``bootstrap_runtime_environment`` used to register a profile into the
    process-wide singleton for every ``draft_model``/``repair_model``, so a job
    that pinned a capability for ``shared-model`` changed the capability of the
    next job using the same model id — and the registry grew without bound.
    """
    from ubt.core.router.registry import get_default_registry

    before = len(get_default_registry().list_profiles())

    job_a = _config(draft_model="shared-model", supports_system_prompt=False)
    job_a.bootstrap_runtime_environment()
    assert registry_from_config(job_a).resolve("shared-model").supports_system_prompt is False

    job_b = _config(draft_model="shared-model")
    job_b.bootstrap_runtime_environment()
    assert registry_from_config(job_b).resolve("shared-model").supports_system_prompt is True

    # The process-wide registry holds only environment-declared profiles, so a
    # config never grows it.
    assert len(get_default_registry().list_profiles()) == before


def test_an_override_preserves_flags_it_does_not_mention() -> None:
    """Setting one flag must not erase another.

    The old copy listed five fields, so any override dropped ``supports_vision``
    and ``deployment_backend`` — the exact flags the visual-scalpel repair and the
    local self-hosted retry read.
    """
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
    """``auto`` means "no opinion", not "an unknown strategy value"."""
    config = _config(draft_model="m", prompt_strategy="auto")
    profile = registry_from_config(config).resolve("m")
    assert profile.prompt_strategy is PromptStrategy.HYBRID

    forced = _config(draft_model="m", prompt_strategy="rich")
    assert registry_from_config(forced).resolve("m").prompt_strategy is PromptStrategy.RICH


# ---------------------------------------------------------------------------
# Environment-declared profiles: per-item tolerance
# ---------------------------------------------------------------------------


def test_one_malformed_env_profile_does_not_discard_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single bad entry must not take the whole list with it.

    The loader wrapped its whole loop in one ``try``, so one invalid enum
    discarded every other profile and logged a truncated 200-character warning.
    """
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
    """``extra="forbid"`` on ModelProfile, applied to the environment path."""
    raw = [{"model_pattern": "typo-one", "supports_temperatur": False}]
    monkeypatch.setenv("UBT_MODEL_PROFILES_JSON", json.dumps(raw))
    monkeypatch.delenv("UBT_MODEL_PROFILES_FILE", raising=False)

    from ubt.core.router.registry import _load_environment_profiles

    assert _load_environment_profiles() == []


def test_the_fallback_profile_claims_nothing_unsafe() -> None:
    """The default must not provoke a 400 on an unknown endpoint."""
    profile = fallback_profile("anything")
    assert profile.supports_temperature is True
    assert profile.supports_system_prompt is True
    # Both of these cost a failed call on every block when assumed wrongly.
    assert profile.supports_reasoning_effort is False
    assert profile.supports_vision is False


def test_a_config_that_declares_nothing_produces_no_profile() -> None:
    """The distinction the diagnostics screen exists to draw.

    Minting a profile for the model id unconditionally made ``resolve`` report
    ``matched_pattern == "gemma4-e4b"`` — an echo of the model id dressed up as a
    resolved capability, and indistinguishable from a real declaration.
    """
    undeclared = _config(draft_model="gemma4-e4b")
    assert undeclared.capability_profiles() == []
    assert registry_from_config(undeclared).resolve("gemma4-e4b").is_fallback

    # Any single declaration is enough to produce a real profile.
    for declared in (
        _config(draft_model="gemma4-e4b", supports_temperature=False),
        _config(draft_model="gemma4-e4b", supports_vision=True),
        _config(draft_model="gemma4-e4b", prompt_strategy="rich"),
        _config(draft_model="gemma4-e4b", deployment_backend="llama-swap"),
    ):
        profiles = declared.capability_profiles()
        assert len(profiles) == 1
        assert not registry_from_config(declared).resolve("gemma4-e4b").is_fallback
