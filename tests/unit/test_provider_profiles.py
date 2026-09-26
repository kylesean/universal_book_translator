"""Unit tests for provider profile loading from TOML files and UBTConfig integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.core.config import UBTConfig
from ubt.core.job_options import apply_config_overrides
from ubt.core.profiles import (
    ProfileNotFoundError,
    find_config_file,
    load_all_profiles,
    load_provider_profile,
)
from ubt.core.router.capabilities import ModelProfile, PromptStrategy
from ubt.core.router.registry import ModelCapabilityRegistry


@pytest.fixture
def sample_toml_file(tmp_path: Path) -> Path:
    toml_content = """
[profiles.deepseek]
api_key = "sk-deepseek-test"
base_url = "https://api.deepseek.com/v1"
draft_model = "deepseek-chat"
repair_model = "deepseek-reasoner"
api_mode = "chat"

[profiles.gemini]
api_key = "AIza-test-key"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
draft_model = "gemini-3.8-flash"
repair_model = "gemini-3.1-pro"
api_mode = "chat"

[profiles.claude]
api_key = "sk-ant-test-key"
base_url = "https://api.anthropic.com"
draft_model = "claude-3-5-haiku"
repair_model = "claude-3-7-sonnet"
api_mode = "anthropic"

[profiles.openai-responses]
api_key = "sk-proj-test-key"
base_url = "https://api.openai.com/v1"
draft_model = "gpt-4o-mini"
repair_model = "o3-mini"
api_mode = "responses"
"""
    file_path = tmp_path / "ubt.toml"
    file_path.write_text(toml_content, encoding="utf-8")
    return file_path


def test_load_all_profiles(sample_toml_file: Path) -> None:
    profiles = load_all_profiles(sample_toml_file)
    assert "deepseek" in profiles
    assert "gemini" in profiles
    assert "claude" in profiles
    assert "openai-responses" in profiles
    assert profiles["claude"]["api_mode"] == "anthropic"


def test_load_provider_profile_success(sample_toml_file: Path) -> None:
    prof = load_provider_profile("gemini", custom_path=sample_toml_file)
    assert prof["api_key"] == "AIza-test-key"
    assert prof["draft_model"] == "gemini-3.8-flash"
    assert prof["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_load_provider_profile_not_found(sample_toml_file: Path) -> None:
    with pytest.raises(ProfileNotFoundError, match="Provider profile 'unknown' not found"):
        load_provider_profile("unknown", custom_path=sample_toml_file)


def test_find_config_file_nonexistent(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        find_config_file(non_existent)


def test_apply_config_overrides_with_provider_profile(
    sample_toml_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ubt.core.profiles.DEFAULT_CONFIG_LOCATIONS", (sample_toml_file,))
    base_cfg = UBTConfig.from_env()

    # 1. Apply claude profile
    overrides = {"provider_profile": "claude"}
    cfg = apply_config_overrides(base_cfg, overrides)
    assert cfg.api_key.get_secret_value() == "sk-ant-test-key"
    assert cfg.base_url == "https://api.anthropic.com"
    assert cfg.api_mode == "anthropic"
    assert cfg.draft_model == "claude-3-5-haiku"
    assert cfg.repair_model == "claude-3-7-sonnet"

    # 2. Explicit CLI flag overrides profile
    overrides_with_flag = {
        "provider_profile": "claude",
        "draft_model": "custom-claude-model",
        "api_key": "sk-override-key",
    }
    cfg2 = apply_config_overrides(base_cfg, overrides_with_flag)
    assert cfg2.draft_model == "custom-claude-model"
    assert cfg2.repair_model == "claude-3-7-sonnet"  # Kept from profile
    assert cfg2.api_key.get_secret_value() == "sk-override-key"


def test_ubt_config_from_env_with_provider_profile(
    sample_toml_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ubt.core.profiles.DEFAULT_CONFIG_LOCATIONS", (sample_toml_file,))
    cfg = UBTConfig.from_env(provider_profile="gemini")
    assert cfg.api_key.get_secret_value() == "AIza-test-key"
    assert cfg.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert cfg.draft_model == "gemini-3.8-flash"
    assert cfg.repair_model == "gemini-3.1-pro"


def test_gemini_endpoint_auto_normalization() -> None:
    # Test short alias
    cfg1 = UBTConfig(base_url="gemini")
    assert cfg1.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

    # Test domain only
    cfg2 = UBTConfig(base_url="https://generativelanguage.googleapis.com")
    assert cfg2.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

    # Test v1beta only
    cfg3 = UBTConfig(base_url="https://generativelanguage.googleapis.com/v1beta")
    assert cfg3.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_anthropic_api_mode_auto_detection() -> None:
    # When base_url contains api.anthropic.com and api_mode is not explicitly specified
    cfg = UBTConfig(base_url="https://api.anthropic.com")
    assert cfg.api_mode == "anthropic"

    # When api_mode is explicitly set, honor the explicit value
    cfg_explicit = UBTConfig(base_url="https://api.anthropic.com", api_mode="chat")
    assert cfg_explicit.api_mode == "chat"


def test_opencode_go_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-test-opencode-key")
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
    cfg = UBTConfig.from_env()
    assert cfg.api_key.get_secret_value() == "sk-test-opencode-key"
    assert cfg.base_url == "https://opencode.ai/zen/go/v1"
    assert cfg.draft_model == "muse-spark-1.3-contributor"
    assert cfg.api_mode == "responses"


def test_muse_spark_auto_responses_mode() -> None:
    # Any muse- model automatically selects responses mode
    cfg = UBTConfig(draft_model="muse-spark-1.3-contributor")
    assert cfg.api_mode == "responses"


def test_api_mode_rederives_after_draft_override() -> None:
    """A request override that moves off a muse- model must drop responses mode.

    ``apply_config_overrides`` assigns one field at a time, so the ``responses``
    api_mode derived from a muse- draft used to latch in ``model_fields_set`` and
    block re-derivation, making the request path disagree with ``from_env``.
    """
    base = UBTConfig(draft_model="muse-spark-1.3-contributor")
    assert base.api_mode == "responses"

    via_request = apply_config_overrides(base, {"draft_model": "gpt-4o"})
    assert via_request.api_mode == "chat"
    assert via_request.api_mode == UBTConfig.from_env(draft_model="gpt-4o").api_mode


def test_api_mode_rederives_after_endpoint_override() -> None:
    """Moving the endpoint to Anthropic must re-select anthropic mode."""
    base = UBTConfig(draft_model="muse-spark-1.3-contributor")
    assert base.api_mode == "responses"

    cfg = apply_config_overrides(
        base, {"draft_model": "gpt-4o", "base_url": "https://api.anthropic.com"}
    )
    assert cfg.api_mode == "anthropic"


def test_explicit_api_mode_still_wins_over_derivation() -> None:
    """An explicit api_mode is never overwritten by the derivation."""
    cfg = apply_config_overrides(
        UBTConfig(draft_model="muse-spark-1.3-contributor"),
        {"draft_model": "gpt-4o", "api_mode": "responses"},
    )
    assert cfg.api_mode == "responses"


def test_profile_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY_ENV", "sk-expanded-secret")
    content = """
[profiles.env-test]
api_key = "${TEST_KEY_ENV}"
base_url = "${TEST_URL:-https://default.example.com/v1}"
draft_model = "muse-spark-1.3-contributor"
"""
    p = tmp_path / "ubt.toml"
    p.write_text(content, encoding="utf-8")
    profiles = load_all_profiles(p)
    assert profiles["env-test"]["api_key"] == "sk-expanded-secret"
    assert profiles["env-test"]["base_url"] == "https://default.example.com/v1"


def test_ubt_config_from_env_reads_profile_from_dotenv(
    sample_toml_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("ubt.core.profiles.DEFAULT_CONFIG_LOCATIONS", (sample_toml_file,))
    monkeypatch.delenv("UBT_PROVIDER_PROFILE", raising=False)
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("UBT_PROVIDER_PROFILE=gemini\n", encoding="utf-8")
    monkeypatch.setitem(UBTConfig.model_config, "env_file", str(env_path))
    cfg = UBTConfig.from_env()
    assert cfg.api_key.get_secret_value() == "AIza-test-key"
    assert cfg.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert cfg.draft_model == "gemini-3.8-flash"
    assert cfg.repair_model == "gemini-3.1-pro"


@pytest.mark.fast
def test_model_profiles_post_forbids_overriding_builtin_profiles() -> None:
    """[HIGH-T4-5] POST /api/v1/model-profiles must forbid overriding existing/built-in profiles
    (override=False -> 409 Conflict) and require verify_api_key when service_api_key is configured."""
    reg = ModelCapabilityRegistry()
    with pytest.raises(ValueError, match="already"):
        reg.register(
            ModelProfile(
                model_pattern="deepseek",
                prompt_strategy=PromptStrategy.MINIMAL,
            ),
            override=False,
        )

    app = create_app(config=UBTConfig(service_api_key=SecretStr("gate-secret-123")))
    client = TestClient(app)

    # Unauthenticated request must be rejected with 401
    res_unauth = client.post(
        "/api/v1/model-profiles",
        json={"model_pattern": "custom-new-model", "prompt_strategy": "minimal"},
    )
    assert res_unauth.status_code == 401

    # Authenticated attempt to override built-in 'deepseek' must be rejected with 409 Conflict
    res_conflict = client.post(
        "/api/v1/model-profiles",
        headers={"X-API-Key": "gate-secret-123"},
        json={"model_pattern": "deepseek", "prompt_strategy": "minimal"},
    )
    assert res_conflict.status_code == 409
