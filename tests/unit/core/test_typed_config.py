"""Tests for strongly typed Pydantic V2 configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ubt.api.provider_editor import ProviderEditError, ProviderEditModel, _coerce_value
from ubt.api.runtime_config import ExposedProviderFields, _safe_block
from ubt.core.config import (
    DefaultsConfig,
    ProviderConfig,
    UBTConfig,
    UBTConfigFile,
)
from ubt.core.providers import (
    ProviderConfigError,
    ProviderNotFoundError,
    load_defaults_config,
    load_layer_config,
    load_provider_config,
    read_config_file,
)

pytestmark = pytest.mark.fast


def test_provider_config_valid_and_forbidden_extra() -> None:
    """ProviderConfig parses valid fields and forbids unknown extra keys."""
    cfg = ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_mode="openai-chat",
        draft_model="gpt-4o",
        api_timeout=60.0,
        supports_vision=True,
    )
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_mode == "openai-chat"
    assert cfg.draft_model == "gpt-4o"
    assert cfg.api_timeout == 60.0
    assert cfg.supports_vision is True

    with pytest.raises(ValidationError):
        ProviderConfig(unknown_field="invalid")  # type: ignore[call-arg]


def test_provider_config_field_constraints() -> None:
    """ProviderConfig enforces field validators and numeric bounds."""
    # api_timeout must be > 0
    with pytest.raises(ValidationError):
        ProviderConfig(api_timeout=0)

    with pytest.raises(ValidationError):
        ProviderConfig(api_timeout=-5.0)

    # qe_judge_pass_sample must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        ProviderConfig(qe_judge_pass_sample=1.5)

    # consistency_enforce must be in Literal["off", "report", "repair"]
    with pytest.raises(ValidationError):
        ProviderConfig(consistency_enforce="invalid_policy")  # type: ignore[arg-type]


def test_ubt_config_file_model(tmp_path: Path) -> None:
    """UBTConfigFile validates defaults and providers dictionary."""
    content = """
    [defaults]
    provider = "custom-openai"
    api_timeout = 90.0

    [providers.custom-openai]
    base_url = "https://custom.api.com/v1"
    api_mode = "openai-chat"
    draft_model = "custom-model"
    """
    config_path = tmp_path / "ubt.toml"
    config_path.write_text(content, encoding="utf-8")

    file_cfg = read_config_file(config_path)
    assert isinstance(file_cfg, UBTConfigFile)
    assert file_cfg.defaults.provider == "custom-openai"
    assert file_cfg.defaults.api_timeout == 90.0
    assert "custom-openai" in file_cfg.providers
    assert file_cfg.providers["custom-openai"].draft_model == "custom-model"


def test_load_defaults_config(tmp_path: Path) -> None:
    """load_defaults_config reads defaults block into DefaultsConfig."""
    config_path = tmp_path / "ubt.toml"
    config_path.write_text(
        """
        [defaults]
        provider = "demo"
        supports_temperature = false
        """,
        encoding="utf-8",
    )
    defaults = load_defaults_config(config_path)
    assert isinstance(defaults, DefaultsConfig)
    assert defaults.provider == "demo"
    assert defaults.supports_temperature is False


def test_load_provider_config_shipped_and_user_overlay(tmp_path: Path) -> None:
    """load_provider_config resolves shipped presets and merges user configuration."""
    # Shipped preset
    shipped = load_provider_config("openai-chat")
    assert isinstance(shipped, ProviderConfig)
    assert shipped.api_mode == "openai-chat"
    assert shipped.base_url == "https://api.openai.com/v1"

    # User overlay over shipped preset
    config_path = tmp_path / "ubt.toml"
    config_path.write_text(
        """
        [providers.openai-chat]
        draft_model = "gpt-4o-mini"
        api_timeout = 45.0
        """,
        encoding="utf-8",
    )
    overlaid = load_provider_config("openai-chat", custom_path=config_path)
    assert overlaid.api_mode == "openai-chat"
    assert overlaid.draft_model == "gpt-4o-mini"
    assert overlaid.api_timeout == 45.0
    assert overlaid.base_url == "https://api.openai.com/v1"

    # Unknown provider
    with pytest.raises(ProviderNotFoundError):
        load_provider_config("nonexistent-provider", custom_path=config_path)

    # Malformed provider block raises ProviderConfigError
    malformed_path = tmp_path / "malformed.toml"
    malformed_path.write_text("[providers.bad]\napi_timeout = -10\n", encoding="utf-8")
    with pytest.raises(ProviderConfigError):
        load_provider_config("bad", custom_path=malformed_path)


def test_load_layer_config(tmp_path: Path) -> None:
    """load_layer_config layers a provider block over the [defaults] baseline."""
    config_path = tmp_path / "ubt.toml"
    config_path.write_text(
        """
        [defaults]
        api_timeout = 100.0
        draft_reasoning_effort = "low"

        [providers.my-provider]
        base_url = "https://my.llm.com/v1"
        api_mode = "openai-chat"
        draft_model = "my-llm"
        api_timeout = 50.0
        """,
        encoding="utf-8",
    )
    layer = load_layer_config("my-provider", custom_path=config_path)
    assert isinstance(layer, ProviderConfig)
    assert layer.base_url == "https://my.llm.com/v1"
    assert layer.draft_model == "my-llm"
    # Layer overrides defaults
    assert layer.api_timeout == 50.0
    # Inherits unchanged defaults
    assert layer.draft_reasoning_effort == "low"


def test_ubt_config_from_provider_config() -> None:
    """UBTConfig.from_provider_config creates a UBTConfig instance directly."""
    p_cfg = ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_mode="openai-chat",
        draft_model="gpt-4o",
        api_timeout=75.0,
        is_free=True,
    )
    cfg = UBTConfig.from_provider_config(p_cfg, overrides={"draft_model": "override-model"})
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_mode == "openai-chat"
    assert cfg.draft_model == "override-model"
    assert cfg.api_timeout == 75.0
    assert cfg.is_free is True


def test_ubt_config_from_config_file(tmp_path: Path) -> None:
    """UBTConfig.from_config_file constructs UBTConfig from a file path."""
    config_path = tmp_path / "ubt.toml"
    config_path.write_text(
        """
        [defaults]
        provider = "local"

        [providers.local]
        base_url = "http://127.0.0.1:8000/v1"
        api_mode = "openai-chat"
        draft_model = "local-llama"
        api_timeout = 30.0
        """,
        encoding="utf-8",
    )
    cfg = UBTConfig.from_config_file(config_path)
    assert cfg.provider == "local"
    assert cfg.base_url == "http://127.0.0.1:8000/v1"
    assert cfg.draft_model == "local-llama"
    assert cfg.api_timeout == 30.0


def test_provider_edit_model_validation() -> None:
    """ProviderEditModel strictly validates inputs and rejects invalid coercion."""
    # Boolean strictly required, strings like "yes" rejected
    with pytest.raises(ValidationError):
        ProviderEditModel(supports_temperature="yes")  # type: ignore[arg-type]

    # api_timeout cannot be boolean or <= 0
    with pytest.raises(ValidationError):
        ProviderEditModel(api_timeout=True)

    with pytest.raises(ValidationError):
        ProviderEditModel(api_timeout=-1.0)

    # fallback_models must be list of strings
    with pytest.raises(ValidationError):
        ProviderEditModel(fallback_models="single-string")  # type: ignore[arg-type]

    # _coerce_value raises ProviderEditError on validation failure
    with pytest.raises(ProviderEditError):
        _coerce_value("supports_temperature", "yes")

    with pytest.raises(ProviderEditError):
        _coerce_value("api_timeout", -10)


def test_exposed_provider_fields_safe_filtering() -> None:
    """ExposedProviderFields ignores unsafe credentials and keeps safe display fields."""
    raw_block = {
        "api_key": "secret-key",
        "extra_headers": {"Authorization": "Bearer secret"},
        "base_url": "https://api.test.com/v1",
        "api_mode": "openai-chat",
        "draft_model": "test-model",
        "is_free": True,
    }
    exposed = ExposedProviderFields.model_validate(raw_block)
    assert exposed.base_url == "https://api.test.com/v1"
    assert exposed.api_mode == "openai-chat"
    assert exposed.draft_model == "test-model"
    assert exposed.is_free is True
    # Credentials were not exposed
    assert not hasattr(exposed, "api_key")
    assert not hasattr(exposed, "extra_headers")

    # _safe_block uses ExposedProviderFields and handles safe display
    safe = _safe_block("openai-chat")
    assert safe["available"] is True
    assert safe["base_url"] == "https://api.openai.com/v1"


def test_provider_config_strict_validation() -> None:
    """ProviderConfig strictly validates enums, URLs, booleans, and formats."""
    # Boolean cannot be string
    with pytest.raises(ValidationError):
        ProviderConfig(supports_temperature="yes")  # type: ignore[arg-type]

    # api_timeout cannot be boolean
    with pytest.raises(ValidationError):
        ProviderConfig(api_timeout=True)

    # api_mode must be one of allowed modes
    with pytest.raises(ValidationError):
        ProviderConfig(api_mode="unsupported-protocol")

    # extraction_strategy must be valid
    with pytest.raises(ValidationError):
        ProviderConfig(extraction_strategy="unknown-strategy")

    # reasoning_dialect must be valid
    with pytest.raises(ValidationError):
        ProviderConfig(reasoning_dialect="invalid-dialect")

    # base_url must start with http:// or https://
    with pytest.raises(ValidationError):
        ProviderConfig(base_url="ftp://server.invalid/v1")

    # fallback_models must be list
    with pytest.raises(ValidationError):
        ProviderConfig(fallback_models="not-a-list")  # type: ignore[arg-type]

    # qe_engine legacy normalisation
    norm = ProviderConfig(qe_engine="comet")
    assert norm.qe_engine == "heuristic"

    # qe_engine invalid choice
    with pytest.raises(ValidationError):
        ProviderConfig(qe_engine="invalid-scorer")


def test_ubt_config_from_provider_config_with_none_values() -> None:
    """None values in ProviderConfig fields do not poison UBTConfig defaults."""
    # When repair_model is None, UBTConfig should not fail with string validation error
    p = ProviderConfig(repair_model=None, qe_engine=None)
    cfg = UBTConfig.from_provider_config(p)
    assert cfg.repair_model == cfg.draft_model
    assert cfg.qe_engine in ("heuristic", "tiered")

    # Empty provider config succeeds
    empty_p = ProviderConfig()
    cfg_empty = UBTConfig.from_provider_config(empty_p)
    assert cfg_empty is not None


def test_validate_block_rejects_invalid_values_in_toml(tmp_path: Path) -> None:
    """_validate_block rejects invalid enum/url values declared in TOML files."""
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text(
        """
        [providers.bad_mode]
        api_mode = "bogus_mode"
        """,
        encoding="utf-8",
    )
    with pytest.raises(ProviderConfigError):
        load_provider_config("bad_mode", custom_path=bad_toml)

    bad_url_toml = tmp_path / "bad_url.toml"
    bad_url_toml.write_text(
        """
        [providers.bad_url]
        base_url = "file:///etc/hosts"
        """,
        encoding="utf-8",
    )
    with pytest.raises(ProviderConfigError):
        load_provider_config("bad_url", custom_path=bad_url_toml)


def test_read_config_file_missing_and_empty(tmp_path: Path) -> None:
    """read_config_file handles missing and empty config files gracefully."""
    # Missing explicit path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        read_config_file(tmp_path / "does_not_exist.toml")

    # Empty file parses defaults and empty providers
    empty_toml = tmp_path / "empty.toml"
    empty_toml.write_text("", encoding="utf-8")
    cfg_file = read_config_file(empty_toml)
    assert isinstance(cfg_file, UBTConfigFile)
    assert cfg_file.providers == {}
