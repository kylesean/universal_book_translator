from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ubt.api.provider_editor import ProviderEditError, ProviderEditModel, _coerce_value
from ubt.api.runtime_config import ExposedProviderFields, _safe_block
from ubt.core.config import (
    DefaultsConfig,
    ProviderConfig,
    UBTConfigFile,
)
from ubt.core.providers import (
    ProviderConfigError,
    ProviderNotFoundError,
    load_defaults_config,
    load_layer_config,
    load_provider_config,
)

pytestmark = pytest.mark.fast


def test_provider_config_valid_and_forbidden_extra() -> None:
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
    with pytest.raises(ValidationError):
        ProviderConfig(api_timeout=0)

    with pytest.raises(ValidationError):
        ProviderConfig(api_timeout=-5.0)

    with pytest.raises(ValidationError):
        ProviderConfig(qe_judge_pass_sample=1.5)

    with pytest.raises(ValidationError):
        ProviderConfig(consistency_enforce="invalid_policy")  # type: ignore[arg-type]


def test_ubt_config_file_model(tmp_path: Path) -> None:
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

    file_cfg = UBTConfigFile(
        defaults=load_defaults_config(config_path),
        providers={"custom-openai": load_provider_config("custom-openai", config_path)},
    )
    assert isinstance(file_cfg, UBTConfigFile)
    assert file_cfg.defaults.provider == "custom-openai"
    assert file_cfg.defaults.api_timeout == 90.0
    assert "custom-openai" in file_cfg.providers
    assert file_cfg.providers["custom-openai"].draft_model == "custom-model"


def test_load_defaults_config(tmp_path: Path) -> None:
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
    shipped = load_provider_config("openai-chat")
    assert isinstance(shipped, ProviderConfig)
    assert shipped.api_mode == "openai-chat"
    assert shipped.base_url == "https://api.openai.com/v1"

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

    with pytest.raises(ProviderNotFoundError):
        load_provider_config("nonexistent-provider", custom_path=config_path)

    malformed_path = tmp_path / "malformed.toml"
    malformed_path.write_text("[providers.bad]\napi_timeout = -10\n", encoding="utf-8")
    with pytest.raises(ProviderConfigError):
        load_provider_config("bad", custom_path=malformed_path)


def test_load_layer_config(tmp_path: Path) -> None:
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
    assert layer.api_timeout == 50.0
    assert layer.draft_reasoning_effort == "low"


def test_provider_edit_model_validation() -> None:
    with pytest.raises(ValidationError):
        ProviderEditModel(supports_temperature="yes")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ProviderEditModel(api_timeout=True)

    with pytest.raises(ValidationError):
        ProviderEditModel(api_timeout=-1.0)

    with pytest.raises(ValidationError):
        ProviderEditModel(fallback_models="single-string")  # type: ignore[arg-type]

    with pytest.raises(ProviderEditError):
        _coerce_value("supports_temperature", "yes")

    with pytest.raises(ProviderEditError):
        _coerce_value("api_timeout", -10)


def test_exposed_provider_fields_safe_filtering() -> None:
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
    assert not hasattr(exposed, "api_key")
    assert not hasattr(exposed, "extra_headers")

    safe = _safe_block("openai-chat")
    assert safe["available"] is True
    assert safe["base_url"] == "https://api.openai.com/v1"


def test_provider_config_strict_validation() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(supports_temperature="yes")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ProviderConfig(api_timeout=True)

    with pytest.raises(ValidationError):
        ProviderConfig(api_mode="unsupported-protocol")

    with pytest.raises(ValidationError):
        ProviderConfig(extraction_strategy="unknown-strategy")

    with pytest.raises(ValidationError):
        ProviderConfig(reasoning_dialect="invalid-dialect")

    with pytest.raises(ValidationError):
        ProviderConfig(base_url="ftp://server.invalid/v1")

    with pytest.raises(ValidationError):
        ProviderConfig(fallback_models="not-a-list")  # type: ignore[arg-type]

    norm = ProviderConfig(qe_engine="comet")
    assert norm.qe_engine == "heuristic"

    with pytest.raises(ValidationError):
        ProviderConfig(qe_engine="invalid-scorer")


def test_validate_block_rejects_invalid_values_in_toml(tmp_path: Path) -> None:
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
