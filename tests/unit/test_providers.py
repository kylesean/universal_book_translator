"""The provider registry: built-ins, user ``[providers.*]`` blocks, one ladder.

A *provider* bundles everything about where and how to call an LLM — endpoint,
wire protocol, default models — and names the environment variable holding its
credential (``api_key_env``). The secret itself never lives in the TOML: a key
committed to a version-controlled file is a leaked key, so a block carrying one
is rejected outright.

``explicit > environment > provider block > [defaults] > field default`` is one
rule, owned by ``merge_provider_under`` and shared by ``UBTConfig.from_env`` and
``apply_config_overrides``; the tests below pin each layer boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from ubt.core.config import UBTConfig
from ubt.core.job_options import apply_config_overrides
from ubt.core.providers import (
    ProviderConfigError,
    ProviderNotFoundError,
    _read_shipped_providers,
    find_config_file,
    list_providers,
    load_defaults_block,
    load_provider_block,
    merge_provider_under,
)

pytestmark = pytest.mark.fast


@pytest.fixture
def config_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A user config: a ``[defaults]`` baseline plus two ``[providers.*]`` blocks.

    ``gemini`` extends the built-in (only its models change); ``custom`` is a
    provider with no built-in base, so it must declare its own endpoint and
    credential variable.
    """
    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\n"
        "api_timeout = 42.0\n"
        "\n"
        "[providers.gemini]\n"
        'draft_model = "gemini-user-draft"\n'
        'repair_model = "gemini-user-repair"\n'
        "\n"
        "[providers.custom]\n"
        'base_url = "https://custom.example/v1"\n'
        'api_mode = "openai-chat"\n'
        'api_key_env = "CUSTOM_API_KEY"\n'
        'draft_model = "custom-draft"\n'
        'repair_model = "custom-repair"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("ubt.core.providers.DEFAULT_CONFIG_LOCATIONS", (path,))
    return path


# -- Registry -----------------------------------------------------------------


def test_shipped_providers_are_pure_wire_protocols() -> None:
    shipped = _read_shipped_providers()
    assert set(shipped) == {
        "openai-chat",
        "openai-responses",
        "anthropic-messages",
        "gemini-native",
    }
    assert "deepseek" not in shipped
    assert "opencode" not in shipped
    # Shipped protocols are pure wire transports: they never hardcode model IDs.
    assert all("draft_model" not in block for block in shipped.values())
    assert all("repair_model" not in block for block in shipped.values())
    # Every shipped protocol names the variable its credential is read from.
    assert all(block.get("api_key_env") for block in shipped.values())


def test_shipped_registry_is_validated_like_a_user_block() -> None:
    """The packaged registry goes through ``_validate_block`` — no secrets, no typos.

    Reading it raises ``ProviderConfigError`` on a leaked key or an unknown
    field, so a successful read is itself the assertion; the checks below pin
    the shape the loader relies on.
    """
    shipped = _read_shipped_providers()
    assert shipped
    for name, block in shipped.items():
        assert "api_key" not in block, name
        assert "api_key_env" in block, name


def test_shipped_registry_ships_inside_the_package() -> None:
    """``packages = ["ubt"]`` must carry the registry into a wheel install."""
    import ubt

    pkg_root = Path(ubt.__file__).resolve().parent
    assert (pkg_root / "resources" / "providers.toml").is_file()


def test_list_providers_merges_builtins_and_declared(config_toml: Path) -> None:
    names = list_providers(config_toml)
    assert "custom" in names  # user-declared
    assert "openai" in names  # built-in
    assert names == sorted(names)


def test_load_builtin_protocol_and_alias() -> None:
    fields, api_key_env = load_provider_block("openai-chat")
    assert api_key_env == "OPENAI_API_KEY"
    assert fields["base_url"] == "https://api.openai.com/v1"
    assert fields["api_mode"] == "openai-chat"
    assert "draft_model" not in fields  # protocols do not hardcode model IDs
    assert "api_key" not in fields  # secret is never in the block

    # Alias 'openai' maps to the canonical 'openai-chat' protocol
    alias_fields, alias_key_env = load_provider_block("openai")
    assert alias_key_env == "OPENAI_API_KEY"
    assert alias_fields == fields


def test_user_block_overrides_a_builtin_field(config_toml: Path) -> None:
    fields, api_key_env = load_provider_block("gemini", config_toml)
    # The user changed only the models; the rest still comes from the built-in.
    assert fields["draft_model"] == "gemini-user-draft"
    assert fields["repair_model"] == "gemini-user-repair"
    assert fields["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert api_key_env == "GEMINI_API_KEY"


def test_declared_provider_without_a_builtin_base(config_toml: Path) -> None:
    fields, api_key_env = load_provider_block("custom", config_toml)
    assert api_key_env == "CUSTOM_API_KEY"
    assert fields == {
        "base_url": "https://custom.example/v1",
        "api_mode": "openai-chat",
        "draft_model": "custom-draft",
        "repair_model": "custom-repair",
    }


def test_provider_block_supports_custom_capabilities_and_pricing(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[providers.enterprise]\n"
        'base_url = "https://llm.corp.example/v1"\n'
        'api_mode = "openai-chat"\n'
        'api_key_env = "CORP_KEY"\n'
        'draft_model = "corp-draft"\n'
        'repair_model = "corp-repair"\n'
        'repair_provider = "anthropic"\n'
        "is_free = true\n"
        "cost_per_mtok = [0.10, 0.40]\n"
        "supports_batch_api = true\n"
        "supports_temperature = false\n"
        "supports_reasoning_effort = true\n"
        'reasoning_dialect = "flat"\n',
        encoding="utf-8",
    )
    fields, api_key_env = load_provider_block("enterprise", path)
    assert api_key_env == "CORP_KEY"
    assert fields["is_free"] is True
    assert fields["cost_per_mtok"] == [0.10, 0.40]
    assert fields["supports_batch_api"] is True
    assert fields["repair_provider"] == "anthropic"
    assert fields["supports_temperature"] is False
    assert fields["supports_reasoning_effort"] is True
    assert fields["reasoning_dialect"] == "flat"


def test_shipped_protocols_batch_api_flags() -> None:
    shipped = _read_shipped_providers()
    assert shipped["openai-chat"]["supports_batch_api"] is True
    assert shipped["anthropic-messages"]["supports_batch_api"] is True
    assert shipped["gemini-native"].get("supports_batch_api", False) is False
    assert shipped["openai-responses"].get("supports_batch_api", False) is False


def test_shipped_gemini_native_speaks_the_native_protocol() -> None:
    """Gemini native protocol points at Google v1beta API."""
    block = _read_shipped_providers()["gemini-native"]
    assert block["api_mode"] == "gemini-native"
    assert block["base_url"] == "https://generativelanguage.googleapis.com/v1beta"


def test_undeclared_vendor_raises_not_found(config_toml: Path) -> None:
    """Third-party providers like deepseek are not hardcoded: user declares them or uses pure protocol."""
    with pytest.raises(ProviderNotFoundError, match="Available providers"):
        load_provider_block("deepseek", config_toml)


def test_unknown_provider_raises_and_lists_available(config_toml: Path) -> None:
    with pytest.raises(ProviderNotFoundError, match="Available providers"):
        load_provider_block("does-not-exist", config_toml)


def test_find_config_file_raises_for_a_missing_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_config_file(tmp_path / "missing.toml")


def test_find_config_file_returns_none_without_a_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ubt.core.providers.DEFAULT_CONFIG_LOCATIONS", (tmp_path / "absent.toml",))
    assert find_config_file() is None


# -- Block validation (fail-closed on secrets) --------------------------------


def test_block_may_not_carry_a_credential(tmp_path: Path) -> None:
    """A key committed to a version-controlled TOML is a leaked key."""
    path = tmp_path / "config.toml"
    path.write_text('[providers.leaky]\napi_key = "sk-leaked"\n', encoding="utf-8")
    with pytest.raises(ProviderConfigError, match="leaked key"):
        load_provider_block("leaky", path)


def test_block_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[providers.typo]\nbase_urll = "https://x.example/v1"\n', encoding="utf-8")
    with pytest.raises(ProviderConfigError, match="unknown field"):
        load_provider_block("typo", path)


def test_block_expands_environment_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_URL_ENV", "https://expanded.example/v1")
    monkeypatch.setenv("TEST_SESSION_ENV", "ses_42")
    path = tmp_path / "config.toml"
    path.write_text(
        "[providers.envtest]\n"
        'base_url = "${TEST_URL_ENV}"\n'
        'draft_model = "${TEST_MODEL_ENV:-fallback-model}"\n'
        'extra_headers = { "x-session" = "${TEST_SESSION_ENV}" }\n',
        encoding="utf-8",
    )
    fields, _ = load_provider_block("envtest", path)
    assert fields["base_url"] == "https://expanded.example/v1"
    assert fields["draft_model"] == "fallback-model"
    # Table values expand too, so a header can name the variable it reads from.
    assert fields["extra_headers"] == {"x-session": "ses_42"}


def test_defaults_block_is_validated_and_never_names_a_credential(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[defaults]\napi_timeout = 30.0\napi_key_env = "SHOULD_BE_IGNORED"\n',
        encoding="utf-8",
    )
    # api_key_env is a provider-only key; in [defaults] it is dropped, not honoured.
    assert load_defaults_block(path) == {"api_timeout": 30.0}


def test_defaults_block_rejects_a_credential(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[defaults]\napi_key = "sk-leaked"\n', encoding="utf-8")
    with pytest.raises(ProviderConfigError, match="leaked key"):
        load_defaults_block(path)


# -- Precedence ladder --------------------------------------------------------


def test_merge_provider_under_explicit_wins_over_the_block() -> None:
    merged = merge_provider_under(
        {"draft_model": "explicit-draft"}, {"draft_model": "block-draft"}, None
    )
    assert merged["draft_model"] == "explicit-draft"


def test_merge_provider_under_env_supplied_field_drops_the_block_value() -> None:
    merged = merge_provider_under(
        {},
        {"draft_model": "block-draft", "base_url": "https://block.example/v1"},
        None,
        env_supplied={"draft_model"},
    )
    assert "draft_model" not in merged
    assert merged["base_url"] == "https://block.example/v1"


def test_merge_provider_under_drops_inherited_repair_when_draft_changes() -> None:
    """A block whose repair only ever mirrored its draft must not pin it."""
    merged = merge_provider_under(
        {"draft_model": "new-draft"},
        {"draft_model": "block-draft", "repair_model": "block-draft"},
        None,
    )
    assert merged["draft_model"] == "new-draft"
    assert "repair_model" not in merged


def test_merge_provider_under_keeps_a_distinct_block_repair() -> None:
    merged = merge_provider_under(
        {"draft_model": "new-draft"},
        {"draft_model": "block-draft", "repair_model": "block-repair"},
        None,
    )
    assert merged["repair_model"] == "block-repair"


def test_merge_provider_under_resolves_credential_from_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")
    assert merge_provider_under({}, {}, "CUSTOM_API_KEY")["api_key"] == "sk-custom"


def test_merge_provider_under_defers_to_the_generic_llm_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``UBT_LLM_API_KEY`` is set, the block's variable is not injected.

    The generic key outranks the block, so the merge leaves ``api_key`` to the
    environment source instead of copying the provider-specific value into the
    init kwargs.
    """
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-generic")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")
    assert "api_key" not in merge_provider_under({}, {}, "CUSTOM_API_KEY")


def test_generic_llm_key_outranks_the_provider_variable(
    config_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-generic")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")
    assert UBTConfig.from_env(provider="custom").api_key.get_secret_value() == "sk-generic"


# -- Integration with UBTConfig ----------------------------------------------


def test_from_env_applies_defaults_then_the_provider_block(config_toml: Path) -> None:
    cfg = UBTConfig.from_env(provider="gemini")
    assert cfg.api_timeout == 42.0  # from [defaults]
    assert cfg.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert cfg.draft_model == "gemini-user-draft"
    assert cfg.repair_model == "gemini-user-repair"


def test_from_env_provider_block_loses_to_the_environment(
    config_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_BASE_URL", "https://env.example/v1")
    cfg = UBTConfig.from_env(provider="gemini")
    assert cfg.base_url == "https://env.example/v1"
    # A field the operator did not pin still comes from the block.
    assert cfg.draft_model == "gemini-user-draft"


def test_apply_config_overrides_with_provider(
    config_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")
    base = UBTConfig.from_env()

    cfg = apply_config_overrides(base, {"provider": "custom"})
    assert cfg.base_url == "https://custom.example/v1"
    assert cfg.draft_model == "custom-draft"
    assert cfg.repair_model == "custom-repair"
    assert cfg.api_key.get_secret_value() == "sk-custom"

    # An explicit request flag outranks the block; a distinct block repair stays.
    explicit = apply_config_overrides(base, {"provider": "custom", "draft_model": "flag-draft"})
    assert explicit.draft_model == "flag-draft"
    assert explicit.repair_model == "custom-repair"


def test_apply_config_overrides_lets_env_outrank_the_provider(
    config_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``--provider`` must not clobber a field the operator pinned in the env."""
    monkeypatch.setenv("UBT_DRAFT_MODEL", "env-draft")
    monkeypatch.setenv("UBT_BASE_URL", "https://env.example/v1")
    base = UBTConfig.from_env()
    assert base.draft_model == "env-draft"

    cfg = apply_config_overrides(base, {"provider": "custom"})
    assert cfg.draft_model == "env-draft"
    assert cfg.base_url == "https://env.example/v1"

    # An explicit request flag still outranks the environment.
    explicit = apply_config_overrides(base, {"provider": "custom", "draft_model": "flag-draft"})
    assert explicit.draft_model == "flag-draft"


def test_from_env_and_overrides_agree_on_the_same_provider_request(
    config_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")
    env_cfg = UBTConfig.from_env(provider="custom")
    req_cfg = apply_config_overrides(UBTConfig.from_env(), {"provider": "custom"})
    assert (env_cfg.draft_model, env_cfg.repair_model, env_cfg.base_url) == (
        req_cfg.draft_model,
        req_cfg.repair_model,
        req_cfg.base_url,
    )


def test_explicit_api_key_override_survives_a_provider(
    config_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-custom")
    cfg = apply_config_overrides(
        UBTConfig.from_env(), {"provider": "custom", "api_key": SecretStr("sk-explicit")}
    )
    assert cfg.api_key.get_secret_value() == "sk-explicit"
