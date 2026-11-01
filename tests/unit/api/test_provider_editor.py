from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast

_BASE_BLOCK = """\
[providers.demo]
base_url = "http://127.0.0.1:9090/v1"
api_mode = "openai-chat"
api_key = "${DEMO_KEY}"
draft_model = "demo-model"
# an operator comment that must survive an edit
is_free = true
"""


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "ubt.toml").write_text(_BASE_BLOCK, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_an_edit_writes_the_block_and_preserves_everything_else(config_dir: Path) -> None:
    from ubt.api.provider_editor import upsert_provider

    result = upsert_provider("demo", {"supports_vision": True, "draft_model": "demo-model-2"})
    assert result["supports_vision"] is True
    assert result["draft_model"] == "demo-model-2"

    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert 'api_key = "${DEMO_KEY}"' in text
    assert "# an operator comment that must survive an edit" in text
    assert "is_free = true" in text
    assert 'base_url = "http://127.0.0.1:9090/v1"' in text


def test_a_credential_is_refused_unless_the_caller_opts_in(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, upsert_provider, validate_edit

    with pytest.raises(ProviderEditError) as excinfo:
        validate_edit("demo", {"api_key": "sk-injected"})
    assert "credential cannot be written" in str(excinfo.value)

    with pytest.raises(ProviderEditError):
        upsert_provider("demo", {"api_key": "sk-injected"})
    assert "sk-injected" not in (config_dir / "ubt.toml").read_text(encoding="utf-8")


def test_a_credential_is_written_when_the_deployment_allows_it(config_dir: Path) -> None:
    from ubt.api.provider_editor import upsert_provider

    result = upsert_provider("demo", {"api_key": "sk-local"}, allow_credential=True)
    assert result["has_credential"] is True
    assert result["credential_hint"] == "********"
    assert "sk-local" not in str(result)
    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert 'api_key = "sk-local"' in text
    assert (config_dir / "ubt.toml").stat().st_mode & 0o777 == 0o600


def test_the_other_credentials_are_still_unwritable(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, validate_edit

    for field in ("ocr_api_key", "service_api_key"):
        with pytest.raises(ProviderEditError) as excinfo:
            validate_edit("demo", {field: "x"}, allow_credential=True)
        assert field in str(excinfo.value)


def test_credential_write_is_gated_on_bind_and_auth_posture() -> None:
    from ubt.api.provider_editor import credential_write_allowed

    assert credential_write_allowed(is_loopback=True, is_strict=False) is True
    assert credential_write_allowed(is_loopback=True, is_strict=True) is False
    assert credential_write_allowed(is_loopback=False, is_strict=False) is False
    assert credential_write_allowed(is_loopback=False, is_strict=True) is False


def test_the_rest_request_model_still_forbids_unknown_fields() -> None:
    from pydantic import ValidationError

    from ubt.api.models import ProviderEditRequest

    with pytest.raises(ValidationError) as excinfo:
        ProviderEditRequest(wibble="x")  # type: ignore[call-arg]
    assert "wibble" in str(excinfo.value)


def test_only_whitelisted_fields_are_writable(config_dir: Path) -> None:
    from ubt.api.provider_editor import WRITABLE_FIELDS, ProviderEditError, validate_edit

    for field in ("db_dir", "prices_file", "service_api_key", "wibble"):
        with pytest.raises(ProviderEditError):
            validate_edit("demo", {field: "x"})

    from ubt.core.providers import PROVIDER_ALLOWED_KEYS

    assert set(WRITABLE_FIELDS) <= PROVIDER_ALLOWED_KEYS
    assert "api_key" in WRITABLE_FIELDS
    assert "ocr_api_key" not in WRITABLE_FIELDS


def test_invalid_values_are_rejected_before_they_reach_the_file(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, validate_edit

    cases: list[dict[str, Any]] = [
        {"api_mode": "nonsense"},
        {"supports_temperature": "yes"},
        {"base_url": "file:///etc/passwd"},
        {"extraction_strategy": "guess"},
        {"reasoning_dialect": "sideways"},
        {"api_timeout": -1},
        {"fallback_models": "not-a-list"},
    ]
    for changes in cases:
        with pytest.raises(ProviderEditError):
            validate_edit("demo", changes)


def test_a_provider_name_cannot_escape_its_table(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, validate_edit

    for name in ("../evil", "a/b", "x]", "", "name with space", 'a"b'):
        with pytest.raises(ProviderEditError):
            validate_edit(name, {"is_free": True})


def test_the_edit_is_atomic_and_tightens_permissions(config_dir: Path) -> None:
    from ubt.api.provider_editor import upsert_provider

    (config_dir / "ubt.toml").chmod(0o644)
    upsert_provider("demo", {"is_free": False})

    mode = (config_dir / "ubt.toml").stat().st_mode & 0o777
    assert mode == 0o600, f"config left at {oct(mode)}; it may hold a literal credential"
    assert not [p for p in config_dir.iterdir() if p.name.startswith(".ubt.toml")]


def test_reading_a_provider_reports_presence_not_the_secret(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import read_provider

    monkeypatch.delenv("DEMO_KEY", raising=False)
    unresolved = read_provider("demo")
    assert unresolved["has_credential"] is False
    assert unresolved["credential_in_file"] is True
    assert unresolved["credential_issue"] == "unresolved_variable"
    assert unresolved["unresolved_vars"] == ["DEMO_KEY"]

    monkeypatch.setenv("DEMO_KEY", "sk-resolved")
    resolved = read_provider("demo")
    assert resolved["has_credential"] is True
    assert resolved["credential_issue"] is None
    assert resolved["unresolved_vars"] == []
    for payload in (unresolved, resolved):
        assert "sk-resolved" not in str(payload)
        assert "${DEMO_KEY}" not in str(payload)


def test_a_keyless_local_endpoint_is_not_reported_as_missing(config_dir: Path) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    upsert_provider(
        "local",
        {
            "base_url": "http://127.0.0.1:9090/v1",
            "api_mode": "openai-chat",
            "draft_model": "local-model",
        },
    )
    detail = read_provider("local")
    assert detail["has_credential"] is False
    assert detail["credential_issue"] is None


def test_a_remote_endpoint_with_no_credential_is_reported_as_missing(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    upsert_provider(
        "remote",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "hosted-model",
        },
    )
    detail = read_provider("remote")
    assert detail["has_credential"] is False
    assert detail["credential_issue"] == "missing"


def test_a_written_credential_reports_that_the_environment_outranks_it(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-from-shell")
    upsert_provider("demo", {"api_key": "sk-written-into-file"}, allow_credential=True)

    detail = read_provider("demo")
    assert detail["has_credential"] is True
    assert detail["credential_shadowed_by_env"] is True

    monkeypatch.delenv("UBT_LLM_API_KEY")
    assert read_provider("demo")["credential_shadowed_by_env"] is False


def test_a_keyless_block_on_a_keyed_deployment_authenticates_through_the_env(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-from-shell")
    upsert_provider(
        "keyless",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "hosted-model",
        },
    )
    detail = read_provider("keyless")
    assert detail["has_credential"] is False
    assert detail["authenticates"] is True
    assert detail["credential_issue"] is None
    assert detail["credential_shadowed_by_env"] is False


def test_an_unresolved_reference_is_reported_even_when_the_env_supplies_a_key(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import read_provider

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-from-shell")
    monkeypatch.delenv("DEMO_KEY", raising=False)
    detail = read_provider("demo")
    assert detail["authenticates"] is True
    assert detail["credential_issue"] == "unresolved_variable"
    assert detail["unresolved_vars"] == ["DEMO_KEY"]


def test_a_new_provider_can_be_created(config_dir: Path) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    result = upsert_provider(
        "fresh",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-responses",
            "draft_model": "some-model",
            "supports_reasoning_effort": True,
        },
    )
    assert result["declared"] is True
    assert result["supports_reasoning_effort"] is True
    assert read_provider("fresh")["has_credential"] is False


def test_reasoning_fields_round_trip_and_clear(config_dir: Path) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider
    from ubt.core.providers import load_provider_block

    upsert_provider(
        "demo",
        {
            "supports_reasoning_effort": True,
            "draft_reasoning_effort": "none",
            "repair_reasoning_effort": "high",
            "reasoning_dialect": "nested",
        },
    )
    block = load_provider_block("demo")
    assert block["draft_reasoning_effort"] == "none"
    assert block["repair_reasoning_effort"] == "high"
    assert block["reasoning_dialect"] == "nested"
    assert read_provider("demo")["draft_reasoning_effort"] == "none"

    upsert_provider("demo", {"draft_reasoning_effort": None, "reasoning_dialect": None})
    block = load_provider_block("demo")
    assert "draft_reasoning_effort" not in block
    assert "reasoning_dialect" not in block
    assert block["repair_reasoning_effort"] == "high"
    assert block["supports_reasoning_effort"] is True


def test_a_provider_block_can_be_deleted(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, delete_provider
    from ubt.core.providers import declared_provider_names

    (config_dir / "ubt.toml").write_text(
        _BASE_BLOCK + '\n[providers.second]\nbase_url = "http://127.0.0.1:9191/v1"\n',
        encoding="utf-8",
    )
    delete_provider("second")
    assert "second" not in declared_provider_names()
    assert 'api_key = "${DEMO_KEY}"' in (config_dir / "ubt.toml").read_text(encoding="utf-8")

    with pytest.raises(ProviderEditError, match="nothing to delete"):
        delete_provider("second")


def test_deleting_a_referenced_provider_is_refused(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, delete_provider

    (config_dir / "ubt.toml").write_text(
        _BASE_BLOCK + '\n[defaults]\nprovider = "demo"\n', encoding="utf-8"
    )
    with pytest.raises(ProviderEditError, match="defaults.provider"):
        delete_provider("demo")
    assert "demo" in (config_dir / "ubt.toml").read_text(encoding="utf-8")


def test_a_dotted_provider_name_is_one_quoted_key(config_dir: Path) -> None:
    from ubt.api.provider_editor import (
        default_provider_name,
        read_provider,
        upsert_provider,
    )
    from ubt.core.providers import declared_provider_names, load_provider_block

    name = "workbuddy-ai.deepseek-v4.1-flash"
    upsert_provider(
        name,
        {"base_url": "http://127.0.0.1:2345/v1", "draft_model": "deepseek-v4.1-flash"},
        make_default=True,
    )

    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert f'[providers."{name}"]' in text
    assert "[providers.workbuddy-ai." not in text
    assert name in declared_provider_names()
    assert load_provider_block(name)["draft_model"] == "deepseek-v4.1-flash"
    assert read_provider(name)["declared"] is True
    assert default_provider_name() == name


def test_the_saved_block_is_what_the_loader_reads_back(config_dir: Path) -> None:
    from ubt.api.provider_editor import upsert_provider
    from ubt.core.providers import load_provider_block

    upsert_provider(
        "demo",
        {
            "supports_temperature": False,
            "supports_vision": True,
            "deployment_backend": "llama-swap",
            "extraction_strategy": "xml_tag",
        },
    )
    block = load_provider_block("demo")
    assert block["supports_temperature"] is False
    assert block["supports_vision"] is True
    assert block["deployment_backend"] == "llama-swap"
    assert block["extraction_strategy"] == "xml_tag"


def test_an_edit_applies_to_the_next_job_without_a_restart(config_dir: Path) -> None:
    from ubt.api.provider_editor import upsert_provider
    from ubt.core.config import UBTConfig

    assert UBTConfig.from_env(provider="demo").draft_model == "demo-model"
    upsert_provider("demo", {"draft_model": "demo-model-3", "supports_reasoning_effort": True})
    reloaded = UBTConfig.from_env(provider="demo")
    assert reloaded.draft_model == "demo-model-3"
    assert reloaded.supports_reasoning_effort is True


def test_a_declared_flag_can_be_cleared_back_to_undeclared(config_dir: Path) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    upsert_provider("demo", {"supports_temperature": False})
    assert read_provider("demo")["supports_temperature"] is False

    upsert_provider("demo", {"supports_temperature": None})
    detail = read_provider("demo")
    assert detail.get("supports_temperature") is None
    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert "supports_temperature" not in text


def test_a_required_field_cannot_be_cleared(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, upsert_provider

    for field in ("base_url", "draft_model", "api_mode"):
        with pytest.raises(ProviderEditError):
            upsert_provider("demo", {field: None})


def test_a_new_block_can_be_made_the_default(config_dir: Path) -> None:
    from ubt.api.provider_editor import default_provider_name, upsert_provider

    assert default_provider_name() is None
    upsert_provider(
        "first",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "m1",
            "api_key": "sk-test",
        },
        allow_credential=True,
        make_default=True,
    )
    assert default_provider_name() == "first"
    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert "[defaults]" in text
    assert 'provider = "first"' in text


def test_the_default_survives_to_what_the_engine_resolves(config_dir: Path) -> None:
    from ubt.api.provider_editor import upsert_provider

    upsert_provider(
        "chosen",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "the-model",
            "api_key": "sk-test",
        },
        allow_credential=True,
        make_default=True,
    )
    from ubt.core.config import UBTConfig, credential_required, has_real_credential

    cfg = UBTConfig.from_env(provider=None)
    assert cfg.provider == "chosen"
    assert cfg.draft_model == "the-model"
    assert credential_required(cfg) and has_real_credential(cfg)


def test_an_explicit_environment_still_outranks_the_file_default(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import set_default_provider, upsert_provider
    from ubt.core.config import UBTConfig

    upsert_provider("other", {"base_url": "https://api.example.com/v1", "draft_model": "o"})
    set_default_provider("other")
    assert UBTConfig.from_env().provider == "other"

    monkeypatch.setenv("UBT_PROVIDER", "demo")
    assert UBTConfig.from_env().provider == "demo"


def test_switching_the_default_does_not_edit_the_block(config_dir: Path) -> None:
    from ubt.api.provider_editor import set_default_provider

    before = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    set_default_provider("demo")
    after = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert 'base_url = "http://127.0.0.1:9090/v1"' in after
    assert "# an operator comment that must survive an edit" in after
    assert after.count("[providers.demo]") == 1
    assert "[defaults]" in after and "[defaults]" not in before


def test_a_name_that_is_not_declared_cannot_become_the_default(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, set_default_provider

    with pytest.raises(ProviderEditError, match="not a declared provider"):
        set_default_provider("ghost")


def test_the_editor_never_writes_outside_the_discovered_config(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api import provider_editor

    target = config_dir / "ubt.toml"
    monkeypatch.setattr(provider_editor, "_config_path", lambda: target)
    assert provider_editor._config_path() == target
    assert provider_editor._config_path().parent == config_dir


def _write_block(tmp_path: Path, body: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "ubt.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return path


_CREDENTIAL_BLOCK = """\
[providers.demo]
base_url = "https://api.example/v1"
api_key = "sk-REAL-SECRET"
draft_model = "demo-model"
"""


def test_a_plaintext_remote_base_url_is_refused_while_a_credential_is_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import ProviderEditError, upsert_provider

    path = _write_block(tmp_path, _CREDENTIAL_BLOCK, monkeypatch)
    with pytest.raises(ProviderEditError, match="cleartext"):
        upsert_provider("demo", {"base_url": "http://attacker.invalid/v1"})
    assert "attacker.invalid" not in path.read_text(encoding="utf-8")
    assert "sk-REAL-SECRET" in path.read_text(encoding="utf-8")


def test_an_https_remote_base_url_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import upsert_provider

    path = _write_block(tmp_path, _CREDENTIAL_BLOCK, monkeypatch)
    upsert_provider("demo", {"base_url": "https://new.example/v1"})
    text = path.read_text(encoding="utf-8")
    assert 'base_url = "https://new.example/v1"' in text
    assert "sk-REAL-SECRET" in text


def test_a_loopback_plaintext_base_url_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import upsert_provider

    path = _write_block(tmp_path, _CREDENTIAL_BLOCK, monkeypatch)
    upsert_provider("demo", {"base_url": "http://127.0.0.1:8080/v1"})
    assert 'base_url = "http://127.0.0.1:8080/v1"' in path.read_text(encoding="utf-8")


def test_a_plaintext_url_without_a_stored_credential_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import upsert_provider

    _write_block(
        tmp_path,
        '[providers.demo]\nbase_url = "https://api.example/v1"\ndraft_model = "m"\n',
        monkeypatch,
    )
    upsert_provider("demo", {"base_url": "http://plain.example/v1"})
    assert 'base_url = "http://plain.example/v1"' in (tmp_path / "ubt.toml").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1/v1", False),
        ("http://localhost:8080/v1", False),
        ("http://[::1]:8080/v1", False),
        ("https://example.com/v1", False),
        ("http://example.com/v1", True),
        ("HTTP://EXAMPLE.COM/v1", True),
    ],
)
def test_the_plaintext_predicate(url: str, expected: bool) -> None:
    from ubt.api.provider_editor import _is_plaintext_nonlocal

    assert _is_plaintext_nonlocal(url) is expected
