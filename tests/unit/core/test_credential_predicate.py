from __future__ import annotations

import pytest
from pydantic import SecretStr

from ubt.core.config import (
    MOCK_API_KEY,
    UBTConfig,
    _env_supplied_field_names,
    credential_required,
    has_real_credential,
)
from ubt.core.engine.job_worker import _should_rehearse

pytestmark = pytest.mark.fast


def test_keyless_worker_rehearses_instead_of_using_the_mock_sentinel() -> None:
    cfg = UBTConfig()
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY
    assert _should_rehearse({}, cfg) is True


def test_worker_with_a_real_key_does_not_rehearse() -> None:
    cfg = UBTConfig(api_key=SecretStr("sk-real-test-key"))
    assert _should_rehearse({}, cfg) is False
    assert _should_rehearse({"dry_run": True}, cfg) is True


def test_empty_env_var_is_not_counted_as_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_BASE_URL", "")
    assert "base_url" not in _env_supplied_field_names()
    monkeypatch.setenv("UBT_BASE_URL", "https://gateway.example.com/v1")
    assert "base_url" in _env_supplied_field_names()


def test_a_loopback_endpoint_needs_no_credential() -> None:
    cfg = UBTConfig(base_url="http://127.0.0.1:9090/v1", draft_model="gemma4-e4b")
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY
    assert has_real_credential(cfg) is False
    assert credential_required(cfg) is False
    assert _should_rehearse({}, cfg) is False


def test_a_remote_endpoint_without_a_credential_still_rehearses() -> None:
    cfg = UBTConfig(base_url="https://api.openai.com/v1", draft_model="gpt-x")
    assert credential_required(cfg) is True
    assert _should_rehearse({}, cfg) is True


def test_a_remote_endpoint_with_a_credential_runs() -> None:
    cfg = UBTConfig(
        base_url="https://api.openai.com/v1",
        draft_model="gpt-x",
        api_key=SecretStr("sk-real-test-key"),
    )
    assert credential_required(cfg) is True
    assert has_real_credential(cfg) is True
    assert _should_rehearse({}, cfg) is False


def test_a_local_endpoint_with_a_credential_still_runs() -> None:
    cfg = UBTConfig(
        base_url="http://127.0.0.1:9090/v1",
        draft_model="m",
        api_key=SecretStr("local-only"),
    )
    assert has_real_credential(cfg) is True
    assert credential_required(cfg) is False
    assert _should_rehearse({}, cfg) is False


def test_an_explicit_dry_run_is_honoured_even_on_a_local_endpoint() -> None:
    cfg = UBTConfig(base_url="http://127.0.0.1:9090/v1", draft_model="m")
    assert _should_rehearse({"dry_run": True}, cfg) is True
    assert _should_rehearse({"dry_run": True, "rehearsal_auto": True}, cfg) is False


def test_the_canonical_runtime_config_agrees_with_the_worker() -> None:
    from ubt.api.runtime_config import describe_runtime_config

    cfg = UBTConfig(base_url="http://127.0.0.1:9090/v1", draft_model="gemma4-e4b")
    payload = describe_runtime_config(cfg)
    assert payload["active"]["can_translate"] is True
    assert payload["active"]["can_translate"] is not _should_rehearse({}, cfg)


def test_a_keyless_local_run_omits_the_authorization_header() -> None:
    from ubt.core.router.transports.anthropic import AnthropicMessagesTransport
    from ubt.core.router.transports.gemini import GeminiTransport
    from ubt.core.router.transports.openai_chat import OpenAIChatTransport

    for key in ("", MOCK_API_KEY):
        for cls, header in (
            (OpenAIChatTransport, "Authorization"),
            (AnthropicMessagesTransport, "x-api-key"),
            (GeminiTransport, "x-goog-api-key"),
        ):
            transport = cls(api_key=key, base_url="http://127.0.0.1:9090/v1", default_model="m")
            headers = transport._auth_headers()
            assert header not in headers, (
                f"{cls.__name__} sent {header!r} for an unconfigured key {key!r}"
            )
            assert not any("mock-key" in str(v) for v in headers.values())

    transport = OpenAIChatTransport(
        api_key="sk-real", base_url="https://api.openai.com/v1", default_model="m"
    )
    assert transport._auth_headers()["Authorization"] == "Bearer sk-real"
