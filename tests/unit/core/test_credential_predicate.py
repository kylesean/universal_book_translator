"""One predicate for "is a real credential configured?" across the entry points.

``UBTConfig`` fills an unconfigured ``api_key`` with the ``MOCK_API_KEY``
sentinel (``"mock-key"``), so "no credential" is *not* an empty string. The API
and MCP surfaces compare against the sentinel; the queue worker used only a
truthiness test and so treated a keyless, air-gapped worker as authenticated —
building a live provider against ``mock-key`` and posting it to
``api.openai.com``. The same "present" confusion appeared in the environment
precedence: ``env_ignore_empty=True`` makes pydantic ignore ``UBT_BASE_URL=""``,
but the supplied-names scan counted it, so a provider block's ``base_url`` was
dropped while its ``api_key`` survived and went to the public endpoint.

The suite's autouse ``clean_ubt_env`` fixture removes every ``UBT_*`` variable,
so ``UBTConfig()`` here observes the true defaults.
"""

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
    """A keyless worker (the documented air-gapped default) must not bill anyone."""
    cfg = UBTConfig()
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY  # the sentinel, not empty
    assert _should_rehearse({}, cfg) is True


def test_worker_with_a_real_key_does_not_rehearse() -> None:
    cfg = UBTConfig(api_key=SecretStr("sk-real-test-key"))
    assert _should_rehearse({}, cfg) is False
    # An explicit user dry_run is still honored.
    assert _should_rehearse({"dry_run": True}, cfg) is True


def test_empty_env_var_is_not_counted_as_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    """``UBT_BASE_URL=""`` is "unset" to pydantic (env_ignore_empty) and must be here too."""
    monkeypatch.setenv("UBT_BASE_URL", "")
    assert "base_url" not in _env_supplied_field_names()
    monkeypatch.setenv("UBT_BASE_URL", "https://gateway.example.com/v1")
    assert "base_url" in _env_supplied_field_names()


def test_a_loopback_endpoint_needs_no_credential() -> None:
    """The reference configuration: a local model, no key, and it runs.

    ``local-gemma4`` points at ``127.0.0.1:9090`` with no ``api_key``. The old
    rule ("no credential ⇒ rehearsal") downgraded it to a zero-token mock while
    the model sat there answering ``/v1/models``, so the operator saw a rehearsal
    they never asked for.
    """
    cfg = UBTConfig(base_url="http://127.0.0.1:9090/v1", draft_model="gemma4-e4b")
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY
    assert has_real_credential(cfg) is False
    # ...but it does not need one, which is the distinction that was missing.
    assert credential_required(cfg) is False
    assert _should_rehearse({}, cfg) is False


def test_a_remote_endpoint_without_a_credential_still_rehearses() -> None:
    """The air-gapped default must not post the sentinel to api.openai.com."""
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
    """A local gateway may want a key; supplying one must not flip it to mock."""
    cfg = UBTConfig(
        base_url="http://127.0.0.1:9090/v1",
        draft_model="m",
        api_key=SecretStr("local-only"),
    )
    assert has_real_credential(cfg) is True
    assert credential_required(cfg) is False
    assert _should_rehearse({}, cfg) is False


def test_an_explicit_dry_run_is_honoured_even_on_a_local_endpoint() -> None:
    """``--dry-run`` is a request, not an inference: it must win either way."""
    cfg = UBTConfig(base_url="http://127.0.0.1:9090/v1", draft_model="m")
    assert _should_rehearse({"dry_run": True}, cfg) is True
    # ...but an *auto*-set rehearsal is overridden, because this worker can run.
    assert _should_rehearse({"dry_run": True, "rehearsal_auto": True}, cfg) is False


def test_the_canonical_runtime_config_agrees_with_the_worker() -> None:
    """``can_translate`` is what the wizard reads; it must match the real decision.

    They were computed independently — the panel tested the credential, the submit
    endpoint tested the credential, and both said "rehearsal" for a working local
    model. One predicate now answers both.
    """
    from ubt.api.runtime_config import describe_runtime_config

    cfg = UBTConfig(base_url="http://127.0.0.1:9090/v1", draft_model="gemma4-e4b")
    payload = describe_runtime_config(cfg)
    assert payload["active"]["can_translate"] is True
    # And it agrees with the worker's own verdict.
    assert payload["active"]["can_translate"] is not _should_rehearse({}, cfg)


def test_a_keyless_local_run_omits_the_authorization_header() -> None:
    """A local server gets no bearer at all, not ``Bearer mock-key``.

    The operator's own config records the failure this prevents: an empty value
    produces the malformed header ``"Bearer "`` (trailing space) and the request
    dies before reaching the gateway.
    """
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

    # A real key is still sent, so the fix cannot silently break authentication.
    transport = OpenAIChatTransport(
        api_key="sk-real", base_url="https://api.openai.com/v1", default_model="m"
    )
    assert transport._auth_headers()["Authorization"] == "Bearer sk-real"
