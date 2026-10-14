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

from ubt.core.config import MOCK_API_KEY, UBTConfig, _env_supplied_field_names
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
