"""The startup warning for a sandbox wide enough to defeat the deny lists.

F8 regression: a sandbox base equal to (or containing) the user's home, or an
``UBT_ALLOWED_DIRS`` entry spanning a system prefix, looked configured while
putting credential-bearing dotfiles / host system trees within reach of the
submit and MCP read surfaces. The warning names the exposure and the knob at
boot, from the same gate that already warns about unauthenticated boots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast

_API_KEY = "test-key-warnings"
_WARN_LOGGER = "ubt.api.security"


def _config(tmp_path: Path, **overrides: object) -> UBTConfig:
    from ubt.core.config import UBTConfig

    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
        **overrides,  # type: ignore[arg-type]
    )


def test_a_home_wide_sandbox_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from ubt.api.security import warn_broad_sandbox

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        warn_broad_sandbox(_config(tmp_path))
    assert any(
        "home directory" in rec.message and "UBT_ALLOWED_DIRS" in rec.message
        for rec in caplog.records
    )


def test_a_root_sandbox_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from ubt.api.security import warn_broad_sandbox

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    from ubt.core.config import UBTConfig

    config = UBTConfig(
        db_dir=tmp_path / "db", allowed_dirs="/", service_api_key=SecretStr(_API_KEY)
    )
    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        warn_broad_sandbox(config)
    assert any("filesystem root" in rec.message for rec in caplog.records)


def test_a_system_prefix_allowlist_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from ubt.api.security import warn_broad_sandbox

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    from ubt.core.config import UBTConfig

    config = UBTConfig(
        db_dir=tmp_path / "db", allowed_dirs="/var", service_api_key=SecretStr(_API_KEY)
    )
    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        warn_broad_sandbox(config)
    assert any(
        "spans system directory" in rec.message and "UBT_ALLOWED_DIRS" in rec.message
        for rec in caplog.records
    )


def test_a_narrow_sandbox_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from ubt.api.security import warn_broad_sandbox

    # A home outside the sandbox base entirely: a book directory + ledger dir
    # sandbox that contains nothing the operator lives in.
    monkeypatch.setenv("HOME", "/nonexistent-home-for-ubt-test")
    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        warn_broad_sandbox(_config(tmp_path))
    assert not [rec for rec in caplog.records if rec.name == _WARN_LOGGER]


def test_the_startup_gate_issues_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The dedicated function is called from the boot lifespan, not just by tests."""
    from ubt.api.app import create_app

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    with (
        caplog.at_level(logging.WARNING, logger=_WARN_LOGGER),
        TestClient(create_app(config=_config(tmp_path))),
    ):
        pass
    assert any("home directory" in rec.message for rec in caplog.records)
