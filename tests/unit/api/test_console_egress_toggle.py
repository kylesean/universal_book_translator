"""The console must be able to flip the page-egress gate — and must not lie about its reach.

Every egress-dependent control (render QC, cloud/VLM OCR) used to name a fix
the reader could not perform: ``UBT_ALLOW_PAGE_UPLOAD=true`` lives in the
server's environment, not on any screen. The toggle closes that loop, and these
guards pin its two honest properties:

* the PUT is a *runtime* override — in force immediately, gone for a fresh
  ConfigSource (the restart story the card discloses),
* the GET reports the live value, not the boot snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.config_source import ConfigSource
from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast

_API_KEY = "egress-test-key"
_AUTH = {"X-API-Key": _API_KEY}


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )


def test_put_flips_the_gate_for_the_running_process(tmp_path: Path) -> None:
    client = TestClient(create_app(config=_config(tmp_path)))

    before = client.get("/system/egress-config", headers=_AUTH).json()
    assert before == {"allow_page_upload": False}

    put = client.put("/system/egress-config", json={"allow_page_upload": True}, headers=_AUTH)
    assert put.status_code == 200
    assert put.json() == {"allow_page_upload": True}

    # The GET is the live value, not the boot snapshot.
    assert client.get("/system/egress-config", headers=_AUTH).json() == {"allow_page_upload": True}


def test_the_override_dies_with_the_process(tmp_path: Path) -> None:
    """A fresh ConfigSource — the restart story — must fall back to the default.

    The card discloses "reverts at restart"; this is that sentence as a test. A
    toggle that silently outlived the process would be a config file the
    operator never wrote.
    """
    source = ConfigSource(_config(tmp_path))
    assert source.config.allow_page_upload is False
    updated = source.set_overrides({"allow_page_upload": True})
    assert updated.allow_page_upload is True
    # A fresh source (the restarted process) reads only the environment.
    assert ConfigSource(_config(tmp_path)).config.allow_page_upload is False


def test_put_survives_an_unrelated_file_reload(tmp_path: Path) -> None:
    """The override merges into the override set, so a file change keeps it.

    An operator editing ``ubt.toml`` mid-run must not silently re-close the
    gate the console just opened: the rebuild carries the runtime choice.
    """
    source = ConfigSource(_config(tmp_path))
    source.set_overrides({"allow_page_upload": True})
    # Simulate the file-triggered rebuild path.
    source.invalidate()
    assert source.config.allow_page_upload is True


def test_put_rejects_a_non_boolean_body(tmp_path: Path) -> None:
    client = TestClient(create_app(config=_config(tmp_path)))
    res = client.put("/system/egress-config", json={"allow_page_upload": "yes"}, headers=_AUTH)
    assert res.status_code == 422
