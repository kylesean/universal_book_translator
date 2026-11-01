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

    assert client.get("/system/egress-config", headers=_AUTH).json() == {"allow_page_upload": True}


def test_the_override_dies_with_the_process(tmp_path: Path) -> None:
    source = ConfigSource(_config(tmp_path))
    assert source.config.allow_page_upload is False
    updated = source.set_overrides({"allow_page_upload": True})
    assert updated.allow_page_upload is True
    assert ConfigSource(_config(tmp_path)).config.allow_page_upload is False


def test_put_survives_an_unrelated_file_reload(tmp_path: Path) -> None:
    source = ConfigSource(_config(tmp_path))
    source.set_overrides({"allow_page_upload": True})
    source.invalidate()
    assert source.config.allow_page_upload is True


def test_put_rejects_a_non_boolean_body(tmp_path: Path) -> None:
    client = TestClient(create_app(config=_config(tmp_path)))
    res = client.put("/system/egress-config", json={"allow_page_upload": "yes"}, headers=_AUTH)
    assert res.status_code == 422
