from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_CONFIG = """\
[defaults]
provider = "demo"

[providers.demo]
base_url = "http://127.0.0.1:9090/v1"
api_mode = "openai-chat"
draft_model = "model-a"
"""


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "ubt.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key in ("UBT_PROVIDER", "UBT_DRAFT_MODEL", "UBT_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_the_file_can_name_the_active_provider(config_dir: Path) -> None:
    from ubt.core.config import UBTConfig

    cfg = UBTConfig.from_env()
    assert cfg.provider == "demo"
    assert cfg.draft_model == "model-a"
    assert cfg.base_url == "http://127.0.0.1:9090/v1"


def test_precedence_is_unchanged_by_the_new_file_key(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.config import UBTConfig

    monkeypatch.setenv("UBT_PROVIDER", "demo")
    monkeypatch.setenv("UBT_DRAFT_MODEL", "from-env")
    assert UBTConfig.from_env().draft_model == "from-env"

    assert UBTConfig.from_env(provider="demo", draft_model="explicit").draft_model == "explicit"


def test_a_missing_provider_key_leaves_selection_unset(config_dir: Path) -> None:
    from ubt.core.config import UBTConfig

    (config_dir / "ubt.toml").write_text(
        '[providers.demo]\nbase_url = "http://x/v1"\ndraft_model = "m"\n', encoding="utf-8"
    )
    assert UBTConfig.from_env().provider is None


def test_the_source_reloads_when_the_file_changes(config_dir: Path) -> None:
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    source = ConfigSource(UBTConfig.from_env())
    assert source.config.draft_model == "model-a"

    path = config_dir / "ubt.toml"
    path.write_text(path.read_text(encoding="utf-8").replace("model-a", "model-b"))
    os.utime(path, ns=(1, 1))

    assert source.config.draft_model == "model-b"


def test_a_broken_file_keeps_the_last_good_configuration(config_dir: Path) -> None:
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    source = ConfigSource(UBTConfig.from_env())
    path = config_dir / "ubt.toml"
    path.write_text("[providers.demo\nbroken", encoding="utf-8")
    os.utime(path, ns=(2, 2))

    assert source.config.draft_model == "model-a"
    assert source.last_error is not None


def test_invalidate_forces_a_rebuild(config_dir: Path) -> None:
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    source = ConfigSource(UBTConfig.from_env())
    path = config_dir / "ubt.toml"
    path.write_text(path.read_text(encoding="utf-8").replace("model-a", "model-z"))
    stat = path.stat()
    source._signature = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    assert source.config.draft_model == "model-a"

    source.invalidate()
    assert source.config.draft_model == "model-z"


def test_deployment_fields_are_reported_as_needing_a_restart(config_dir: Path) -> None:
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    source = ConfigSource(UBTConfig.from_env())
    path = config_dir / "ubt.toml"
    path.write_text(
        path.read_text(encoding="utf-8") + '\ndb_dir = ".ubt/moved"\n', encoding="utf-8"
    )
    os.utime(path, ns=(3, 3))

    changed = source.frozen_fields_changed()
    assert "db_dir" in changed
    assert str(source.config.db_dir).endswith("ledgers")


def test_the_source_never_second_guesses_a_callers_config(config_dir: Path) -> None:
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    explicit = UBTConfig.from_env(draft_model="caller-model")
    source = ConfigSource(explicit)
    assert source.config.draft_model == "caller-model"


def test_the_api_reports_the_live_config_not_the_boot_snapshot(config_dir: Path) -> None:
    from fastapi.testclient import TestClient

    from ubt.api.app import create_app
    from ubt.core.config import UBTConfig

    client = TestClient(create_app(UBTConfig.from_env()))
    before = client.get("/system/runtime-config").json()["active"]["draft_model"]
    assert before == "model-a"

    response = client.put("/system/providers/demo", json={"draft_model": "model-live"})
    assert response.status_code == 200, response.text

    after = client.get("/system/runtime-config").json()
    assert after["active"]["draft_model"] == "model-live"
    row = next(p for p in after["providers"] if p["name"] == "demo")
    assert row["draft_model"] == after["active"]["draft_model"]


def test_a_file_selected_provider_is_read_back_onto_the_config(config_dir: Path) -> None:
    from ubt.core.config import UBTConfig

    cfg = UBTConfig.from_env()
    assert cfg.draft_model == "model-a"
    assert cfg.provider == "demo"
    assert "provider" in cfg.model_fields_set
