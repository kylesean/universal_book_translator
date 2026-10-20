"""One owner for "what configuration is in force".

Configuration had five lifetimes at once (boot snapshot, per-job override, env
merge, per-call file read, bootstrap side effect), so a console save wrote the
file successfully and the panel kept showing the pre-save value — in the same
response, ``active.draft_model`` held the old value while
``providers[name].draft_model`` held the new one.
"""

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
    """A throwaway CWD holding one ``ubt.toml``.

    ``ubt.core.providers`` resolves the file relative to the process CWD, so the
    test chdirs — the same way the real server finds the operator's file.
    """
    (tmp_path / "ubt.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key in ("UBT_PROVIDER", "UBT_DRAFT_MODEL", "UBT_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_the_file_can_name_the_active_provider(config_dir: Path) -> None:
    """Selection used to be CLI/env only, so the console could not make a block live.

    ``[defaults] provider`` was rejected as an unknown field, which meant the file
    could describe ten providers and still not say which one runs.
    """
    from ubt.core.config import UBTConfig

    cfg = UBTConfig.from_env()
    assert cfg.provider == "demo"
    assert cfg.draft_model == "model-a"
    assert cfg.base_url == "http://127.0.0.1:9090/v1"


def test_precedence_is_unchanged_by_the_new_file_key(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``overrides > env > [providers] > [defaults] > default`` still holds."""
    from ubt.core.config import UBTConfig

    # The environment outranks the file.
    monkeypatch.setenv("UBT_PROVIDER", "demo")
    monkeypatch.setenv("UBT_DRAFT_MODEL", "from-env")
    assert UBTConfig.from_env().draft_model == "from-env"

    # An explicit argument outranks the environment.
    assert UBTConfig.from_env(provider="demo", draft_model="explicit").draft_model == "explicit"


def test_a_missing_provider_key_leaves_selection_unset(config_dir: Path) -> None:
    """Absent means "no provider", not "the last one someone mentioned"."""
    from ubt.core.config import UBTConfig

    (config_dir / "ubt.toml").write_text(
        '[providers.demo]\nbase_url = "http://x/v1"\ndraft_model = "m"\n', encoding="utf-8"
    )
    assert UBTConfig.from_env().provider is None


def test_the_source_reloads_when_the_file_changes(config_dir: Path) -> None:
    """The boot snapshot is what made a save invisible until a restart."""
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    source = ConfigSource(UBTConfig.from_env())
    assert source.config.draft_model == "model-a"

    path = config_dir / "ubt.toml"
    path.write_text(path.read_text(encoding="utf-8").replace("model-a", "model-b"))
    # mtime granularity: force a distinct signature rather than sleeping.
    os.utime(path, ns=(1, 1))

    assert source.config.draft_model == "model-b"


def test_a_broken_file_keeps_the_last_good_configuration(config_dir: Path) -> None:
    """One bad save must not take the whole console down."""
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    source = ConfigSource(UBTConfig.from_env())
    path = config_dir / "ubt.toml"
    path.write_text("[providers.demo\nbroken", encoding="utf-8")
    os.utime(path, ns=(2, 2))

    # Still serving the last configuration that parsed …
    assert source.config.draft_model == "model-a"
    # … and able to say why it stopped reloading.
    assert source.last_error is not None


def test_invalidate_forces_a_rebuild(config_dir: Path) -> None:
    """A save must be visible to the very next request, not the next tick."""
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    source = ConfigSource(UBTConfig.from_env())
    path = config_dir / "ubt.toml"
    path.write_text(path.read_text(encoding="utf-8").replace("model-a", "model-z"))
    # Pin the mtime the source recorded, so the signature genuinely does not move
    # and only ``invalidate()`` can force the rebuild. Without this the test would
    # pass on the mtime check alone and prove nothing about invalidation.
    stat = path.stat()
    source._signature = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    assert source.config.draft_model == "model-a"  # signature unchanged

    source.invalidate()
    assert source.config.draft_model == "model-z"


def test_deployment_fields_are_reported_as_needing_a_restart(config_dir: Path) -> None:
    """A ledger dir changed in the file is not in force, and must say so."""
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
    # The running configuration still uses the boot value.
    assert str(source.config.db_dir).endswith("ledgers")


def test_the_source_never_second_guesses_a_callers_config(config_dir: Path) -> None:
    """``create_app(config=...)`` is an explicit statement, not a suggestion."""
    from ubt.api.config_source import ConfigSource
    from ubt.core.config import UBTConfig

    explicit = UBTConfig.from_env(draft_model="caller-model")
    source = ConfigSource(explicit)
    assert source.config.draft_model == "caller-model"


def test_the_api_reports_the_live_config_not_the_boot_snapshot(config_dir: Path) -> None:
    """The end-to-end guard for the staleness bug, through the real route.

    Reverting ``live_config()`` to the boot snapshot must fail here. Without this
    the unit tests above pass while the panel is still frozen, because they test
    the source directly and never the route that reads it.
    """
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
    # And the response must not contradict itself: the row and the header read the
    # same source now.
    row = next(p for p in after["providers"] if p["name"] == "demo")
    assert row["draft_model"] == after["active"]["draft_model"]


def test_a_file_selected_provider_is_read_back_onto_the_config(config_dir: Path) -> None:
    """``merge_provider_under`` only carries keys the caller passed.

    So a provider name that came from the file was dropped from the resulting
    config, which then claimed no provider while running one — the same
    "configuration that disagrees with itself" the whole module exists to remove.
    """
    from ubt.core.config import UBTConfig

    cfg = UBTConfig.from_env()
    # The block's values were applied …
    assert cfg.draft_model == "model-a"
    # … and the selection that produced them is reported, not lost.
    assert cfg.provider == "demo"
    assert "provider" in cfg.model_fields_set
