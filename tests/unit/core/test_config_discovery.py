from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_BLOCK = """\
[defaults]
provider = "mine"

[providers.mine]
base_url = "http://127.0.0.1:1234/v1"
api_mode = "openai-chat"
draft_model = "my-model"
"""


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "ubt.toml").write_text(_BLOCK, encoding="utf-8")
    nested = tmp_path / "web" / "src"
    nested.mkdir(parents=True)
    monkeypatch.setattr(
        "ubt.core.providers.Path.home", classmethod(lambda cls: tmp_path / "nowhere")
    )
    monkeypatch.chdir(nested)
    return nested


def test_the_project_file_is_found_from_a_subdirectory(project: Path) -> None:
    from ubt.core.providers import find_config_file

    assert Path.cwd() != project.parent.parent
    found = find_config_file()
    assert found is not None
    assert found == project.parent.parent / "ubt.toml"


def test_a_provider_declared_upstream_resolves_from_a_subdirectory(project: Path) -> None:
    from ubt.core.config import UBTConfig

    cfg = UBTConfig.from_env()
    assert cfg.provider == "mine"
    assert cfg.draft_model == "my-model"
    assert cfg.base_url == "http://127.0.0.1:1234/v1"


def test_the_nearest_file_wins(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.core.providers import find_config_file

    nested_block = _BLOCK.replace("my-model", "nested-model")
    (project / "ubt.toml").write_text(nested_block, encoding="utf-8")

    found = find_config_file()
    assert found == project / "ubt.toml"


def test_the_failure_message_names_the_file_that_was_read(project: Path) -> None:
    from ubt.core.providers import ProviderNotFoundError, load_provider_block

    with pytest.raises(ProviderNotFoundError) as excinfo:
        load_provider_block("does-not-exist")
    message = str(excinfo.value)
    assert "the configuration file read was" in message
    assert "ubt.toml" in message
    assert "mine" in message


def test_the_failure_message_lists_the_search_path_when_no_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.providers import ProviderNotFoundError, load_provider_block

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(
        "ubt.core.providers.Path.home", classmethod(lambda cls: tmp_path / "nowhere")
    )

    with pytest.raises(ProviderNotFoundError) as excinfo:
        load_provider_block("whatever")
    message = str(excinfo.value)
    assert "no configuration file was found" in message
    assert "searched:" in message
