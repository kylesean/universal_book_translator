from __future__ import annotations

from pathlib import Path

import pytest

from ubt.api.qe_settings import describe_qe_settings, env_overrides
from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CARD = _REPO_ROOT / "web" / "src" / "views" / "doctor" / "QESettingsCard.tsx"


def _source(path: Path) -> str:
    if not path.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    return path.read_text(encoding="utf-8")


def test_the_heuristic_engine_is_reported_as_unmeasured() -> None:
    settings = describe_qe_settings(UBTConfig(qe_engine="heuristic", qe_judge_enabled=False))
    assert settings["measurable"] is False
    assert settings["engine"] == "heuristic"
    assert settings["judge_enabled"] is False


def test_the_tiered_label_without_the_judge_is_unmeasured() -> None:
    settings = describe_qe_settings(
        UBTConfig(qe_engine="tiered", qe_judge_enabled=False, repair_model="m")
    )
    assert settings["measurable"] is False


def test_a_judge_with_a_model_is_measured() -> None:
    settings = describe_qe_settings(
        UBTConfig(qe_engine="tiered", qe_judge_enabled=True, repair_model="judge-model")
    )
    assert settings["measurable"] is True
    assert settings["judge_model"] == "judge-model"


def test_a_judge_without_a_model_is_unmeasured() -> None:
    settings = describe_qe_settings(
        UBTConfig(qe_engine="tiered", qe_judge_enabled=True, repair_model="")
    )
    assert settings["judge_model"] == ""
    assert settings["measurable"] is False


def test_the_judge_model_comes_from_its_own_provider_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ubt.toml").write_text(
        '[providers.judge]\nbase_url = "https://judge.example/v1"\nrepair_model = "hosted-judge"\n',
        encoding="utf-8",
    )
    settings = describe_qe_settings(
        UBTConfig(
            qe_engine="tiered",
            qe_judge_enabled=True,
            qe_judge_provider="judge",
            repair_model="local-translator",
        )
    )
    assert settings["judge_provider"] == "judge"
    assert settings["judge_model"] == "hosted-judge"
    assert settings["measurable"] is True


def test_an_environment_variable_is_named_as_an_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UBT_QE_ENGINE", "heuristic")
    assert "qe_engine" in env_overrides()
    monkeypatch.delenv("UBT_QE_ENGINE")
    assert "qe_engine" not in env_overrides()


def test_the_edit_form_seeds_from_the_raw_judge_model() -> None:
    inherited = describe_qe_settings(
        UBTConfig(qe_engine="tiered", qe_judge_enabled=True, repair_model="inherited")
    )
    assert inherited["judge_model"] == "inherited"
    assert inherited["judge_model_explicit"] is None

    pinned = describe_qe_settings(
        UBTConfig(
            qe_engine="tiered",
            qe_judge_enabled=True,
            repair_model="inherited",
            qe_judge_model="pinned",
        )
    )
    assert pinned["judge_model"] == "pinned"
    assert pinned["judge_model_explicit"] == "pinned"


def test_the_console_writes_the_engine_and_the_judge_together() -> None:
    source = _source(_CARD)
    assert "qe_engine: tiered ? 'tiered' : 'heuristic'" in source
    assert "qe_judge_enabled: tiered" in source


def test_the_write_keeps_the_other_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import update_qe_settings

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "ubt.toml"
    config.write_text('[defaults]\nprovider = "acme"\n', encoding="utf-8")

    update_qe_settings({"qe_engine": "tiered", "qe_judge_enabled": True, "qe_threshold": 0.8})

    text = config.read_text(encoding="utf-8")
    assert 'provider = "acme"' in text
    assert 'qe_engine = "tiered"' in text
    assert "qe_judge_enabled = true" in text
    assert "qe_threshold = 0.8" in text


def test_a_judge_provider_that_names_nothing_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import ProviderEditError, update_qe_settings

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ubt.toml").write_text("[defaults]\n", encoding="utf-8")

    with pytest.raises(ProviderEditError, match="not a declared provider"):
        update_qe_settings({"qe_judge_provider": "no-such-block"})


def test_an_unknown_qe_key_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.api.provider_editor import ProviderEditError, update_qe_settings

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ubt.toml").write_text("[defaults]\n", encoding="utf-8")

    with pytest.raises(ProviderEditError, match="cannot be set from the console"):
        update_qe_settings({"qe_judge_api_key": "sk-nope"})
