"""The console must be able to turn quality measurement on — and must say when it is off.

Every finished job said only "QE not measured (heuristic engine)" — jargon the
operator could not act on — and stopped there. The remedy lived in ``ubt.toml``
with no control anywhere in the console, so the sentence named a fix the reader
could not perform on the screen they were on. The card exists to close that
loop, and these guards cover the three ways it could lie:

* report "measured" for a configuration that scores nothing,
* write only one of the two keys, leaving the documented-but-useless
  ``qe_engine = "tiered"`` + judge-off state,
* claim a save took effect while ``UBT_QE_*`` outranks the file.
"""

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
    """The default configuration scores nothing, and 0.0 is a placeholder.

    This is the exact state the reference job ran in. Reporting it as
    "measurable" would put a score-shaped zero in front of an operator who never
    asked for one.
    """
    settings = describe_qe_settings(UBTConfig(qe_engine="heuristic", qe_judge_enabled=False))
    assert settings["measurable"] is False
    assert settings["engine"] == "heuristic"
    assert settings["judge_enabled"] is False


def test_the_tiered_label_without_the_judge_is_unmeasured() -> None:
    """``qe_engine = "tiered"`` alone is the documented no-op.

    The pipeline wraps the runner on ``qe_judge_enabled`` and only logs a warning
    about the mismatch, so a surface that trusted the label would promise a
    measurement the run never takes. This is the state the card's single selector
    exists to make unreachable.
    """
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
    """The wrapper needs something to call.

    ``repair_model`` is the implicit fallback, so the only way to have no judge
    model at all is a configuration that names neither — which is a real state on
    a fresh install, and one where "judge on" would otherwise read as working.
    """
    settings = describe_qe_settings(
        UBTConfig(qe_engine="tiered", qe_judge_enabled=True, repair_model="")
    )
    assert settings["judge_model"] == ""
    assert settings["measurable"] is False


def test_the_judge_model_comes_from_its_own_provider_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge on another endpoint answers with that block's model.

    Checking only ``qe_judge_model or repair_model`` printed the translator's
    model while the judge endpoint had never heard of it — measured at runtime as
    twelve ``404 no such model`` answers in one run.
    """
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
    """A save cannot change what an exported variable decides.

    ``UBT_QE_ENGINE`` outranks ``ubt.toml``. A form that reported "saved" here
    would be describing a file the next job never reads.
    """
    monkeypatch.setenv("UBT_QE_ENGINE", "heuristic")
    assert "qe_engine" in env_overrides()
    monkeypatch.delenv("UBT_QE_ENGINE")
    assert "qe_engine" not in env_overrides()


def test_the_edit_form_seeds_from_the_raw_judge_model() -> None:
    """Seeding the form from the resolved id would pin an inherited model.

    ``judge_model`` is the id the run will call — usually the translator's
    ``repair_model``. An edit that only changed the threshold would then write
    that inherited name back as an explicit ``qe_judge_model``, silently
    detaching the judge from the block it was inheriting.
    """
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
    """One selector, two keys.

    Two independent controls would let an operator save ``tiered`` with the judge
    off — a state that reads "semantic scoring" and scores nothing. The card
    therefore submits both keys from the same ``tiered`` value.
    """
    source = _source(_CARD)
    assert "qe_engine: tiered ? 'tiered' : 'heuristic'" in source
    assert "qe_judge_enabled: tiered" in source


def test_the_write_keeps_the_other_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A QE save must not drop ``[defaults] provider``.

    The two settings share one table. Rewriting the table instead of updating
    keys in it would leave a working install pointing at nothing — the failure
    ``set_default_provider`` refuses to create.
    """
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
    """Writing a name the loader cannot resolve fails every judge call.

    The console's select only offers real blocks; this guards the direct API
    caller, and matches the rule ``set_default_provider`` already enforces for
    ``[defaults] provider``.
    """
    from ubt.api.provider_editor import ProviderEditError, update_qe_settings

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ubt.toml").write_text("[defaults]\n", encoding="utf-8")

    with pytest.raises(ProviderEditError, match="not a declared provider"):
        update_qe_settings({"qe_judge_provider": "no-such-block"})


def test_an_unknown_qe_key_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge credential belongs to the provider block that owns it.

    Accepting ``qe_judge_api_key`` here would give the console a second place to
    write a secret, bypassing the deployment-aware gate the provider editor
    applies.
    """
    from ubt.api.provider_editor import ProviderEditError, update_qe_settings

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ubt.toml").write_text("[defaults]\n", encoding="utf-8")

    with pytest.raises(ProviderEditError, match="cannot be set from the console"):
        update_qe_settings({"qe_judge_api_key": "sk-nope"})
