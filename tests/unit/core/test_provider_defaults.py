"""The [defaults] TOML block accepts run-level UBTConfig fields like db_dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.providers import PROVIDER_ALLOWED_KEYS, ProviderConfigError, _validate_block

pytestmark = pytest.mark.fast


def test_db_dir_is_allowed_in_a_toml_block() -> None:
    # db_dir is a real UBTConfig field (--db-dir / UBT_DB_DIR); it must be
    # settable from the config file too, or the same key is settable on every
    # surface except TOML.
    assert "db_dir" in PROVIDER_ALLOWED_KEYS
    assert _validate_block("[defaults]", {"db_dir": "/tmp/ledgers"}) == {"db_dir": "/tmp/ledgers"}


def test_the_shared_glossary_is_a_defaults_only_setting() -> None:
    """A provider block cannot swap the shared terminology for its endpoint.

    ``[defaults]`` owns deployment-wide assets. Allowing a provider block to
    point ``glossary_path`` at a different file would make enforced terminology
    change with the selected model provider.
    """
    assert _validate_block("[defaults]", {"glossary_path": "/tmp/terms.csv"}) == {
        "glossary_path": Path("/tmp/terms.csv")
    }
    with pytest.raises(ProviderConfigError, match="global field"):
        _validate_block("[providers.demo]", {"glossary_path": "/tmp/terms.csv"})


def test_an_unknown_key_is_still_rejected() -> None:
    with pytest.raises(ProviderConfigError):
        _validate_block("[defaults]", {"not_a_real_field": "x"})


def test_a_config_file_can_describe_the_whole_quality_policy() -> None:
    """Every knob a quality policy needs must be settable from a file.

    ``consistency_enforce`` / ``consistency_max_repairs`` were readable by the
    stage but offered on no documented surface: a user could not turn the
    feature on from a config file at all, only by exporting an environment
    variable the docs do not mention.
    """
    policy = {
        "qe_threshold": 0.85,
        "max_repair_rounds": 3,
        "consistency_enforce": "report",
        "consistency_max_repairs": 25,
        "qe_judge_sentences": True,
        "qe_judge_pass_sample": 0.15,
        "qe_judge_provider": "my-judge",
    }
    assert _validate_block("[defaults]", policy) == policy


def test_the_removed_judge_knobs_are_now_rejected() -> None:
    """A dead knob must fail loudly, not be silently accepted.

    ``qe_judge_gray_low``/``_high`` were inert (the heuristic emits twelve
    discrete values and the only one in the default band was already judged), and
    ``qe_judge_allow_upgrade`` let an opinion lift a real failure over the pass
    line. All three are gone; a config still carrying one should say so rather
    than quietly doing nothing.
    """
    for key in (
        "qe_judge_gray_low",
        "qe_judge_gray_high",
        "qe_judge_allow_upgrade",
    ):
        assert key not in PROVIDER_ALLOWED_KEYS
        with pytest.raises(ProviderConfigError):
            _validate_block("[defaults]", {key: 0.5})
