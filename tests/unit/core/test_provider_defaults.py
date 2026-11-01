from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.providers import PROVIDER_ALLOWED_KEYS, ProviderConfigError, _validate_block

pytestmark = pytest.mark.fast


def test_db_dir_is_allowed_in_a_toml_block() -> None:
    assert "db_dir" in PROVIDER_ALLOWED_KEYS
    assert _validate_block("[defaults]", {"db_dir": "/tmp/ledgers"}) == {"db_dir": "/tmp/ledgers"}


def test_the_shared_glossary_is_a_defaults_only_setting() -> None:
    assert _validate_block("[defaults]", {"glossary_path": "/tmp/terms.csv"}) == {
        "glossary_path": Path("/tmp/terms.csv")
    }
    with pytest.raises(ProviderConfigError, match="global field"):
        _validate_block("[providers.demo]", {"glossary_path": "/tmp/terms.csv"})


def test_an_unknown_key_is_still_rejected() -> None:
    with pytest.raises(ProviderConfigError):
        _validate_block("[defaults]", {"not_a_real_field": "x"})


def test_a_config_file_can_describe_the_whole_quality_policy() -> None:
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
    for key in (
        "qe_judge_gray_low",
        "qe_judge_gray_high",
        "qe_judge_allow_upgrade",
    ):
        assert key not in PROVIDER_ALLOWED_KEYS
        with pytest.raises(ProviderConfigError):
            _validate_block("[defaults]", {key: 0.5})
