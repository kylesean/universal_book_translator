from __future__ import annotations

import pytest

from ubt.core.config import UBTConfig
from ubt.core.exceptions import UBTError
from ubt.core.job_options import RUN_ONLY_KEYS, overrides_from_request

pytestmark = pytest.mark.fast


def test_run_only_keys_are_not_config_fields() -> None:
    config_fields = set(UBTConfig.model_fields)
    collisions = sorted(set(RUN_ONLY_KEYS) & config_fields)
    assert not collisions, (
        "run-only keys must never also be UBTConfig fields: "
        f"{collisions} — rename the field or drop it from RUN_ONLY_KEYS"
    )


def test_run_only_keys_are_dropped_from_overrides() -> None:
    overrides = overrides_from_request(
        {"input_path": "/tmp/in.pdf", "target_lang": "zh", "max_chapters": 3}
    )
    for key in RUN_ONLY_KEYS:
        assert key not in overrides


@pytest.mark.parametrize(
    "key",
    ["base_url", "api_mode", "api_key", "ocr_endpoint"],
)
def test_endpoint_and_credential_keys_are_rejected_on_untrusted_surfaces(key: str) -> None:
    with pytest.raises(UBTError, match="not accepted in job payloads"):
        overrides_from_request({key: "attacker.example"}, allow_provider_keys=False)


def test_a_payload_may_name_a_declared_provider() -> None:
    from ubt.core.providers import list_providers

    declared = list_providers()
    assert declared, "the test needs at least one usable provider name"
    chosen = declared[0]
    overrides = overrides_from_request({"provider": chosen}, allow_provider_keys=False)
    assert overrides["provider"] == chosen


def test_a_payload_cannot_invent_a_provider_name() -> None:
    with pytest.raises(UBTError, match="Unknown provider"):
        overrides_from_request({"provider": "attacker.example"}, allow_provider_keys=False)

    for bad in ("", "   "):
        with pytest.raises(UBTError, match="non-empty provider name"):
            overrides_from_request({"provider": bad}, allow_provider_keys=False)


def test_provider_keys_are_allowed_on_the_operator_surface() -> None:
    overrides = overrides_from_request(
        {"base_url": "http://localhost:9090"}, allow_provider_keys=True
    )
    assert overrides.get("base_url") == "http://localhost:9090"


def test_an_alias_maps_to_the_config_field() -> None:
    overrides = overrides_from_request({"glossary": "/tmp/g.csv"})
    assert overrides.get("glossary_path") == "/tmp/g.csv"


def test_unknown_keys_are_dropped() -> None:
    overrides = overrides_from_request({"definitely_not_a_field": 1})
    assert "definitely_not_a_field" not in overrides


def test_selecting_a_provider_changes_the_endpoint_the_job_uses() -> None:
    import os
    import tempfile
    from pathlib import Path

    from ubt.core.config import UBTConfig
    from ubt.core.job_options import apply_config_overrides

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "ubt.toml").write_text(
            '[defaults]\nprovider = "alpha"\n\n'
            '[providers.alpha]\nbase_url = "http://127.0.0.1:1111/v1"\n'
            'api_mode = "openai-chat"\ndraft_model = "alpha-model"\n\n'
            '[providers.beta]\nbase_url = "http://127.0.0.1:2222/v1"\n'
            'api_mode = "openai-chat"\ndraft_model = "beta-model"\n',
            encoding="utf-8",
        )
        previous = Path.cwd()
        os.chdir(tmp)
        try:
            os.environ.pop("UBT_PROVIDER", None)
            base = UBTConfig.from_env()
            assert base.draft_model == "alpha-model"

            job = apply_config_overrides(
                base, overrides_from_request({"provider": "beta"}, allow_provider_keys=False)
            )
            assert job.base_url == "http://127.0.0.1:2222/v1"
            assert job.draft_model == "beta-model"
        finally:
            os.chdir(previous)
