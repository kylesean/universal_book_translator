from __future__ import annotations

from typing import get_args

import pytest

from ubt.core.config import DeliveryMode, SpillPolicy, UBTConfig
from ubt.core.providers import PROVIDER_ALLOWED_KEYS

pytestmark = pytest.mark.fast


def test_every_provider_allowed_key_is_a_config_field() -> None:
    fields = set(UBTConfig.model_fields)
    unknown = sorted(PROVIDER_ALLOWED_KEYS - fields)
    assert not unknown, f"provider allow-list names non-fields: {unknown}"


def test_the_provider_allow_list_covers_the_engine_knobs_a_provider_sets() -> None:
    for required in (
        "api_key",
        "base_url",
        "api_mode",
        "draft_model",
        "repair_model",
        "draft_reasoning_effort",
        "repair_reasoning_effort",
        "fallback_models",
        "api_timeout",
        "prompt_caching_enabled",
    ):
        assert required in PROVIDER_ALLOWED_KEYS, f"{required} missing from the allow-list"
    for name in PROVIDER_ALLOWED_KEYS:
        assert name in UBTConfig.model_fields


def test_the_provider_allow_list_cannot_reach_secrets_or_unrelated_knobs() -> None:
    assert "service_api_key" not in PROVIDER_ALLOWED_KEYS


def test_delivery_mode_has_target_and_legacy_vocabulary() -> None:
    assert set(get_args(DeliveryMode)) == {
        "auto",
        "readable",
        "faithful",
        "reflow",
        "facsimile",
        "page_locked",
    }
    config = UBTConfig()
    assert config.delivery_mode == "auto"
    assert config.allow_degrade is False
    assert UBTConfig(delivery_mode="auto").delivery_mode == "auto"


def test_spill_policy_is_a_closed_vocabulary() -> None:
    assert set(SpillPolicy.__args__) == {  # type: ignore[attr-defined]
        "error",
        "warn",
        "appendix",
        "source_kept_warn",
    }
    config = UBTConfig()
    assert config.spill_policy == "error"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UBTConfig(spill_policy="nonsense")


def test_the_export_stage_accepts_exactly_the_declared_values() -> None:
    allowed = set(SpillPolicy.__args__)  # type: ignore[attr-defined]
    spill_warning_accepts = {"warn", "appendix", "source_kept_warn"}
    appendix_accepts = {"appendix", "warn", "source_kept_warn"}
    assert spill_warning_accepts <= allowed
    assert appendix_accepts <= allowed
    assert allowed - spill_warning_accepts == {"error"}
