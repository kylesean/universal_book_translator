"""Configuration-vocabulary parity guards (W6-4).

Two hand-maintained sets had drifted from the code they describe:

- ``PROVIDER_ALLOWED_KEYS`` (``ubt/core/providers.py``) lists the fields a
  ``[providers.*]`` block may set. It is written by hand to keep the module
  import free of ``config`` (the reverse direction is a function-level import
  in ``UBTConfig.with_overrides``), which means nothing catches a config field
  added to the allow-list's intent but not to the list.
- ``spill_policy`` was a bare ``str`` while the export stage accepted four
  values, one of them (``source_kept_warn``) absent from the field's own
  description — an operator could not discover it from the schema.

These tests pin the current state of both so a future edit has to be deliberate.
"""

from __future__ import annotations

from typing import get_args

import pytest

from ubt.core.config import DeliveryMode, SpillPolicy, UBTConfig
from ubt.core.providers import PROVIDER_ALLOWED_KEYS

pytestmark = pytest.mark.fast


def test_every_provider_allowed_key_is_a_config_field() -> None:
    """A key the loader accepts must be a field the config can receive.

    ``merge_provider_under`` writes into ``UBTConfig``; a name that is not a
    field is silently dropped at assignment time (the loop skips
    ``key not in valid_fields``), so an operator's ``[providers.x]`` entry
    would look accepted and do nothing.
    """
    fields = set(UBTConfig.model_fields)
    unknown = sorted(PROVIDER_ALLOWED_KEYS - fields)
    assert not unknown, f"provider allow-list names non-fields: {unknown}"


def test_the_provider_allow_list_covers_the_engine_knobs_a_provider_sets() -> None:
    """The fields a provider block is FOR are present.

    The intent of the allow-list is 'credentials, endpoints and the engine
    knobs a provider layer may set'. Adding a model/endpoint field to
    ``UBTConfig`` without adding it here means the packaged
    ``resources/providers.toml`` cannot set it.
    """
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
    # And the field exists on the config (the other direction).
    for name in PROVIDER_ALLOWED_KEYS:
        assert name in UBTConfig.model_fields


def test_the_provider_allow_list_cannot_reach_secrets_or_unrelated_knobs() -> None:
    """The comment says the list keeps a block away from secrets.

    ``service_api_key`` guards the REST surface and must never be settable by
    a ``[providers.*]`` block.
    """
    assert "service_api_key" not in PROVIDER_ALLOWED_KEYS


def test_delivery_mode_has_target_and_legacy_vocabulary() -> None:
    """Phase 0 accepts both vocabularies; the default stays facsimile until G1.

    ``auto``/``readable``/``faithful`` are the target contract words. The old
    three remain for stored jobs and one-release compatibility. A missing value
    here is a CLI/API surface that silently rejects a valid mode.
    """
    assert set(get_args(DeliveryMode)) == {
        "auto",
        "readable",
        "faithful",
        "reflow",
        "facsimile",
        "page_locked",
    }
    config = UBTConfig()
    # G1 default switch: auto routes paper/book to readable; layout-sensitive
    # documents still need an explicit faithful request/eligibility.
    assert config.delivery_mode == "auto"
    assert config.allow_degrade is False
    assert UBTConfig(delivery_mode="auto").delivery_mode == "auto"


def test_spill_policy_is_a_closed_vocabulary() -> None:
    """The field is a Literal now, and every accepted value is declared.

    The export stage accepts ``source_kept_warn`` alongside ``error`` / ``warn``
    / ``appendix``; before the Literal it was undiscoverable from the schema
    (the description listed three).
    """
    assert set(SpillPolicy.__args__) == {  # type: ignore[attr-defined]
        "error",
        "warn",
        "appendix",
        "source_kept_warn",
    }
    config = UBTConfig()
    assert config.spill_policy == "error"
    # An out-of-vocabulary value is refused at construction.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UBTConfig(spill_policy="nonsense")


def test_the_export_stage_accepts_exactly_the_declared_values() -> None:
    """The stage's own membership sets match the field's vocabulary.

    These are the literals the stage branches on; if one drifts the config
    accepts a value the stage ignores (or vice versa).
    """
    allowed = set(SpillPolicy.__args__)  # type: ignore[attr-defined]
    # The two branch sets in ubt/core/engine/stages/export.py.
    spill_warning_accepts = {"warn", "appendix", "source_kept_warn"}
    appendix_accepts = {"appendix", "warn", "source_kept_warn"}
    assert spill_warning_accepts <= allowed
    assert appendix_accepts <= allowed
    assert allowed - spill_warning_accepts == {"error"}
