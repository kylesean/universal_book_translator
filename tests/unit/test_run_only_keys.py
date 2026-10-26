"""Job-request → config mapping, and the run-only key contract.

``overrides_from_request`` is the single funnel every non-CLI entry point goes
through (REST, job queue, MCP). It decides which keys in a request payload may
reach ``UBTConfig``. Two invariants there are load-bearing and were previously
only stated in prose:

1. **Run-only keys never become config.** ``input_path``, ``target_lang`` and
   friends are consumed by the orchestrator's ``run`` call, not by
   ``UBTConfig``. The filter works today because those names happen not to be
   config fields — a coincidence, not an enforced rule. Adding a config field
   named ``profile`` (plausible: there is already a ``preset``) would silently
   let a job payload set it.
2. **Provider/credential keys are rejected on untrusted surfaces.** A job
   payload must not be able to point the pipeline at an attacker's endpoint or
   swap the API key.

Both are security-adjacent, so they are pinned rather than trusted.
"""

from __future__ import annotations

import pytest

from ubt.core.config import UBTConfig
from ubt.core.exceptions import UBTError
from ubt.core.job_options import RUN_ONLY_KEYS, overrides_from_request

pytestmark = pytest.mark.fast


def test_run_only_keys_are_not_config_fields() -> None:
    """The invariant ``RUN_ONLY_KEYS`` exists to document.

    ``overrides_from_request`` drops a key by checking ``UBTConfig.model_fields``.
    That only keeps run-only keys out while none of them is also a config field;
    this asserts the two sets stay disjoint, so adding such a field fails here
    rather than silently letting a payload set it.
    """
    config_fields = set(UBTConfig.model_fields)
    collisions = sorted(set(RUN_ONLY_KEYS) & config_fields)
    assert not collisions, (
        "run-only keys must never also be UBTConfig fields: "
        f"{collisions} — rename the field or drop it from RUN_ONLY_KEYS"
    )


def test_run_only_keys_are_dropped_from_overrides() -> None:
    """A payload carrying them yields no config override for them."""
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
    """A payload may not choose *where* a request goes or *what secret* it carries.

    ``provider`` and ``repair_provider`` are absent from this list on purpose:
    both are names that must already exist in the operator's file, so they select
    among endpoints the operator declared rather than introducing one. See
    :func:`test_a_payload_may_name_a_declared_provider`.
    """
    with pytest.raises(UBTError, match="not accepted in job payloads"):
        overrides_from_request({key: "attacker.example"}, allow_provider_keys=False)


def test_a_payload_may_name_a_declared_provider() -> None:
    """Selecting among declared blocks is not credential injection.

    The endpoint, the protocol and the credential all still come from the block the
    name points at. Refusing the name meant the console could edit and activate
    provider blocks while no job could ever pick a different one — a capability the
    UI presented and the API silently ignored.
    """
    from ubt.core.providers import list_providers

    declared = list_providers()
    assert declared, "the test needs at least one usable provider name"
    chosen = declared[0]
    overrides = overrides_from_request({"provider": chosen}, allow_provider_keys=False)
    assert overrides["provider"] == chosen


def test_a_payload_cannot_invent_a_provider_name() -> None:
    """The name is a lookup key, so it is validated against the declared set."""
    with pytest.raises(UBTError, match="Unknown provider"):
        overrides_from_request({"provider": "attacker.example"}, allow_provider_keys=False)

    for bad in ("", "   "):
        with pytest.raises(UBTError, match="non-empty provider name"):
            overrides_from_request({"provider": bad}, allow_provider_keys=False)


def test_provider_keys_are_allowed_on_the_operator_surface() -> None:
    """The operator's own CLI/env path may set them (allow_provider_keys=True)."""
    overrides = overrides_from_request(
        {"base_url": "http://localhost:9090"}, allow_provider_keys=True
    )
    assert overrides.get("base_url") == "http://localhost:9090"


def test_an_alias_maps_to_the_config_field() -> None:
    """``glossary`` is the request spelling of the ``glossary_path`` field."""
    overrides = overrides_from_request({"glossary": "/tmp/g.csv"})
    assert overrides.get("glossary_path") == "/tmp/g.csv"


def test_unknown_keys_are_dropped() -> None:
    """A payload cannot inject an arbitrary attribute."""
    overrides = overrides_from_request({"definitely_not_a_field": 1})
    assert "definitely_not_a_field" not in overrides


def test_selecting_a_provider_changes_the_endpoint_the_job_uses() -> None:
    """The end-to-end claim the UI now makes.

    A name is only worth offering if it actually re-routes the job. This pins the
    whole path — payload → overrides → config → endpoint — so a future "validate
    but ignore" regression fails here rather than in a user's failed job.
    """
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
