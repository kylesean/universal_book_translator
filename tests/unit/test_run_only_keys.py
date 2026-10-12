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
    ["provider", "repair_provider", "base_url", "api_mode", "api_key", "ocr_endpoint"],
)
def test_provider_keys_are_rejected_on_untrusted_surfaces(key: str) -> None:
    """REST/queue/MCP payloads may not select a provider or supply a credential."""
    with pytest.raises(UBTError, match="not accepted in job payloads"):
        overrides_from_request({key: "attacker.example"}, allow_provider_keys=False)


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
