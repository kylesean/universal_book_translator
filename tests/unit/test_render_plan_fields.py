"""Where the render decision lives: on the plan, never back on the manifest bus.

The bilingual/engine mode cluster was the last inter-stage bus on
``manifest.run``: the PDF renderer read and rewrote it across the adapter
boundary. It now lives in typed values -- ``RenderPlan`` (the advisories'
decision, threaded to the renderer) and ``RenderOutcome`` (what the renderer
actually used). The old manifest fields were deleted only after the migration
shadow proved the two channels equivalent.

This is the permanent guard that the bus does not come back: the mode cluster
must be on the plan/outcome, and ``RunMetadata`` must never carry it again. It
is a field-home assertion, so it fails the moment a field is added to the wrong
value rather than after a runtime divergence.
"""

from __future__ import annotations

import dataclasses

import pytest

from ubt.core.ir.render_plan import RenderOutcome, RenderPlan
from ubt.core.ir.run_metadata import RunMetadata

pytestmark = pytest.mark.fast

#: The cluster that used to be a ``manifest.run`` bus.
_MODE_CLUSTER = (
    "bilingual_mode",
    "effective_dual_mode",
    "dual_mode_downgraded",
    "facing_spread",
    "translate_chrome",
    "cover_mode",
)

#: The purely-advisory outputs that moved to the plan in the earlier cut.
_ADVISORY_OUTPUTS = (
    "bilingual_advisory",
    "emit_secondary_mode",
)

#: The renderer's result channel: the mode it actually used.
_RESULT_CHANNEL = ("bilingual_mode", "effective_dual_mode", "dual_mode_downgraded")

_PLAN_FIELDS = frozenset(field.name for field in dataclasses.fields(RenderPlan))
_OUTCOME_FIELDS = frozenset(field.name for field in dataclasses.fields(RenderOutcome))
_RUN_FIELDS = frozenset(RunMetadata.model_fields)


@pytest.mark.parametrize("name", _MODE_CLUSTER + _ADVISORY_OUTPUTS)
def test_the_mode_cluster_lives_on_the_render_plan(name: str) -> None:
    assert name in _PLAN_FIELDS


@pytest.mark.parametrize("name", _MODE_CLUSTER + _ADVISORY_OUTPUTS)
def test_the_mode_cluster_is_gone_from_run_metadata(name: str) -> None:
    assert name not in _RUN_FIELDS


@pytest.mark.parametrize("name", _RESULT_CHANNEL)
def test_the_result_channel_lives_on_the_render_outcome(name: str) -> None:
    assert name in _OUTCOME_FIELDS


def test_plan_and_outcome_are_dataclasses() -> None:
    assert dataclasses.is_dataclass(RenderPlan)
    assert dataclasses.is_dataclass(RenderOutcome)
