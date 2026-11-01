from __future__ import annotations

import dataclasses

import pytest

from ubt.core.ir.render_plan import RenderOutcome, RenderPlan
from ubt.core.ir.run_metadata import RunMetadata

pytestmark = pytest.mark.fast

_MODE_CLUSTER = (
    "bilingual_mode",
    "effective_dual_mode",
    "dual_mode_downgraded",
    "translate_chrome",
)

_ADVISORY_OUTPUTS = (
    "bilingual_advisory",
    "emit_secondary_mode",
)

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
