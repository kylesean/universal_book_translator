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


def test_adapter_mode_prefers_the_adapter_domain_value() -> None:
    plan = RenderPlan(bilingual_mode="monolingual", effective_dual_mode="dual")
    assert plan.adapter_mode() == "monolingual"


def test_adapter_mode_translates_an_advisor_only_plan() -> None:
    assert RenderPlan(effective_dual_mode="dual").adapter_mode() == "bilingual"
    assert RenderPlan(effective_dual_mode="mono").adapter_mode() == "monolingual"


def test_adapter_mode_falls_back_when_the_plan_carries_no_mode() -> None:
    assert RenderPlan().adapter_mode(fallback="monolingual") == "monolingual"
    assert RenderPlan().adapter_mode() is None


def test_preflight_and_export_resolve_the_same_mode() -> None:
    # The two callers must not disagree about which vocabulary they were handed.
    plan = RenderPlan(effective_dual_mode="mono")
    assert plan.adapter_mode() == plan.adapter_mode(fallback="bilingual")
