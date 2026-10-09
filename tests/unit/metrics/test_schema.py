"""Contract tests for the versioned KPI schema (definitions + artifact model)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ubt.core.metrics.schema import (
    KPI_BY_NAME,
    KPI_DEFINITIONS,
    SCHEMA_VERSION,
    Direction,
    KpiSet,
)

pytestmark = pytest.mark.fast


def test_schema_version_is_pinned() -> None:
    assert SCHEMA_VERSION == 3


def test_definition_names_are_unique_and_indexed() -> None:
    names = [d.name for d in KPI_DEFINITIONS]
    assert len(names) == len(set(names))
    assert set(KPI_BY_NAME) == set(names)
    assert len(KPI_BY_NAME) == len(KPI_DEFINITIONS)


def test_every_definition_is_documented() -> None:
    for definition in KPI_DEFINITIONS:
        assert definition.definition.strip()
        assert definition.source.strip()
        assert definition.tolerance >= 0.0


def test_units_are_from_the_declared_vocabulary() -> None:
    allowed = {"ratio", "score", "count", "usd_per_1k_chars"}
    assert {d.unit for d in KPI_DEFINITIONS} <= allowed


def test_floor_applies_only_to_higher_is_better() -> None:
    for definition in KPI_DEFINITIONS:
        if definition.floor is not None:
            assert definition.direction is Direction.HIGHER_IS_BETTER, definition.name


def test_ceiling_applies_only_to_lower_is_better() -> None:
    for definition in KPI_DEFINITIONS:
        if definition.ceiling is not None:
            assert definition.direction is Direction.LOWER_IS_BETTER, definition.name


def test_the_defect_bounds_are_the_expected_ones() -> None:
    assert KPI_BY_NAME["placeholder_retention"].floor == 1.0
    assert KPI_BY_NAME["render_skip_rate"].ceiling == 0.0
    assert KPI_BY_NAME["untranslated_leak_rate"].ceiling == 0.0
    assert KPI_BY_NAME["term_consistency"].tolerance == 0.02
    assert KPI_BY_NAME["term_consistency"].direction is Direction.HIGHER_IS_BETTER
    assert KPI_BY_NAME["cost_per_1k_chars"].unit == "usd_per_1k_chars"


def test_kpi_set_defaults_to_an_unknown_version() -> None:
    kpi_set = KpiSet()
    assert kpi_set.schema_version == 0  # 0 = legacy/unknown
    assert kpi_set.job == {}
    assert kpi_set.kpis == {}
    assert kpi_set.details == {}


def test_kpi_set_is_frozen() -> None:
    kpi_set = KpiSet(schema_version=SCHEMA_VERSION)
    with pytest.raises(ValidationError):
        kpi_set.schema_version = 1
