"""Contract tests for the direction-aware KPI regression gate."""

from __future__ import annotations

import pytest

from ubt.core.metrics.compare import (
    _STRUCTURAL_DETAIL_KEYS,
    check_thresholds,
    compare_kpi_sets,
    compare_kpis,
)
from ubt.core.metrics.schema import KpiSet

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# compare_kpis
# --------------------------------------------------------------------------- #


def test_a_higher_is_better_drop_beyond_tolerance_regresses() -> None:
    report = compare_kpis({"term_precision": 0.9}, {"term_precision": 0.8})
    assert report.passed is False
    (delta,) = report.deltas
    assert delta.name == "term_precision"
    assert delta.baseline == 0.9
    assert delta.candidate == 0.8
    assert delta.direction == "higher_is_better"
    assert delta.regressed is True
    assert report.violations == ["term_precision: 0.9 -> 0.8 (higher_is_better, tolerance 0.02)"]


def test_a_drop_within_tolerance_does_not_regress() -> None:
    report = compare_kpis({"term_precision": 0.9}, {"term_precision": 0.89})
    assert report.passed is True
    assert report.deltas[0].regressed is False
    assert report.violations == []


def test_a_higher_is_better_rise_is_never_a_regression() -> None:
    report = compare_kpis({"term_precision": 0.8}, {"term_precision": 0.95})
    assert report.passed is True
    assert report.deltas[0].delta == pytest.approx(0.15)


def test_a_lower_is_better_rise_regresses() -> None:
    report = compare_kpis({"render_skip_rate": 0.0}, {"render_skip_rate": 0.5})
    assert report.passed is False
    assert report.deltas[0].direction == "lower_is_better"
    assert report.deltas[0].regressed is True


def test_a_lower_is_better_fall_does_not_regress() -> None:
    report = compare_kpis({"render_skip_rate": 0.1}, {"render_skip_rate": 0.0})
    assert report.passed is True


def test_an_unknown_kpi_in_the_candidate_is_listed_and_never_regresses() -> None:
    report = compare_kpis({}, {"brand_new": 1.0})
    assert report.passed is True
    assert report.unknown_kpis == ["brand_new"]
    assert report.violations == []


def test_strict_names_turns_a_missing_baseline_kpi_into_a_violation() -> None:
    report = compare_kpis({}, {"brand_new": 1.0}, strict_names=True)
    assert report.passed is False
    assert report.violations == [
        "brand_new: missing from baseline (stale golden — re-record deliberately)"
    ]


def test_a_kpi_missing_from_the_candidate_is_a_violation() -> None:
    report = compare_kpis({"gone": 1.0}, {})
    assert report.passed is False
    assert report.violations == ["gone: missing from candidate"]


def test_a_non_finite_kpi_is_a_violation() -> None:
    report = compare_kpis({"term_precision": float("nan")}, {"term_precision": 1.0})
    assert report.passed is False
    assert report.violations == ["term_precision: non-finite KPI (nan -> 1.0)"]


def test_tolerance_override_can_excuse_a_regression() -> None:
    report = compare_kpis(
        {"term_precision": 0.9}, {"term_precision": 0.8}, tolerances={"term_precision": 0.5}
    )
    assert report.passed is True
    assert report.deltas[0].tolerance == 0.5


def test_an_unknown_kpi_name_defaults_to_higher_is_better_with_zero_tolerance() -> None:
    report = compare_kpis({"mystery": 1.0}, {"mystery": 0.5})
    (delta,) = report.deltas
    assert delta.direction == "higher_is_better"
    assert delta.tolerance == 0.0
    assert delta.regressed is True


def test_deltas_are_sorted_by_name() -> None:
    report = compare_kpis({"z": 1.0, "a": 1.0}, {"z": 1.0, "a": 1.0})
    assert [d.name for d in report.deltas] == ["a", "z"]


# --------------------------------------------------------------------------- #
# compare_kpi_sets
# --------------------------------------------------------------------------- #


def test_identical_kpi_sets_pass() -> None:
    baseline = KpiSet(schema_version=2, kpis={"term_precision": 0.9})
    assert compare_kpi_sets(baseline, baseline.model_copy()).passed is True


def test_a_structural_detail_change_is_a_violation() -> None:
    baseline = KpiSet(schema_version=2, details={"total_blocks": 16})
    candidate = KpiSet(schema_version=2, details={"total_blocks": 17})
    report = compare_kpi_sets(baseline, candidate)
    assert report.passed is False
    assert report.violations == ["details.total_blocks: 16 -> 17"]


def test_a_schema_version_mismatch_is_a_violation() -> None:
    baseline = KpiSet(schema_version=2)
    candidate = KpiSet(schema_version=1)
    report = compare_kpi_sets(baseline, candidate)
    assert report.passed is False
    assert report.baseline_schema == 2
    assert report.candidate_schema == 1
    assert any(v.startswith("schema_version mismatch:") for v in report.violations)


def test_structural_detail_keys_are_the_documented_set() -> None:
    assert "total_blocks" in _STRUCTURAL_DETAIL_KEYS


def test_non_structural_details_are_ignored() -> None:
    baseline = KpiSet(schema_version=2, details={"skip_families": {"a": 1}})
    candidate = KpiSet(schema_version=2, details={"skip_families": {"a": 2}})
    assert compare_kpi_sets(baseline, candidate).passed is True


# --------------------------------------------------------------------------- #
# check_thresholds
# --------------------------------------------------------------------------- #


def test_a_value_below_a_floor_violates() -> None:
    assert check_thresholds({"placeholder_retention": 0.9}) == [
        "placeholder_retention: 0.9 below floor 1"
    ]


def test_a_value_above_a_ceiling_violates() -> None:
    assert check_thresholds({"render_skip_rate": 0.5}) == ["render_skip_rate: 0.5 above ceiling 0"]


def test_a_value_on_its_bound_passes() -> None:
    assert check_thresholds({"placeholder_retention": 1.0, "render_skip_rate": 0.0}) == []


def test_explicit_bounds_override_the_schema() -> None:
    assert check_thresholds({"term_precision": 0.5}, floors={"term_precision": 0.9}) == [
        "term_precision: 0.5 below floor 0.9"
    ]
    assert check_thresholds({"term_precision": 0.95}, ceilings={"term_precision": 0.9}) == [
        "term_precision: 0.95 above ceiling 0.9"
    ]


def test_a_non_finite_threshold_value_violates() -> None:
    assert check_thresholds({"placeholder_retention": float("inf")}) == [
        "placeholder_retention: non-finite KPI (inf)"
    ]


def test_absent_kpis_are_skipped() -> None:
    assert check_thresholds({}) == []
