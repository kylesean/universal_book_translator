"""Unit tests for the versioned KPI metrics layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.reporter import QualityReport
from ubt.core.metrics.collect import collect_kpis, load_kpis, save_metrics_report
from ubt.core.metrics.compare import check_thresholds, compare_kpi_sets, compare_kpis
from ubt.core.metrics.schema import KPI_BY_NAME, SCHEMA_VERSION, Direction, KpiSet


def _report(**overrides: Any) -> QualityReport:
    base: dict[str, Any] = {
        "job_id": "job_test",
        "doc_id": "doc_test",
        "book_title": "Test",
        "source_path": "/in.pdf",
        "output_path": "/out.pdf",
        "target_lang": "zh",
        "summary": {
            "total_blocks": 100,
            "completed_blocks": 95,
            "repaired_blocks": 10,
            "failed_blocks": 5,
            "pass_rate": 0.95,
            "estimated_cost_usd": 1.0,
            "needs_human_blocks": 5,
        },
        "score_metrics": {
            "avg_qe": 0.8,
            "min_qe": 0.4,
            "max_qe": 0.95,
            "p10_qe": 0.5,
            "p50_qe": 0.8,
            "p90_qe": 0.9,
            "bottom_15_avg_qe": 0.5,
            "scored_blocks": 80,
        },
        "repair_breakdown": {
            "direct_pass_count": 85,
            "round_1_repaired_count": 8,
            "round_2_repaired_count": 2,
            "exhausted_count": 5,
        },
        "terminology": {
            "terms_expected": 10,
            "terms_rendered": 9,
            "term_precision": 0.9,
            "fuzzy_term_precision": 0.95,
            "term_recall": 0.9,
        },
        "entity_consistency": {"terms_audited": 10, "terms_with_drift": 2},
        "placeholder": {
            "masked_spans": 20,
            "corrupt_spans": 1,
            "retention_rate": 0.95,
            "masked_blocks": 8,
            "corrupt_blocks": 1,
        },
        "render_coverage": {
            "rendered_blocks": 93,
            "skipped_blocks": 2,
            "fail_closed_blocks": 2,
            "preserved_blocks": 0,
            "render_coverage": 0.98,
        },
        "route": {
            "mode": "long",
            "pages": 10,
            "chars": 10000,
            "reason": "default",
            "formula_mode": "witness",
        },
        "defect_flags": {"untranslated:alpha": 3, "math_token_corrupt": 1},
        "formula_witness_fallbacks": ["eq1", "eq2"],
        "formula_blocks": 8,
    }
    base.update(overrides)
    return QualityReport.model_validate(base)


def test_collect_core_values() -> None:
    kpis = collect_kpis(_report())
    assert kpis.schema_version == SCHEMA_VERSION
    assert kpis.kpis["term_consistency"] == pytest.approx(0.8)
    assert kpis.kpis["term_precision"] == pytest.approx(0.9)
    assert kpis.kpis["term_fuzzy_precision"] == pytest.approx(0.95)
    assert kpis.kpis["term_recall"] == pytest.approx(0.9)
    assert kpis.kpis["render_skip_rate"] == pytest.approx(0.02)
    assert kpis.kpis["placeholder_retention"] == pytest.approx(0.95)
    assert kpis.kpis["pass_rate"] == pytest.approx(0.95)
    assert kpis.kpis["avg_qe"] == pytest.approx(0.8)
    assert kpis.kpis["repair_rate"] == pytest.approx(0.1)
    assert kpis.kpis["needs_human_rate"] == pytest.approx(0.05)
    assert kpis.kpis["formula_fidelity"] == pytest.approx(0.75)
    assert kpis.kpis["formula_substitutions"] == 2.0
    assert kpis.kpis["untranslated_leak_rate"] == pytest.approx(0.03)
    assert kpis.kpis["cost_per_1k_chars"] == pytest.approx(0.1)
    # No visual report => visual rates are 0, never a crash or a phantom finding.
    assert kpis.kpis["visual_critical_rate"] == 0.0
    assert kpis.kpis["visual_major_rate"] == 0.0
    assert kpis.details["visual_report"] is False


def test_collect_is_vacuous_safe_on_an_empty_job() -> None:
    empty = _report(
        summary={
            "total_blocks": 0,
            "completed_blocks": 0,
            "repaired_blocks": 0,
            "failed_blocks": 0,
            "pass_rate": 1.0,
            "estimated_cost_usd": 0.0,
            "needs_human_blocks": 0,
        },
        terminology={},
        entity_consistency={},
        placeholder={
            "masked_spans": 0,
            "corrupt_spans": 0,
            "retention_rate": 1.0,
            "masked_blocks": 0,
            "corrupt_blocks": 0,
        },
        render_coverage={"rendered_blocks": 0, "skipped_blocks": 0, "render_coverage": 1.0},
        route=None,
        defect_flags={},
        formula_witness_fallbacks=[],
        formula_blocks=0,
    )
    kpis = collect_kpis(empty)
    assert kpis.kpis["term_consistency"] == 1.0
    assert kpis.kpis["formula_fidelity"] == 1.0
    assert kpis.kpis["formula_substitutions"] == 0.0
    assert kpis.kpis["render_skip_rate"] == 0.0
    assert kpis.kpis["cost_per_1k_chars"] == 0.0
    assert kpis.kpis["untranslated_leak_rate"] == 0.0


def test_visual_findings_become_page_rates() -> None:
    visual = {
        "findings": [
            {"severity": "critical", "code": "blank_page", "page": 1},
            {"severity": "major", "code": "overlap", "page": 2},
            {"severity": "info", "code": "note"},
        ]
    }
    kpis = collect_kpis(_report(), visual)
    assert kpis.kpis["visual_critical_rate"] == pytest.approx(0.1)
    assert kpis.kpis["visual_major_rate"] == pytest.approx(0.1)
    assert kpis.details["visual_findings"] == {"info": 1, "major": 1, "critical": 1}
    assert kpis.details["visual_pages"] == 10
    assert kpis.details["visual_report"] is True


def test_registry_covers_exactly_the_collected_kpis() -> None:
    """A KPI added to the collector without a definition (or vice versa) fails here."""
    collected = set(collect_kpis(_report()).kpis)
    assert collected == set(KPI_BY_NAME)


def test_save_load_round_trip(tmp_path: Path) -> None:
    kpis = collect_kpis(_report())
    path = save_metrics_report(kpis, tmp_path / "out_metrics.json")
    assert path.exists()
    again = load_kpis(path)
    assert again.kpis == kpis.kpis
    assert again.schema_version == kpis.schema_version
    assert again.job["job_id"] == "job_test"


def test_compare_flags_regressions_in_both_directions() -> None:
    report = compare_kpis(
        {"term_consistency": 0.90, "render_skip_rate": 0.01},
        {"term_consistency": 0.80, "render_skip_rate": 0.05},
    )
    assert not report.passed
    assert any(v.startswith("term_consistency") for v in report.violations)
    assert any(v.startswith("render_skip_rate") for v in report.violations)


def test_compare_ignores_changes_within_tolerance() -> None:
    assert compare_kpis({"term_consistency": 0.90}, {"term_consistency": 0.89}).passed
    assert compare_kpis({"render_skip_rate": 0.01}, {"render_skip_rate": 0.02}).passed


def test_compare_tolerance_override() -> None:
    assert compare_kpis({"avg_qe": 0.90}, {"avg_qe": 0.80}, tolerances={"avg_qe": 0.15}).passed


def test_compare_missing_kpi_is_a_violation() -> None:
    report = compare_kpis({"avg_qe": 0.8, "pass_rate": 1.0}, {"avg_qe": 0.8})
    assert not report.passed
    assert any(v.startswith("pass_rate") and "missing" in v for v in report.violations)


def test_compare_reports_unknown_kpi_without_guessing_direction() -> None:
    report = compare_kpis({}, {"mystery_kpi": 1.0})
    assert report.unknown_kpis == ["mystery_kpi"]
    assert report.passed


def test_strict_names_makes_a_missing_baseline_kpi_fail() -> None:
    """A metric the golden predates gates nothing at all; strict mode says so."""
    report = compare_kpis({"pass_rate": 1.0}, {"pass_rate": 1.0, "formula_fidelity": 0.5})
    assert report.passed
    strict = compare_kpis(
        {"pass_rate": 1.0}, {"pass_rate": 1.0, "formula_fidelity": 0.5}, strict_names=True
    )
    assert not strict.passed
    assert strict.unknown_kpis == ["formula_fidelity"]
    assert any(
        v.startswith("formula_fidelity") and "missing from baseline" in v for v in strict.violations
    )


def test_compare_kpi_sets_threads_strict_names() -> None:
    baseline = KpiSet(kpis={"pass_rate": 1.0})
    candidate = KpiSet(kpis={"pass_rate": 1.0, "avg_qe": 0.9})
    assert compare_kpi_sets(baseline, candidate).passed
    assert not compare_kpi_sets(baseline, candidate, strict_names=True).passed


def test_compare_kpi_sets_rejects_schema_drift() -> None:
    baseline = KpiSet(kpis={"avg_qe": 0.8})
    candidate = KpiSet(schema_version=SCHEMA_VERSION + 1, kpis={"avg_qe": 0.8})
    report = compare_kpi_sets(baseline, candidate)
    assert not report.passed
    assert any("schema_version" in v for v in report.violations)


def test_check_thresholds_floors_and_ceilings() -> None:
    assert check_thresholds({"pass_rate": 0.5}, floors={"pass_rate": 0.9})
    assert check_thresholds({"pass_rate": 0.95}, floors={"pass_rate": 0.9}) == []
    assert check_thresholds({"render_skip_rate": 0.5}, ceilings={"render_skip_rate": 0.1})
    assert check_thresholds({"render_skip_rate": 0.05}, ceilings={"render_skip_rate": 0.1}) == []
    # A KPI with no bound configured stays report-only, never a false gate.
    assert check_thresholds({"pass_rate": 1.0}) == []


def test_registry_bounds_are_defect_shaped() -> None:
    """Every configured bound must be the value a defect-free run produces.

    A floor above 1.0 or a ceiling below 0.0 is unreachable, and anything else
    encodes a tolerance that belongs in ``tolerance``, not in an absolute gate.
    """
    for definition in KPI_BY_NAME.values():
        if definition.floor is not None:
            assert definition.direction is Direction.HIGHER_IS_BETTER, definition.name
            assert 0.0 <= definition.floor <= 1.0, definition.name
        if definition.ceiling is not None:
            assert definition.direction is Direction.LOWER_IS_BETTER, definition.name
            assert definition.ceiling == 0.0, definition.name


def test_goldens_satisfy_absolute_bounds() -> None:
    """Fast-tier guard for the checked-in baselines' KPI artifacts.

    Two failures this turns red instead of leaving silent: a golden that does
    not carry the whole registry (a metric added afterwards gates nothing,
    because the compare skips names the baseline lacks), and a golden that
    records a defect — leaked source, a dropped placeholder, a quarantined
    block — as its own baseline. Both would otherwise only surface inside a
    minutes-long end-to-end run, and only if that corpus were re-recorded.
    """
    golden_dir = Path(__file__).resolve().parents[1] / "baselines"
    goldens = sorted(golden_dir.glob("*/metrics.golden.json"))
    assert goldens, f"no golden artifact found under {golden_dir}"
    for path in goldens:
        golden = load_kpis(path)
        assert set(golden.kpis) == set(KPI_BY_NAME), (
            f"{path.relative_to(golden_dir.parent)} is out of step with the "
            f"registry: extra={sorted(set(golden.kpis) - set(KPI_BY_NAME))} "
            f"missing={sorted(set(KPI_BY_NAME) - set(golden.kpis))} "
            "(re-record deliberately with UBT_UPDATE_GOLDENS=1)"
        )
        violations = check_thresholds(golden.kpis)
        assert not violations, f"{path.name} records a defect:\n" + "\n".join(violations)


@pytest.mark.fast
def test_metrics_formula_fidelity_zero_when_no_blocks_but_substitutions() -> None:
    """A run with substitutions but no counted formula blocks must not score 1.0.

    The old form re-derived the branch inline and asserted its own literal, so
    it stayed green even if ``collect_kpis`` regressed to ``1 - 0/0 == 1.0``.
    This drives the real collector: ``formula_blocks=0`` with two witness
    fallbacks is a defect the KPI must report as 0.0, not a perfect score.
    """
    report = _report(formula_blocks=0, formula_witness_fallbacks=["eq1", "eq2"])
    kpis = collect_kpis(report)
    assert kpis.kpis["formula_fidelity"] == 0.0
    assert kpis.kpis["formula_substitutions"] == 2.0


@pytest.mark.fast
def test_metrics_json_errors_go_to_stderr(tmp_path: Path) -> None:
    missing = tmp_path / "missing_metrics.json"
    commands = [
        [sys.executable, "-m", "ubt", "metrics", "show", str(missing), "--json"],
        [
            sys.executable,
            "-m",
            "ubt",
            "metrics",
            "compare",
            str(missing),
            str(missing),
            "--json",
        ],
    ]
    for command in commands:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert proc.returncode == 1
        assert proc.stdout.strip() == ""
        assert "not found" in proc.stderr


def test_compare_and_thresholds_reject_non_finite_kpis() -> None:
    """A NaN KPI must violate, not silently pass (all NaN comparisons are False)."""
    report = compare_kpis({"avg_qe": 0.8}, {"avg_qe": float("nan")})
    assert not report.passed
    assert any("non-finite" in v for v in report.violations)
    assert check_thresholds({"avg_qe": float("nan")})
    assert any("non-finite" in v for v in check_thresholds({"avg_qe": float("nan")}))


def test_missing_schema_version_is_legacy_and_mismatches() -> None:
    """An artifact without schema_version must not be stamped as current."""
    legacy = KpiSet(kpis={"avg_qe": 0.8})
    assert legacy.schema_version == 0
    report = compare_kpi_sets(legacy, KpiSet(schema_version=SCHEMA_VERSION, kpis={"avg_qe": 0.8}))
    assert any("schema_version" in v for v in report.violations)


def test_page_less_visual_findings_cannot_exceed_a_rate_of_one() -> None:
    """Findings that name no page must not make visual_*_rate exceed 1.0.

    ``route=None`` (no known page population) is the branch the original
    hard-coded denominator of 1 broke, so the test must exercise *that*, not a
    report that already carries ``route.pages`` (which made the old test pass
    whether or not the fix was present).
    """
    visual = {"findings": [{"severity": "major", "flag": "banned_unicode_dash"}] * 5}
    kpis = collect_kpis(_report(route=None), visual).kpis
    assert kpis["visual_major_rate"] == pytest.approx(1.0)
    assert kpis["visual_critical_rate"] <= 1.0
    # Unrecognised severities are not counted and must not dilute the rates.
    diluted = {
        "findings": [{"severity": "major"}] * 2 + [{"severity": "bogus"}] * 8
    }
    assert collect_kpis(_report(route=None), diluted).kpis["visual_major_rate"] == pytest.approx(
        1.0
    )
