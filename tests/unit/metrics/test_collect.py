"""Contract tests for deriving the KPI set from a finished job's report artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.reporter import (
    QualityReport,
    ReportEntityConsistency,
    ReportFidelity,
    ReportPlaceholderMetrics,
    ReportRenderCoverage,
    ReportRepairBreakdown,
    ReportRouteInfo,
    ReportScoreMetrics,
    ReportSummary,
    ReportTerminologyMetrics,
)
from ubt.core.metrics.collect import (
    _div,
    _flag_count,
    _severity_counts,
    _visual_page_count,
    collect_kpis,
    load_kpis,
    save_metrics_report,
)
from ubt.core.metrics.schema import SCHEMA_VERSION

pytestmark = pytest.mark.fast


def _report(
    *,
    total_blocks: int = 10,
    repaired_blocks: int = 2,
    needs_human_blocks: int = 1,
    blocked_human_blocks: int = 0,
    pass_rate: float = 0.8,
    estimated_cost_usd: float | None = 0.5,
    avg_qe: float = 0.9,
    chars: int = 0,
    pages: int = 0,
    terms_audited: int = 4,
    terms_with_drift: int = 1,
    fail_closed_blocks: int = 0,
    preserved_blocks: int = 0,
    skipped_blocks: int = 0,
    formula_blocks: int = 0,
    defect_flags: dict[str, int] | None = None,
    retention_rate: float = 1.0,
    non_text_residual: float = 0.0,
    painted_coverage: float = 0.0,
) -> QualityReport:
    return QualityReport(
        job_id="job-1",
        doc_id="doc-1",
        book_title="Book",
        source_path="src.epub",
        output_path="out.epub",
        target_lang="zh",
        summary=ReportSummary(
            total_blocks=total_blocks,
            completed_blocks=total_blocks - 2,
            repaired_blocks=repaired_blocks,
            failed_blocks=1,
            pass_rate=pass_rate,
            estimated_cost_usd=estimated_cost_usd,
            needs_human_blocks=needs_human_blocks,
            blocked_human_blocks=blocked_human_blocks,
        ),
        score_metrics=ReportScoreMetrics(
            avg_qe=avg_qe,
            min_qe=0.1,
            max_qe=1.0,
            p10_qe=0.5,
            p50_qe=0.9,
            p90_qe=1.0,
            bottom_15_avg_qe=0.6,
        ),
        repair_breakdown=ReportRepairBreakdown(
            direct_pass_count=5,
            round_1_repaired_count=2,
            round_2_repaired_count=1,
            exhausted_count=0,
        ),
        terminology=ReportTerminologyMetrics(
            term_precision=0.9, fuzzy_term_precision=0.95, term_recall=0.8
        ),
        entity_consistency=ReportEntityConsistency(
            terms_audited=terms_audited, terms_with_drift=terms_with_drift
        ),
        placeholder=ReportPlaceholderMetrics(
            masked_spans=3,
            corrupt_spans=0,
            retention_rate=retention_rate,
            masked_blocks=1,
            corrupt_blocks=0,
        ),
        render_coverage=ReportRenderCoverage(
            rendered_blocks=total_blocks,
            skipped_blocks=skipped_blocks,
            fail_closed_blocks=fail_closed_blocks,
            preserved_blocks=preserved_blocks,
            render_coverage=0.0,
            skip_families={},
        ),
        fidelity=ReportFidelity(
            non_text_residual=non_text_residual, painted_coverage=painted_coverage, pages_measured=0
        ),
        defect_flags=defect_flags or {},
        formula_blocks=formula_blocks,
        route=ReportRouteInfo(mode="short", pages=pages, chars=chars),
    )


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_div_is_guarded_against_a_zero_denominator() -> None:
    assert _div(1, 0) == 0.0
    assert _div(3, 2) == 1.5
    assert _div(0, 5) == 0.0


def test_flag_count_sums_by_prefix() -> None:
    flags = {"untranslated:a": 2, "other": 5, "untranslated:b": 1}
    assert _flag_count(flags, "untranslated") == 3
    assert _flag_count(flags, "absent") == 0


def test_severity_counts_defaults_and_ignores_noise() -> None:
    assert _severity_counts(None) == {"info": 0, "major": 0, "critical": 0}
    counts = _severity_counts(
        {
            "findings": [
                {"severity": "critical"},
                {"severity": "major"},
                {"severity": "bogus"},
                "not-a-mapping",
                {"severity": "info"},
            ]
        }
    )
    assert counts == {"info": 1, "major": 1, "critical": 1}


def test_severity_counts_tolerates_a_non_list_findings_field() -> None:
    assert _severity_counts({"findings": "nope"}) == {"info": 0, "major": 0, "critical": 0}


def test_visual_page_count_prefers_the_route_page_count() -> None:
    assert _visual_page_count(None, 5) == 0
    assert _visual_page_count({}, 5) == 5


def test_visual_page_count_uses_the_largest_named_page() -> None:
    report: dict[str, Any] = {"findings": [{"page": 3}, {"page": 7}]}
    assert _visual_page_count(report, 0) == 7


def test_visual_page_count_falls_back_to_severity_findings() -> None:
    assert _visual_page_count({}, 0) == 1  # historic denominator
    assert _visual_page_count({"findings": [{"severity": "critical"}]}, 0) == 1
    assert (
        _visual_page_count({"findings": [{"severity": "critical"}, {"severity": "major"}]}, 0) == 2
    )
    # Unrecognised severity is excluded and cannot dilute the rate.
    assert _visual_page_count({"findings": [{"severity": "bogus"}]}, 0) == 1


# --------------------------------------------------------------------------- #
# collect_kpis
# --------------------------------------------------------------------------- #


def test_collect_kpis_derives_the_core_ratios() -> None:
    kpis = collect_kpis(_report())
    assert kpis.schema_version == SCHEMA_VERSION
    assert kpis.kpis["term_consistency"] == pytest.approx(0.75)  # 1 - 1/4
    assert kpis.kpis["repair_rate"] == pytest.approx(0.2)  # 2/10
    assert kpis.kpis["needs_human_rate"] == pytest.approx(0.1)
    assert kpis.kpis["blocked_human_rate"] == 0.0
    assert kpis.kpis["pass_rate"] == 0.8
    assert kpis.kpis["avg_qe"] == 0.9
    assert kpis.kpis["term_precision"] == 0.9
    assert kpis.kpis["term_fuzzy_precision"] == 0.95
    assert kpis.kpis["term_recall"] == 0.8


def test_collect_kpis_guards_the_empty_denominator() -> None:
    kpis = collect_kpis(_report(total_blocks=0, terms_audited=0, terms_with_drift=0))
    assert kpis.kpis["term_consistency"] == 1.0  # vacuously consistent
    assert kpis.kpis["repair_rate"] == 0.0
    assert kpis.kpis["render_skip_rate"] == 0.0


def test_collect_kpis_scales_cost_by_chars() -> None:
    kpis = collect_kpis(_report(chars=2500, estimated_cost_usd=0.5))
    assert kpis.kpis["cost_per_1k_chars"] == pytest.approx(0.2)  # 0.5 / 2.5
    assert collect_kpis(_report(chars=0, estimated_cost_usd=0.5)).kpis["cost_per_1k_chars"] == 0.0


def test_collect_kpis_counts_untranslated_flags() -> None:
    kpis = collect_kpis(
        _report(
            total_blocks=10,
            defect_flags={"untranslated:x": 2, "unknown:y": 5, "other": 9},
        )
    )
    # Only the 'untranslated' family counts; 'unknown' must not leak in.
    assert kpis.kpis["untranslated_leak_rate"] == pytest.approx(0.2)


def test_collect_kpis_visual_rates_use_the_page_denominator() -> None:
    visual = {
        "findings": [
            {"severity": "critical", "page": 1},
            {"severity": "major", "page": 2},
            {"severity": "major", "page": 3},
        ]
    }
    kpis = collect_kpis(_report(pages=10), visual)
    assert kpis.kpis["visual_critical_rate"] == pytest.approx(0.1)
    assert kpis.kpis["visual_major_rate"] == pytest.approx(0.2)
    assert kpis.details["visual_report"] is True
    assert kpis.details["visual_findings"] == {"info": 0, "major": 2, "critical": 1}


def test_collect_kpis_without_a_visual_report_zeroes_the_visual_rates() -> None:
    kpis = collect_kpis(_report(pages=10))
    assert kpis.kpis["visual_critical_rate"] == 0.0
    assert kpis.kpis["visual_major_rate"] == 0.0
    assert kpis.details["visual_report"] is False


def test_collect_kpis_records_structural_details() -> None:
    kpis = collect_kpis(
        _report(
            total_blocks=7,
            terms_audited=4,
            terms_with_drift=1,
            fail_closed_blocks=1,
            preserved_blocks=2,
            skipped_blocks=3,
            chars=1234,
        )
    )
    details = kpis.details
    assert details["total_blocks"] == 7
    assert details["terms_audited"] == 4
    assert details["terms_with_drift"] == 1
    assert details["fail_closed_blocks"] == 1
    assert details["preserved_blocks"] == 2
    assert details["skipped_blocks"] == 3
    assert details["chars"] == 1234


def test_collect_kpis_records_the_job_block() -> None:
    kpis = collect_kpis(_report())
    assert kpis.job["job_id"] == "job-1"
    assert kpis.job["doc_id"] == "doc-1"
    assert kpis.job["target_lang"] == "zh"
    assert kpis.job["output_path"] == "out.epub"
    assert kpis.job["typst_version"] == ""  # None folds to empty string


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_metrics_report_round_trips_through_disk(tmp_path: Path) -> None:
    kpis = collect_kpis(_report(chars=1000, estimated_cost_usd=0.25))
    path = save_metrics_report(kpis, tmp_path / "nested" / "metrics.json")
    assert path.exists()
    assert load_kpis(path) == kpis
