"""Derive the KPI set from the artifacts a finished job already wrote.

Pure functions over :class:`~ubt.core.engine.reporter.QualityReport` plus the
optional visual report: no new instrumentation, no text re-scan, no LLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.metrics.schema import SCHEMA_VERSION, KpiSet

if TYPE_CHECKING:
    # Annotation-only: a module-level edge would make the metrics package
    # statically depend back on core.engine, which calls into it at export.
    from ubt.core.engine.reporter import QualityReport

_SEVERITIES = ("info", "major", "critical")


def _div(numerator: float, denominator: float) -> float:
    """Guarded division: an empty denominator yields 0.0, never a crash."""
    return numerator / denominator if denominator else 0.0


def _flag_count(flags: Mapping[str, int], prefix: str) -> int:
    return sum(int(value) for key, value in flags.items() if key.startswith(prefix))


def _severity_counts(visual_report: Mapping[str, Any] | None) -> dict[str, int]:
    counts = dict.fromkeys(_SEVERITIES, 0)
    if not visual_report:
        return counts
    findings = visual_report.get("findings")
    if not isinstance(findings, list):
        return counts
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        severity = str(finding.get("severity", ""))
        if severity in counts:
            counts[severity] += 1
    return counts


def _visual_page_count(visual_report: Mapping[str, Any] | None, route_pages: int) -> int:
    """The denominator for the visual rate KPIs — pages a finding could hit.

    Prefers the real page count when the report knows it (route_pages, or the
    largest page number any finding names). When findings name no page and the
    route carries no page count, there is no page population to divide by; the
    denominator then falls back to the number of *severity-bearing* findings so
    the rate stays a bounded share in [0, 1] rather than a count divided by a
    single page (which let ``visual_*_rate`` exceed 1.0). Findings whose
    severity is unrecognised are excluded so they cannot dilute the rates, and
    a report with no counted findings keeps the historic denominator of 1.
    """
    if visual_report is None:
        return 0
    if route_pages > 0:
        return route_pages
    page_numbers: list[int] = []
    severity_findings = 0
    for finding in visual_report.get("findings") or []:
        if not isinstance(finding, Mapping):
            continue
        page = finding.get("page")
        if isinstance(page, int):
            page_numbers.append(page)
        if str(finding.get("severity", "")) in _SEVERITIES:
            severity_findings += 1
    if page_numbers:
        # max(), not len(): findings can repeat a page and can be numbered
        # sparsely, so the count was never the population size.
        return max(page_numbers)
    return max(1, severity_findings)


def collect_kpis(report: QualityReport, visual_report: Mapping[str, Any] | None = None) -> KpiSet:
    """Collect the KPI v1 set from a finished job's report artifacts."""
    total_blocks = report.summary.total_blocks
    chars = report.route.chars if report.route is not None else 0
    route_pages = report.route.pages if report.route is not None else 0
    entity = report.entity_consistency
    terminology = report.terminology
    coverage = report.render_coverage
    substitutions = len(report.formula_witness_fallbacks)
    formula_blocks = report.formula_blocks
    severities = _severity_counts(visual_report)
    visual_pages = _visual_page_count(visual_report, route_pages)
    fidelity = report.fidelity

    kpis: dict[str, float] = {
        "term_consistency": 1.0 - _div(entity.terms_with_drift, entity.terms_audited),
        "term_precision": terminology.term_precision,
        "term_fuzzy_precision": terminology.fuzzy_term_precision,
        "term_recall": terminology.term_recall,
        "render_skip_rate": _div(coverage.fail_closed_blocks, total_blocks),
        "render_preserved_rate": _div(coverage.preserved_blocks, total_blocks),
        "placeholder_retention": report.placeholder.retention_rate,
        "pass_rate": report.summary.pass_rate,
        "avg_qe": report.score_metrics.avg_qe,
        "repair_rate": _div(report.summary.repaired_blocks, total_blocks),
        "needs_human_rate": _div(report.summary.needs_human_blocks, total_blocks),
        "blocked_human_rate": _div(report.summary.blocked_human_blocks, total_blocks),
        "formula_fidelity": (
            0.0
            if (formula_blocks == 0 and substitutions > 0)
            else max(0.0, 1.0 - _div(substitutions, formula_blocks))
        ),
        "formula_substitutions": float(substitutions),
        "visual_critical_rate": _div(severities["critical"], visual_pages),
        "visual_major_rate": _div(severities["major"], visual_pages),
        "fidelity_non_text_residual": fidelity.non_text_residual,
        "rigid_painted_coverage": fidelity.painted_coverage,
        "untranslated_leak_rate": _div(
            _flag_count(report.defect_flags, "untranslated"), total_blocks
        ),
        # KPIs are float-only (``KpiSet.kpis: dict[str, float]``) and every
        # checked-in golden carries this key, so an unpriced run still lands on
        # 0.0 *here* — the quality report beside it says "unknown" instead.
        # Making this null means dropping the key, which ``compare_kpis`` treats
        # as a hard "missing from candidate" violation: that is a golden
        # regeneration sweep over every baseline, deliberately not taken now.
        "cost_per_1k_chars": _div(report.summary.estimated_cost_usd or 0.0, chars / 1000.0),
    }

    details: dict[str, Any] = {
        "total_blocks": total_blocks,
        "terms_audited": entity.terms_audited,
        "terms_with_drift": entity.terms_with_drift,
        "skipped_blocks": coverage.skipped_blocks,
        "fail_closed_blocks": coverage.fail_closed_blocks,
        "preserved_blocks": coverage.preserved_blocks,
        "skip_families": dict(coverage.skip_families),
        "formula_blocks": formula_blocks,
        "formula_substitutions": substitutions,
        "visual_report": visual_report is not None,
        "visual_pages": visual_pages,
        "visual_findings": severities,
        "needs_human_blocks": report.summary.needs_human_blocks,
        "blocked_human_blocks": report.summary.blocked_human_blocks,
        "chars": chars,
        "estimated_cost_usd": report.summary.estimated_cost_usd,
    }

    job = {
        "job_id": report.job_id,
        "doc_id": report.doc_id,
        "target_lang": report.target_lang,
        "output_path": report.output_path,
        "generated_at": report.generated_at.isoformat(),
        "typst_version": report.typst_version or "",
    }

    return KpiSet(schema_version=SCHEMA_VERSION, job=job, kpis=kpis, details=details)


def save_metrics_report(kpis: KpiSet, path: Path | str) -> Path:
    """Write the KPI artifact (JSON) and return its path."""
    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(kpis.model_dump_json(indent=2), encoding="utf-8")
    return out_file


def load_kpis(path: Path | str) -> KpiSet:
    """Read a KPI artifact written by :func:`save_metrics_report`."""
    return KpiSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
