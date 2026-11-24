"""Quality report and versioned KPI artifact builders for the export stage."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from ubt.core.engine.reporter import (
    QualityReport,
    ReportEntityConsistency,
    ReportTerminologyMetrics,
    build_quality_report,
    save_quality_report,
)
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import IRBlock
from ubt.core.job_options import RUN_REPORT_KINDS, sidecar_path
from ubt.core.metrics.collect import collect_kpis, save_metrics_report
from ubt.core.qe.term_metrics import evaluate_terms, summarize_drift
from ubt.core.router.pricing import cache_hit_rate_from_usage, estimate_cost_usd

logger = logging.getLogger(__name__)


def _drop_stale_run_reports(rendered_path: Path) -> None:
    """Delete the previous run's derived reports sitting beside a new deliverable.

    The names come from ``rendered_path.stem`` exactly as the writers build them
    (:func:`_build_reports` in this module, ``artifact_and_report_paths`` on the
    read side), so what gets cleared is precisely what would otherwise be read
    back. The deliverable itself is never touched, and a missing file is not an
    error.
    """
    for kind in RUN_REPORT_KINDS:
        with contextlib.suppress(OSError):
            sidecar_path(rendered_path, kind).unlink(missing_ok=True)


def _font_delivery_for(ctx: StageContext) -> dict[str, Any] | None:
    """The font verdict the report carries, or None when nothing was measured.

    A deliverable that cannot render the target script is a delivery-level
    fact, not a log line: ``script_available=False`` means the PDF contains
    tofu squares. Reporting it here puts it in the quality report the console
    and the KPI artifact already read, so the reader learns it without
    watching stderr.

    ``None`` (rather than a guessed verdict) when the probe could not answer —
    no typst binary and no fontconfig. An unmeasured fact must not be rendered
    as a measured one.
    """
    try:
        from ubt.core.engine.document_services import resolve_font_for_render as resolve_for_render

        stack = resolve_for_render(
            ctx.manifest.target_lang,
            font_override=getattr(ctx.config, "font_family", None),
        )
    except Exception as exc:  # a reporting extra must never sink the delivery
        logger.debug("font verdict unavailable for job %s: %s", ctx.job_id, exc)
        return None
    if not stack.probed:
        return None
    return {
        "probed": stack.probed,
        "families": list(stack.families),
        "unavailable": list(stack.unavailable),
        "substituted": list(stack.substituted),
        # False = the delivered file cannot render the target script.
        "script_available": stack.script_available,
    }


async def _build_reports(
    ctx: StageContext,
    final_blocks: list[IRBlock],
    *,
    rendered_path: Path,
    artifact_report_path: Path | None,
    glossary_dicts: list[dict[str, Any]],
    enforced_spans: int,
    run_usage: Any,
    delivery_contract: dict[str, Any] | None = None,
    mode_advisory: dict[str, Any] | None = None,
    companion_artifacts: list[dict[str, Any]] | None = None,
) -> tuple[QualityReport, Path]:
    """Write the quality report + versioned KPI artifact; return both handles.

    Cost/cache must describe the JOB across every resume, not this process:
    the pipeline persists each priced delta into job_meta.usage_totals, so
    the ledger holds the whole bill and a book finished in N sittings stops
    reporting only the N-th one. Provider counters reset on launch, and
    charging a job the router's cumulative totals over-reports every job
    after the first whenever one router serves several (the API does exactly
    that) -- so the order is ledger, then the caller's run delta, then the
    router's own totals, and a job with no recorded usage reports unknown,
    never $0. Terminology metrics run over the DELIVERED blocks so the
    report measures the artifact the reader gets; ``enforced_spans`` says
    how much of that was mechanically corrected.
    """
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    # Ledger first (its ``usage_totals`` spans every sitting of the job), then
    # the caller's run delta, then the router's process-wide totals — the order
    # this docstring documents. The previous ``if run_usage is not None`` always
    # took the run delta (``measure_run_usage`` never returns None), so a
    # resumed job reported only its final sitting's spend.
    usage = await asyncio.to_thread(ledger.get_job_usage, actual_job_id)
    if not usage and run_usage:
        usage = run_usage
    if not usage and ctx.router is not None:
        usage = ctx.router.usage_totals_by_model()
    token_cost_usd: float | None = (
        estimate_cost_usd(
            usage,
            base_url=ctx.config.base_url,
            # The same attribution the budget preflight bills against (see
            # ``pipeline._bill_session``): the OCR-channel map merged with the
            # router's per-model fallback endpoints. Passing only the former
            # priced a model that a local fallback endpoint actually served at
            # the primary endpoint's cloud rate, so the report and the budget
            # quoted two different totals for one run.
            endpoint_map={
                **ctx.config.remote_billing_models(),
                **(ctx.router.billing_endpoint_map() if ctx.router is not None else {}),
            },
        )
        if usage
        else None
    )
    cache_hit_rate: float | None = cache_hit_rate_from_usage(usage) if usage else None

    if enforced_spans:
        logger.info(
            "Deterministic glossary enforcement rewrote %d span(s) before the "
            "terminology metrics were computed for job %s",
            enforced_spans,
            actual_job_id,
        )
    term_metrics = await asyncio.to_thread(evaluate_terms, final_blocks, glossary_dicts)
    terminology_metrics = ReportTerminologyMetrics(
        terms_expected=term_metrics.terms_expected,
        terms_rendered=term_metrics.terms_rendered,
        term_precision=term_metrics.term_precision,
        fuzzy_term_precision=term_metrics.fuzzy_term_precision,
        term_recall=term_metrics.term_recall,
    )
    # Document-level terminology drift, derived from the same scan (no second
    # pass). Terms without a canonical rendering are not auditable and excluded.
    drifts = await asyncio.to_thread(summarize_drift, term_metrics)
    entity_consistency = ReportEntityConsistency(
        terms_audited=term_metrics.terms_expected,
        terms_with_drift=len(drifts),
        top_drifted=[
            {
                "source": d.source,
                "expected": d.expected,
                "occurrences": d.occurrences,
                "exact_renderings": d.exact_renderings,
                "drift_rate": d.drift_rate,
                "block_ids": list(d.drifted_block_ids[:20]),
            }
            for d in drifts[:20]
        ],
    )

    report = await asyncio.to_thread(
        build_quality_report,
        ledger=ledger,
        job_id=actual_job_id,
        manifest=ctx.manifest,
        output_path=rendered_path,
        token_cost_usd=token_cost_usd,
        cache_hit_rate=cache_hit_rate,
        terminology_metrics=terminology_metrics,
        entity_consistency=entity_consistency,
        enforced_spans=enforced_spans,
        delivery_contract=delivery_contract,
        mode_advisory=mode_advisory,
        font_delivery=_font_delivery_for(ctx),
        companion_artifacts=companion_artifacts,
    )
    if report.summary.failed_blocks > 0:
        logger.warning(
            "Job %s completed with %d FAILED block(s) (pass_rate=%.3f); see %s",
            actual_job_id,
            report.summary.failed_blocks,
            report.summary.pass_rate,
            rendered_path,
        )
    report_path = sidecar_path(rendered_path, "quality_report.json")
    await asyncio.to_thread(save_quality_report, report, report_path)

    # Versioned KPI artifact: a pure derivation from the quality report plus
    # the optional visual report (no LLM, no text re-scan). Checked-in golden
    # baselines and the CI regression gate compare against it.
    artifact_report: dict[str, Any] | None = None
    if artifact_report_path is not None and artifact_report_path.exists():
        try:
            artifact_report = json.loads(artifact_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "KPI collection ignoring unreadable visual report %s: %s",
                artifact_report_path,
                exc,
            )
    metrics_path = sidecar_path(rendered_path, "metrics.json")
    kpis = await asyncio.to_thread(collect_kpis, report, artifact_report)
    await asyncio.to_thread(save_metrics_report, kpis, metrics_path)
    return report, report_path
