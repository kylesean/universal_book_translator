"""Hard integrity validation & render output with warning annotations."""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ubt.core.cleaners.cjk_spacing import normalize_publishing_cjk
from ubt.core.config import DualMode
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pe_queue import PEQueueResult, export_pe_queue
from ubt.core.engine.reporter import (
    QualityReport,
    ReportEntityConsistency,
    ReportTerminologyMetrics,
    build_quality_report,
    save_quality_report,
)
from ubt.core.engine.stage_context import StageContext
from ubt.core.exceptions import (
    IntegrityViolationError,
    JobInterruptedError,
    RenderBlocksNotImplementedError,
    UBTError,
)
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, IRBlock
from ubt.core.job_options import resolve_target_output, sidecar_path
from ubt.core.metrics.collect import collect_kpis, save_metrics_report
from ubt.core.policy.bilingual_advisor import SECONDARY_SUFFIX
from ubt.core.policy.layout_policy import LENGTH_OVERFLOW_TO_HUMAN, LENGTH_POLICY_PAGE_KINDS
from ubt.core.ports import (
    DocumentAdapter,
    blocking_gate_tripped,
    crashed_visual_gate_result,
    get_last_render_skips,
    is_pdf_engine_adapter,
)

if TYPE_CHECKING:
    from ubt.adapters.pdf.visual_gate import VisualGateResult
    from ubt.core.content.contract import ReconciliationReport
from ubt.core.qe.defect_taxonomy import (
    INTENTIONAL_PRESERVED_SKIP_PREFIXES as _INTENTIONAL_PRESERVED_SKIP_PREFIXES,
)
from ubt.core.qe.defect_taxonomy import UNTRANSLATED_PREFIX
from ubt.core.qe.term_metrics import evaluate_terms, summarize_drift
from ubt.core.router.pricing import cache_hit_rate_from_usage, estimate_cost_usd
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer
from ubt.core.validators.html_delta import HTMLDeltaValidator
from ubt.core.validators.math_guard import apply_math_guards

logger = logging.getLogger(__name__)


async def _render_adapter_output(
    adapter: DocumentAdapter,
    manifest: BookManifest,
    ledger: SQLiteJobLedger,
    blocks: list[IRBlock],
    target_lang: str,
    output_path: Path,
    job_id: str | None = None,
    bilingual_mode: str | None = None,
    render_engine: str | None = None,
) -> Path:
    """Render via ``render_blocks`` with fallback to ``render_output``.

    Standard adapters implement ``render_blocks(manifest, blocks, ...)``;
    external or third-party adapters implementing ``render_output(manifest, ledger, ...)``
    are invoked through the fallback branch.
    """
    if bilingual_mode is None and manifest and manifest.run:
        bilingual_mode = manifest.run.bilingual_mode
    if render_engine is None and manifest and manifest.run:
        render_engine = manifest.run.render_engine

    try:
        if is_pdf_engine_adapter(adapter):
            pdf_adapter = cast(Any, adapter)
            res = await pdf_adapter.render_blocks(
                manifest=manifest,
                blocks=blocks,
                target_lang=target_lang,
                output_path=output_path,
                bilingual_mode=bilingual_mode,
                render_engine=render_engine,
            )
            return cast(Path, res)
        return await adapter.render_blocks(
            manifest=manifest,
            blocks=blocks,
            target_lang=target_lang,
            output_path=output_path,
            bilingual_mode=bilingual_mode,
        )
    except RenderBlocksNotImplementedError:
        pass
    # Fallback path for adapters implementing render_output(manifest, ledger, ...).
    if is_pdf_engine_adapter(adapter):
        pdf_adapter = cast(Any, adapter)
        res = await pdf_adapter.render_output(
            manifest=manifest,
            ledger=ledger,
            target_lang=target_lang,
            output_path=output_path,
            job_id=job_id,
            bilingual_mode=bilingual_mode,
            render_engine=render_engine,
        )
        return cast(Path, res)
    return await adapter.render_output(
        manifest=manifest,
        ledger=ledger,
        target_lang=target_lang,
        output_path=output_path,
        job_id=job_id,
        bilingual_mode=bilingual_mode,
    )


def apply_render_skip_flags(
    blocks: list[IRBlock],
    skips: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Sync ``render_skip:{reason}`` error flags with this render's skips.

    Pure helper (unit-testable): maps the adapter's ``(block_id, reason)``
    side channel onto the finalized blocks. Unknown block ids are ignored
    and flags already present are not duplicated. Stale flags from a
    previous render of the same job are dropped first: a render-only rerun
    (same ledger, different engine result) can render a block that an older
    artifact skipped, and the kept flag would make the quality report
    describe an artifact that no longer exists. Legacy ledgers may carry the
    ``inplace_skip:`` spelling, so both prefixes are dropped. The
    quality report picks the flags up via ``defect_flags`` with no further
    wiring.
    """
    by_id = {b.id: b for b in blocks}
    wanted_by_block: dict[str, set[str]] = {}
    for block_id, reason in skips:
        wanted_by_block.setdefault(block_id, set()).add(f"render_skip:{reason}")
    for block in blocks:
        keep = wanted_by_block.get(block.id, set())
        block.error_flags = [
            flag
            for flag in block.error_flags
            if not flag.startswith(("inplace_skip:", "render_skip:")) or flag in keep
        ]
    checkpoints: list[dict[str, Any]] = []
    for block_id, reason in skips:
        fb = by_id.get(block_id)
        if fb is None:
            continue
        flag = f"render_skip:{reason}"
        if flag in fb.error_flags:
            continue
        fb.error_flags.append(flag)
        checkpoints.append(
            {
                "block_id": fb.id,
                "status": fb.status,
                "error_flags": fb.error_flags,
            }
        )
    return checkpoints


def apply_length_policy_flags(
    blocks: list[IRBlock],
    skipped_ids: set[str],
) -> int:
    """Flip overflow skips on length-policy pages to NEEDS_HUMAN.

    Pure helper (unit-testable): only blocks whose provenance ``page_kind``
    is in ``LENGTH_POLICY_PAGE_KINDS`` move; everything else keeps the
    flag-only behaviour. Returns the flipped count.
    """
    by_id = {b.id: b for b in blocks}
    flipped = 0
    for block_id in skipped_ids:
        fb = by_id.get(block_id)
        if fb is None or fb.skip_translate:
            continue
        provenance = fb.provenance if isinstance(fb.provenance, dict) else {}
        if str(provenance.get("page_kind", "")) not in LENGTH_POLICY_PAGE_KINDS:
            continue
        reason = next(
            (
                f.split(":", 1)[1]
                for f in fb.error_flags
                if f.startswith(("inplace_skip:", "render_skip:"))
            ),
            "overflow",
        )
        if not reason.startswith(("overflow", "spill")):
            continue
        flag = f"length_overflow:{reason}"
        if flag not in fb.error_flags:
            fb.error_flags.append(flag)
        if fb.status != BlockStatus.NEEDS_HUMAN:
            fb.status = BlockStatus.NEEDS_HUMAN
        flipped += 1
    return flipped


def _terminology_and_structure_pass(
    blocks: list[IRBlock],
    *,
    glossary_enforcer: DeterministicGlossaryEnforcer | None,
    glossary_validator: GlossaryConsistencyValidator,
    html_validator: HTMLDeltaValidator,
    target_lang: str,
) -> tuple[list[dict[str, Any]], int]:
    """Edit every final block once and return the checkpoints its edits changed.

    Runs in a worker thread: over a book's worth of blocks this is the longest
    synchronous stretch of the export, and the loop it would otherwise hold also
    serves SSE subscribers and other concurrent jobs. Safe to
    move because the blocks are this stage's own copies from
    ``ledger.get_all_blocks``, and nothing else reads them mid-pass.
    """
    modified_checkpoints: list[dict[str, Any]] = []
    enforced_spans = 0

    for fb in blocks:
        # Verbatim-skipped blocks (watermarks, bibliography, symbol
        # debris) ship byte-identical: glossary substitution inside them
        # creates Chinglish mash ("A. Vaswani et al. 注意力 Is All You
        # Need"), so both enforcement and validation skip them.
        if fb.target_text and not fb.skip_translate:
            # 0. Glossary enforcement (opt-in)
            # Establish canonical terminology first so subsequent publishing polish formats it properly.
            if glossary_enforcer:
                enforced_text, records = glossary_enforcer.enforce(fb.target_text)
                if records:
                    enforced_spans += len(records)
                    fb.target_text = enforced_text
                    modified_checkpoints.append(
                        {
                            "block_id": fb.id,
                            "target_text": fb.target_text,
                            "status": fb.status,
                            "error_flags": fb.error_flags,
                        }
                    )

            # 1. CJK spacing and publishing punctuation normalization
            spaced = normalize_publishing_cjk(fb.target_text, target_lang=target_lang)
            if spaced != fb.target_text:
                fb.target_text = spaced
                modified_checkpoints.append(
                    {
                        "block_id": fb.id,
                        "target_text": fb.target_text,
                        "status": fb.status,
                        "error_flags": fb.error_flags,
                    }
                )

            glossary_res = glossary_validator.validate(fb.source_text, fb.target_text)
            if not glossary_res.is_valid:
                flag = f"glossary_inconsistency: {glossary_res.message}"
                if flag not in fb.error_flags:
                    fb.error_flags.append(flag)
                    modified_checkpoints.append(
                        {
                            "block_id": fb.id,
                            "target_text": fb.target_text,
                            "status": fb.status,
                            "error_flags": fb.error_flags,
                        }
                    )

            # 2. Structural HTML delta validation (preserve draft with warning mark)
            html_res = html_validator.validate(fb.source_text, fb.target_text)
            if not html_res.is_valid:
                # Dedup like every other flag append here — without
                # it a resume/re-export double-appends the flag and nests the
                # <mark> warning wrap around an already-wrapped target.
                flag = f"html_structure_mismatch: {html_res.message}"
                if flag not in fb.error_flags:
                    fb.error_flags.append(flag)
                    clean_msg = html.escape((html_res.message or "").replace('"', "'"), quote=True)
                    fb.target_text = (
                        f'<mark class="ubt-failed-draft" title="html_structure_mismatch: {clean_msg}">'
                        f"{fb.target_text}</mark>"
                    )
                    # Keep the in-memory block in step with the checkpoint: the
                    # checkpoint records FAILED, so leaving ``fb.status`` at its
                    # prior value made the ledger and ``final_blocks`` disagree
                    # for the rest of the stage.
                    fb.status = BlockStatus.FAILED
                    modified_checkpoints.append(
                        {
                            "block_id": fb.id,
                            "target_text": fb.target_text,
                            "status": BlockStatus.FAILED,
                            "error_flags": fb.error_flags,
                        }
                    )
            else:
                # Advisory tier: emphasis-tag drift / dropped anchors are
                # normal in fluent translation, so they get a review flag and
                # a checkpoint -- never a <mark> wrap and never FAILED.
                for warning in html_res.details.get("formatting_warnings", []):
                    wflag = f"html_formatting_drift: {warning}"
                    if wflag not in fb.error_flags:
                        fb.error_flags.append(wflag)
                        modified_checkpoints.append(
                            {
                                "block_id": fb.id,
                                "target_text": fb.target_text,
                                "status": fb.status,
                                "error_flags": fb.error_flags,
                            }
                        )

    return modified_checkpoints, enforced_spans


async def _force_stale_blocks_terminal(
    ledger: SQLiteJobLedger, actual_job_id: str
) -> list[IRBlock]:
    """Fail every still-non-terminal block, then return the job's blocks.

    Any block in a non-terminal status here means a stage crashed mid-flight.
    Rendering them as-is would ship untranslated source while the job reports
    "completed"; forcing FAILED first surfaces them in stats, the quality
    report, and logs.
    """
    stale_block_ids = await asyncio.to_thread(
        ledger.fail_non_terminal_blocks,
        actual_job_id,
        f"{UNTRANSLATED_PREFIX} block was still non-terminal when export started",
    )
    if stale_block_ids:
        logger.warning(
            "Export forced %d stale block(s) to FAILED for job %s: %s",
            len(stale_block_ids),
            actual_job_id,
            ", ".join(stale_block_ids[:10]),
        )
    final_blocks: list[IRBlock] = await asyncio.to_thread(ledger.get_all_blocks, actual_job_id)
    return final_blocks


def _check_completion_ratio(
    actual_job_id: str, final_blocks: list[IRBlock], min_completion_ratio: float
) -> None:
    """Refuse to ship a mostly-untranslated book.

    The renderers fall back to ``source_text`` when a block carries no
    target, so a run in which most blocks failed would finalize as
    "completed" with an untranslated book. Refusing here keeps the ledger
    intact, so a resume retries exactly the blocks that have no target.

    ``BLOCKED_HUMAN`` blocks carry a non-empty ``<mark>`` placeholder wrapping
    the *source*, not a translation, so they must not count as delivered — an
    all-quarantined book would otherwise pass a completion floor of 1.0.

    The floor is measured over *translatable* blocks only. A ``skip_translate``
    block (formula, image, preserved marker) is rendered verbatim by design, so
    leaving it in the denominator let a mostly-skipped book clear the floor
    without a single translated sentence.
    """
    if min_completion_ratio <= 0 or not final_blocks:
        return

    translatable = [fb for fb in final_blocks if not fb.skip_translate]
    if not translatable:
        return

    untranslated = [
        fb.id
        for fb in translatable
        if not (fb.target_text or "")
        or fb.status in (BlockStatus.BLOCKED_HUMAN, BlockStatus.FAILED)
        or any(
            isinstance(f, str)
            and f.startswith(("render_skip:", "inplace_skip:"))
            and not f.startswith(_INTENTIONAL_PRESERVED_SKIP_PREFIXES)
            for f in fb.error_flags
        )
    ]
    completed_ratio = 1.0 - len(untranslated) / len(translatable)
    if completed_ratio < min_completion_ratio:
        raise IntegrityViolationError(
            f"Export blocked for job {actual_job_id}: only {completed_ratio:.0%} of "
            f"{len(translatable)} translatable block(s) carry a translation, below the "
            f"{min_completion_ratio:.0%} floor (UBT_EXPORT_MIN_COMPLETION_RATIO). "
            f"{len(untranslated)} block(s) have no target, e.g. "
            f"{', '.join(untranslated[:10])}. Resume the job to retry them, "
            "or lower the floor to ship the partial book knowingly."
        )


def _partition_render_skip_counts(
    skip_checkpoints: list[dict[str, Any]],
) -> tuple[int, int]:
    """Separate true fail-closed skips (spill, no_zone, math_unrenderable) from intentional preserved elements."""
    fail_closed = 0
    preserved = 0
    for cp in skip_checkpoints:
        flags = cp.get("error_flags") or []
        skip_flags = [
            f
            for f in flags
            if isinstance(f, str) and f.startswith(("render_skip:", "inplace_skip:"))
        ]
        if skip_flags and all(
            f.startswith(_INTENTIONAL_PRESERVED_SKIP_PREFIXES) for f in skip_flags
        ):
            preserved += 1
        else:
            fail_closed += 1
    return fail_closed, preserved


async def _apply_render_skip_ledger_pass(
    ctx: StageContext,
    adapter: DocumentAdapter,
    manifest: BookManifest,
    final_blocks: list[IRBlock],
) -> None:
    """Land fail-closed render skips (and length-policy flips) on the ledger.

    Skips collected by the adapter's side channel go on the blocks'
    error_flags BEFORE the quality report is built, so the KDP audit lists
    every source-visible segment. Secondary (complementary) renders are
    excluded: their skips describe the same blocks under a different
    artifact. Flag removal must reach the ledger too: the report re-reads
    blocks from it, and a render-only rerun can render what an older
    artifact skipped — a kept stale flag would make the report describe a
    file that no longer exists.
    """
    actual_job_id = ctx.job_id
    ledger = ctx.ledger
    stale_skips = {
        b.id: [f for f in b.error_flags if f.startswith(("inplace_skip:", "render_skip:"))]
        for b in final_blocks
    }
    render_skip_checkpoints = apply_render_skip_flags(final_blocks, get_last_render_skips(adapter))
    # Persist a block whose skip set *changed*, partial removals included: the
    # old filter only wrote when every skip flag was gone, so a block that kept
    # one current reason while losing a stale one stayed wrong in the ledger the
    # quality report re-reads.
    cleared_skips = [
        {"block_id": b.id, "status": b.status, "error_flags": b.error_flags}
        for b in final_blocks
        if any(f not in b.error_flags for f in stale_skips[b.id])
    ]
    # Length conservation: overflow skips on length-policy pages
    # (resume_dense/poster_fixed — fit-to-page, never repaginate) flip to
    # NEEDS_HUMAN so they reach the PE queue instead of shipping
    # source-visible with only an error flag. Statuses stay terminal either way.
    length_human = 0
    if LENGTH_OVERFLOW_TO_HUMAN and render_skip_checkpoints:
        skipped_ids = {c["block_id"] for c in render_skip_checkpoints}
        length_human = apply_length_policy_flags(final_blocks, skipped_ids)
        by_id = {b.id: b for b in final_blocks}
        for checkpoint in render_skip_checkpoints:
            flipped = by_id.get(checkpoint["block_id"])
            if flipped is not None:
                checkpoint["status"] = flipped.status
                checkpoint["error_flags"] = flipped.error_flags
    if length_human:
        logger.warning(
            "Job %s routed %d length-policy overflow block(s) to NEEDS_HUMAN",
            actual_job_id,
            length_human,
        )
    manifest.run.length_policy = {
        "overflow_to_human": length_human,
        "enforce": bool(LENGTH_OVERFLOW_TO_HUMAN),
    }
    if render_skip_checkpoints or cleared_skips:
        if render_skip_checkpoints:
            fail_closed_count, preserved_count = _partition_render_skip_counts(
                render_skip_checkpoints
            )
            if fail_closed_count > 0:
                logger.warning(
                    "Job %s rendered with %d fail-closed skipped block(s) left source-visible "
                    "(%d source element(s) intentionally preserved)",
                    actual_job_id,
                    fail_closed_count,
                    preserved_count,
                )
            elif preserved_count > 0:
                logger.info(
                    "Job %s preserved %d source element(s) intact (0 fail-closed skips)",
                    actual_job_id,
                    preserved_count,
                )
        await asyncio.to_thread(
            ledger.save_checkpoints_batch, render_skip_checkpoints + cleared_skips
        )


async def _run_visual_gate(
    ctx: StageContext,
    adapter: DocumentAdapter,
    final_blocks: list[IRBlock],
    rendered_path: Path,
) -> tuple[Path, Path | None, Any]:
    """Post-render visual self-healing gate; never fatal except on cancel.

    T0/T1 deterministic checks + optional pixel confirmation + sampled T2
    VLM + ReflowControlLoop, all driven by the config bound on ``ctx``.
    Returns the (possibly re-rendered) artifact, its visual report path, and
    the gate object (None when the gate is off or not a PDF).
    """
    gate: Any = None
    visual_report_path: Path | None = None
    if not ctx.config.visual_gate_enabled or rendered_path.suffix.lower() != ".pdf":
        return rendered_path, visual_report_path, gate
    try:
        from ubt.core.engine.reflow_loop import ReflowControlLoop

        reflow_loop = ReflowControlLoop(
            adapter=adapter,
            manifest=ctx.manifest,
            ledger=ctx.ledger,
            job_id=ctx.job_id,
            target_lang=ctx.target_lang,
            render_fn=_render_adapter_output,
            router=ctx.router,
            sample_pages=max(0, ctx.config.visual_sample_pages),
            max_vlm_pages=max(0, ctx.config.visual_max_vlm_pages),
            visual_judge_enabled=ctx.config.visual_judge_enabled,
            visual_judge_model=ctx.config.visual_judge_model,
            render_fidelity_enabled=ctx.config.render_fidelity_enabled,
            cancel_token=ctx.cancel_token,
        )
        rendered_path, visual_report_path, gate = await reflow_loop.run(
            rendered_path=rendered_path,
            blocks=final_blocks,
        )
        ctx.check_cancelled()
        bad = [f for f in gate.findings if getattr(f, "severity", "") in ("major", "critical")]
        if bad:
            logger.warning(
                "Visual gate for job %s: %d warning(s), see %s",
                ctx.job_id,
                len(bad),
                visual_report_path,
            )
        elif gate.skipped_reason:
            logger.info("Visual gate for job %s skipped: %s", ctx.job_id, gate.skipped_reason)
    except JobInterruptedError:
        # Cooperative cancellation (user cancel or lost lease) is raised from
        # inside reflow_loop.run and by the check below it; it must reach the
        # pipeline's handler, not be reported as a non-fatal gate hiccup that
        # then renders, writes the report and stamps the job "completed".
        raise
    except Exception as exc:
        logger.warning(
            "Visual gate self-healing failed (non-fatal) for job %s: %s", ctx.job_id, exc
        )
        # A crashed gate must read as failed, not absent: blocking enforcement
        # and the KPI collector both key off this object, and None silently
        # skipped every rejection layer while metrics recorded a perfect pass.
        gate = crashed_visual_gate_result(
            f"visual gate did not complete: {type(exc).__name__}: {exc}"[:300]
        )
        visual_report_path = await _persist_crashed_gate_report(ctx, rendered_path, gate, exc)
    return rendered_path, visual_report_path, gate


async def _persist_crashed_gate_report(
    ctx: StageContext,
    rendered_path: Path,
    gate: VisualGateResult,
    exc: Exception,
) -> Path | None:
    """Write the crash gate where a live run would have written its report.

    The sidecar, the ledger record and the KPI collector all read this state;
    best-effort only, the caller is already inside the failure path.
    """
    try:
        from ubt.core.log_aggregate import noise_report

        payload: dict[str, Any] = {
            **gate.report_payload(),
            "self_healed": False,
            "healing_strategy": "none",
            "healing_skipped_reason": f"visual gate crashed: {type(exc).__name__}: {exc}"[:300],
            "parse_noise": noise_report(),
        }
        visual_report_path = sidecar_path(rendered_path, "visual_report.json")
        await asyncio.to_thread(
            visual_report_path.write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        await asyncio.to_thread(ctx.ledger.record_visual_report, ctx.job_id, payload)
        return visual_report_path
    except Exception as persist_exc:
        logger.debug(
            "crashed-gate report persistence failed for job %s: %s", ctx.job_id, persist_exc
        )
        return None


def _drop_stale_run_reports(rendered_path: Path) -> None:
    """Delete the previous run's derived reports sitting beside a new deliverable.

    The names come from ``rendered_path.stem`` exactly as the writers build them
    (:func:`_build_reports` in this module, ``artifact_and_report_paths`` on the
    read side), so what gets cleared is precisely what would otherwise be read
    back. The deliverable itself is never touched, and a missing file is not an
    error.
    """
    for kind in ("quality_report.json", "quality_report.md", "metrics.json", "visual_report.json"):
        with contextlib.suppress(OSError):
            sidecar_path(rendered_path, kind).unlink(missing_ok=True)


def _enforce_blocking_gate(
    actual_job_id: str,
    gate: Any,
    visual_report_path: Path | None,
    blocking_enabled: bool,
    *,
    rehearsal: bool = False,
) -> None:
    """Refuse export on CRITICAL visual findings (opt-in), after persistence.

    Runs after the quality report is written so a refusal still leaves the
    ``*_quality_report.json`` / ``*_metrics.json`` audit trail beside the
    deliverable for diagnostic inspection.
    An unreadable rendered PDF refuses here too: the gate reports
    the pdf_page_count -1 sentinel and ``blocking_gate_tripped`` fails
    closed on it even while the opt-in blocking gate is off.

    ``rehearsal=True`` (mock/dry-run provider) downgrades the refusal to a
    warning: a rehearsal echoes source text, so ``target_language_sparse`` is
    guaranteed to fire and every ``--dry-run`` would exit 1 without ever
    proving the render path. Real runs never pass
    this flag.
    """
    if gate is None:
        return
    total_pages = 0
    stats = getattr(gate, "stats", None)
    if isinstance(stats, dict):
        try:
            total_pages = int(stats.get("total_pages", 0))
        except (TypeError, ValueError):
            total_pages = 0
    tripped = blocking_gate_tripped(getattr(gate, "findings", []), total_pages, blocking_enabled)
    if not tripped:
        return
    codes = sorted({str(getattr(f, "code", "defect")) for f in tripped})
    pages = sorted(
        {page for page in (getattr(f, "page", None) for f in tripped) if isinstance(page, int)}
    )
    if rehearsal:
        logger.warning(
            "Rehearsal run: visual blocking gate would have refused job %s "
            "(%d finding(s), codes: %s); downgraded to warning because a mock "
            "draft echoes source text by design.",
            actual_job_id,
            len(tripped),
            ", ".join(codes),
        )
        return
    raise UBTError(
        f"Visual blocking gate refused export for job {actual_job_id}: "
        f"{len(tripped)} blocking finding(s) (severity: {', '.join(sorted({str(getattr(f, 'severity', 'critical')) for f in tripped}))}) "
        f"on page(s) {pages or 'document level'} "
        f"(codes: {', '.join(codes)}). See {visual_report_path}."
    )


def _enforce_syntax_fallback_gate(
    actual_job_id: str,
    syntax_fallbacks: list[str] | tuple[str, ...] | None,
    config: Any,
    *,
    rehearsal: bool = False,
) -> None:
    """Refuse export when Typst self-healing removed too much translation.

    Runs after the quality report is on disk (same placement as the visual
    gate) so a refusal keeps its audit trail. ``export_max_syntax_fallbacks``
    is the ceiling; 0 fails on any removal. Rehearsal downgrades to warning.
    """
    fallbacks = list(syntax_fallbacks or [])
    if not fallbacks:
        return
    max_allowed = int(getattr(config, "export_max_syntax_fallbacks", 5))
    if len(fallbacks) <= max_allowed:
        return
    preview = "; ".join(fallbacks[:3])
    if rehearsal:
        logger.warning(
            "Rehearsal run: syntax-fallback gate would have refused job %s "
            "(%d removed line(s) > max %d, e.g. %s); downgraded to warning.",
            actual_job_id,
            len(fallbacks),
            max_allowed,
            preview,
        )
        return
    raise UBTError(
        f"Typst syntax-fallback gate refused export for job {actual_job_id}: "
        f"{len(fallbacks)} translated line(s) commented out (max {max_allowed}), "
        f"e.g. {preview}. See quality_report.syntax_fallbacks."
    )


async def _render_complementary_artifact(
    ctx: StageContext,
    adapter: DocumentAdapter,
    final_blocks: list[IRBlock],
    target_output: Path,
    rendered_path: Path | None = None,
) -> Path | None:
    """Render the requested complementary dual-mode or secondary-engine PDF, or None.

    Same translated blocks, no extra LLM cost; PDF only (other adapters
    have no render-mode override plumbing), and never fatal: a failed
    secondary render must not sink the delivered primary artifact.
    """
    manifest = ctx.manifest
    secondary_render = str(manifest.run.emit_secondary_mode or "")
    secondary_engine = str(manifest.run.emit_secondary_engine or "")
    secondary_path: Path | None = None
    # Companion renders are intentional second passes; the forced-engine
    # warning ("...auto dispatch would route...") is for a user-forced *primary*
    # render, so suppress it while a companion is produced.
    if isinstance(getattr(manifest, "metadata", None), dict):
        manifest.metadata["suppress_render_engine_warning"] = True
    if secondary_render and str(manifest.run.render_engine_effective or "") == "rigid":
        # The rigid engine is monolingual: a complementary mode would
        # produce a byte-identical mono artifact. Record the skip instead.
        logger.info("Complementary dual render skipped: 'rigid' is monolingual")
        secondary_render = ""
    if secondary_render:
        primary_mode = str(manifest.run.effective_dual_mode or "inline")
        suffix = SECONDARY_SUFFIX.get(cast(DualMode, primary_mode), "_secondary")
        ext = target_output.suffix
        candidate = target_output.with_name(f"{target_output.stem}{suffix}{ext}")
        try:
            secondary_path = await _render_adapter_output(
                adapter=adapter,
                manifest=manifest,
                ledger=ctx.ledger,
                blocks=final_blocks,
                target_lang=ctx.target_lang,
                output_path=candidate,
                job_id=ctx.job_id,
                bilingual_mode=secondary_render,
            )
            manifest.run.companion_output_path = str(secondary_path)
            logger.info("Dual output rendered complementary artifact: %s", secondary_path)
        except Exception as exc:
            logger.warning("Complementary dual render failed (non-fatal): %s", exc)
            secondary_path = None
    elif (
        secondary_engine == "rigid"
        and str(manifest.run.render_engine_effective or "") != "rigid"
        and is_pdf_engine_adapter(adapter)
        and target_output.suffix.lower() == ".pdf"
    ):
        candidate = target_output.with_name(f"{target_output.stem}_rigid.pdf")
        saved_effective = manifest.run.render_engine_effective
        saved_meta_effective = manifest.metadata.get("render_engine_effective")
        saved_skips = list(getattr(adapter, "last_render_skips", ()))
        try:
            secondary_path = await _render_adapter_output(
                adapter=adapter,
                manifest=manifest,
                ledger=ctx.ledger,
                blocks=final_blocks,
                target_lang=ctx.target_lang,
                output_path=candidate,
                job_id=ctx.job_id,
                bilingual_mode="monolingual",
                render_engine="rigid",
            )
            manifest.run.companion_output_path = str(secondary_path)
            logger.info(
                "Zero-cost companion rigid PDF rendered alongside forced reflow artifact: %s",
                secondary_path,
            )
        except Exception as exc:
            logger.warning("Companion rigid render failed (non-fatal): %s", exc)
            secondary_path = None
        finally:
            manifest.run.render_engine_effective = saved_effective
            if saved_meta_effective is not None:
                manifest.metadata["render_engine_effective"] = saved_meta_effective
            elif "render_engine_effective" in manifest.metadata:
                del manifest.metadata["render_engine_effective"]
            if hasattr(adapter, "last_render_skips"):
                adapter.last_render_skips = saved_skips
    elif (
        secondary_engine == "rigid_bilingual"
        and str(manifest.run.render_engine_effective or "") == "rigid"
        and target_output.suffix.lower() == ".pdf"
        and ctx.source_pdf_path is not None
        and rendered_path is not None
    ):
        # Auto-routed rigid: the primary is a monolingual overlay. The bilingual
        # companion interleaves the SOURCE pages with the rigid TARGET pages, so
        # the reader gets 1:1 fidelity and the translation side by side without a
        # second re-typeset (fidelity and bilingual are no longer a tradeoff). An
        # auto-named primary is ``<stem>_mono``; restore the canonical
        # ``<stem>_bilingual`` name for the companion.
        stem = target_output.stem
        base = stem[: -len("_mono")] if stem.endswith("_mono") else stem
        candidate = target_output.with_name(f"{base}_bilingual{target_output.suffix}")
        try:
            from ubt.adapters.pdf.alternator import BilingualAlternator

            result = await BilingualAlternator().interleave_pages_async(
                source_pdf=ctx.source_pdf_path,
                translated_pdf=rendered_path,
                output_pdf=candidate,
                facing_spread=bool(manifest.run.facing_spread),
            )
            secondary_path = Path(result.output_path)
            manifest.run.companion_output_path = str(secondary_path)
            logger.info(
                "Zero-cost rigid bilingual companion (source + target pages) rendered: %s",
                secondary_path,
            )
        except Exception as exc:
            logger.warning("Rigid bilingual interleave failed (non-fatal): %s", exc)
            secondary_path = None
    manifest.metadata.pop("suppress_render_engine_warning", None)
    return secondary_path


def _reconcile_delivery_contract(
    ctx: StageContext, blocks: list[IRBlock], rendered_path: Path
) -> ReconciliationReport:
    """Build the content graph, reconcile the two ledgers, persist the contract.

    Always writes the standalone ``*_contract.json`` and returns the report. The
    report is advisory unless ``config.strict_contract`` is set, in which case
    the caller aborts on an ERROR-severity violation; ``ubt verify`` applies the
    same contract to a delivered artifact or a finished job's ledger.
    """
    from ubt.core.content import graph_from_blocks, reconcile

    manifest = ctx.manifest
    engine = str(
        getattr(manifest.run, "render_engine_effective", "")
        or ctx.config.render_engine
        or "publication"
    )
    graph = graph_from_blocks(
        blocks,
        engine=engine,
        doc_id=str(getattr(manifest, "doc_id", "") or ""),
        title=str(getattr(manifest, "title", "") or ""),
        source_path=str(getattr(manifest, "source_path", "") or ""),
        witness_findings=[
            str(item) for item in (manifest.metadata.get("formula_witness_findings") or []) if item
        ],
        table_fallbacks=[
            str(item) for item in (manifest.metadata.get("table_fallback_findings") or []) if item
        ],
    )
    contract = reconcile(graph)
    payload = contract.model_dump(mode="json")
    contract_path = sidecar_path(rendered_path, "contract.json")
    try:
        contract_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:  # a missing audit file must not sink the artifact
        logger.warning("Could not write delivery contract for %s: %s", ctx.job_id, exc)
        return contract
    if contract.passed:
        logger.info("Delivery contract %s: %s", contract_path.name, contract.summary_line())
    else:
        logger.warning(
            "Delivery contract FAILED for job %s: %s (%d error(s)); see %s",
            ctx.job_id,
            contract.summary_line(),
            len(contract.errors),
            contract_path,
        )
    return contract


def _write_xliff_companion(
    ctx: StageContext, rendered_path: Path, blocks: list[IRBlock]
) -> Path | None:
    """Write a bilingual XLIFF beside the artifact for review (best-effort).

    Read-only by construction: it serializes the *delivered* blocks -- each
    source with its protected spans masked into inline codes, and the target that
    actually shipped -- so an editor opens source and target side by side in any
    CAT tool or XLIFF viewer. It never touches the rendered artifact, so any
    failure is logged and skipped rather than allowed to sink the delivery.
    """
    if not ctx.config.emit_xliff_companion:
        return None
    try:
        from ubt.core.cleaners.citation_masker import CitationMasker
        from ubt.core.cleaners.code_masker import CodeMasker
        from ubt.core.cleaners.math_masker import MathMasker
        from ubt.core.cleaners.soup_math import SoupMathMasker
        from ubt.core.job_options import companion_path
        from ubt.segment.document import segments_from_blocks
        from ubt.segment.placeholders import PlaceholderEngine
        from ubt.segment.xliff import to_xliff

        engine = PlaceholderEngine(
            code=CodeMasker(),
            math=MathMasker(),
            soup=SoupMathMasker(),
            citation=CitationMasker(),
        )
        segments = segments_from_blocks(blocks, engine=engine)
        if not segments:
            return None
        xml = to_xliff(
            segments,
            src_lang=ctx.source_lang or "en",
            trg_lang=ctx.target_lang,
            original=Path(ctx.input_path).name,
        )
        path = companion_path(rendered_path, ".xliff")
        path.write_text(xml, encoding="utf-8")
        logger.info(
            "XLIFF companion for job %s: %s (%d segment(s))",
            ctx.job_id,
            path.name,
            len(segments),
        )
        return path
    except Exception as exc:  # a companion must never sink the delivery
        logger.warning("XLIFF companion skipped for job %s: %s", ctx.job_id, exc)
        return None


def _attestation_payload(report: Any) -> dict[str, Any]:
    """Serialize an :class:`~ubt.pipeline.attest.AttestationReport` for the shadow."""
    return {
        "total": report.total,
        "text": dict(report.text),
        "assets": dict(report.assets),
        "violations": list(report.violations),
        "summary": report.summary_line(),
    }


def _write_attestation_shadow(
    ctx: StageContext, rendered_path: Path, blocks: list[IRBlock]
) -> Path | None:
    """Write the realize()-based attestation shadow beside the artifact (best-effort).

    Migration shadow (ADR-0001 Phase 3): the content-graph contract still decides
    delivery; this records the per-element attestations the ADR will replace it
    with, built from the *same* delivered blocks, so the two can be compared on
    real deliveries. Read-only and off-loop; a failure is logged and skipped and
    can never sink the delivery.
    """
    if not ctx.config.emit_attestation_shadow:
        return None
    try:
        from ubt.analyze.bridge import document_from_blocks
        from ubt.core.content.adapt import _all_intentional, _skip_flags
        from ubt.core.job_options import companion_path
        from ubt.pipeline.attest import attest_document
        from ubt.render.typst_backend import TypstBackend
        from ubt.verify.verifier import build_verifiers

        def _kept(block: IRBlock) -> bool:
            """Deliberately kept in the source: no delivered realization to attest.

            ``skip_translate`` and an intentional render-skip (page chrome) both
            record VERBATIM in the contract. A rigid run places assets opaque
            instead of reconstructing them, so their markup is not a realization
            either -- the engine is the whole-document choice the per-element
            backends replace, and it drives the backend here.
            """
            if block.block_type in (BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE):
                return engine == "rigid"
            return block.skip_translate or _all_intentional(_skip_flags(block))

        engine = str(
            getattr(ctx.manifest.run, "render_engine_effective", "")
            or ctx.config.render_engine
            or "publication"
        )
        document = document_from_blocks(
            blocks, doc_id=str(getattr(ctx.manifest, "doc_id", "") or "")
        )
        backend = TypstBackend(
            translations={
                block.id: block.target_text
                for block in blocks
                if block.target_text and not _kept(block)
            }
        )
        report = attest_document(document, backend, build_verifiers(ctx.fast_pass))
        path = companion_path(rendered_path, "_attestations.json")
        path.write_text(
            json.dumps(_attestation_payload(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Attestation shadow for job %s: %s", ctx.job_id, report.summary_line())
        return path
    except Exception as exc:  # a migration shadow must never sink the delivery
        logger.warning("Attestation shadow skipped for job %s: %s", ctx.job_id, exc)
        return None


async def _build_reports(
    ctx: StageContext,
    final_blocks: list[IRBlock],
    *,
    rendered_path: Path,
    visual_report_path: Path | None,
    glossary_dicts: list[dict[str, Any]],
    enforced_spans: int,
    run_usage: Any,
    delivery_contract: dict[str, Any] | None = None,
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
            endpoint_map=ctx.config.remote_billing_models(),
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
    await asyncio.to_thread(
        save_quality_report, report, report_path, write_markdown=ctx.config.kdp_audit_markdown
    )

    # Versioned KPI artifact: a pure derivation from the quality report plus
    # the optional visual report (no LLM, no text re-scan). Checked-in golden
    # baselines and the CI regression gate compare against it.
    visual_report: dict[str, Any] | None = None
    if visual_report_path is not None and visual_report_path.exists():
        try:
            visual_report = json.loads(visual_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "KPI collection ignoring unreadable visual report %s: %s",
                visual_report_path,
                exc,
            )
    metrics_path = sidecar_path(rendered_path, "metrics.json")
    kpis = await asyncio.to_thread(collect_kpis, report, visual_report)
    await asyncio.to_thread(save_metrics_report, kpis, metrics_path)
    return report, report_path


async def run_export_stage(
    ctx: StageContext,
) -> AsyncIterator[TranslationProgressEvent]:
    """Perform integrity checks, apply glossary enforcement (opt-in), and render output document."""
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    # NOTE: the run's usage snapshot is deliberately taken at the very end of
    # this stage, not here. The post-render visual gate below makes its own paid
    # VLM calls; capturing before it left that spend out of the quality report,
    # so the report disagreed with the bill the job carries. The pipeline lets
    # the final event through the budget check on purpose ("so the artifact and
    # report still say what the job spent") — which only holds if the report is
    # built from the post-gate figure.
    manifest = ctx.manifest
    adapter = ctx.require_adapter()
    output_path = ctx.output_path
    input_path = ctx.input_path
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    glossary_dicts = ctx.glossary_dicts
    html_validator = ctx.html_validator
    create_event_fn = ctx.create_event
    pe_queue_enabled = ctx.config.pe_queue_enabled
    pe_export_format = ctx.config.pe_export_format
    # Read the short-chain-derived policy from its single owner (AdaptivePolicy,
    # also what manifest.run.adaptive_policy reports) instead of re-deriving
    # from ctx.short_chain here — a second derivation could silently drift from
    # the value the report already published.
    visual_blocking_gate_enabled = ctx.adaptive_policy.visual_blocking
    min_completion_ratio = ctx.config.export_min_completion_ratio
    deterministic_glossary_enforce = ctx.adaptive_policy.deterministic_glossary
    ctx.check_cancelled()
    final_blocks = await _force_stale_blocks_terminal(ledger, actual_job_id)
    _check_completion_ratio(actual_job_id, final_blocks, min_completion_ratio)

    glossary_enforcer = (
        DeterministicGlossaryEnforcer(
            glossary=glossary_dicts,
            target_lang=target_lang,
            source_lang=source_lang,
        )
        if deterministic_glossary_enforce
        else None
    )
    glossary_validator = GlossaryConsistencyValidator(glossary=glossary_dicts)
    modified_checkpoints, enforced_spans = await asyncio.to_thread(
        _terminology_and_structure_pass,
        final_blocks,
        glossary_enforcer=glossary_enforcer,
        glossary_validator=glossary_validator,
        html_validator=html_validator,
        target_lang=target_lang,
    )

    if modified_checkpoints:
        await asyncio.to_thread(ledger.save_checkpoints_batch, modified_checkpoints)

    # Gate 3 (math closed loop, CAT translate="no" + QA): formula targets must
    # be source LaTeX verbatim; FAILED math-debris blocks render source instead
    # of hallucinated drafts. Counted + flagged, statuses untouched.
    math_checkpoints, math_counts = await asyncio.to_thread(apply_math_guards, final_blocks)
    if math_checkpoints:
        await asyncio.to_thread(ledger.save_checkpoints_batch, math_checkpoints)
    if (
        math_counts["formula_invariant_repairs"]
        or math_counts["math_debris_fallbacks"]
        or math_counts["c_text_accepted"]
    ):
        logger.warning(
            "Math guard for job %s: formula_invariant_repairs=%d "
            "math_debris_fallbacks=%d c_text_accepted=%d (formula pages should "
            "use docling; see pdf_parser_engine in job manifest)",
            actual_job_id,
            math_counts["formula_invariant_repairs"],
            math_counts["math_debris_fallbacks"],
            math_counts["c_text_accepted"],
        )

    # Default output is isolated under tmp/output/ so a forgotten -o
    # never pollutes the input directory (or docs/). All derived artifacts
    # (secondary render, quality/visual reports, .typ sidecar, PE queue)
    # hang off target_output and follow it automatically. A monolingual
    # (rigid) primary gets the honest ``_mono`` default name; an explicit -o is
    # respected as given.
    is_monolingual_output = (
        str(manifest.run.effective_dual_mode or manifest.run.bilingual_mode or "") == "monolingual"
    )
    target_output = resolve_target_output(
        output_path, input_path, monolingual=is_monolingual_output
    )
    target_output.parent.mkdir(parents=True, exist_ok=True)

    ctx.check_cancelled()
    effective_bilingual_mode = manifest.run.bilingual_mode if manifest else None
    rendered_path = await _render_adapter_output(
        adapter=adapter,
        manifest=manifest,
        ledger=ledger,
        blocks=final_blocks,
        target_lang=target_lang,
        output_path=target_output,
        job_id=actual_job_id,
        bilingual_mode=effective_bilingual_mode,
    )

    # Render skip pass-through + length conservation, before the report is
    # built (see _apply_render_skip_ledger_pass).
    await _apply_render_skip_ledger_pass(ctx, adapter, manifest, final_blocks)
    _check_completion_ratio(actual_job_id, final_blocks, min_completion_ratio)

    # Clean up any stale quality/metrics reports from prior runs so that
    # the persisted report files always reflect the current render.
    _drop_stale_run_reports(rendered_path)

    # Delivery contract: reconcile the content and asset ledgers for the primary
    # artifact. Written beside it and embedded in the quality report; an opt-in
    # hard gate (UBT_STRICT_CONTRACT) aborts a knowingly-broken delivery.
    contract = _reconcile_delivery_contract(ctx, final_blocks, rendered_path)
    if ctx.config.strict_contract and not contract.passed:
        raise IntegrityViolationError(
            f"Export blocked for job {ctx.job_id}: delivery contract failed with "
            f"{len(contract.errors)} error(s) - {contract.summary_line()}. The "
            "content/asset ledgers do not balance (dropped text or lost assets). "
            "Fix the loss, or unset UBT_STRICT_CONTRACT to ship knowingly."
        )
    delivery_contract = contract.model_dump(mode="json")

    # Bilingual XLIFF companion (source + delivered target). Read-only,
    # best-effort: it cannot affect the artifact, only add a file beside it.
    await asyncio.to_thread(_write_xliff_companion, ctx, rendered_path, final_blocks)

    # realize()-based attestation shadow (migration, ADR-0001 Phase 3): the same
    # delivered blocks, accounted for per element. Read-only, best-effort.
    await asyncio.to_thread(_write_attestation_shadow, ctx, rendered_path, final_blocks)

    # Post-render visual gate (self-healing loop): T0/T1 deterministic +
    # optional pixel confirmation + sampled T2 VLM + ReflowControlLoop.
    # Enforcement is deferred until after _build_reports (below): a refusal
    # must still leave the quality/metrics reports on disk for audit.
    rendered_path, visual_report_path, gate = await _run_visual_gate(
        ctx, adapter, final_blocks, rendered_path
    )

    # Human PE (HITL) queue export: NEEDS_HUMAN /
    # BLOCKED_HUMAN segments go to CSV / XLIFF 2.1 for human post-editing.
    # BLOCKED_HUMAN drafts were already quarantined by the triage stage (the
    # rendered target holds a source-only placeholder), so the machine output
    # of Critical blocks never ships in the document itself.
    # Runs after the render/visual-gate/reflow section so blocks quarantined
    # as NEEDS_HUMAN in this run reach the human queue.
    pe_result: PEQueueResult | None = None
    if pe_queue_enabled:
        pe_result = await asyncio.to_thread(
            export_pe_queue,
            final_blocks,
            target_output,
            pe_export_format,
            source_lang,
            target_lang,
            actual_job_id,
        )
        if pe_result is not None:
            logger.info(
                "PE queue exported for job %s: %s (%d segment(s), %s)",
                actual_job_id,
                pe_result.path,
                pe_result.segment_count,
                pe_result.fmt,
            )

    # Dual output (BabelDOC no-dual/no-mono style): when the pipeline asked
    # for a complementary artifact, render it from the same translated
    # blocks (no extra LLM cost). PDF only — other adapters have no
    # render-mode override plumbing.
    secondary_path = await _render_complementary_artifact(
        ctx, adapter, final_blocks, target_output, rendered_path
    )

    # Persist this run's spend to the ledger before the report reads it: the
    # visual gate and complementary render above made paid calls after the last
    # progress event, so the ledger's lifetime figure would otherwise omit them
    # (the terminal event is the one the budget check exempts).
    await ctx.bill_run_usage()
    run_usage = ctx.measure_run_usage()
    report, report_path = await _build_reports(
        ctx,
        final_blocks,
        rendered_path=rendered_path,
        visual_report_path=visual_report_path,
        glossary_dicts=glossary_dicts,
        enforced_spans=enforced_spans,
        run_usage=run_usage,
        delivery_contract=delivery_contract,
    )

    # Enforce AFTER the reports are on disk so a refused job keeps its audit
    # trail, and downgrade to a warning for rehearsal (mock) runs — a mock
    # draft echoes source text, so the sparse-target finding would fail every
    # --dry-run before it could prove the render path.
    _enforce_blocking_gate(
        actual_job_id,
        gate,
        visual_report_path,
        visual_blocking_gate_enabled,
        rehearsal=ctx.is_mock_run,
    )
    _enforce_syntax_fallback_gate(
        actual_job_id,
        report.syntax_fallbacks,
        ctx.config,
        rehearsal=ctx.is_mock_run,
    )

    await asyncio.to_thread(ledger.finalize_job, actual_job_id, status="completed")

    pe_suffix = (
        f" | PE queue: {pe_result.path} ({pe_result.segment_count} segment(s))"
        if pe_result is not None
        else ""
    )
    dual_suffix = f" | Complementary artifact: {secondary_path}" if secondary_path else ""
    is_bilingual = (
        getattr(getattr(ctx, "manifest", None), "run", None) is not None
        and getattr(ctx.manifest.run, "effective_dual_mode", None) not in ("monolingual", None)
        and getattr(ctx.manifest.run, "bilingual_mode", None) != "monolingual"
    )
    doc_label = "Bilingual document" if is_bilingual else "Translated document"
    event = await create_event_fn(
        EventType.EXPORT_COMPLETED,
        actual_job_id,
        ledger,
        message=f"{doc_label} rendered: {rendered_path}{pe_suffix}{dual_suffix}",
        artifact_path=str(rendered_path),
    )

    yield event
