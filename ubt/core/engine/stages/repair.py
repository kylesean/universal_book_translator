"""Stage 5: Targeted Repair Loop with Option B Preserved Draft Fallbacks."""

import asyncio
import html
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Scoring, Terminology
from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext, check_stage_cancelled
from ubt.core.exceptions import (
    BudgetExceededError,
    JobInterruptedError,
    LedgerWriterLeaseLostError,
)
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.memory.glossary_table import (
    build_chunk_glossary_table,
    build_global_glossary_table,
)
from ubt.core.qe.defect_taxonomy import REPAIR_ERROR_PREFIX
from ubt.core.qe.error_summary import summarize_provider_error
from ubt.core.qe.score_policy import is_shippable

logger = logging.getLogger(__name__)


def failed_draft_checkpoint(cand: IRBlock, res: BaseException) -> dict[str, Any]:
    """The checkpoint written when a repair call fails for one block.

    Option B: preserve the draft/target with a failure notice instead of
    replacing it with the bare source.

    The ``title`` keeps a one-line summary, not the provider's whole JSON
    envelope. The envelope was ~700 escaped characters on the reference job, and
    the review workbench renders ``target_text`` as plain text — so the attribute
    became content and broke out of a 446px column as an 828px line. The full
    detail is already on the ledger; this attribute exists to tell a human what
    happened.

    Module level, not a closure: as a closure it ran only on a live provider
    failure, so nothing in the fast tier exercised it and the coverage gate
    flagged the module it calls.
    """
    fallback_text = cand.target_text or cand.draft_text or ""
    summary = html.escape(summarize_provider_error(res), quote=True)
    target_str = (
        f'<mark class="ubt-failed-draft" title="{summary}">{fallback_text}</mark>'
        if fallback_text
        else cand.source_text
    )
    return {
        "block_id": cand.id,
        "target_text": target_str,
        "status": BlockStatus.FAILED,
        "mtqe_score": cand.mtqe_score,
        "repair_rounds": cand.repair_rounds,
        # The ledger keeps the full error: only the rendered notice is shortened.
        "error_flags": [*cand.error_flags, f"{REPAIR_ERROR_PREFIX} {res}"],
    }


async def run_repair_stage(
    ctx: StageContext,
    services: RunServices,
    terminology: Terminology,
    scoring: Scoring,
    defer_unresolved_to_triage: bool = True,
    chapter_id: str | None = None,
) -> AsyncGenerator[TranslationProgressEvent, None]:
    """Execute iterative surgical repairs on candidate blocks with Option B draft retention."""
    check_stage_cancelled(ctx)
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    # Short chain caps at 1 round: the deterministic glossary enforcer at export
    # fixes terminology drift without a second LLM pass.
    max_repair_rounds = (
        min(ctx.config.max_repair_rounds, 1) if ctx.short_chain else ctx.config.max_repair_rounds
    )
    # Production always defers: Critical/Major defects are classified by MQM
    # triage rather than force-finalized here. The flag survives as an argument
    # only because the fallback path (finalize as FAILED) is still reachable and
    # still tested.
    repair_loop = scoring.repair_loop or services.repair_loop
    glossary_dicts = terminology.glossary_dicts
    abbreviation_entries = terminology.abbreviation_entries
    global_glossary_table = build_global_glossary_table(
        glossary_dicts or [], getattr(ctx.config, "glossary_max_global_entries", 80)
    )
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    fast_pass = services.fast_pass
    concurrency_sem = services.concurrency_sem
    create_event_fn = ctx.create_event
    source_pdf_path = ctx.source_pdf_path

    resolved_pdf_path: Path | None = source_pdf_path
    if resolved_pdf_path is None:
        try:
            snapshot = await asyncio.to_thread(ledger.get_job_snapshot, actual_job_id)
            if snapshot and snapshot.get("source_path"):
                p = Path(str(snapshot["source_path"]))
                if p.suffix.lower() == ".pdf" and p.exists():
                    resolved_pdf_path = p
        except Exception as exc:
            # No resolvable source PDF only degrades the visual evidence a
            # repair round can attach; say why instead of failing silently.
            logger.warning(
                "Could not resolve the source PDF snapshot for %s; "
                "visual repair evidence is unavailable: %s",
                actual_job_id,
                exc,
            )

    for round_idx in range(max_repair_rounds):
        check_stage_cancelled(ctx)
        repair_eligible = await asyncio.to_thread(
            ledger.fetch_repair_eligible_blocks,
            actual_job_id,
            max_repair_rounds,
            chapter_id=chapter_id,
        )
        repair_candidates = repair_loop.select_repair_candidates(repair_eligible)
        if not repair_candidates:
            break

        event = await create_event_fn(
            EventType.REPAIR_STARTED,
            actual_job_id,
            ledger,
            message=f"Starting targeted repair round {round_idx + 1} for {len(repair_candidates)} candidate blocks",
        )
        yield event

        async def _repair_single_candidate(cand: IRBlock) -> dict[str, Any]:
            glossary_table = build_chunk_glossary_table(
                glossary_dicts, abbreviation_entries, cand.source_text
            )

            async with concurrency_sem:
                repaired = await repair_loop.repair_single_block(
                    block=cand,
                    glossary_table=glossary_table,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    fast_pass=fast_pass,
                    glossary_entries=glossary_dicts,
                    source_pdf_path=resolved_pdf_path,
                    global_glossary=global_glossary_table,
                )
            return {
                "block_id": repaired.id,
                "target_text": repaired.target_text or "",
                "status": repaired.status,
                "mtqe_score": repaired.mtqe_score,
                "repair_rounds": repaired.repair_rounds,
                "error_flags": repaired.error_flags,
            }

        def _record_failure(cand: IRBlock, res: BaseException) -> dict[str, Any]:
            return failed_draft_checkpoint(cand, res)

        flusher = CheckpointBatchFlusher(
            ledger=ledger,
            flush_interval=float(getattr(ctx.config, "ledger_flush_interval", 0.25)),
            max_batch_size=int(getattr(ctx.config, "ledger_flush_batch_size", 50)),
        )

        async def _repair_and_record(
            cand: IRBlock, flusher: CheckpointBatchFlusher = flusher
        ) -> None:
            """Persist each candidate the moment its repair returns.

            Recording per candidate ensures that cancellations or failures during
            a repair round preserve work that has already succeeded, preventing
            duplicate processing and redundant LLM spend on resumption.
            """
            try:
                update = await _repair_single_candidate(cand)
            except (asyncio.CancelledError, BudgetExceededError, JobInterruptedError):
                # A cancel/budget-cap is not a failed repair; laundering it into a FAILED
                # checkpoint would strand the block behind a defect marker and swallow hard stops.
                raise
            except BaseException as res:  # mirrors return_exceptions=True
                logger.warning("Repair failed for block %s: %s", cand.id, res)
                update = _record_failure(cand, res)
            await flusher.enqueue(update)

        batch_size = max(5, int(getattr(ctx.config, "repair_batch_size", 10)))
        processed_candidates = 0
        try:
            for chunk_start in range(0, len(repair_candidates), batch_size):
                chunk = repair_candidates[chunk_start : chunk_start + batch_size]
                check_stage_cancelled(ctx)
                results = await asyncio.gather(
                    *[_repair_and_record(cand) for cand in chunk],
                    return_exceptions=True,
                )
                for outcome in results:
                    if isinstance(
                        outcome,
                        (
                            BudgetExceededError,
                            JobInterruptedError,
                            asyncio.CancelledError,
                            LedgerWriterLeaseLostError,
                        ),
                    ):
                        raise outcome
                    if isinstance(outcome, BaseException):
                        # Without return_exceptions=True the first ledger-write failure
                        # propagated while sibling repairs kept calling the paid model
                        # and writing after the stage unwound.
                        logger.warning("repair task failed: %s", outcome)

                processed_candidates += len(chunk)
                if processed_candidates < len(repair_candidates):
                    progress_event = await create_event_fn(
                        EventType.REPAIR_PROGRESS,
                        actual_job_id,
                        ledger,
                        message=f"Targeted repair round {round_idx + 1}: {processed_candidates}/{len(repair_candidates)} candidate blocks processed",
                        active_block_id=chunk[-1].id,
                    )
                    yield progress_event
        finally:
            await flusher.close()

        event = await create_event_fn(
            EventType.REPAIR_BATCH_COMPLETED,
            actual_job_id,
            ledger,
            message=f"Targeted repair round {round_idx + 1} completed for {len(repair_candidates)} candidate blocks",
        )
        yield event

    # Ensure all remaining REPAIR_PENDING blocks reach terminal status. With
    # ``defer_unresolved_to_triage`` the leftovers stay REPAIR_PENDING so the
    # MQM triage stage can classify them by severity
    # (Critical -> escalated repair / BLOCKED_HUMAN; Major -> NEEDS_HUMAN;
    # Minor -> auto-pass) instead of being force-finalized here.
    repair_pending_blocks = await asyncio.to_thread(
        ledger.fetch_blocks_by_status, actual_job_id, BlockStatus.REPAIR_PENDING, chapter_id
    )
    # Standalone-API escape hatch: callers driving repair without triage
    # still finalize unresolved blocks.
    if repair_pending_blocks and not defer_unresolved_to_triage:
        terminal_updates: list[dict[str, Any]] = []
        # Same shared "shippable" rule as triage and the repair loop.
        for b in repair_pending_blocks:
            final_status = (
                BlockStatus.MTQE_PASSED
                if is_shippable(b.mtqe_score, ctx.config.qe_threshold, list(b.error_flags))
                else BlockStatus.FAILED
            )
            raw_text = b.target_text or b.draft_text or ""
            if final_status == BlockStatus.FAILED and raw_text:
                # Option B: keep translation draft with warning mark for human review
                target = f'<mark class="ubt-failed-draft" title="QE check not passed (score: {b.mtqe_score})">{raw_text}</mark>'
            else:
                target = raw_text or b.source_text

            terminal_updates.append(
                {
                    "block_id": b.id,
                    "target_text": target,
                    "status": final_status,
                    "mtqe_score": b.mtqe_score,
                    "repair_rounds": b.repair_rounds,
                    "error_flags": b.error_flags,
                }
            )
        await asyncio.to_thread(ledger.save_checkpoints_batch, terminal_updates)
