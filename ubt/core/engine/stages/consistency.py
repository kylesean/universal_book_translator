"""Post-repair stage: enforce document-level terminology consistency.

Drift is detected deterministically (the same scan the quality report uses) and
repaired with the repair stage's machinery: the constraint rides in as an error
flag, so the model sees it under "Issues to Fix" and the result is accepted
only when it passes the identical structural + QE gates. Default mode is
``report`` (plan + log, no LLM spend); ``repair`` runs the tasks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BlockStatus
from ubt.core.memory.glossary_table import build_chunk_glossary_table
from ubt.core.qe.comet_runner import GLOSSARY_VIOLATION_MARKER
from ubt.core.qe.consistency_enforce import ConsistencyTask, plan_consistency_tasks
from ubt.core.qe.term_metrics import evaluate_terms
from ubt.core.validators.consistency import GlossaryConsistencyValidator

logger = logging.getLogger(__name__)


async def run_consistency_stage(
    ctx: StageContext,
) -> AsyncIterator[TranslationProgressEvent]:
    """Detect terminology drift and (mode='repair') re-translate drifted blocks."""
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    repair_loop = ctx.repair_loop
    glossary_dicts = ctx.glossary_dicts
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    fast_pass = ctx.fast_pass
    concurrency_sem = ctx.concurrency_sem
    create_event_fn = ctx.create_event
    source_pdf_path = ctx.source_pdf_path
    mode = ctx.config.consistency_enforce
    max_repairs = ctx.config.consistency_max_repairs
    if mode == "off":
        # Explicit off must skip the whole O(blocks x terms) scan, not run it
        # and throw the metrics away, avoiding unnecessary full-book scans when consistency enforcement is disabled.
        return
    all_blocks = await asyncio.to_thread(ledger.get_all_blocks, actual_job_id)
    # On a fresh run triage has not run yet, so these states do not appear here;
    # they only show up on a RESUMED ledger. Consistency must not re-translate
    # them: BLOCKED_HUMAN carries a source-only quarantine placeholder (not a
    # draft) as its target, and repairing a NEEDS_HUMAN/FAILED block to
    # REPAIRED drops it out of the human queue — triage later only re-reads
    # REPAIR_PENDING/FAILED, so a Critical could ship as clean. Only machine
    # drafts (DRAFTED / REPAIR_PENDING) and machine-passed blocks
    # (MTQE_PASSED / REPAIRED) are legitimate terminology-drift targets.
    excluded = {BlockStatus.BLOCKED_HUMAN, BlockStatus.NEEDS_HUMAN, BlockStatus.FAILED}
    deliverable = [
        b
        for b in all_blocks
        if (b.target_text or "") and not b.skip_translate and b.status not in excluded
    ]
    if not deliverable:
        return

    metrics = await asyncio.to_thread(evaluate_terms, deliverable, glossary_dicts)
    tasks = plan_consistency_tasks(metrics, max_repairs)
    if not tasks:
        return

    if mode != "repair":
        logger.info(
            "Terminology consistency (report only): %d block(s) would be re-translated; e.g. %s",
            len({task.block_id for task in tasks}),
            " | ".join(task.flag for task in tasks[:3]),
        )
        return

    by_id = {b.id: b for b in deliverable}
    glossary_validator = GlossaryConsistencyValidator(glossary=glossary_dicts)
    per_block: dict[str, list[ConsistencyTask]] = {}
    for task in tasks:
        per_block.setdefault(task.block_id, []).append(task)

    async def _repair_block(
        block_id: str, block_tasks: list[ConsistencyTask]
    ) -> dict[str, Any] | None:
        block = by_id.get(block_id)
        if block is None:
            return None
        flagged = block.model_copy(
            update={
                "error_flags": [*block.error_flags, *[task.flag for task in block_tasks]],
                "status": BlockStatus.REPAIR_PENDING,
            }
        )
        glossary_table = build_chunk_glossary_table(glossary_dicts, [], block.source_text)
        async with concurrency_sem:
            repaired = await repair_loop.repair_single_block(
                block=flagged,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
                fast_pass=fast_pass,
                glossary_entries=glossary_dicts,
                source_pdf_path=source_pdf_path,
            )
        status = repaired.status
        error_flags = list(repaired.error_flags)
        # A terminology repair is accepted only when the rendering is actually
        # restored. The repair loop's acceptance check is term-blind unless its
        # QE runner is glossary-aware, so on a comet/neural/subprocess engine the
        # constraint flag is dropped and a still-drifted block would be promoted
        # REPAIRED — which triage never re-reads. Verify the postcondition here.
        if status is BlockStatus.REPAIRED:
            check = glossary_validator.validate(repaired.source_text, repaired.target_text)
            if not check.is_valid:
                status = BlockStatus.REPAIR_PENDING
                error_flags.append(f"{GLOSSARY_VIOLATION_MARKER}: {check.message}")
        return {
            "block_id": repaired.id,
            "target_text": repaired.target_text or "",
            "status": status,
            "mtqe_score": repaired.mtqe_score,
            "repair_rounds": repaired.repair_rounds,
            "error_flags": error_flags,
        }

    results = await asyncio.gather(
        *[_repair_block(block_id, block_tasks) for block_id, block_tasks in per_block.items()],
        return_exceptions=True,
    )
    updates = [res for res in results if isinstance(res, dict)]
    for block_id, res in zip(per_block, results, strict=True):
        if not isinstance(res, dict):
            logger.warning("Consistency repair failed for block %s: %s", block_id, res)
    if updates:
        await asyncio.to_thread(ledger.save_checkpoints_batch, updates)

    event = await create_event_fn(
        EventType.REPAIR_BATCH_COMPLETED,
        actual_job_id,
        ledger,
        message=(
            f"Terminology consistency: re-translated {len(updates)}/{len(per_block)} "
            f"drifted block(s)"
        ),
    )
    yield event
