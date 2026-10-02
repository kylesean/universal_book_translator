"""Stage 3.5 (C-track): translate natural-language spans inside formulas.

FORMULA blocks are finalized verbatim at ingest (Gate 3's closed loop), so
the draft stage never sees them — its claim query excludes ``skip_translate``
by construction. C-track therefore runs as its own small stage after draft:
for each ``FORMULA`` block with translatable ``\\text{…}`` phrases, translate
the span bodies via the draft-tier LLM and splice them back.

Safety contract (C-era invariant):

- only span *contents* may change (``skeleton_holds`` enforced inside
  :func:`translate_math_text` AND re-checked here before persisting);
- every failure fails closed: the block keeps source-verbatim target;
- block statuses are never touched (MTQE_PASSED in → MTQE_PASSED out);
- Gate 3 accepts skeleton-identical targets (``formula_skeleton_intact``),
  so legal C translations survive export while math drift still repairs.

Off by default (``c_text_enabled``); runs only when enabled.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ubt.core.cleaners.math_text import (
    get_math_text_system_prompt,
    skeleton_holds,
    translatable_spans,
    translate_math_text,
)
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)


def _already_translated(block: IRBlock) -> bool:
    """True when a previous C-track run already rewrote this formula's target.

    Ingest finalizes FORMULA blocks with ``target_text == source_text``
    (verbatim); C-track is the only writer that changes the target. A
    non-identical target therefore means the span translation already happened,
    so a resume/re-export must not pay for it again nor overwrite the accepted
    result.
    """
    source = block.source_text or ""
    target = block.target_text or ""
    return bool(target) and target != source


async def _translate_block(
    router: ModelRouter,
    target_lang: str,
    sem: asyncio.Semaphore,
    block: IRBlock,
    source_lang: str = "en",
) -> tuple[IRBlock, str | None]:
    """Translate one formula's spans; (block, new_target or None)."""
    source = block.source_text or ""
    if not translatable_spans(source):
        return block, None

    system_prompt = get_math_text_system_prompt(source_lang, target_lang)

    async def _complete(inner: str, _lang: str) -> str:
        return await router.complete_raw(
            system_prompt=system_prompt,
            user_prompt=f"Target language: {target_lang}\nText: {inner}",
            temperature=0.0,
        )

    try:
        async with sem:
            translated = await translate_math_text(source, target_lang, _complete)
    except Exception as exc:
        logger.warning("C-track span translation failed for %s: %s", block.id, exc)
        return block, None
    if translated is None:
        return block, None
    # Defense in depth: re-verify the invariant at the persistence boundary
    # even though translate_math_text already checked it.
    if not skeleton_holds(source, translated):
        logger.warning("C-track skeleton re-check failed for %s (not persisted)", block.id)
        return block, None
    return block, translated


async def run_c_text_stage(
    ctx: StageContext,
    services: RunServices,
    chapter_id: str | None = None,
) -> AsyncIterator[TranslationProgressEvent]:
    """Translate formula text-spans; yields its completion event when enabled.

    ``chapter_id`` scopes the fetch to one chapter. The chapter-streaming QE
    worker calls this once per chapter; without it chapter 1 translated the
    whole book and a formula that failed closed was re-sent on every later
    chapter (N-times billing).
    """
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    router = ctx.router
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    create_event_fn = ctx.create_event
    blocks = await asyncio.to_thread(
        ledger.fetch_blocks_by_type, actual_job_id, BlockType.FORMULA, chapter_id=chapter_id
    )
    formulas = [b for b in blocks if (b.source_text or "").strip()]
    pending = [b for b in formulas if not _already_translated(b)]
    counts = {"translated": 0, "failed_closed": 0, "skipped": len(formulas) - len(pending)}
    if not pending:
        return
    # Share the run-wide in-flight cap: a private semaphore let this stage run
    # concurrently with the draft/QE workers in chapter streaming and exceed the
    # configured max_concurrency by up to 2x.
    sem = services.concurrency_sem
    results = await asyncio.gather(
        *(_translate_block(router, target_lang, sem, b, source_lang=source_lang) for b in pending)
    )
    checkpoints: list[dict[str, Any]] = []
    for block, new_target in results:
        if new_target is None:
            counts["failed_closed"] += 1
            continue
        counts["translated"] += 1
        checkpoints.append(
            {
                "block_id": block.id,
                "target_text": new_target,
                "status": block.status,
            }
        )
    if checkpoints:
        await asyncio.to_thread(ledger.save_checkpoints_batch, checkpoints)
    logger.info(
        "C-track stage for job %s: %d formula(s) span-translated, %d failed closed",
        actual_job_id,
        counts["translated"],
        counts["failed_closed"],
    )
    event: TranslationProgressEvent = await create_event_fn(
        EventType.CTEXT_COMPLETED,
        actual_job_id,
        ledger,
        message=(
            f"C-track: {counts['translated']} formula(s) span-translated, "
            f"{counts['failed_closed']} failed closed, "
            f"{counts['skipped']} already translated"
        ),
    )
    yield event
