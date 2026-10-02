"""Chapter-Streaming Pipeline: Decoupled producer-consumer stages for multi-chapter books."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Scoring, Terminology
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.ctext import run_c_text_stage
from ubt.core.engine.stages.draft import run_draft_stage
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.engine.stages.repair import run_repair_stage
from ubt.core.ir.models import ChapterMeta

logger = logging.getLogger(__name__)

_DONE = object()


async def run_chapter_streaming_pipeline(
    ctx: StageContext,
    services: RunServices,
    terminology: Terminology,
    scoring: Scoring,
) -> AsyncIterator[TranslationProgressEvent]:
    """Execute streaming pipelined chapter processing across stages.

    Decouples the book-wide monolithic stage barriers (all-draft -> all-qe -> all-repair)
    into a concurrent two-worker pipeline:
      - Worker 1 (Draft Worker): drafts chapters in spine order (C_1, C_2, ...), maintaining
        rolling memory context across chapters, then enqueues each completed chapter to qe_queue.
      - Worker 2 (QE & Repair Worker): consumes drafted chapters from qe_queue and executes
        Quality Gate and Repair stages concurrently while Worker 1 is drafting the next chapter.

    Multiplexes progress events onto an asynchronous event queue so callers receive real-time
    progress updates as blocks and chapters complete.
    """
    chapters: list[ChapterMeta] = getattr(ctx.manifest, "chapters", [])
    if chapters:
        start_idx = max(0, ctx.start_chapter - 1)
        if ctx.max_chapters is not None:
            chapters = chapters[start_idx : start_idx + ctx.max_chapters]
        elif start_idx > 0:
            chapters = chapters[start_idx:]
    if not chapters:
        # Fallback to monolithic stage execution when no chapter metadata is present
        async for event in run_draft_stage(ctx, services, terminology):
            yield event
        if ctx.config.c_text_enabled:
            async for event in run_c_text_stage(ctx, services):
                yield event
        async for event in run_quality_gate_stage(ctx, services, terminology, scoring):
            yield event
        async for event in run_repair_stage(ctx, services, terminology, scoring):
            yield event
        return

    queue_size = getattr(ctx.config, "chapter_streaming_queue_size", 2)
    qe_queue: asyncio.Queue[ChapterMeta | None] = asyncio.Queue(maxsize=max(1, int(queue_size)))
    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    c_text_enabled = bool(ctx.config.c_text_enabled)

    async def draft_worker() -> None:
        try:
            for ch in chapters:
                ctx.check_cancelled()
                logger.info(
                    "Chapter-Streaming [Draft]: starting chapter %d (%s)",
                    ch.spine_index,
                    ch.chapter_id,
                )
                async for event in run_draft_stage(
                    ctx, services, terminology, chapter_id=ch.chapter_id
                ):
                    await event_queue.put(event)
                await qe_queue.put(ch)
        finally:
            await qe_queue.put(None)

    async def qe_repair_worker() -> None:
        while True:
            ch = await qe_queue.get()
            if ch is None:
                break
            try:
                ctx.check_cancelled()
                logger.info(
                    "Chapter-Streaming [QE & Repair]: processing chapter %d (%s)",
                    ch.spine_index,
                    ch.chapter_id,
                )
                if c_text_enabled:
                    async for event in run_c_text_stage(ctx, services, chapter_id=ch.chapter_id):
                        await event_queue.put(event)
                async for event in run_quality_gate_stage(
                    ctx, services, terminology, scoring, chapter_id=ch.chapter_id
                ):
                    await event_queue.put(event)
                async for event in run_repair_stage(
                    ctx, services, terminology, scoring, chapter_id=ch.chapter_id
                ):
                    await event_queue.put(event)

                # Emit CHAPTER_COMPLETED milestone event
                chapter_done_event = await ctx.create_event(
                    EventType.CHAPTER_COMPLETED,
                    ctx.job_id,
                    ctx.ledger,
                    message=f"Chapter {ch.spine_index} completed: {ch.title}",
                    active_block_id=ch.chapter_id,
                )
                await event_queue.put(chapter_done_event)
            finally:
                qe_queue.task_done()

    draft_task = asyncio.create_task(draft_worker(), name=f"chapter_draft_{ctx.job_id}")
    qe_task = asyncio.create_task(qe_repair_worker(), name=f"chapter_qe_{ctx.job_id}")

    async def supervisor() -> None:
        try:
            # Concurrently await both workers so failure in either triggers immediate reaction
            await asyncio.gather(draft_task, qe_task)
        except BaseException as exc:
            # Cancel sibling tasks on unhandled error
            if not draft_task.done():
                draft_task.cancel()
            if not qe_task.done():
                qe_task.cancel()
            await event_queue.put(exc)
        finally:
            await event_queue.put(_DONE)

    supervisor_task = asyncio.create_task(supervisor(), name=f"chapter_supervisor_{ctx.job_id}")

    try:
        while True:
            item = await event_queue.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, TranslationProgressEvent):
                yield item
    finally:
        for t in (supervisor_task, qe_task, draft_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(supervisor_task, qe_task, draft_task, return_exceptions=True)
