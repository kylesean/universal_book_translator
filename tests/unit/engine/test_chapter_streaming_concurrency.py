from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Scoring, Terminology
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.chapter_streaming import _DONE, run_chapter_streaming_pipeline
from ubt.core.ir.models import BookManifest, ChapterMeta

pytestmark = pytest.mark.fast


def _make_event(event_type: EventType, message: str) -> TranslationProgressEvent:
    return TranslationProgressEvent(
        event_type=event_type,
        job_id="test_job",
        total_blocks=10,
        completed_blocks=1,
        message=message,
    )


def _make_context(
    chapters: list[ChapterMeta], queue_size: int = 1
) -> tuple[StageContext, SimpleNamespace]:
    manifest = BookManifest(
        doc_id="test_doc", title="Test Book", source_path="dummy.epub", chapters=chapters
    )
    config = SimpleNamespace(
        chapter_streaming_queue_size=queue_size,
        c_text_enabled=False,
    )

    async def create_event(
        event_type: EventType,
        job_id: str,
        ledger: Any,
        message: str = "",
        active_block_id: str | None = None,
    ) -> TranslationProgressEvent:
        return TranslationProgressEvent(
            event_type=event_type,
            job_id=job_id,
            total_blocks=10,
            completed_blocks=1,
            message=message,
            active_block_id=active_block_id,
        )

    ctx = SimpleNamespace(
        job_id="test_job",
        manifest=manifest,
        config=config,
        ledger=None,
        start_chapter=1,
        max_chapters=None,
        check_cancelled=lambda: None,
        create_event=create_event,
    )
    return ctx, config  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_chapter_streaming_normal_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    chapters = [
        ChapterMeta(chapter_id="c1", spine_index=1, title="Chapter 1"),
        ChapterMeta(chapter_id="c2", spine_index=2, title="Chapter 2"),
    ]
    ctx, _ = _make_context(chapters, queue_size=2)

    async def mock_draft(ctx: Any, services: Any, term: Any, chapter_id: str | None = None) -> Any:
        yield _make_event(EventType.DRAFT_BATCH_COMPLETED, f"drafted {chapter_id}")

    async def mock_qg(
        ctx: Any, services: Any, term: Any, scoring: Any, chapter_id: str | None = None
    ) -> Any:
        yield _make_event(EventType.MTQE_EVALUATED, f"qg {chapter_id}")

    async def mock_repair(
        ctx: Any, services: Any, term: Any, scoring: Any, chapter_id: str | None = None
    ) -> Any:
        yield _make_event(EventType.REPAIR_BATCH_COMPLETED, f"repair {chapter_id}")

    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_draft_stage", mock_draft)
    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_quality_gate_stage", mock_qg)
    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_repair_stage", mock_repair)

    events: list[TranslationProgressEvent] = []
    async for event in run_chapter_streaming_pipeline(
        ctx,
        SimpleNamespace(),  # type: ignore[arg-type]
        Terminology(glossary_dicts=[], abbreviation_entries=[]),
        Scoring(),
    ):
        events.append(event)

    types = [e.event_type for e in events]
    assert EventType.DRAFT_BATCH_COMPLETED in types
    assert EventType.CHAPTER_COMPLETED in types
    chapter_done_events = [e for e in events if e.event_type is EventType.CHAPTER_COMPLETED]
    assert len(chapter_done_events) == 2


@pytest.mark.asyncio
async def test_chapter_streaming_consumer_crash_does_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chapters = [
        ChapterMeta(chapter_id=f"c{i}", spine_index=i, title=f"Chapter {i}") for i in range(1, 6)
    ]
    ctx, _ = _make_context(chapters, queue_size=1)

    async def mock_draft(ctx: Any, services: Any, term: Any, chapter_id: str | None = None) -> Any:
        yield _make_event(EventType.DRAFT_BATCH_COMPLETED, f"drafted {chapter_id}")
        await asyncio.sleep(0.01)

    async def mock_qg(
        ctx: Any, services: Any, term: Any, scoring: Any, chapter_id: str | None = None
    ) -> Any:
        if chapter_id == "c1":
            raise RuntimeError("Consumer exploded!")
        yield _make_event(EventType.MTQE_EVALUATED, f"qg {chapter_id}")

    async def mock_repair(
        ctx: Any, services: Any, term: Any, scoring: Any, chapter_id: str | None = None
    ) -> Any:
        yield _make_event(EventType.REPAIR_BATCH_COMPLETED, f"repair {chapter_id}")

    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_draft_stage", mock_draft)
    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_quality_gate_stage", mock_qg)
    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_repair_stage", mock_repair)

    with pytest.raises(RuntimeError, match="Consumer exploded!"):
        async with asyncio.timeout(2.0):
            async for _ in run_chapter_streaming_pipeline(
                ctx,
                SimpleNamespace(),  # type: ignore[arg-type]
                Terminology(glossary_dicts=[], abbreviation_entries=[]),
                Scoring(),
            ):
                pass


@pytest.mark.asyncio
async def test_a_cancelled_stream_drains_the_per_chapter_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chapters = [
        ChapterMeta(chapter_id="c1", spine_index=1, title="Chapter 1"),
        ChapterMeta(chapter_id="c2", spine_index=2, title="Chapter 2"),
    ]
    ctx, _ = _make_context(chapters, queue_size=2)

    closed: list[str] = []

    async def mock_draft(ctx: Any, services: Any, term: Any, chapter_id: str | None = None) -> Any:
        try:
            yield _make_event(EventType.DRAFT_BATCH_COMPLETED, f"drafted {chapter_id}")
            await asyncio.sleep(10)
        finally:
            closed.append(str(chapter_id))

    async def mock_pass(ctx: Any, *a: Any, chapter_id: str | None = None, **kw: Any) -> Any:
        if False:  # pragma: no cover - async generator
            yield _make_event(EventType.MTQE_EVALUATED, "unused")

    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_draft_stage", mock_draft)
    monkeypatch.setattr(
        "ubt.core.engine.stages.chapter_streaming.run_quality_gate_stage", mock_pass
    )
    monkeypatch.setattr("ubt.core.engine.stages.chapter_streaming.run_repair_stage", mock_pass)

    gen = run_chapter_streaming_pipeline(
        ctx,
        SimpleNamespace(),  # type: ignore[arg-type]
        Terminology(glossary_dicts=[], abbreviation_entries=[]),
        Scoring(),
    )
    await anext(gen)
    await gen.aclose()

    await asyncio.sleep(0.05)
    assert "c1" in closed, "the in-flight chapter's stage generator was not drained"


@pytest.mark.asyncio
async def test_chapter_streaming_event_queue_backpressure() -> None:
    event_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
    await event_queue.put("pre-existing-event")

    async def drain_event() -> None:
        await asyncio.sleep(0.05)
        await event_queue.get()
        event_queue.task_done()

    drain_task = asyncio.create_task(drain_event())

    put_done_coro = event_queue.put(_DONE)
    await asyncio.wait_for(put_done_coro, timeout=1.0)
    await drain_task

    item = await event_queue.get()
    assert item is _DONE
