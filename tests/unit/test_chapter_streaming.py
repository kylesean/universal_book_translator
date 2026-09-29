"""Unit tests for the chapter-streaming pipeline (decoupled stage barriers)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.chapter_streaming import run_chapter_streaming_pipeline
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def test_ledger_fetch_by_chapter_id(tmp_path: Path) -> None:
    """Test that ledger query methods filter strictly by chapter_id."""
    db_path = tmp_path / "test_ledger.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_test_chapter_filter"

    ch1_blocks = [
        IRBlock(
            id="ch01#p01",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Chapter 1 Paragraph 1",
            status=BlockStatus.PENDING,
        ),
        IRBlock(
            id="ch01#p02",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Chapter 1 Paragraph 2",
            status=BlockStatus.DRAFTED,
        ),
    ]
    ch2_blocks = [
        IRBlock(
            id="ch02#p01",
            flow_id=FlowID.MAIN_STORY,
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            source_text="Chapter 2 Paragraph 1",
            status=BlockStatus.PENDING,
        ),
        IRBlock(
            id="ch02#p02",
            flow_id=FlowID.MAIN_STORY,
            spine_index=4,
            block_type=BlockType.NARRATIVE,
            source_text="Chapter 2 Paragraph 2",
            status=BlockStatus.DRAFTED,
        ),
    ]

    ledger.append_chapter(
        job_id,
        ChapterIR(doc_id="doc1", chapter_id="ch01", title="Ch1", spine_index=1, blocks=ch1_blocks),
    )
    ledger.append_chapter(
        job_id,
        ChapterIR(doc_id="doc1", chapter_id="ch02", title="Ch2", spine_index=2, blocks=ch2_blocks),
    )

    # 1. fetch_pending_blocks with chapter_id
    pending_ch1 = ledger.fetch_pending_blocks(job_id, chapter_id="ch01")
    assert len(pending_ch1) == 1
    assert pending_ch1[0].id == "ch01#p01"

    pending_ch2 = ledger.fetch_pending_blocks(job_id, chapter_id="ch02")
    assert len(pending_ch2) == 1
    assert pending_ch2[0].id == "ch02#p01"

    # 2. fetch_blocks_by_status with chapter_id
    drafted_ch1 = ledger.fetch_blocks_by_status(job_id, BlockStatus.DRAFTED, chapter_id="ch01")
    assert len(drafted_ch1) == 1
    assert drafted_ch1[0].id == "ch01#p02"

    drafted_ch2 = ledger.fetch_blocks_by_status(job_id, BlockStatus.DRAFTED, chapter_id="ch02")
    assert len(drafted_ch2) == 1
    assert drafted_ch2[0].id == "ch02#p02"

    # 3. fetch_repair_eligible_blocks with chapter_id
    ledger.save_checkpoint(block_id="ch01#p02", status=BlockStatus.MTQE_PASSED)
    ledger.save_checkpoint(block_id="ch02#p02", status=BlockStatus.MTQE_PASSED)
    ledger.save_checkpoint(
        block_id="ch01#p01", status=BlockStatus.REPAIR_PENDING, draft_text="Draft ch1"
    )
    ledger.save_checkpoint(
        block_id="ch02#p01", status=BlockStatus.REPAIR_PENDING, draft_text="Draft ch2"
    )

    repair_ch1 = ledger.fetch_repair_eligible_blocks(job_id, chapter_id="ch01")
    assert len(repair_ch1) == 1
    assert repair_ch1[0].id == "ch01#p01"

    repair_ch2 = ledger.fetch_repair_eligible_blocks(job_id, chapter_id="ch02")
    assert len(repair_ch2) == 1
    assert repair_ch2[0].id == "ch02#p01"

    ledger.close()


@pytest.mark.asyncio
async def test_chapter_streaming_pipeline_end_to_end(tmp_path: Path) -> None:
    """Test full multi-chapter execution through PipelineOrchestrator with chapter streaming enabled."""
    db_dir = tmp_path / "ledgers"
    input_file = tmp_path / "multi_chapter.md"
    output_file = tmp_path / "multi_chapter_bilingual.md"

    md_content = """# Chapter 1: First Chapter

This is the content of the first chapter.

# Chapter 2: Second Chapter

This is the content of the second chapter.

# Chapter 3: Third Chapter

This is the content of the third chapter.
"""
    input_file.write_text(md_content, encoding="utf-8")

    config = UBTConfig(
        db_dir=db_dir,
        draft_model="mock-draft",
        repair_model="mock-repair",
        chapter_streaming_enabled=True,
    )

    mock_provider = MockModelProvider(
        default_response="这是流式章节翻译测试的通用输出。",
        custom_responses={
            "First Chapter": "# 第一章：第一章",
            "first chapter": "这是第一章的正文内容。",
            "Second Chapter": "# 第二章：第二章",
            "second chapter": "这是第二章的正文内容。",
            "Third Chapter": "# 第三章：第三章",
            "third chapter": "这是第三章的正文内容。",
        },
    )
    router = ModelRouter(
        provider=mock_provider, draft_model="mock-draft", repair_model="mock-repair"
    )
    qe = MockQERunner(default_score=0.92)

    orchestrator = PipelineOrchestrator(
        config=config,
        router=router,
        qe_runner=qe,
    )

    events: list[TranslationProgressEvent] = []
    async for event in orchestrator.run(
        input_path=input_file,
        output_path=output_file,
        target_lang="zh",
        job_id="job_streaming_e2e",
    ):
        events.append(event)

    event_types = [e.event_type for e in events]
    assert EventType.JOB_STARTED in event_types
    assert EventType.PREPROCESSING_DONE in event_types
    assert EventType.BIBLE_EXTRACTED in event_types
    # Verify CHAPTER_COMPLETED milestone events were emitted
    chapter_events = [e for e in events if e.event_type == EventType.CHAPTER_COMPLETED]
    assert len(chapter_events) == 3
    assert EventType.EXPORT_COMPLETED in event_types

    assert output_file.exists()
    rendered_text = output_file.read_text(encoding="utf-8")
    assert "Chapter 1: First Chapter" in rendered_text
    assert "Chapter 2: Second Chapter" in rendered_text
    assert "Chapter 3: Third Chapter" in rendered_text


@pytest.mark.asyncio
async def test_chapter_streaming_empty_manifest_fallback(tmp_path: Path) -> None:
    """Test that chapter streaming gracefully falls back when manifest chapters is empty."""
    db_path = tmp_path / "fallback.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_empty_fallback"

    block = IRBlock(
        id="single_block",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Single paragraph without chapter structure.",
        status=BlockStatus.PENDING,
    )
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id="doc1", chapter_id="single", title="Single", spine_index=1, blocks=[block]
        ),
    )

    manifest = BookManifest(
        doc_id="doc1",
        title="Single Doc",
        source_path=str(tmp_path / "doc.md"),
        chapters=[],  # empty chapters
    )

    mock_provider = MockModelProvider(
        default_response="这是没有章节结构的单段落完整中文翻译测试语句，用于验证阶段回退。"
    )
    router = ModelRouter(
        provider=mock_provider, draft_model="mock-draft", repair_model="mock-repair"
    )
    ctx = build_stage_ctx(
        ledger=ledger,
        router=router,
        job_id=job_id,
        manifest=manifest,
    )

    events: list[TranslationProgressEvent] = []
    async for event in run_chapter_streaming_pipeline(ctx):
        events.append(event)

    assert len(events) >= 1
    # Block should be drafted and passed through quality gate
    final_blocks = ledger.get_all_blocks(job_id)
    assert len(final_blocks) == 1
    assert final_blocks[0].status in (BlockStatus.DRAFTED, BlockStatus.MTQE_PASSED)

    ledger.close()


@pytest.mark.asyncio
async def test_chapter_streaming_cancellation(tmp_path: Path) -> None:
    """Test that cancellation token terminates chapter streaming cleanly without orphaned tasks."""
    db_path = tmp_path / "cancel.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_cancel_streaming"

    blocks = [
        IRBlock(
            id=f"ch01#p{i:02d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            block_type=BlockType.NARRATIVE,
            source_text=f"Paragraph {i}",
            status=BlockStatus.PENDING,
        )
        for i in range(1, 10)
    ]
    ledger.append_chapter(
        job_id,
        ChapterIR(doc_id="doc1", chapter_id="ch01", title="Ch1", spine_index=1, blocks=blocks),
    )

    manifest = BookManifest(
        doc_id="doc1",
        title="Test Book",
        source_path=str(tmp_path / "book.md"),
        chapters=[ChapterMeta(chapter_id="ch01", title="Ch1", spine_index=1)],
    )

    cancel_token = asyncio.Event()
    cancel_token.set()  # Cancel immediately

    mock_provider = MockModelProvider(default_response="Test")
    router = ModelRouter(
        provider=mock_provider, draft_model="mock-draft", repair_model="mock-repair"
    )
    ctx = build_stage_ctx(
        ledger=ledger,
        router=router,
        job_id=job_id,
        manifest=manifest,
        cancel_token=cancel_token,
    )

    from ubt.core.exceptions import JobInterruptedError

    with pytest.raises(JobInterruptedError):
        async for _ in run_chapter_streaming_pipeline(ctx):
            pass

    ledger.close()


@pytest.mark.asyncio
async def test_hard_task_cancellation_marks_job_cancelled(tmp_path: Path) -> None:
    """task.cancel() (Ctrl-C / server shutdown) must persist a terminal
    status. CancelledError is a BaseException, so without a dedicated handler in
    PipelineOrchestrator.run the ``except Exception`` branch never fires, the
    ledger row stays "running", and JobManager counts it forever against
    ``max_running_jobs``."""

    class _BlockingProvider(MockModelProvider):
        def __init__(self, **kw: object) -> None:
            super().__init__(**kw)  # type: ignore[arg-type]
            self.entered = asyncio.Event()
            self._release = asyncio.Event()  # never set: cancellation breaks the await

        async def generate(self, *args: object, **kwargs: object) -> str:
            self.entered.set()
            await self._release.wait()
            return ""  # unreachable

    db_dir = tmp_path / "ledgers"
    input_file = tmp_path / "book.md"
    input_file.write_text("# Chapter 1\n\nSome prose to translate.\n", encoding="utf-8")

    provider = _BlockingProvider(default_response="译文")
    router = ModelRouter(provider=provider, draft_model="m", repair_model="m")
    orch = PipelineOrchestrator(
        config=UBTConfig(db_dir=db_dir, draft_model="m", repair_model="m"),
        router=router,
        qe_runner=MockQERunner(default_score=0.95),
    )

    async def _drive() -> None:
        async for _ in orch.run(
            input_path=input_file,
            output_path=tmp_path / "out.md",
            target_lang="zh",
            job_id="job_hard_cancel",
        ):
            pass

    task = asyncio.create_task(_drive())
    await asyncio.wait_for(provider.entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    probe = SQLiteJobLedger(db_dir / "job_hard_cancel.sqlite")
    try:
        assert probe.get_job_status("job_hard_cancel") == "cancelled"
    finally:
        probe.close()


@pytest.mark.fast
async def test_chapter_streaming_filters_chapters_by_window(tmp_path: Path) -> None:
    """run_chapter_streaming_pipeline filters chapters according to ctx.start_chapter and max_chapters."""
    config = UBTConfig()
    manifest = BookManifest(
        doc_id="doc1",
        title="Title",
        source_path=str(tmp_path / "input.epub"),
        chapters=[
            ChapterMeta(chapter_id="ch1", title="Chapter 1", spine_index=0),
            ChapterMeta(chapter_id="ch2", title="Chapter 2", spine_index=1),
            ChapterMeta(chapter_id="ch3", title="Chapter 3", spine_index=2),
            ChapterMeta(chapter_id="ch4", title="Chapter 4", spine_index=3),
        ],
    )
    ledger = SQLiteJobLedger(tmp_path / "test.sqlite")
    ledger.init_job_from_manifest("job_test", manifest)

    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_test",
        input_path=tmp_path / "input.epub",
        config=config,
        manifest=manifest,
        ledger=ledger,
        start_chapter=2,
        max_chapters=2,
    )

    drafted_chapters: list[str] = []

    async def fake_draft_stage(ctx: StageContext, chapter_id: str | None = None) -> Any:
        if chapter_id:
            drafted_chapters.append(chapter_id)
        if False:
            yield None

    async def fake_stage(ctx: StageContext, chapter_id: str | None = None) -> Any:
        if False:
            yield None

    with (
        patch(
            "ubt.core.engine.stages.chapter_streaming.run_draft_stage", side_effect=fake_draft_stage
        ),
        patch(
            "ubt.core.engine.stages.chapter_streaming.run_quality_gate_stage",
            side_effect=fake_stage,
        ),
        patch("ubt.core.engine.stages.chapter_streaming.run_repair_stage", side_effect=fake_stage),
    ):
        async for _ in run_chapter_streaming_pipeline(ctx):
            pass

    # Should only draft ch2 and ch3 (start_chapter=2, max_chapters=2)
    assert drafted_chapters == ["ch2", "ch3"]
