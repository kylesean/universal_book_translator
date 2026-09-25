"""RED tests for Round 2 Batch 5: Core Engine, Stages, Ledger, CLI & API fixes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_base import _upsert_blocks_batch
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.advisory import run_difficulty_advisory_stage
from ubt.core.engine.stages.chapter_streaming import run_chapter_streaming_pipeline
from ubt.core.engine.stages.ingest import run_ingest_stage
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.policy.bilingual_advisor import Advisory, ModeScore


@pytest.mark.fast
async def test_advisory_rigid_engine_monolingual_in_difficulty_stage(tmp_path: Path) -> None:
    """run_difficulty_advisory_stage preserves 'monolingual' for rigid engines under auto mode."""
    config = UBTConfig(render_engine="rigid", dual_mode="auto")
    manifest = BookManifest(
        doc_id="doc1", title="Title", source_path=str(tmp_path / "input.pdf"), chapters=[]
    )
    ledger = SQLiteJobLedger(tmp_path / "test.sqlite")
    ledger.init_job_from_manifest("job_test", manifest)

    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_test",
        input_path=tmp_path / "input.pdf",
        config=config,
        manifest=manifest,
        ledger=ledger,
        short_chain=False,
    )
    ctx.tier_basis = "inline"
    ctx.enforcement = "auto"
    ctx.advisory = Advisory(
        requested="inline",
        tier="ok",
        ranking=(ModeScore(mode="inline", score=1.0),),
        reasons=(),
    )
    manifest.run.effective_dual_mode = "monolingual"

    await run_difficulty_advisory_stage(ctx)

    # For rigid engine, effective_mode must remain monolingual, never inline
    assert manifest.run.effective_dual_mode == "monolingual"


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


@pytest.mark.fast
async def test_draft_stage_restores_memory_on_fresh_job(tmp_path: Path) -> None:
    """run_draft_stage restores rolling memory even if config.fresh is True."""
    from ubt.core.engine.stages.draft import run_draft_stage

    config = UBTConfig(fresh=True, enable_rolling_summary=True)
    manifest = BookManifest(
        doc_id="doc1",
        title="Title",
        source_path=str(tmp_path / "input.epub"),
        chapters=[
            ChapterMeta(chapter_id="ch1", title="Chapter 1", spine_index=0),
            ChapterMeta(chapter_id="ch2", title="Chapter 2", spine_index=1),
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
    )

    restore_mock = MagicMock()
    with (
        patch("ubt.core.engine.stages.draft._restore_memory_state", restore_mock),
        patch("ubt.core.engine.stages.draft.resolve_draft_policy", return_value=(True, False, 10)),
        patch.object(ledger, "fetch_pending_blocks", return_value=[]),
    ):
        async for _ in run_draft_stage(ctx, chapter_id="ch2"):
            pass

    restore_mock.assert_called_once()


@pytest.mark.fast
async def test_ingest_stage_raises_document_parse_error_on_zero_blocks(tmp_path: Path) -> None:
    """run_ingest_stage raises DocumentParseError when 0 blocks are parsed."""
    config = UBTConfig()
    input_file = tmp_path / "empty.txt"
    input_file.write_text("")
    manifest = BookManifest(doc_id="doc1", title="Title", source_path=str(input_file), chapters=[])
    ledger = SQLiteJobLedger(tmp_path / "test.sqlite")
    ledger.init_job_from_manifest("job_test", manifest)

    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_test",
        input_path=input_file,
        config=config,
        manifest=manifest,
        ledger=ledger,
    )

    class DummyAdapter:
        async def parse_stream(self, path: Path, selected_pages: Any = None) -> Any:
            from ubt.core.ir.models import ChapterIR

            yield ChapterIR(
                doc_id="doc1", chapter_id="ch0", spine_index=0, title="Empty", blocks=[]
            )

    ctx.__dict__["require_adapter"] = lambda: DummyAdapter()

    with pytest.raises(DocumentParseError, match="0 content blocks"):
        async for _ in run_ingest_stage(ctx):
            pass


@pytest.mark.fast
def test_ledger_upsert_blocks_batch_includes_mqm_fields(tmp_path: Path) -> None:
    """_upsert_blocks_batch writes mqm_severity and mqm_spans_json."""
    db_path = tmp_path / "test_ledger.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_mqm"
    manifest = BookManifest(
        doc_id="doc1", title="Title", source_path=str(tmp_path / "doc.txt"), chapters=[]
    )
    ledger.init_job_from_manifest(job_id, manifest)

    span = {
        "start": 5,
        "end": 10,
        "severity": "critical",
        "category": "accuracy",
        "explanation": "Mistranslation",
    }
    block = IRBlock(
        id="b001",
        spine_index=0,
        flow_id=FlowID.MAIN_STORY,
        block_type=BlockType.NARRATIVE,
        source_text="Hello world",
        target_text="你好世界",
        status=BlockStatus.DRAFTED,
        mqm_severity="critical",
        mqm_spans=[span],
    )

    with ledger._get_conn() as conn:
        cursor = conn.cursor()
        _upsert_blocks_batch(cursor, job_id, [block])

    loaded = ledger.get_block("b001")
    assert loaded is not None
    assert loaded.mqm_severity == "critical"
    assert len(loaded.mqm_spans) == 1
    assert loaded.mqm_spans[0]["severity"] == "critical"
    assert loaded.mqm_spans[0]["category"] == "accuracy"
