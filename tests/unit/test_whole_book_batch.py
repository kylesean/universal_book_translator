"""Unit and integration tests for Phase 4: Whole-Book Offline Batch API Mode."""

import asyncio
from pathlib import Path
from typing import Any

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.draft import run_draft_stage
from ubt.core.exceptions import ModelProviderError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterMeta,
    DocumentIR,
    FlowID,
    IRBlock,
)
from ubt.core.memory.tm import TMPendingEntry, TranslationMemory, compute_tm_context
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


class WholeBookBatchProvider(MockModelProvider):
    """Mock Batch API provider tracking all batch submissions."""

    @property
    def supports_batch_api(self) -> bool:
        return True

    def __init__(self, partial_fail_ids: set[str] | None = None) -> None:
        super().__init__(default_response="[INTERACTIVE-DRAFTED]")
        self.batch_submissions: list[list[dict[str, Any]]] = []
        self.partial_fail_ids = partial_fail_ids or set()
        self.created_batch_ids: list[str] = []

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        self.batch_submissions.append(requests)
        batch_id = f"batch-job-{len(self.batch_submissions)}"
        self.created_batch_ids.append(batch_id)
        return batch_id

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        return {
            "id": batch_id,
            "status": "completed",
            "request_counts": {
                "total": len(self.batch_submissions[-1]),
                "completed": len(self.batch_submissions[-1]),
                "failed": 0,
            },
        }

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for req in self.batch_submissions[-1]:
            cid = req["custom_id"]
            if cid in self.partial_fail_ids:
                results[cid] = {"content": None, "error": "Simulated batch line error"}
            else:
                results[cid] = {"content": f"[BATCH-DRAFTED: {cid}]", "error": None}
        return results


class BrokenBatchProvider(MockModelProvider):
    """Mock Batch API provider that fails batch submission."""

    @property
    def supports_batch_api(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__(default_response="[FALLBACK-DRAFTED]")

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        raise ModelProviderError("500 Batch API service unavailable")


def _make_multi_chapter_ir(total_blocks: int, num_chapters: int = 3) -> DocumentIR:
    blocks: list[IRBlock] = []
    blocks_per_ch = total_blocks // num_chapters
    for i in range(total_blocks):
        ch_idx = min(i // blocks_per_ch + 1, num_chapters)
        blocks.append(
            IRBlock(
                id=f"ch{ch_idx}#b{i:04d}",
                flow_id=FlowID.MAIN_STORY,
                spine_index=i + 1,
                block_type=BlockType.NARRATIVE,
                source_text=f"Sentence {i} in chapter {ch_idx}.",
                status=BlockStatus.PENDING,
            )
        )
    return DocumentIR(
        doc_id="test_whole_book_doc",
        source_path="/tmp/test_book.epub",
        format_type="epub",
        metadata={"title": "Test Whole Book"},
        blocks=blocks,
    )


def _make_manifest(num_chapters: int = 3) -> BookManifest:
    return BookManifest(
        doc_id="test_whole_book_doc",
        title="Test Whole Book",
        source_path="/tmp/test_book.epub",
        chapters=[
            ChapterMeta(
                chapter_id=f"ch{c}",
                title=f"Chapter {c}",
                source_file=f"ch{c}.html",
                spine_index=c,
            )
            for c in range(1, num_chapters + 1)
        ],
        metadata={},
    )


async def _run_stage(
    ledger: SQLiteJobLedger,
    job_id: str,
    manifest: BookManifest,
    router: ModelRouter,
    config: UBTConfig,
    tm: TranslationMemory | None = None,
) -> None:
    ctx = build_stage_ctx(
        job_id=job_id,
        ledger=ledger,
        router=router,
        manifest=manifest,
        config=config,
        tm=tm,
    )
    async for _ in run_draft_stage(ctx):
        pass


def test_whole_book_batch_single_job_for_entire_book(tmp_path: Path) -> None:
    """Whole-book batch mode should submit exactly ONE batch job for all blocks across chapters."""
    db_path = tmp_path / "job.sqlite"
    ledger = SQLiteJobLedger(db_path)
    doc_ir = _make_multi_chapter_ir(60, num_chapters=3)
    ledger.init_job("job_wb_1", doc_ir, target_lang="zh")

    provider = WholeBookBatchProvider()
    router = ModelRouter(provider=provider, draft_model="test-batch-model")

    config = UBTConfig(
        offline_batch_enabled=True,
        batch_limit=15,  # Deliberately small: pagination would have required 4 batches
        batch_poll_interval=0.01,
        batch_poll_timeout=5.0,
    )

    asyncio.run(_run_stage(ledger, "job_wb_1", _make_manifest(3), router, config))

    # Verification: EXACTLY ONE batch submission containing all 60 blocks!
    assert len(provider.batch_submissions) == 1
    assert len(provider.batch_submissions[0]) == 60
    assert provider.call_history == []  # Interactive path not called

    blocks = ledger.get_all_blocks("job_wb_1")
    assert len(blocks) == 60
    assert all(b.status == BlockStatus.DRAFTED for b in blocks)
    assert all("[BATCH-DRAFTED:" in (b.target_text or "") for b in blocks)
    ledger.close()


def test_whole_book_batch_with_skips_and_tm(tmp_path: Path) -> None:
    """Static skips (images/formula) and exact TM hits are processed and filtered before batch submission."""
    db_path = tmp_path / "job.sqlite"
    ledger = SQLiteJobLedger(db_path)
    doc_ir = _make_multi_chapter_ir(20, num_chapters=2)
    # Make block 0 an image (static skip)
    doc_ir.blocks[0].block_type = BlockType.IMAGE
    # Make block 1 formula (static skip)
    doc_ir.blocks[1].block_type = BlockType.FORMULA
    # Block 2 without digits for TM exact hit and FastPass
    doc_ir.blocks[2].source_text = "Standard paragraph without digits."
    ledger.init_job("job_wb_skips", doc_ir, target_lang="zh")

    # Seed TM with exact hit for block 2
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    tm_ctx = compute_tm_context("v1", "general", "", "en", "zh", "")
    tm.writeback(
        [
            TMPendingEntry(
                src_lang="en",
                tgt_lang="zh",
                source_text="Standard paragraph without digits.",
                target_text="标准无数字段落。",
                domain="general",
                context_hash=tm_ctx,
            )
        ]
    )

    provider = WholeBookBatchProvider()
    router = ModelRouter(provider=provider, draft_model="test-batch-model")

    config = UBTConfig(
        offline_batch_enabled=True,
        batch_poll_interval=0.01,
        batch_poll_timeout=5.0,
    )

    asyncio.run(_run_stage(ledger, "job_wb_skips", _make_manifest(2), router, config, tm=tm))

    # 20 blocks total: 1 image + 1 skip + 1 TM hit = 3 non-batch, 17 sent to batch!
    assert len(provider.batch_submissions) == 1
    assert len(provider.batch_submissions[0]) == 17

    blocks = {b.id: b for b in ledger.get_all_blocks("job_wb_skips")}
    assert blocks[doc_ir.blocks[0].id].status == BlockStatus.MTQE_PASSED
    assert blocks[doc_ir.blocks[1].id].status == BlockStatus.MTQE_PASSED
    assert blocks[doc_ir.blocks[2].id].status == BlockStatus.MTQE_PASSED
    assert blocks[doc_ir.blocks[2].id].target_text == "标准无数字段落。"
    # Remaining blocks drafted via batch
    for b in doc_ir.blocks[3:]:
        assert blocks[b.id].status == BlockStatus.DRAFTED
    ledger.close()


def test_whole_book_batch_fallback_on_provider_error(tmp_path: Path) -> None:
    """When batch API submission fails, whole-book mode falls back to interactive drafting."""
    db_path = tmp_path / "job.sqlite"
    ledger = SQLiteJobLedger(db_path)
    doc_ir = _make_multi_chapter_ir(10, num_chapters=2)
    ledger.init_job("job_wb_err", doc_ir, target_lang="zh")

    provider = BrokenBatchProvider()
    router = ModelRouter(provider=provider, draft_model="test-batch-model")

    config = UBTConfig(
        offline_batch_enabled=True,
        enable_rolling_summary=False,
        batch_limit=5,
        batch_poll_interval=0.01,
        batch_poll_timeout=5.0,
    )

    asyncio.run(_run_stage(ledger, "job_wb_err", _make_manifest(2), router, config))

    # All 10 blocks drafted interactively
    blocks = ledger.get_all_blocks("job_wb_err")
    assert len(blocks) == 10
    assert all(b.status == BlockStatus.DRAFTED for b in blocks)
    assert all(b.target_text == "[FALLBACK-DRAFTED]" for b in blocks)
    assert len(provider.call_history) == 10
    ledger.close()


def test_whole_book_batch_partial_failures_redrafted_interactively(tmp_path: Path) -> None:
    """Lines failing inside the batch output are retried interactively."""
    db_path = tmp_path / "job.sqlite"
    ledger = SQLiteJobLedger(db_path)
    doc_ir = _make_multi_chapter_ir(12, num_chapters=2)
    ledger.init_job("job_wb_partial", doc_ir, target_lang="zh")

    failed_ids = {doc_ir.blocks[2].id, doc_ir.blocks[5].id}
    provider = WholeBookBatchProvider(partial_fail_ids=failed_ids)
    router = ModelRouter(provider=provider, draft_model="test-batch-model")

    config = UBTConfig(
        offline_batch_enabled=True,
        batch_poll_interval=0.01,
        batch_poll_timeout=5.0,
    )

    asyncio.run(_run_stage(ledger, "job_wb_partial", _make_manifest(2), router, config))

    blocks = {b.id: b for b in ledger.get_all_blocks("job_wb_partial")}
    # The 2 failed lines fell back to interactive draft
    assert blocks[doc_ir.blocks[2].id].target_text == "[INTERACTIVE-DRAFTED]"
    assert blocks[doc_ir.blocks[5].id].target_text == "[INTERACTIVE-DRAFTED]"
    # The other 10 lines were drafted via batch
    for b in doc_ir.blocks:
        if b.id not in failed_ids:
            assert "[BATCH-DRAFTED:" in (blocks[b.id].target_text or "")
        assert blocks[b.id].status == BlockStatus.DRAFTED

    assert len(provider.call_history) == 2
    ledger.close()


def test_whole_book_batch_idempotent_resume(tmp_path: Path) -> None:
    """A re-run over an already-drafted job creates no new batch (idempotent resume)."""
    db_path = tmp_path / "job.sqlite"
    ledger = SQLiteJobLedger(db_path)
    doc_ir = _make_multi_chapter_ir(10, num_chapters=1)
    ledger.init_job("job_wb_resume", doc_ir, target_lang="zh")

    class ResumeBatchProvider(WholeBookBatchProvider):
        def __init__(self) -> None:
            super().__init__()
            self.create_calls = 0

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            self.create_calls += 1
            return await super().create_batch_job(requests)

    provider = ResumeBatchProvider()
    router = ModelRouter(provider=provider, draft_model="test-batch-model")

    config = UBTConfig(
        offline_batch_enabled=True,
        batch_poll_interval=0.01,
        batch_poll_timeout=5.0,
    )

    # First run completes
    asyncio.run(_run_stage(ledger, "job_wb_resume", _make_manifest(1), router, config))
    assert provider.create_calls == 1

    # Blocks are all drafted
    blocks = ledger.get_all_blocks("job_wb_resume")
    assert all(b.status == BlockStatus.DRAFTED for b in blocks)

    # Running a second time on the drafted job creates 0 new batches
    asyncio.run(_run_stage(ledger, "job_wb_resume", _make_manifest(1), router, config))
    assert provider.create_calls == 1
    ledger.close()
