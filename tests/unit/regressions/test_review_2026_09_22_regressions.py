"""Regression guards for the 2026-09-22 review round (round 5).

One test per defect found by that round, each reproducing the user-visible
outcome rather than the internals, so reverting the fix fails the test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.mock_providers import TokenEchoMockProvider
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.ir.models import BlockStatus, ChapterIR, DocumentIR, IRBlock
from ubt.core.job_options import artifact_and_report_paths, sidecar_path
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.router import ModelRouter


def _doc_ir(count: int) -> tuple[DocumentIR, ChapterIR]:
    """A one-chapter document whose block ids match what the tests stamp."""
    chapter = ChapterIR(
        doc_id="reg_doc",
        chapter_id="ch01",
        title="One",
        spine_index=1,
        blocks=[
            IRBlock(id=f"ch01#b{i:03d}", spine_index=i, source_text=f"Paragraph {i}.")
            for i in range(1, count + 1)
        ],
    )
    return (
        DocumentIR(
            doc_id="reg_doc",
            source_path="/tmp/reg.md",
            format_type="markdown",
            blocks=chapter.blocks,
        ),
        chapter,
    )


_BOOK = """# Chapter One

The mill ran through the night, and the river carried the noise away.

# Chapter Two

By morning the wheel had stopped, and the builder came down to look at it.
"""


async def _translate(
    src: Path, out: Path, db_dir: Path, job_id: str, *, reply: str = "离线译文段落。"
) -> None:
    router = ModelRouter(
        provider=TokenEchoMockProvider(default_response=reply),
        draft_model="mock-draft",
        repair_model="mock-repair",
        rate_limiter=AdaptiveTokenBucket(
            initial_rpm=1_000_000,
            max_rpm=1_000_000,
            initial_tpm=1_000_000_000,
            max_tpm=1_000_000_000,
        ),
    )
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=db_dir, rate_limit_rpm=100_000, tm_enabled=False),
        router=router,
        qe_runner=MockQERunner(default_score=0.92),
    )
    async for _ in orchestrator.run(
        input_path=src, output_path=out, target_lang="zh", job_id=job_id
    ):
        pass


def _sidecars(out: Path) -> list[Path]:
    return [
        sidecar_path(out, "quality_report.json"),
        sidecar_path(out, "metrics.json"),
        sidecar_path(out, "visual_report.json"),
    ]


@pytest.mark.asyncio
async def test_interrupted_export_leaves_no_report_for_a_deliverable_it_never_saw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that dies after the render must not leave the last run's reports.

    Reproduction of the round-5 finding: the deliverable is written first and the
    reports later, so an interrupt (or the blocking visual gate raising) in that
    window left a ``*_quality_report.json`` claiming a finished book whose text it
    had never seen -- and ``artifact_and_report_paths`` re-attaches those names to
    later runs, so the stale report outlived the process that orphaned it.
    """
    src = tmp_path / "book.md"
    src.write_text(_BOOK, encoding="utf-8")
    out = tmp_path / "out_bilingual.md"

    await _translate(src, out, tmp_path / "db1", "job_sidecar_first")
    written = [path for path in _sidecars(out) if path.exists()]
    assert written, "the first run wrote no sidecar reports; the test proves nothing"
    first_deliverable = out.read_bytes()

    async def _die(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt("operator stopped the run after the render")

    monkeypatch.setattr(
        "ubt.core.engine.stages.export._build_reports",
        _die,
        raising=True,
    )
    with pytest.raises(BaseException):  # noqa: B017 - KeyboardInterrupt is the point
        await _translate(src, out, tmp_path / "db2", "job_sidecar_second", reply="第二轮离线译文。")

    assert out.read_bytes() != first_deliverable, "the second run never replaced the deliverable"
    survivors = [path.name for path in _sidecars(out) if path.exists()]
    assert not survivors, f"stale reports describe the new deliverable: {survivors}"


class _OneFastThenHangingRepairLoop:
    """Repairs one block immediately and never returns for the rest."""

    def __init__(self, fast_block_id: str) -> None:
        self._fast = fast_block_id

    def select_repair_candidates(self, eligible: list[Any]) -> list[Any]:
        return list(eligible)

    async def repair_single_block(self, *, block: Any, **kwargs: Any) -> Any:
        if block.id != self._fast:
            await asyncio.Event().wait()  # never resolves: the round is cancelled
        return block.model_copy(
            update={
                "target_text": "已付费的修复译文。",
                "status": BlockStatus.MTQE_PASSED,
                "repair_rounds": 1,
                "mtqe_score": 0.82,
            }
        )


@pytest.mark.asyncio
async def test_cancel_mid_repair_round_keeps_the_repairs_already_paid_for(
    tmp_path: Path,
) -> None:
    """A cancel must not throw away a repair the provider already charged for.

    The round used to collect every candidate's update in memory and write them
    in one batch after ``gather`` returned, so cancelling (or any failure that
    aborted the round) discarded the completed ones: the block stayed
    REPAIR_PENDING with its old draft and the resume re-sent the identical
    prompt. Each candidate is recorded the moment it returns now.
    """
    from tests.stage_ctx_factory import build_stage_ctx
    from ubt.core.engine.stages.repair import run_repair_stage

    doc, chapter = _doc_ir(3)
    ledger = SQLiteJobLedger(tmp_path / "repair.sqlite")
    ledger.init_job("job_repair", doc, target_lang="zh")
    ledger.append_chapter("job_repair", chapter)
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": f"ch01#b00{i}",
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": f"OLD DRAFT {i}",
                "mtqe_score": 0.31,
            }
            for i in (1, 2, 3)
        ]
    )
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_repair",
        repair_loop=_OneFastThenHangingRepairLoop("ch01#b001"),
    )

    async def _drain() -> None:
        async for _ in run_repair_stage(ctx, defer_unresolved_to_triage=True):
            pass

    async def _first_persisted_target() -> str:
        """Wait for b001's repair to reach the ledger (or time out)."""
        for _ in range(200):
            await asyncio.sleep(0.01)
            rows_now = {b.id: b for b in ledger.get_all_blocks("job_repair")}
            if rows_now["ch01#b001"].target_text != "OLD DRAFT 1":
                return str(rows_now["ch01#b001"].target_text)
        return ""

    task = asyncio.create_task(_drain())
    # The scenario only exists once the first paid repair has landed; before
    # that there is nothing for the cancel to lose.
    assert await _first_persisted_target() == "已付费的修复译文。", (
        "the round's first paid repair never reached the ledger"
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    rows = {b.id: b for b in ledger.get_all_blocks("job_repair")}
    assert rows["ch01#b001"].target_text == "已付费的修复译文。", (
        "the paid repair was lost with the cancel, so a resume bills it again"
    )
    assert rows["ch01#b002"].target_text == "OLD DRAFT 2"
    ledger.close()


def test_deliverables_sharing_a_stem_do_not_share_a_report(tmp_path: Path) -> None:
    """``book.epub`` and ``book.md`` are two documents, not two names for one.

    Sidecar names used to hang off the bare stem, so both defaults
    (``book_bilingual.epub`` / ``book_bilingual.md``, and the same trap for a
    zh-then-ja re-run of one file) resolved to a single
    ``book_bilingual_quality_report.json``. Whichever run finished last owned
    the name: the other's report was overwritten by a document whose
    ``output_path`` pointed at a file no reader finds beside it, while
    ``artifact_and_report_paths`` kept handing both jobs the same JSON.
    """
    epub = sidecar_path(tmp_path / "book_bilingual.epub", "quality_report.json")
    markdown = sidecar_path(tmp_path / "book_bilingual.md", "quality_report.json")
    assert epub != markdown, f"two deliverables share {epub.name}"

    # The reader resolves the same names the writers use.
    epub.touch()
    _, report, _ = artifact_and_report_paths(tmp_path / "book_bilingual.epub")
    assert report == epub
    _, missing, _ = artifact_and_report_paths(tmp_path / "book_bilingual.md")
    assert missing is None, "the reader crossed over into the other document's report"

    metrics = sidecar_path(tmp_path / "book_bilingual.epub", "metrics.json")
    assert metrics.name == "book_bilingual_epub_metrics.json"
