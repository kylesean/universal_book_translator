"""The queue projection must not turn failed blocks into progress."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.api.job_catalog import summarize_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockStatus, BookManifest, ChapterIR, IRBlock
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast


def _block(index: int, status: BlockStatus) -> IRBlock:
    element = Paragraph(
        id=f"p{index}",
        spine_index=index,
        span=Span(page=1),
        text="Hello.",
    )
    return IRBlock(element=element, target_text="你好", status=status)


def test_failed_blocks_do_not_advance_the_job_list_progress(tmp_path: Path) -> None:
    job = "failedprogress"
    db = tmp_path / f"{job}.sqlite"
    with SQLiteJobLedger(db) as ledger:
        ledger.init_job_from_manifest(
            job,
            BookManifest(doc_id=job, title="t", source_path="in.pdf", target_lang="zh"),
        )
        ledger.append_chapter(
            job,
            ChapterIR(
                doc_id=job,
                chapter_id="c1",
                title="C",
                spine_index=0,
                blocks=[
                    _block(1, BlockStatus.MTQE_PASSED),
                    _block(2, BlockStatus.NEEDS_HUMAN),
                    _block(3, BlockStatus.FAILED),
                    _block(4, BlockStatus.FAILED),
                ],
            ),
        )

    summary = summarize_job(db)

    assert summary is not None
    assert summary["completed_blocks"] == 1
    assert summary["needs_human_blocks"] == 1
    assert summary["failed_blocks"] == 2
    assert summary["progress_percent"] == 50.0


def test_job_list_reports_blocked_human_blocks(tmp_path: Path) -> None:
    job = "blockedprogress"
    db = tmp_path / f"{job}.sqlite"
    with SQLiteJobLedger(db) as ledger:
        ledger.init_job_from_manifest(
            job,
            BookManifest(doc_id=job, title="t", source_path="in.pdf", target_lang="zh"),
        )
        ledger.append_chapter(
            job,
            ChapterIR(
                doc_id=job,
                chapter_id="c1",
                title="C",
                spine_index=0,
                blocks=[
                    _block(1, BlockStatus.MTQE_PASSED),
                    _block(2, BlockStatus.BLOCKED_HUMAN),
                ],
            ),
        )

    summary = summarize_job(db)

    assert summary is not None
    assert summary["blocked_human_blocks"] == 1
    assert summary["progress_percent"] == 100.0
