from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ubt.api.job_catalog import _reportable_status, summarize_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterIR,
    IRBlock,
)
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast

_QUIET_FOR = 1000.0


def _block(index: int, status: BlockStatus) -> IRBlock:
    element = Paragraph(id=f"p{index}", spine_index=index, span=Span(page=1), text="Hello.")
    return IRBlock(element=element, target_text="你好", status=status)


def _ledger(tmp_path: Path, job: str, *, status: str, block: IRBlock) -> Path:
    db = tmp_path / f"{job}.sqlite"
    with SQLiteJobLedger(db) as ledger:
        ledger.init_job_from_manifest(
            job, BookManifest(doc_id=job, title="t", source_path="in.pdf", target_lang="zh")
        )
        ledger.append_chapter(
            job,
            ChapterIR(doc_id=job, chapter_id="c1", title="C", spine_index=0, blocks=[block]),
        )
        ledger.set_job_metadata_value(job, "estimated_cost_usd", 1.0)
        if status != "initialized":
            ledger.finalize_job(job, status=status)
    return db


def _status(db: Path) -> str:
    summary = summarize_job(db)
    assert summary is not None, "the fixture must be a real job ledger"
    return str(summary["status"])


def _age(db: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(db, (stamp, stamp))


def test_a_crashed_row_is_reported_failed(tmp_path: Path) -> None:
    db = _ledger(tmp_path, "deadjob", status="initialized", block=_block(1, BlockStatus.DRAFTED))
    _age(db, _QUIET_FOR)
    assert _status(db) == "failed"


def test_a_live_writer_overrides_a_quiet_ledger(tmp_path: Path) -> None:
    db = _ledger(tmp_path, "longcall", status="initialized", block=_block(1, BlockStatus.DRAFTED))
    _age(db, _QUIET_FOR)
    lock = LedgerWriterLock(db, "longcall")
    lock.acquire()
    try:
        assert _status(db) == "running"
    finally:
        lock.release()
    assert _status(db) == "failed"


def test_a_live_writer_on_a_fresh_ledger_reads_running(tmp_path: Path) -> None:
    db = _ledger(tmp_path, "livejob", status="initialized", block=_block(1, BlockStatus.DRAFTED))
    lock = LedgerWriterLock(db, "livejob")
    lock.acquire()
    try:
        assert _status(db) == "running"
    finally:
        lock.release()


def test_a_terminal_row_is_passed_through(tmp_path: Path) -> None:
    for stored in ("completed", "failed", "cancelled"):
        block = _block(1, BlockStatus.MTQE_PASSED)
        db = _ledger(tmp_path, f"{stored}job", status=stored, block=block)
        _age(db, _QUIET_FOR)
        assert _status(db) == stored


def test_a_fully_processed_row_reads_completed_even_before_finalize(tmp_path: Path) -> None:
    db = _ledger(
        tmp_path, "allblocks", status="initialized", block=_block(1, BlockStatus.MTQE_PASSED)
    )
    _age(db, _QUIET_FOR)
    assert _status(db) == "completed"


@pytest.mark.parametrize(
    ("stored", "live", "total", "done", "expected"),
    [
        ("completed", False, 0, 0, "completed"),
        ("failed", True, 5, 1, "failed"),
        ("cancelled", True, 5, 1, "cancelled"),
        ("initialized", True, 5, 1, "running"),
        ("initialized", False, 5, 1, "failed"),
        ("initialized", False, 5, 5, "completed"),
        ("initialized", True, 5, 5, "completed"),
        ("initialized", False, 0, 0, "failed"),
        ("initialized", True, 0, 0, "running"),
    ],
)
def test_the_status_rule(stored: str, live: bool, total: int, done: int, expected: str) -> None:
    # ``done`` counts every terminal block (completed + failed + needs_human +
    # blocked_human) — the same accounting /status and the MCP fallback apply.
    assert _reportable_status(stored, has_live_writer=live, total=total, done=done) == expected
