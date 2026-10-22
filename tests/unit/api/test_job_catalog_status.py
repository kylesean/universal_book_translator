"""`/jobs` and `/status` must agree about one ledger.

`job_meta.status` has no `running` value: the row is written `initialized` at
ingest and stays that way for the whole run, so the column alone cannot tell a
live job from one whose process died. `summarize_job` passed it through verbatim,
so `/jobs` showed `initialized` (which the console renders as a live run) for a
job that had crashed hours earlier, while `/status` for the same id — which does
normalize — reported it failed.

The rule is the one `/status` already applied: a terminal row passes through, a
row whose blocks are all accounted for is completed, and anything else is
`running` while a live writer holds the lease and `failed` once nothing does.

**The lease, not the file's mtime.** A first attempt used "the ledger has not
been written for N seconds" and was wrong in the one case that matters: a run
spends minutes inside a single model call without writing a checkpoint, so a
working job was reported dead. `test_a_live_writer_overrides_a_quiet_ledger`
pins that, and is the reason the mtime rule must not come back.
"""

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

#: Longer than any lease, so a "quiet ledger" cannot be mistaken for a live one.
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
    """The regression: a crashed run read as live forever."""
    db = _ledger(tmp_path, "deadjob", status="initialized", block=_block(1, BlockStatus.DRAFTED))
    _age(db, _QUIET_FOR)
    assert _status(db) == "failed"


def test_a_live_writer_overrides_a_quiet_ledger(tmp_path: Path) -> None:
    """The bug the mtime rule caused: a long model call is not a crash.

    A run can sit inside one provider call for longer than any staleness window
    without touching the ledger. The lease is the only honest signal.
    """
    db = _ledger(tmp_path, "longcall", status="initialized", block=_block(1, BlockStatus.DRAFTED))
    _age(db, _QUIET_FOR)
    lock = LedgerWriterLock(db, "longcall")
    lock.acquire()
    try:
        assert _status(db) == "running"
    finally:
        lock.release()
    # Once the lease is gone, nothing is working on it any more.
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
    """Every block accounted for is a finished run whatever the row says.

    That is exactly the condition ``finalize_job("completed")`` is guarded on,
    and the export stage finalizes *before* it records the rendered digest, so a
    process lost in that window leaves a fully translated book behind.
    """
    db = _ledger(
        tmp_path, "allblocks", status="initialized", block=_block(1, BlockStatus.MTQE_PASSED)
    )
    _age(db, _QUIET_FOR)
    assert _status(db) == "completed"


@pytest.mark.parametrize(
    ("stored", "live", "total", "completed", "expected"),
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
def test_the_status_rule(
    stored: str, live: bool, total: int, completed: int, expected: str
) -> None:
    assert (
        _reportable_status(stored, has_live_writer=live, total=total, completed=completed)
        == expected
    )
