"""`/jobs` must not report a crashed run as live.

`job_meta.status` has no `running` value: the row is written `initialized` at
ingest and stays that way for the whole run, so the column alone cannot tell a
live job from one whose process died. `summarize_job` passed it through verbatim,
so `/jobs` showed `initialized` (which the console renders as a live run) for a
job that had crashed hours earlier, while `/status` for the same id — which does
normalize — reported it failed.

The rule here mirrors MCP's `ubt_status`: a non-terminal row is only called
interrupted once the ledger has gone quiet, because a live run in another
process keeps writing checkpoints.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ubt.api.job_catalog import LEDGER_STALENESS_SECONDS, _reportable_status, summarize_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterIR,
    IRBlock,
)
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast


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


def test_a_fresh_non_terminal_row_stays_as_stored(tmp_path: Path) -> None:
    """It may be a live run in another process; `initialized` is the honest answer."""
    db = _ledger(tmp_path, "freshjob", status="initialized", block=_block(1, BlockStatus.DRAFTED))
    _age(db, 10)
    assert _status(db) == "initialized"


def test_a_quiet_non_terminal_row_is_reported_failed(tmp_path: Path) -> None:
    """The regression: a crashed run read as live forever."""
    db = _ledger(tmp_path, "deadjob", status="initialized", block=_block(1, BlockStatus.DRAFTED))
    _age(db, LEDGER_STALENESS_SECONDS + 600)
    assert _status(db) == "failed"


def test_a_terminal_row_is_passed_through(tmp_path: Path) -> None:
    db = _ledger(tmp_path, "donejob", status="completed", block=_block(1, BlockStatus.MTQE_PASSED))
    _age(db, LEDGER_STALENESS_SECONDS + 600)
    assert _status(db) == "completed"


def test_a_fully_processed_row_reads_completed_even_before_finalize(tmp_path: Path) -> None:
    """Every block accounted for is a finished run whatever the row says."""
    db = _ledger(
        tmp_path, "allblocks", status="initialized", block=_block(1, BlockStatus.MTQE_PASSED)
    )
    _age(db, 10)
    assert _status(db) == "completed"


@pytest.mark.parametrize(
    ("stored", "quiet", "total", "completed", "expected"),
    [
        ("completed", None, 0, 0, "completed"),
        ("failed", 10.0, 5, 1, "failed"),
        ("cancelled", 10.0, 5, 1, "cancelled"),
        ("initialized", 10.0, 5, 1, "initialized"),
        ("initialized", None, 5, 1, "initialized"),
        ("initialized", 10_000.0, 5, 1, "failed"),
        ("initialized", 10.0, 5, 5, "completed"),
        ("initialized", 10_000.0, 0, 0, "failed"),
    ],
)
def test_the_status_rule(
    stored: str, quiet: float | None, total: int, completed: int, expected: str
) -> None:
    assert (
        _reportable_status(stored, quiet_seconds=quiet, total=total, completed=completed)
        == expected
    )
