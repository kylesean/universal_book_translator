"""Re-export must not race a live writer or own another run's blocks.

Two regressions for the single-writer invariant:

1. A re-export over a job with non-terminal blocks must be REFUSED. The old
   behavior force-FAILED them (``fail_non_terminal_blocks``) and finalized the
   job ``completed`` — a re-export issued while a CLI resume was mid-flight
   destroyed the resume's work and flipped the job status behind its back
   (review 2026-10-14, W1-1).
2. A re-export must take the job's ``LedgerWriterLock``. Without it, the
   pre-check (queue rows + in-process manager) cannot see a CLI-started run,
   so the API happily re-exported a job another process was writing.

Both run at fast tier with no typst: the refusal and the lock conflict fire
before the render path is ever reached.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.re_export import ReExportError, re_export_job
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import IntegrityViolationError, LedgerWriterLockConflictError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterIR,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast

_JOB = "reexport-race-job"


def _block(index: int, source: str, target: str, status: BlockStatus) -> IRBlock:
    element = make_element(
        id=f"b{index:03d}",
        spine_index=index,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        bbox=BoundingBox(page=1, x0=50.0, y0=700.0 - index * 20, x1=350.0, y1=712.0 - index * 20),
    )
    block = IRBlock(element=element)
    block.target_text = target
    block.status = status
    return block


def _seed(
    tmp_path: Path, blocks: list[IRBlock], *, job_status: str = "running"
) -> tuple[UBTConfig, Path]:
    """A ledger whose blocks/status are whatever the caller plants."""
    config = UBTConfig(db_dir=tmp_path / "db", allowed_dirs=str(tmp_path))
    config.db_dir.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "in.md"
    source.write_text("# t\n\nDone sentence.\n", encoding="utf-8")
    db_path = config.db_dir / f"{_JOB}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            _JOB, BookManifest(doc_id=_JOB, title="t", source_path=str(source))
        )
        ledger.set_job_metadata_value(_JOB, "source_lang", "en")
        ledger.append_chapter(
            _JOB,
            ChapterIR(doc_id=_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=blocks),
        )
        if job_status != "initialized":
            # finalize_job refuses non-'completed' overrides on non-terminal
            # blocks, and the seeded blocks below are terminal; go through the
            # same terminal write the ledger exposes for a realistic mid-flight
            # status ('failed'/'cancelled'), or the raw column write for
            # 'running' (there is no public setter for an in-flight status).
            ledger.finalize_job(_JOB, status=job_status)
    return config, db_path


async def test_re_export_refuses_non_terminal_blocks_instead_of_failing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old path force-FAILED in-flight blocks and finalized 'completed'."""
    blocks = [
        _block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED),
        _block(2, "Still drafting here.", "", BlockStatus.DRAFTED),
    ]
    config, db_path = _seed(tmp_path, blocks, job_status="failed")
    # Precondition resolution (source, output path) legitimately precedes the
    # refusal; plant a recorded output so the run reaches the export stage,
    # and stub the collaborators so no provider is constructed (a re-export
    # never calls the router — every block here is non-terminal on purpose).
    with SQLiteJobLedger(db_path) as ledger:
        ledger.set_job_metadata_value(_JOB, "output_file", str(tmp_path / "out.pdf"))
    import ubt.core.engine.re_export as re_export_module

    dummy_router, dummy_qe = object(), object()
    monkeypatch.setattr(
        re_export_module,
        "_default_collaborators",
        lambda config, router, qe: (dummy_router, dummy_qe),
    )
    with pytest.raises(IntegrityViolationError, match="non-terminal"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)
    # The in-flight block must be untouched: not force-FAILED, still drafted.
    with SQLiteJobLedger(db_path, read_only=True) as ledger:
        row = ledger.get_block("b002", _JOB)
        assert row is not None
        assert row.status is BlockStatus.DRAFTED
        assert row.error_flags == []
        # And the job's terminal status must be untouched (the seed wrote
        # 'failed'; the old path would have superseded it to 'completed').
        assert ledger.get_job_status(_JOB) == "failed"


async def test_re_export_holds_the_writer_lock_against_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held writer lock (CLI run the queue cannot see) refuses the re-export."""
    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks)
    lock = LedgerWriterLock(db_path, _JOB)
    lock.acquire()  # simulates the running owner
    try:
        # A lock conflict is a distinct class from an integrity violation: the
        # caller may retry the first and must not retry the second.
        with pytest.raises(LedgerWriterLockConflictError, match="already being written"):
            await re_export_job(db_path=db_path, job_id=_JOB, config=config)
    finally:
        lock.release()


async def test_re_export_still_reports_missing_output_after_the_terminal_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully-terminal job without a recorded output path fails with its own
    reason — proving the non-terminal gate does not mask ordinary ReExportError
    contracts."""
    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks)
    with pytest.raises(ReExportError, match="no recorded output path"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)
