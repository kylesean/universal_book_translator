"""The ledger's late writes must verify the writer lease in their own transaction.

These tests bind a real ``LedgerWriterLock`` to a real ``SQLiteJobLedger``,
reclaim the lease behind the original owner's back, and then exercise every
write path that can run after a render. Removing any of the ownership checks
inside those transactions makes this file fail.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import LedgerWriterLeaseLostError
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

_JOB = "writer-ownership-job"


def _block() -> IRBlock:
    element = make_element(
        id="b001",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Done sentence.",
        bbox=BoundingBox(page=1, x0=50.0, y0=700.0, x1=350.0, y1=712.0),
    )
    block = IRBlock(element=element)
    block.target_text = "已完成。"
    block.status = BlockStatus.MTQE_PASSED
    return block


def _owner(tmp_path: Path) -> tuple[Path, LedgerWriterLock, SQLiteJobLedger]:
    db_path = tmp_path / "ledger.sqlite"
    lock = LedgerWriterLock(db_path, _JOB, lease_seconds=0.05)
    lock.acquire()
    ledger = SQLiteJobLedger(db_path)
    ledger.bind_writer_lock(lock)
    ledger.init_job_from_manifest(
        _JOB, BookManifest(doc_id=_JOB, title="t", source_path=str(tmp_path / "in.md"))
    )
    ledger.set_job_metadata_value(_JOB, "source_lang", "en")
    ledger.append_chapter(
        _JOB,
        ChapterIR(doc_id=_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=[_block()]),
    )
    return db_path, lock, ledger


def _reclaim(db_path: Path) -> LedgerWriterLock:
    """Take the lease with a different owner token, as a successor would."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("UPDATE job_locks SET lease_expires_at = 0 WHERE job_id = ?", (_JOB,))
    conn.close()
    successor = LedgerWriterLock(db_path, _JOB, lease_seconds=60.0)
    successor.acquire()
    return successor


def test_every_late_write_is_rejected_after_the_lease_is_reclaimed(tmp_path: Path) -> None:
    db_path, owner, ledger = _owner(tmp_path)
    successor = _reclaim(db_path)
    try:
        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.set_job_metadata_value(_JOB, "probe_after_loss", "written")

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.save_checkpoints_batch(
                [{"block_id": "b001", "status": "repaired", "target_text": "失租后写入"}],
                job_id=_JOB,
            )

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.record_page_geometry(
                _JOB, [{"page_no": 1, "tier": "L1", "delivered_size_pt": 9.5}]
            )

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.atomic_increment_job_usage(_JOB, {"m": {"prompt_tokens": 10}})

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.finalize_job(_JOB, status="completed")

        # Every rejected call must have rolled back: no state may have leaked
        # through before the ownership SELECT.
        assert ledger.get_job_metadata_value(_JOB, "probe_after_loss") is None
        block = ledger.get_block("b001", _JOB)
        assert block is not None and block.target_text == "已完成。"
        assert ledger.get_job_status(_JOB) == "initialized"
        assert ledger.retrieve_page_geometry(_JOB) == {}
        assert ledger.get_job_usage(_JOB) == {}
    finally:
        ledger.close()
        successor.release()
        owner.release()


def test_the_check_runs_on_the_write_transaction_not_the_standalone_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, owner, ledger = _owner(tmp_path)
    successor = _reclaim(db_path)
    # If a future refactor moves the check back to the pre-transaction
    # ``assert_owned()`` probe, this no-op would let the write through. The
    # in-transaction path uses ``assert_owned_in_transaction`` and still rejects.
    monkeypatch.setattr(LedgerWriterLock, "assert_owned", lambda self: None)
    try:
        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.set_job_metadata_value(_JOB, "probe_after_loss", "written")
        assert ledger.get_job_metadata_value(_JOB, "probe_after_loss") is None
    finally:
        ledger.close()
        successor.release()
        owner.release()
