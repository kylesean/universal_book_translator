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

        # Deliberately exempt from the ownership SELECT: the additive merge can
        # only add real spend, never overwrite the successor's figures, and
        # refusing it would under-count the run that died mid-flight.
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
        assert ledger.get_job_usage(_JOB) == {"m": {"prompt_tokens": 10}}, (
            "the loser's real spend must survive the reclaim"
        )
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


def test_the_destructive_reset_paths_are_rejected_after_the_lease_is_reclaimed(
    tmp_path: Path,
) -> None:
    """The reset/merge/fail family can destroy a successor's paid work (force-
    FAIL its in-flight rows, NULL its checkpoints, zero its repair rounds), so
    each must carry the same in-transaction ownership assertion as the
    checkpoint paths. Removing any of these guards makes this test fail."""
    db_path, owner, ledger = _owner(tmp_path)
    # A transient-failure row is what gives ``reset_transient_failures``
    # something to write; without one it exits before reaching its
    # transaction and the guard there cannot fire.
    failed = IRBlock(
        element=make_element(
            id="b002",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="Another sentence.",
            bbox=BoundingBox(page=1, x0=50.0, y0=712.0, x1=350.0, y1=724.0),
        ),
    )
    failed.status = BlockStatus.FAILED
    failed.error_flags = ["Drafting error: provider 500"]
    ledger.append_chapter(
        _JOB,
        ChapterIR(doc_id=_JOB, chapter_id="c0", title="C0", spine_index=-1, blocks=[failed]),
    )
    successor = _reclaim(db_path)
    try:
        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.fail_non_terminal_blocks(_JOB, "lease lost")

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.reset_blocks_to_pending(["b001"], job_id=_JOB)

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.reset_blocks_to_repair(["b001"], job_id=_JOB)

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.reset_transient_failures(_JOB)

        rejected = IRBlock(
            element=make_element(
                id="b900",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Must never persist.",
                bbox=BoundingBox(page=1, x0=50.0, y0=724.0, x1=350.0, y1=736.0),
            ),
        )
        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.append_chapter(
                _JOB,
                ChapterIR(
                    doc_id=_JOB, chapter_id="c2", title="C2", spine_index=1, blocks=[rejected]
                ),
            )

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.clear_job_blocks(_JOB)

        with pytest.raises(LedgerWriterLeaseLostError):
            ledger.reserve_batch_job("idem-1", "batch-job-1")

        # Every rejected call rolled back: the successor's state is untouched.
        block = ledger.get_block("b001", _JOB)
        assert block is not None and block.status == BlockStatus.MTQE_PASSED
        assert block.target_text == "已完成。"
        failed_after = ledger.get_block("b002", _JOB)
        assert failed_after is not None and failed_after.status == BlockStatus.FAILED
        assert failed_after.error_flags == ["Drafting error: provider 500"]
        assert ledger.get_job_status(_JOB) == "initialized"
        assert ledger.get_block("b900", _JOB) is None, (
            "the rejected append_chapter must have rolled back"
        )
    finally:
        ledger.close()
        successor.release()
        owner.release()


def test_the_additive_usage_flush_survives_a_reclaim_and_never_resets(tmp_path: Path) -> None:
    """The lease-loser's in-flight tail is real provider spend: the additive
    accounting path must accept it after a reclaim, merge with whatever the
    successor has already recorded, and never be able to reset figures."""
    db_path, owner, ledger = _owner(tmp_path)
    successor = _reclaim(db_path)
    try:
        # The successor bills a progress event of its own first.
        successor_ledger = SQLiteJobLedger(db_path)
        try:
            successor_ledger.atomic_increment_job_usage(_JOB, {"m": {"prompt_tokens": 700}})
        finally:
            successor_ledger.close()

        # The stale owner flushes its in-flight tail.
        ledger.atomic_increment_job_usage(_JOB, {"m": {"prompt_tokens": 40}})

        usage = ledger.get_job_usage(_JOB)
        assert usage == {"m": {"prompt_tokens": 740}}, (
            "both writers' real spend must merge additively"
        )
    finally:
        ledger.close()
        successor.release()
        owner.release()
