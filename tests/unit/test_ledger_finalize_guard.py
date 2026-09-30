"""RED: finalize_job must not silently mark non-terminal work completed."""

from pathlib import Path

import pytest

from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import LedgerError
from ubt.core.ir.models import BlockStatus, FlowID, IRBlock

pytestmark = pytest.mark.fast


def _doc(n: int = 3) -> SeedDoc:
    blocks = [
        IRBlock(
            id=f"b{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"s{i}",
        )
        for i in range(1, n + 1)
    ]
    return SeedDoc(
        doc_id="d-finalize-guard",
        source_path="/tmp/b.md",
        format_type="md",
        metadata={},
        blocks=blocks,
    )


def test_finalize_unknown_job_raises(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    with pytest.raises(LedgerError):
        ledger.finalize_job("no_such_job", status="completed")


def test_finalize_completed_with_non_terminal_blocks_raises(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    # default init status is pending/drafted (non-terminal)
    with pytest.raises(LedgerError):
        ledger.finalize_job("j1", status="completed")
    # job must not have flipped to completed
    assert ledger.get_job_status("j1") != "completed"


def test_finalize_completed_after_all_terminal_succeeds(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    ledger.save_checkpoints_batch(
        [
            {"block_id": "b001", "status": BlockStatus.MTQE_PASSED.value, "target_text": "t1"},
            {"block_id": "b002", "status": BlockStatus.REPAIRED.value, "target_text": "t2"},
        ]
    )
    ledger.finalize_job("j1", status="completed")
    assert ledger.get_job_status("j1") == "completed"


def test_finalize_failed_allows_non_terminal(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    ledger.finalize_job("j1", status="failed")
    assert ledger.get_job_status("j1") == "failed"


def test_finalize_completed_supersedes_a_cancelled_job(tmp_path: Path) -> None:
    """A rerun that finishes after a cancel flips the status to completed.

    A cancel stays authoritative only against a late *abort* (failed). The
    concurrent-cancel race this pin used to guard is closed at the
    coordination layer — REST and MCP cancels take the job-level writer lock
    and re-check the status under it — while the ledger-level refusal
    stranded reruns that genuinely delivered their artifact under a dead
    ``cancelled`` status forever.
    """
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    doc = SeedDoc(
        doc_id="d-supersede-cancelled",
        source_path="/tmp/b.md",
        format_type="md",
        metadata={},
        blocks=[
            IRBlock(
                id=f"b{i:03d}",
                flow_id=FlowID.MAIN_STORY,
                spine_index=i,
                source_text=f"s{i}",
                status=BlockStatus.MTQE_PASSED,
            )
            for i in (1, 2)
        ],
    )
    seed_job(ledger, "j1", doc, target_lang="zh")
    ledger.finalize_job("j1", status="cancelled")
    ledger.finalize_job("j1", status="completed")
    assert ledger.get_job_status("j1") == "completed"


def test_finalize_completed_supersedes_an_earlier_failure(tmp_path: Path) -> None:
    """A real completion must supersede a non-authoritative ``failed``.

    That failure can come from a stale owner whose lease was reclaimed, or from
    a previous attempt the user resumed. Refusing the override left delivered
    jobs reading ``failed`` and made every resume re-run the paid export.
    """
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    ledger.finalize_job("j1", status="failed")
    _all_terminal(ledger)
    ledger.finalize_job("j1", status="completed")
    assert ledger.get_job_status("j1") == "completed"


def _all_terminal(ledger: "SQLiteJobLedger") -> None:
    ledger.save_checkpoints_batch(
        [
            {"block_id": "b001", "status": BlockStatus.MTQE_PASSED.value, "target_text": "t1"},
            {"block_id": "b002", "status": BlockStatus.REPAIRED.value, "target_text": "t2"},
        ]
    )


def test_stale_failure_cannot_overwrite_completed(tmp_path: Path) -> None:
    """A stale owner's abort write must not re-mark a job the new owner finished."""
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    _all_terminal(ledger)
    ledger.finalize_job("j1", status="completed")

    ledger.finalize_job("j1", status="failed")
    assert ledger.get_job_status("j1") == "completed"


def test_finalize_failed_cannot_overwrite_cancelled(tmp_path: Path) -> None:
    """A worker abort/failure must not overwrite a user-initiated cancellation."""
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    ledger.finalize_job("j1", status="cancelled")
    ledger.finalize_job("j1", status="failed")
    assert ledger.get_job_status("j1") == "cancelled"


def test_completed_supersedes_a_cancelled_job(tmp_path: Path) -> None:
    """A user re-run that finishes after a cancel must flip the status.

    ``cancelled`` was authoritative against a later completion, so a rerun
    that actually delivered the artifact left the official status cancelled
    forever (a cancel stays authoritative only while nothing new finished).
    """
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    job_id = "job_cancelled_then_done"
    doc = SeedDoc(
        doc_id="d-cancelled-then-done",
        source_path="/tmp/b.md",
        format_type="md",
        metadata={},
        blocks=[
            IRBlock(
                id="b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                source_text="s1",
                status=BlockStatus.MTQE_PASSED,
            )
        ],
    )
    seed_job(ledger, job_id, doc, target_lang="zh")
    ledger.finalize_job(job_id, status="cancelled")
    # The rerun finished its (single, terminal) block before finalizing.
    ledger.finalize_job(job_id, status="completed")
    assert ledger.get_job_status(job_id) == "completed"
