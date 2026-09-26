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


def test_finalize_completed_cannot_overwrite_cancelled(tmp_path: Path) -> None:
    """An export that finishes after a concurrent cancel must not report success."""
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    ledger.finalize_job("j1", status="cancelled")
    ledger.finalize_job("j1", status="completed")
    assert ledger.get_job_status("j1") == "cancelled"


def test_finalize_completed_cannot_overwrite_failed(tmp_path: Path) -> None:
    """Lease loss / abort marks the ledger failed; a late export must not flip it."""
    ledger = SQLiteJobLedger(tmp_path / "l.sqlite")
    seed_job(ledger, "j1", _doc(2), target_lang="zh")
    ledger.finalize_job("j1", status="failed")
    ledger.finalize_job("j1", status="completed")
    assert ledger.get_job_status("j1") == "failed"
