"""Tests for SQLiteJobLedger record_visual_report and get_visual_report."""

from pathlib import Path

from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import FlowID, IRBlock


def test_ledger_record_and_get_visual_report(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_vis_001"

    doc_ir = SeedDoc(
        doc_id="doc_vis_001",
        source_path="test.pdf",
        format_type="pdf",
        blocks=[IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="hello")],
    )
    seed_job(ledger, job_id, doc_ir, target_lang="zh")

    # Initial report is None
    assert ledger.get_visual_report(job_id) is None

    # Record report
    sample_report = {
        "passed": True,
        "self_healed": False,
        "findings": [],
        "stats": {"total_pages": 10},
    }
    ledger.record_visual_report(job_id, sample_report)

    retrieved = ledger.get_visual_report(job_id)
    assert retrieved is not None
    assert retrieved["passed"] is True
    assert retrieved["stats"]["total_pages"] == 10
