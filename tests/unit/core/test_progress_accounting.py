from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import ProgressSnapshot, live_progress_percent

pytestmark = pytest.mark.fast


@pytest.fixture()
def ledger(tmp_path: Path) -> SQLiteJobLedger:
    return SQLiteJobLedger(tmp_path / "ledger.sqlite")


def _seed(ledger: SQLiteJobLedger, job_id: str, rows: list[tuple[str, int]]) -> None:
    with ledger._get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            "INSERT INTO job_meta (job_id, doc_id, source_path, target_lang, "
            "total_blocks, status, created_at) "
            "VALUES (?, 'doc', 'book.pdf', 'zh-CN', ?, 'running', '2026-01-01')",
            (job_id, len(rows)),
        )
        conn.executemany(
            "INSERT INTO blocks (block_id, job_id, flow_id, spine_index, "
            "block_type, source_text, status, skip_translate) "
            "VALUES (?, ?, 'main', ?, 'narrative', 'src', ?, ?)",
            [(f"b{i}", job_id, i, status, skip) for i, (status, skip) in enumerate(rows)],
        )
        conn.execute("COMMIT;")


def test_skipped_blocks_are_excluded_from_total_and_completed(
    ledger: SQLiteJobLedger,
) -> None:
    job_id = "job_progress_skip"
    _seed(
        ledger,
        job_id,
        [
            ("mtqe_passed", 0),
            ("mtqe_passed", 0),
            ("failed", 0),
            ("needs_human", 0),
            ("mtqe_passed", 1),
            ("mtqe_passed", 1),
            ("mtqe_passed", 1),
        ],
    )

    stats = ledger.get_job_stats(job_id)
    assert stats["total_all"] == 7, "the source really has 7 blocks"
    assert stats["skipped"] == 3
    assert stats["total"] == 4, "only translatable blocks form the denominator"
    assert stats["completed"] == 2, "a skipped block is not a completion"
    assert stats["failed"] == 1
    assert stats["needs_human"] == 1


def test_a_job_of_only_skipped_blocks_reports_no_work(ledger: SQLiteJobLedger) -> None:
    job_id = "job_all_skipped"
    _seed(ledger, job_id, [("mtqe_passed", 1), ("mtqe_passed", 1)])

    stats = ledger.get_job_stats(job_id)
    assert stats["total"] == 0
    assert stats["skipped"] == 2
    assert stats["completed"] == 0
    assert stats["total_all"] == 2


def test_the_real_job_shape_matches_the_reported_workload(
    ledger: SQLiteJobLedger,
) -> None:
    rows: list[tuple[str, int]] = []
    rows += [("mtqe_passed", 0)] * 90
    rows += [("needs_human", 0)] * 15
    rows += [("mtqe_passed", 1)] * 154
    _seed(ledger, "job_real", rows)

    stats = ledger.get_job_stats("job_real")
    assert stats["total"] == 105
    assert stats["skipped"] == 154
    assert stats["completed"] == 90
    assert stats["needs_human"] == 15
    processed = stats["completed"] + stats["failed"] + stats["needs_human"]
    assert processed == 105, "every translatable block reached a disposition"


def test_live_progress_uses_drafts_but_reserves_100_for_finalization() -> None:
    assert live_progress_percent(0, 50, 100) == 50.0
    assert live_progress_percent(20, 80, 100) == 80.0
    assert live_progress_percent(80, 0, 100) == 80.0
    assert live_progress_percent(100, 0, 100) == 100.0
    assert live_progress_percent(0, 100, 100) == 99.9
    assert live_progress_percent(0, 0, 0) == 0.0


def test_failed_blocks_never_advance_the_progress_bar() -> None:
    assert live_progress_percent(0, 0, 200) == 0.0
    assert live_progress_percent(10, 0, 200) == 5.0
    assert live_progress_percent(10, 50, 200) == 25.0


def test_needs_human_is_successful_progress_but_failed_is_not(
    ledger: SQLiteJobLedger,
) -> None:
    job_id = "job_progress_failed_vs_human"
    _seed(
        ledger,
        job_id,
        [
            ("mtqe_passed", 0),
            ("needs_human", 0),
            ("failed", 0),
            ("failed", 0),
        ],
    )

    snapshot = ProgressSnapshot.from_ledger(ledger.get_job_stats(job_id))

    assert snapshot.processed_blocks == 4
    assert snapshot.completed_blocks == 1
    assert snapshot.needs_human_blocks == 1
    assert snapshot.failed_blocks == 2
    assert snapshot.progress_percent == 50.0


def test_repair_pending_is_live_progress_not_a_regression(ledger: SQLiteJobLedger) -> None:
    job_id = "job_repair_pending_live"
    _seed(
        ledger,
        job_id,
        [
            ("mtqe_passed", 0),
            ("failed", 0),
            ("repair_pending", 0),
            ("repair_pending", 0),
            ("repair_pending", 0),
        ],
    )

    snapshot = ProgressSnapshot.from_ledger(ledger.get_job_stats(job_id))
    assert snapshot.processed_blocks == 2
    assert snapshot.repair_pending_blocks == 3
    assert snapshot.progress_percent == 60.0


def test_progress_snapshot_exposes_a_machine_readable_event_type() -> None:
    event = TranslationProgressEvent(
        event_type=EventType.BIBLE_EXTRACTED,
        job_id="job-activity",
        total_blocks=10,
        completed_blocks=0,
        message="Translation Bible established with 30 entries",
    )

    snapshot = ProgressSnapshot.from_event(event)

    assert snapshot.event_type == "bible_extracted"
    assert snapshot.message == "Translation Bible established with 30 entries"


def test_qe_measured_rides_the_snapshot_projection() -> None:
    """A 0.0 ``current_avg_qe`` must be tellable from "QE scored nothing".

    The queue/SSE projection dropped the ledger's ``qe_measured`` sentinel and
    reported the placeholder 0.0 as if it were a measurement.
    """
    unmeasured = ProgressSnapshot.from_ledger({"avg_qe_score": 0.0, "qe_measured": False})
    assert unmeasured.current_avg_qe == 0.0
    assert unmeasured.qe_measured is False

    measured = ProgressSnapshot.from_ledger({"avg_qe_score": 0.92, "qe_measured": True})
    assert measured.current_avg_qe == 0.92
    assert measured.qe_measured is True


def test_qe_measured_rides_the_event_projection() -> None:
    event = TranslationProgressEvent(
        event_type=EventType.DRAFT_BATCH_COMPLETED,
        job_id="job_qe_measured",
        total_blocks=4,
        completed_blocks=4,
        qe_measured=True,
    )
    snapshot = ProgressSnapshot.from_event(event)
    assert snapshot.qe_measured is True
    assert snapshot.to_payload()["qe_measured"] is True
