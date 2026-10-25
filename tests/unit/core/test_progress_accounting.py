"""Progress must count translation work, not blocks the skip rule finished.

On a real 26-page PDF (259 blocks) the skip rule marked 154 of them
``mtqe_passed`` without translating anything — images, list entries and short
headings. The old query counted every row on both sides of the fraction, so
skipped blocks were "done" by construction:

    total     = COUNT(*)                                   -> 259
    completed = SUM(status IN ('mtqe_passed','repaired'))  -> 244
    percent   -> 100.0%   for 90 translated blocks out of 105

Two books with identical translation load therefore reported different progress
depending on how many images they happened to contain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import ProgressSnapshot, live_progress_percent

pytestmark = pytest.mark.fast


@pytest.fixture()
def ledger(tmp_path: Path) -> SQLiteJobLedger:
    return SQLiteJobLedger(tmp_path / "ledger.sqlite")


def _seed(ledger: SQLiteJobLedger, job_id: str, rows: list[tuple[str, int]]) -> None:
    """Write one job with ``rows`` of ``(status, skip_translate)``."""
    with ledger._get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            "INSERT INTO job_meta (job_id, doc_id, source_path, target_lang, "
            "total_blocks, status, created_at) "
            "VALUES (?, 'doc', 'book.pdf', 'zh-CN', ?, 'running', '2026-01-01')",
            (job_id, len(rows)),
        )
        # Only the NOT NULL columns the query touches: this fixture is about the
        # aggregate, not about block fidelity (which the ledger's own tests cover).
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
            # Translated / terminal work.
            ("mtqe_passed", 0),
            ("mtqe_passed", 0),
            ("failed", 0),
            ("needs_human", 0),
            # Skipped: never translated, must not count as work.
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
    """No division by zero, and no "100% of nothing" claim either."""
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
    """The measured distribution: 105 translatable, 154 skipped, 15 to a human.

    The panel used to say 244/259 = 100% for this. The honest reading is that all
    105 translatable blocks reached a disposition, 15 of them needing a person.
    """
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


def test_repair_pending_is_live_progress_not_a_regression(ledger: SQLiteJobLedger) -> None:
    """Starting repair must not erase already-credited live work.

    A real run reached 86.6% while 504 blocks were drafted; repair then moved
    all of them to repair_pending. The old snapshot only credited drafted, so
    the next repair heartbeat fell to 31.4% even though the same blocks were
    still queued for work.
    """
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
