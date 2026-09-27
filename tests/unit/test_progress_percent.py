"""Guards for the monotone progress projection.

The old display numerator was ``completed + drafted``: it read 100% while the
draft stage was still running and fell back to ``0/12 0%`` on the export event
of a run whose blocks all landed in the human queue. The snapshot's
``progress_percent`` must never decrease within a run, and its
``completed_blocks`` half must stay the same number the quality report's
``summary.completed_blocks`` reads (both come from the ledger's job stats).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.progress import ProgressSnapshot


def _event(
    event_type: EventType = EventType.MTQE_EVALUATED, **overrides: Any
) -> TranslationProgressEvent:
    fields: dict[str, Any] = {
        "event_type": event_type,
        "job_id": "job_mono",
        "timestamp": datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC),
        "total_blocks": 12,
        "completed_blocks": 0,
        "drafted_blocks": 0,
        "failed_blocks": 0,
        "needs_human_blocks": 0,
        "blocked_human_blocks": 0,
        "message": "",
    }
    fields.update(overrides)
    return TranslationProgressEvent(**fields)


def test_draft_stage_is_not_credited_as_one_hundred_percent() -> None:
    """Drafted blocks are in-flight work: the bar must not claim 100% yet."""
    snap = ProgressSnapshot.from_event(_event(drafted_blocks=12))
    assert snap.progress_percent == 0.0
    assert snap.processed_blocks == 0


def test_export_of_a_run_routed_to_the_human_queue_stays_at_one_hundred() -> None:
    """The E2E regression: every block NEEDS_HUMAN -> old display said 0/12 0%."""
    draft_phase = ProgressSnapshot.from_event(_event(drafted_blocks=12))
    export_phase = ProgressSnapshot.from_event(
        _event(
            EventType.EXPORT_COMPLETED,
            needs_human_blocks=12,
            artifact_path="/tmp/book_bilingual.md",
        )
    )
    assert export_phase.completed_blocks == 0  # ledger "completed" semantics kept
    assert export_phase.processed_blocks == 12
    assert export_phase.progress_percent == 100.0
    assert export_phase.progress_percent >= draft_phase.progress_percent


def test_percent_never_decreases_across_a_normal_run() -> None:
    """completed climbs while drafts are finalized; nothing may pull it back."""
    sequence = [
        _event(),
        _event(drafted_blocks=6),
        _event(drafted_blocks=12),
        _event(completed_blocks=8, drafted_blocks=4),
        _event(completed_blocks=10, needs_human_blocks=1, failed_blocks=1),
        # Export: drafted count is gone, two blocks were routed to humans.
        _event(
            EventType.EXPORT_COMPLETED,
            completed_blocks=10,
            needs_human_blocks=1,
            failed_blocks=1,
            artifact_path="/tmp/book_bilingual.md",
        ),
    ]
    percents = [ProgressSnapshot.from_event(e).progress_percent for e in sequence]
    assert percents == sorted(percents), percents
    assert percents[-1] == 100.0


def test_quarantined_blocks_count_as_processed_but_not_as_completed() -> None:
    """BLOCKED_HUMAN is finished processing; it is still not a "completion"."""
    snap = ProgressSnapshot.from_event(_event(blocked_human_blocks=12))
    assert snap.completed_blocks == 0
    assert snap.processed_blocks == 12
    assert snap.progress_percent == 100.0


def test_ledger_fold_matches_the_event_fold_for_the_same_counters() -> None:
    """from_event and from_ledger project one job the same way."""
    stats = {
        "total": 12,
        "completed": 9,
        "repaired": 3,
        "failed": 1,
        "needs_human": 2,
        "blocked_human": 0,
        "avg_qe_score": 0.88,
        "bottom_15_avg_qe": 0.61,
    }
    from_ledger = ProgressSnapshot.from_ledger(stats)
    from_event = ProgressSnapshot.from_event(
        _event(completed_blocks=9, failed_blocks=1, needs_human_blocks=2)
    )
    assert from_ledger.processed_blocks == from_event.processed_blocks == 12
    assert from_ledger.progress_percent == from_event.progress_percent == 100.0


def test_unknown_total_reports_zero_percent_not_a_division_error() -> None:
    snap = ProgressSnapshot.from_event(_event(total_blocks=0, completed_blocks=3))
    assert snap.progress_percent == 0.0


def test_progress_percent_survives_the_queue_row_round_trip() -> None:
    """The persisted row must carry the projection, not recompute it downstream."""
    snap = ProgressSnapshot.from_event(_event(completed_blocks=6, drafted_blocks=6))
    row = snap.to_payload()
    assert row["progress_percent"] == 50.0
    assert row["processed_blocks"] == 6
    assert ProgressSnapshot.model_validate(row) == snap
