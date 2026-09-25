"""Guards for the single progress projection shared by queue, API and MCP.

These pin two things the four delivery surfaces used to get to drift apart on:
the key names persisted in ``job_queue.progress_json`` (rows written by an older
build must stay readable) and the rule that every telemetry field the engine
emits is exposed by every surface that claims to mirror it.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.progress import ARTIFACT_KEYS, ProgressSnapshot
from ubt.core.job_options import sidecar_path


def _event(
    event_type: EventType = EventType.MTQE_EVALUATED,
    artifact_path: str | None = None,
    **overrides: Any,
) -> TranslationProgressEvent:
    fields: dict[str, Any] = {
        "event_type": event_type,
        "job_id": "job_snap",
        "timestamp": datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC),
        "total_blocks": 10,
        "completed_blocks": 6,
        "drafted_blocks": 2,
        "repaired_blocks": 3,
        "failed_blocks": 1,
        "current_avg_qe": 0.91,
        "bottom_15_avg_qe": 0.74,
        "estimated_cost_usd": 0.0312,
        "message": "evaluating",
        "artifact_path": artifact_path,
    }
    fields.update(overrides)
    return TranslationProgressEvent(**fields)


def test_queue_row_key_names_are_an_on_disk_contract() -> None:
    """``progress_json`` keys are written by one build and read by the next."""
    payload = ProgressSnapshot(
        total_blocks=1,
        estimated_cost_usd=0.0,
        output_file="/tmp/book_bilingual.pdf",
        report_file="/tmp/book_bilingual_quality_report.json",
        visual_report_file="/tmp/book_bilingual_visual_report.json",
    ).to_payload()
    assert set(payload) == {
        "total_blocks",
        "completed_blocks",
        "repaired_blocks",
        "failed_blocks",
        "needs_human_blocks",
        "blocked_human_blocks",
        "processed_blocks",
        "progress_percent",
        "current_avg_qe",
        "bottom_15_avg_qe",
        "estimated_cost_usd",
        "output_file",
        "report_file",
        "visual_report_file",
    }


def test_mid_run_row_keeps_the_counters_and_nothing_else() -> None:
    """A non-export event must not grow null artifact keys into the row."""
    payload = ProgressSnapshot.from_event(_event()).to_payload()
    assert set(payload) == {
        "total_blocks",
        "completed_blocks",
        "repaired_blocks",
        "failed_blocks",
        "needs_human_blocks",
        "blocked_human_blocks",
        "processed_blocks",
        "progress_percent",
        "current_avg_qe",
        "bottom_15_avg_qe",
        "estimated_cost_usd",
    }


def test_unknown_cost_stays_a_visible_null_and_artifacts_stay_absent() -> None:
    """The queue row keeps every counter key (SSE forwards it verbatim) and adds
    artifact paths only once an export names them."""
    payload = ProgressSnapshot.from_event(_event(estimated_cost_usd=None)).to_payload()
    assert payload["estimated_cost_usd"] is None
    assert not set(payload) & set(ARTIFACT_KEYS)
    assert set(payload) == {
        "total_blocks",
        "completed_blocks",
        "repaired_blocks",
        "failed_blocks",
        "needs_human_blocks",
        "blocked_human_blocks",
        "processed_blocks",
        "progress_percent",
        "current_avg_qe",
        "bottom_15_avg_qe",
        "estimated_cost_usd",
    }


@pytest.mark.parametrize("with_reports", [True, False])
def test_export_event_folds_the_artifact_triple(tmp_path: Path, with_reports: bool) -> None:
    output = tmp_path / "book_bilingual.pdf"
    output.write_bytes(b"%PDF-1.4")
    if with_reports:
        sidecar_path(output, "quality_report.json").write_text("{}", encoding="utf-8")
        sidecar_path(output, "visual_report.json").write_text("{}", encoding="utf-8")

    snap = ProgressSnapshot.from_event(
        _event(EventType.EXPORT_COMPLETED, artifact_path=str(output))
    )

    assert snap.output_file == str(output)
    assert (snap.report_file is not None) is with_reports
    assert (snap.visual_report_file is not None) is with_reports


def test_export_event_without_an_artifact_location_still_folds() -> None:
    """A malformed export event must not fail a finished job."""
    snap = ProgressSnapshot.from_event(
        _event(EventType.EXPORT_COMPLETED, artifact_path=None, message="")
    )
    assert snap.completed_blocks == 6
    assert snap.output_file is None


def test_ledger_column_names_are_translated_once() -> None:
    """The ledger spells these columns differently from the event; one map serves all."""
    stats = {
        "total": 12,
        "completed": 12,
        "repaired": 4,
        "failed": 0,
        "avg_qe_score": 0.88,
        "bottom_15_avg_qe": 0.61,
    }
    meta = {"estimated_cost_usd": 1.5, "output_file": "/tmp/a.pdf", "report_file": None}

    snap = ProgressSnapshot.from_ledger(stats, lambda key: meta.get(key))

    assert (snap.total_blocks, snap.completed_blocks) == (12, 12)
    assert snap.current_avg_qe == 0.88
    assert snap.bottom_15_avg_qe == 0.61
    assert snap.estimated_cost_usd == 1.5
    assert snap.output_file == "/tmp/a.pdf"
    assert snap.report_file is None
    assert snap.visual_report_file is None


def test_queue_row_round_trips_to_the_same_projection() -> None:
    """What the worker writes is what /status reads back, unchanged."""
    original = ProgressSnapshot.from_event(
        _event(EventType.EXPORT_COMPLETED, artifact_path="/tmp/book_bilingual.pdf")
    )
    assert ProgressSnapshot.model_validate(original.to_payload()) == original


def test_api_status_exposes_every_snapshot_field() -> None:
    """Adding a metric to the snapshot without exposing it here is the old drift."""
    from ubt.api.app import JobStatusResponse

    missing = set(ProgressSnapshot.model_fields) - set(JobStatusResponse.model_fields)
    assert not missing
    assert "visual_report_file" in JobStatusResponse.model_fields


def test_worker_and_api_project_the_same_dict_for_one_event() -> None:
    """The queue payload and the response body must not disagree on one event."""
    from ubt.api.app import JobStatusResponse

    event = _event(EventType.EXPORT_COMPLETED, artifact_path="/tmp/book_bilingual.pdf")
    payload = ProgressSnapshot.from_event(event).to_payload()
    body = JobStatusResponse(
        **payload, job_id="job_snap", status="running", created_at=event.timestamp
    )
    dumped = body.model_dump()
    for key, value in payload.items():
        assert dumped[key] == value
