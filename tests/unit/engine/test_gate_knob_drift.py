"""A resume with changed gate knobs must say so out loud (W2-3).

Gate knobs decide what the export stage is willing to ship, but terminal
blocks are never re-evaluated on resume: every stage reads only the statuses
it owns (the quality gate claims ``drafted`` blocks, triage reads
``repair_pending``/``failed``). So a resume that raised ``--qe-threshold``
re-graded 0 blocks while the report's config snapshot recorded the new value
as if it had applied — the operator believed a paid re-run was re-grading the
book. The ingest stage now records the knobs per run and warns on a mismatch.
"""

from __future__ import annotations

import logging

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.ingest import _GATE_KNOB_KEY, _warn_gate_knob_drift
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast


def _ledger(tmp_path) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job_from_manifest("job", BookManifest(doc_id="doc", title="t", source_path="s.pdf"))
    return ledger


def test_first_run_records_gate_knobs_without_warning(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    ledger = _ledger(tmp_path)
    try:
        with caplog.at_level(logging.WARNING):
            _warn_gate_knob_drift(ledger, "job", UBTConfig())
        assert caplog.records == []
        stored = ledger.get_job_metadata_value("job", _GATE_KNOB_KEY)
        assert isinstance(stored, dict) and "qe_threshold" in stored
    finally:
        ledger.close()


def test_a_changed_threshold_warns_and_names_the_knob(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    ledger = _ledger(tmp_path)
    try:
        _warn_gate_knob_drift(ledger, "job", UBTConfig(qe_threshold=0.5))
        with caplog.at_level(logging.WARNING):
            _warn_gate_knob_drift(ledger, "job", UBTConfig(qe_threshold=0.9))
        assert any("gate knobs changed" in r.message for r in caplog.records)
        assert any("qe_threshold" in r.message for r in caplog.records)
        assert any("NOT re-evaluated" in r.message for r in caplog.records)
    finally:
        ledger.close()


def test_an_unchanged_resume_is_silent(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    ledger = _ledger(tmp_path)
    try:
        config = UBTConfig(qe_threshold=0.8)
        _warn_gate_knob_drift(ledger, "job", config)
        with caplog.at_level(logging.WARNING):
            _warn_gate_knob_drift(ledger, "job", UBTConfig(qe_threshold=0.8))
        assert caplog.records == []
    finally:
        ledger.close()
