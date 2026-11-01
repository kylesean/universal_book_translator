from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.ingest import _GATE_KNOB_KEY, _warn_gate_knob_drift
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast


def _ledger(tmp_path: Path) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job_from_manifest("job", BookManifest(doc_id="doc", title="t", source_path="s.pdf"))
    return ledger


def test_first_run_records_gate_knobs_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
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
    tmp_path: Path, caplog: pytest.LogCaptureFixture
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


def test_an_unchanged_resume_is_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ledger = _ledger(tmp_path)
    try:
        config = UBTConfig(qe_threshold=0.8)
        _warn_gate_knob_drift(ledger, "job", config)
        with caplog.at_level(logging.WARNING):
            _warn_gate_knob_drift(ledger, "job", UBTConfig(qe_threshold=0.8))
        assert caplog.records == []
    finally:
        ledger.close()
