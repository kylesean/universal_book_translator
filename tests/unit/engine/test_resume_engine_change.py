"""A provider fix must not strand a paid ledger behind the old model identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.ingest import _guard_run_identity
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast


def _ledger(tmp_path: Path, *, status: str) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job_from_manifest(
        "job-1", BookManifest(doc_id="job-1", title="Book", source_path="book.pdf")
    )
    ledger.set_job_metadata_value(
        "job-1",
        "run_identity",
        {
            "profile_name": "paper",
            "engine_signature": "dmold-model-rmold-model-psrich",
            "mock_run": False,
        },
    )
    if status != "initialized":
        ledger.finalize_job("job-1", status=status)
    return ledger


def test_a_terminal_job_may_resume_after_the_provider_model_changes(
    tmp_path: Path,
) -> None:
    """The first job failed because its model was missing; fixing it is the fix.

    Refusing the resume made the operator's only recovery a destructive
    re-parse, even though every pending block and the source fingerprint were
    still valid.
    """
    with _ledger(tmp_path, status="failed") as ledger:
        _guard_run_identity(
            ledger,
            "job-1",
            "paper",
            "dmindex-translate-9b-rmindex-translate-9b-psrich",
            False,
        )

        identity = ledger.get_job_metadata_value("job-1", "run_identity")
        assert identity["engine_signature"] == ("dmindex-translate-9b-rmindex-translate-9b-psrich")
        assert ledger.get_job_metadata_value("job-1", "resume_engine_change") == {
            "from_engine_signature": "dmold-model-rmold-model-psrich",
            "to_engine_signature": "dmindex-translate-9b-rmindex-translate-9b-psrich",
        }


def test_a_live_job_still_refuses_an_engine_change(tmp_path: Path) -> None:
    with (
        _ledger(tmp_path, status="initialized") as ledger,
        pytest.raises(DocumentParseError, match="failed or cancelled"),
    ):
        _guard_run_identity(ledger, "job-1", "paper", "new-engine", False)


def test_profile_and_rehearsal_mode_remain_hard_guards(tmp_path: Path) -> None:
    with _ledger(tmp_path, status="failed") as ledger:
        with pytest.raises(DocumentParseError, match="different profile"):
            _guard_run_identity(ledger, "job-1", "textbook", "new-engine", False)
        with pytest.raises(DocumentParseError, match="different profile"):
            _guard_run_identity(ledger, "job-1", "paper", "new-engine", True)
