"""Pipeline failures must leave a durable, operator-safe terminal reason."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import _mark_failed_unless_completed
from ubt.core.ir.models import BookManifest
from ubt.core.qe.defect_taxonomy import RETRYABLE_CONFIG_DRAFT_PREFIX

pytestmark = pytest.mark.fast


def test_abort_persists_the_safe_error_before_the_terminal_status(tmp_path: Path) -> None:
    """A restart must still be able to explain why the job failed.

    The in-memory ``JobRecord.error`` disappears with the process. Before this,
    the pipeline wrote only ``failed`` to the ledger, so ``/status`` fell back
    to a generic "server restarted" message after a restart.
    """
    db_path = tmp_path / "job.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            "job-1", BookManifest(doc_id="job-1", title="Book", source_path="book.pdf")
        )

        asyncio.run(
            _mark_failed_unless_completed(
                ledger,
                "job-1",
                status="failed",
                error="IntegrityViolationError: Resume the job to retry.",
            )
        )

        assert ledger.get_job_status("job-1") == "failed"
        assert ledger.get_job_metadata_value("job-1", "last_error") == (
            "IntegrityViolationError: Resume the job to retry."
        )


def test_a_completed_job_is_never_downgraded_by_the_abort_path(tmp_path: Path) -> None:
    db_path = tmp_path / "job.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            "job-1", BookManifest(doc_id="job-1", title="Book", source_path="book.pdf")
        )
        ledger.finalize_job("job-1", status="completed")

        asyncio.run(
            _mark_failed_unless_completed(
                ledger,
                "job-1",
                status="failed",
                error="late failure",
            )
        )

        assert ledger.get_job_status("job-1") == "completed"
        assert ledger.get_job_metadata_value("job-1", "last_error") is None


def test_completion_clears_the_previous_attempts_error(tmp_path: Path) -> None:
    """A resumed job that finishes must not keep the old failure attached.

    ``last_error`` drives the console's failure banner. Leaving it on a
    completed job made the operator read an outdated IntegrityViolationError
    beside a delivered artifact, with no way to tell whether recovery had
    actually worked.
    """
    db_path = tmp_path / "job.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            "job-1", BookManifest(doc_id="job-1", title="Book", source_path="book.pdf")
        )
        ledger.set_job_metadata_value(
            "job-1", "last_error", "IntegrityViolationError: recovery needed"
        )
        ledger.finalize_job("job-1", status="completed")

        assert ledger.get_job_status("job-1") == "completed"
        assert ledger.get_job_metadata_value("job-1", "last_error") is None


def test_config_failure_is_retryable_but_source_failure_is_not(tmp_path: Path) -> None:
    """A fixed model route must not leave 30 blocks permanently poisoned."""
    from ubt.core.engine.stages.draft import _draft_failure_prefix
    from ubt.core.exceptions import ModelProviderError
    from ubt.core.qe.defect_taxonomy import NON_RETRYABLE_DRAFT_PREFIX

    missing_model = ModelProviderError(
        "Model API error (404): no router for requested model",
        details={"status_code": 404},
    )
    malformed_source = ModelProviderError(
        "Model API error (422): unsupported source structure",
        details={"status_code": 422},
    )
    unreachable_endpoint = ModelProviderError(
        "HTTP connection failed: All connection attempts failed",
        details={"fail_fast": True, "reason": "transport_unreachable"},
    )

    assert _draft_failure_prefix(missing_model) == RETRYABLE_CONFIG_DRAFT_PREFIX
    assert _draft_failure_prefix(unreachable_endpoint) == RETRYABLE_CONFIG_DRAFT_PREFIX
    assert _draft_failure_prefix(malformed_source) == NON_RETRYABLE_DRAFT_PREFIX
