"""Unit tests for `ubt re-export` CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.re_export import ReExportError
from ubt.core.exceptions import LedgerWriterLockConflictError

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_re_export_cli_invalid_job_id() -> None:
    result = runner.invoke(app, ["re-export", "../traversal/bad"])
    assert result.exit_code != 0
    assert "Invalid job id" in result.output


def test_re_export_cli_missing_ledger(tmp_path: Path) -> None:
    result = runner.invoke(app, ["re-export", "nonexistent-job-id", "--db-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "Ledger database not found" in result.output


def test_re_export_cli_success(tmp_path: Path) -> None:
    job_id = "test-job-123"
    db_file = tmp_path / f"{job_id}.sqlite"
    db_file.write_bytes(b"mock sqlite")
    mock_event = TranslationProgressEvent(
        job_id=job_id,
        event_type=EventType.EXPORT_COMPLETED,
        total_blocks=10,
        completed_blocks=10,
        artifact_path=str(tmp_path / "output.pdf"),
        message="Re-export complete",
    )

    with patch("ubt.core.engine.re_export.re_export_job", new_callable=AsyncMock) as mock_fn:
        mock_fn.return_value = mock_event
        result = runner.invoke(app, ["re-export", job_id, "--db-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Deliverable re-exported to:" in result.output
        mock_fn.assert_awaited_once()


def test_re_export_cli_lock_conflict(tmp_path: Path) -> None:
    job_id = "locked-job-123"
    db_file = tmp_path / f"{job_id}.sqlite"
    db_file.write_bytes(b"mock sqlite")

    with patch(
        "ubt.core.engine.re_export.re_export_job",
        side_effect=LedgerWriterLockConflictError("lock busy"),
    ):
        result = runner.invoke(app, ["re-export", job_id, "--db-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "refusing to re-export" in result.output


def test_re_export_cli_engine_error(tmp_path: Path) -> None:
    job_id = "error-job-123"
    db_file = tmp_path / f"{job_id}.sqlite"
    db_file.write_bytes(b"mock sqlite")

    with patch(
        "ubt.core.engine.re_export.re_export_job", side_effect=ReExportError("manifest broken")
    ):
        result = runner.invoke(app, ["re-export", job_id, "--db-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "Re-export failed:" in result.output
