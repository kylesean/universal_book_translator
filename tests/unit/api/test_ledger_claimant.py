"""The ledger-level output-path claimant — the cross-process half of the claim protocol.

The in-memory claimant cannot see a run another process started: the MCP
server and the REST server each hold their own manager over the same
``db_dir``, and a CLI ``ubt translate`` writes ledgers there too, so two
pipelines could write (and bill) one deliverable. The per-job ledger is the
durable record every writer shares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.routes.shared import ledger_output_path_claimant
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.writer_lock import LedgerWriterLock

pytestmark = pytest.mark.fast

_API_KEY = "unit-test-key"
_AUTH = {"X-API-Key": _API_KEY}


def _ledger(
    db_dir: Path,
    job_id: str,
    output_path: str,
    *,
    agent_state: str | None = None,
    terminal: str | None = None,
) -> Path:
    db_path = db_dir / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_placeholder(
            job_id,
            "/tmp/book.md",
            "zh",
            submitted_request={
                "input_path": "/tmp/book.md",
                "job_id": job_id,
                "output_path": output_path,
            },
        )
        if agent_state:
            ledger.set_agent_session_state(job_id, agent_state, now=0.0)
        if terminal:
            ledger.finalize_job(job_id, status=terminal)
    return db_path


def _held(tmp_path: Path, job_id: str) -> Any:
    """Acquire the writer lease so the row reads as actively written."""
    lock = LedgerWriterLock(tmp_path / f"{job_id}.sqlite", job_id)
    lock.acquire()
    return lock


def test_a_live_writer_claiming_the_output_path_is_reported(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    claimed = str((tmp_path / "out.md").resolve())
    _ledger(db_dir, "liverun01", claimed)
    lock = _held(db_dir, "liverun01")
    try:
        assert ledger_output_path_claimant(db_dir, None, Path(claimed)) == "liverun01"
    finally:
        lock.release()


def test_a_dead_run_does_not_block_the_path_forever(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    claimed = str((tmp_path / "out.md").resolve())
    _ledger(db_dir, "deadrun01", claimed)
    lock = _held(db_dir, "deadrun01")
    lock.release()  # the process died: the lease is gone

    assert ledger_output_path_claimant(db_dir, None, Path(claimed)) is None


def test_a_terminal_ledger_is_never_a_claimant(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    claimed = str((tmp_path / "out.md").resolve())
    _ledger(db_dir, "donerun01", claimed, terminal="completed")

    assert ledger_output_path_claimant(db_dir, None, Path(claimed)) is None


def test_a_parked_agent_session_counts_as_live(tmp_path: Path) -> None:
    """Agent jobs are intentionally lock-free between tool calls."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    claimed = str((tmp_path / "out.md").resolve())
    _ledger(db_dir, "agentrun1", claimed, agent_state="awaiting_agent")

    assert ledger_output_path_claimant(db_dir, None, Path(claimed)) == "agentrun1"


def test_other_output_paths_do_not_match(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    _ledger(db_dir, "otherrun1", str((tmp_path / "other.md").resolve()))
    lock = _held(db_dir, "otherrun1")
    try:
        assert ledger_output_path_claimant(db_dir, None, Path(str(tmp_path / "out.md"))) is None
    finally:
        lock.release()


def test_the_own_id_is_excluded(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    claimed = str((tmp_path / "out.md").resolve())
    _ledger(db_dir, "samescan", claimed)
    lock = _held(db_dir, "samescan")
    try:
        assert ledger_output_path_claimant(db_dir, "samescan", Path(claimed)) is None
    finally:
        lock.release()


def test_the_rest_submit_refuses_a_foreign_live_claim(tmp_path: Path) -> None:
    """A live run in ANOTHER process now blocks a REST submit on the same path.

    The in-memory claimant never saw cross-process embedded jobs; the ledger
    scan closes the MCP/CLI gap from the REST side too.
    """
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    doc = tmp_path / "book.md"
    doc.write_text("# Chapter 1\n\nClaimed path contract.\n", encoding="utf-8")
    claimed_out = tmp_path / "foreign_out.md"
    db_dir = config.db_dir
    db_dir.mkdir(parents=True, exist_ok=True)
    _ledger(db_dir, "foreignrun", str(claimed_out))
    lock = _held(db_dir, "foreignrun")
    try:
        with TestClient(create_app(config=config)) as client:
            response = client.post(
                "/jobs/submit",
                json={"input_path": str(doc), "output_path": str(claimed_out)},
                headers=_AUTH,
            )
    finally:
        lock.release()

    assert response.status_code == 409, response.text
    assert "foreignrun" in response.text
