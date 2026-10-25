"""`/jobs` and `/status` must answer the same thing about the same ledger.

They are two views of one durable store, read by different code paths — the
catalog builds summaries from `db_dir/*.sqlite`, the status route has an
in-memory branch and a disk fallback. When they disagree the console shows a
crashed job as live in the list and as failed on its own page, and the operator
has no way to tell which one is lying.

The cases below are the ones that actually differ in behaviour: a crash, a live
writer holding the lease through a long call, and a book whose blocks all
reached a terminal state before the row was finalized.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.job_catalog import summarize_job
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.ir.models import BlockStatus, BookManifest, ChapterIR, IRBlock
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast

_API_KEY = "status-agreement-key"
_JOB = "agreejob001"
_QUIET_FOR = 1000.0


def _block(status: BlockStatus) -> IRBlock:
    element = Paragraph(id="p1", spine_index=1, span=Span(page=1), text="Hello.")
    return IRBlock(element=element, target_text="你好", status=status)


def _seed(db_dir: Path, *, block: BlockStatus, age: float) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    db = db_dir / f"{_JOB}.sqlite"
    with SQLiteJobLedger(db) as ledger:
        ledger.init_job_from_manifest(
            _JOB,
            BookManifest(doc_id=_JOB, title="t", source_path="in.pdf", target_lang="zh"),
        )
        ledger.append_chapter(
            _JOB,
            ChapterIR(
                doc_id=_JOB, chapter_id="c1", title="C", spine_index=0, blocks=[_block(block)]
            ),
        )
        ledger.set_job_metadata_value(
            _JOB,
            "agent_session",
            {"state": "awaiting_agent", "packages": {}},
        )
    stamp = time.time() - age
    os.utime(db, (stamp, stamp))
    return db


def _client(tmp_path: Path, db_dir: Path) -> TestClient:
    config = UBTConfig(
        db_dir=db_dir,
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    return TestClient(create_app(config))


def _both(tmp_path: Path, db_dir: Path, db: Path) -> tuple[str, str]:
    response = _client(tmp_path, db_dir).get(
        f"/jobs/{_JOB}/status", headers={"X-API-Key": _API_KEY}
    )
    assert response.status_code == 200, response.text
    summary = summarize_job(db)
    assert summary is not None
    return str(response.json()["status"]), str(summary["status"])


def test_a_crashed_job_reads_failed_in_both(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db = _seed(db_dir, block=BlockStatus.DRAFTED, age=_QUIET_FOR)
    with SQLiteJobLedger(db) as ledger:
        ledger.set_job_metadata_value(_JOB, "agent_session", {"state": "running"})
    status, catalog = _both(tmp_path, db_dir, db)
    assert status == catalog == "failed"


def test_a_live_writer_reads_running_in_both(tmp_path: Path) -> None:
    """The case an mtime rule got wrong: quiet ledger, working process."""
    db_dir = tmp_path / "db"
    db = _seed(db_dir, block=BlockStatus.DRAFTED, age=_QUIET_FOR)
    with SQLiteJobLedger(db) as ledger:
        ledger.set_job_metadata_value(_JOB, "agent_session", {"state": "running"})
    lock = LedgerWriterLock(db, _JOB)
    lock.acquire()
    try:
        status, catalog = _both(tmp_path, db_dir, db)
    finally:
        lock.release()
    assert status == catalog == "running"


def test_a_fully_processed_book_reads_completed_in_both(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db = _seed(db_dir, block=BlockStatus.MTQE_PASSED, age=_QUIET_FOR)
    status, catalog = _both(tmp_path, db_dir, db)
    assert status == catalog == "completed"


def test_an_agent_job_waiting_for_input_reads_awaiting_agent_in_both(tmp_path: Path) -> None:
    """An Agent job holds no writer lease between MCP calls by design."""
    db_dir = tmp_path / "db"
    db = _seed(db_dir, block=BlockStatus.PENDING, age=_QUIET_FOR)
    status, catalog = _both(tmp_path, db_dir, db)
    assert status == catalog == "awaiting_agent"
