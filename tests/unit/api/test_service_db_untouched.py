from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.core.config import UBTConfig
from ubt.core.engine.delivery_freshness import (
    RENDERED_TEXT_DIGEST_KEY,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast

_API_KEY = "svc-db-test-key"
_AUTH = {"X-API-Key": _API_KEY}


def _config(tmp_path: Path) -> UBTConfig:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    return config


def _tables(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [row[0] for row in conn.execute("select name from sqlite_master where type='table'")]


def _client(config: UBTConfig) -> TestClient:
    return TestClient(create_app(config))


def test_a_download_for_a_service_database_id_leaves_it_untouched(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service_db = config.db_dir / "job_queue.sqlite"
    sqlite3.connect(service_db).close()
    before = _tables(service_db)

    response = _client(config).get("/jobs/job_queue/download", headers=_AUTH)
    assert response.status_code in (404, 409)
    assert _tables(service_db) == before


def test_a_download_for_a_translation_memory_id_leaves_it_untouched(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tm_db = config.db_dir / "tm.sqlite"
    sqlite3.connect(tm_db).close()
    before = _tables(tm_db)

    response = _client(config).get("/jobs/tm/download", headers=_AUTH)
    assert response.status_code in (404, 409)
    assert _tables(tm_db) == before


def test_a_real_job_ledger_is_still_probed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    job = "realjob001"
    out = tmp_path / "out.pdf"
    out.write_bytes(b"%PDF-1.4\nfake")

    with SQLiteJobLedger(config.db_dir / f"{job}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            job, BookManifest(doc_id=job, title="t", source_path=str(tmp_path / "in.md"))
        )
        block = IRBlock(
            element=make_element(
                id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="Hello"
            )
        )
        block.status = BlockStatus.MTQE_PASSED
        block.target_text = "你好"
        ledger.append_chapter(
            job,
            ChapterIR(doc_id=job, chapter_id="c1", title="C1", spine_index=0, blocks=[block]),
        )
        ledger.set_job_metadata_value(job, "output_file", str(out))
        ledger.set_job_metadata_value(job, RENDERED_TEXT_DIGEST_KEY, "stale-digest")
        ledger.finalize_job(job, status="completed")

    response = _client(config).get(f"/jobs/{job}/download/primary", headers=_AUTH)
    assert response.status_code == 409
    assert "re-render" in response.text.lower() or "stale" in response.text.lower()
