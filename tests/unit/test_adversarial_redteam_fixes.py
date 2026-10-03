from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import LedgerError
from ubt.core.ir.models import BookManifest
from ubt.core.job_options import apply_config_overrides
from ubt.core.router.rate_limiter import SqliteTokenBucket

pytestmark = pytest.mark.fast


def test_ledger_dotted_metadata_keys(tmp_path: Path) -> None:
    """Verify that model names with dots like claude-3.5-sonnet or gemini-1.5-pro are stored and retrieved without splitting."""
    db_path = tmp_path / "test_ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "test-job-dotted"
    ledger.init_job_from_manifest(
        job_id, BookManifest(doc_id="doc1", title="Title", source_path="input.epub")
    )

    key = "claude-3.5-sonnet"
    val = {"cost": 0.42, "calls": 5}
    ledger.set_job_metadata_value(job_id, key, val)

    # Read back through ledger API
    retrieved = ledger.get_job_metadata_value(job_id, key)
    assert retrieved == val, f"Expected {val}, got {retrieved}"

    # Check raw JSON structure: key should NOT be nested like 'claude-3': {'5-sonnet': ...}
    with ledger._get_conn() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM job_meta WHERE job_id = ?", (job_id,)
        ).fetchone()
        assert row is not None
        raw = json.loads(row["metadata_json"])
        assert key in raw, (
            f"Key '{key}' was not found in raw metadata root keys: {list(raw.keys())}"
        )
        assert raw[key] == val


def test_atomic_increment_job_usage_corrupt_metadata(tmp_path: Path) -> None:
    """Verify that atomic_increment_job_usage rejects corrupt or non-dict metadata cleanly."""
    db_path = tmp_path / "test_ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "test-job-corrupt"
    ledger.init_job_from_manifest(
        job_id, BookManifest(doc_id="doc1", title="Title", source_path="input.epub")
    )

    # Corrupt metadata with non-dict JSON array
    with ledger._get_conn() as conn:
        conn.execute("UPDATE job_meta SET metadata_json = '[1, 2, 3]' WHERE job_id = ?", (job_id,))
        conn.commit()

    with pytest.raises(LedgerError, match="not a JSON object"):
        ledger.atomic_increment_job_usage(job_id, {"model": {"prompt_tokens": 10}})


def test_apply_config_overrides_preserves_base_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that apply_config_overrides keeps base.provider even when UBT_PROVIDER env var is set."""
    base = UBTConfig.from_env(bootstrap=False, provider="claude")
    monkeypatch.setenv("UBT_PROVIDER", "openai")

    res = apply_config_overrides(base, {"temperature": 0.7})
    assert res.provider == "claude"
    assert res.base_url == "https://api.anthropic.com"


@pytest.mark.asyncio
async def test_sqlite_token_bucket_close_safety(tmp_path: Path) -> None:
    """Verify that closing SqliteTokenBucket while tasks may be waiting does not raise ProgrammingError."""
    db_path = tmp_path / "bucket.db"
    bucket = SqliteTokenBucket(
        path=db_path,
        initial_rpm=10,
        initial_tpm=1000,
        bucket_key="test_bucket",
    )

    bucket.close()
    assert bucket._closed is True

    # acquire should exit cleanly without calling into closed SQLite
    await bucket.acquire(estimated_tokens=500)


def test_writer_lock_fail_closed_on_missing_platform_module(tmp_path: Path) -> None:
    """Verify that writer lock raises RuntimeError when platform file-locking is unavailable."""
    db_path = tmp_path / "job.db"
    lock = LedgerWriterLock(db_path, "job-fail-closed")

    with (
        patch(
            "ubt.core.engine.writer_lock._try_exclusive_lock",
            side_effect=ImportError("No locking module"),
        ),
        pytest.raises(RuntimeError, match="platform file-locking module is unavailable"),
    ):
        lock.acquire()
