"""``page_geometry``: an additive table, and the guarantee that it costs nothing.

The v2.1 design doc originally proposed this as a ``blocks.geometry_json``
column and accepted a ``--fresh`` for it. Both halves were wrong, and these
tests pin the correction rather than the conclusion:

* **Page-level, not block-level.** One page carries ~20 blocks; a per-block
  column stores the same page's facts 20 times and makes one page's update a
  20-row consistency problem. There is one row per page here.
* **Additive, so no data loss.** ``CREATE TABLE IF NOT EXISTS`` touches no
  existing row, which is not the in-place-migration shape the ledger forbids
  (that rule is about rewriting a half-migrated ``blocks`` table). So an old
  ledger gains the table on open and keeps every translation — the test for
  that is the whole point of this file.

``SCHEMA_VERSION`` is deliberately not bumped: the stamp is what *refuses* an
older ledger, and gaining an empty table is not a reason to refuse one. If a
future change does bump it, ``test_the_schema_version_is_not_bumped_by_this_table``
fails, and that failure is the signal that a ``--fresh`` became mandatory —
which is exactly the trade this file exists to forbid silently.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_base import (
    _PAGE_GEOMETRY_DDL,
    _REQUIRED_PAGE_GEOMETRY_COLUMNS,
    SCHEMA_VERSION,
)

pytestmark = pytest.mark.fast


@pytest.fixture(name="ledger")
def _ledger(tmp_path: Path) -> Iterator[SQLiteJobLedger]:
    """A fresh ledger per test.

    Declared here rather than shared: ``tests/unit/test_ledger.py`` defines its
    own local fixture of this name, and a conftest-level global would silently
    become every ledger test's database shape.
    """
    with SQLiteJobLedger(tmp_path / "ledger.db") as db:
        yield db


def _page(page_no: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "page_no": page_no,
        "page_w_pt": 595.0,
        "page_h_pt": 842.0,
        "measured_size_pt": 9.0,
        "measured_leading_pt": 12.0,
        "delivered_size_pt": 9.0,
        "delivered_leading_em": 0.33,
        "tracking_em": 0.0,
        "tier": "L1",
        "fits": True,
    }
    row.update(overrides)
    return row


def _job(ledger: SQLiteJobLedger, job_id: str = "job-1") -> str:
    """A job_meta row, because page_geometry has a FK to it."""
    with ledger._get_conn() as conn:
        conn.execute(
            "INSERT INTO job_meta (job_id, doc_id, source_path, target_lang, "
            "total_blocks, status) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, "doc", "/tmp/x.pdf", "zh", 0, "initialized"),
        )
    return job_id


# --------------------------------------------------------------------------- #
# The additive-table guarantee
# --------------------------------------------------------------------------- #
def test_a_fresh_ledger_has_the_table(ledger: SQLiteJobLedger) -> None:
    with ledger._get_conn() as conn:
        columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(page_geometry);").fetchall()
        }
    assert columns >= _REQUIRED_PAGE_GEOMETRY_COLUMNS


def test_an_old_ledger_gains_the_table_without_fresh(tmp_path: Path) -> None:
    """A ledger written before this table existed must open and keep its rows.

    Built at the current ``SCHEMA_VERSION`` *without* ``page_geometry``, which is
    precisely the file an older build produced. Opening it must succeed, add the
    table, and leave the translations in place.
    """
    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(f"""
        CREATE TABLE job_meta (job_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
            source_path TEXT NOT NULL, target_lang TEXT NOT NULL,
            total_blocks INTEGER NOT NULL, status TEXT NOT NULL,
            metadata_json TEXT, created_at TIMESTAMP, updated_at TIMESTAMP);
        CREATE TABLE blocks (block_id TEXT NOT NULL, job_id TEXT NOT NULL,
            flow_id TEXT NOT NULL, spine_index INTEGER NOT NULL,
            block_type TEXT, bbox_json TEXT, style_json TEXT,
            source_text TEXT NOT NULL, draft_text TEXT, target_text TEXT,
            status TEXT NOT NULL, skip_translate INTEGER DEFAULT 0,
            glossary_hits_json TEXT, mtqe_score REAL, repair_rounds INTEGER DEFAULT 0,
            error_flags_json TEXT, tm_hit INTEGER DEFAULT 0,
            mqm_severity TEXT, mqm_spans_json TEXT, layout_role TEXT,
            policy_translate INTEGER, policy_reason TEXT, provenance_json TEXT,
            level INTEGER, marker TEXT, confidence TEXT, decorative INTEGER,
            span_chars_json TEXT, updated_at TIMESTAMP,
            PRIMARY KEY (job_id, block_id));
        CREATE TABLE batch_jobs (batch_id TEXT PRIMARY KEY);
        CREATE TABLE job_locks (job_id TEXT PRIMARY KEY, owner_pid INTEGER NOT NULL,
            owner_token TEXT NOT NULL, lease_expires_at REAL NOT NULL,
            acquired_at REAL NOT NULL);
        CREATE INDEX idx_blocks_spine ON blocks(job_id, spine_index);
        INSERT INTO job_meta VALUES ('job-1','doc','/tmp/x.pdf','zh',1,'completed',
            '{{}}','2026-01-01','2026-01-01');
        INSERT INTO blocks (block_id, job_id, flow_id, spine_index, block_type,
            source_text, status, target_text)
            VALUES ('b1','job-1','main_story',0,'narrative','Source','mtqe_passed',
                '已付费的译文');
        PRAGMA user_version = {SCHEMA_VERSION};
    """)
    conn.commit()
    conn.close()

    opened = SQLiteJobLedger(db_file)

    with opened._get_conn() as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert "page_geometry" in tables
    # The paid translation survived — the whole point of being additive.
    block = opened.get_block("b1", "job-1")
    assert block is not None and block.target_text == "已付费的译文"


def test_the_schema_version_is_not_bumped_by_this_table() -> None:
    """The stamp is what refuses an older ledger. A table that costs nothing must
    not make every existing ledger unusable."""
    assert SCHEMA_VERSION == 1000


def test_a_malformed_page_geometry_table_is_refused(tmp_path: Path) -> None:
    """``IF NOT EXISTS`` finding a *wrong* table must not be mistaken for success:
    a missing column would otherwise fail deep inside a resume."""
    db_file = tmp_path / "broken.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(f"""
        CREATE TABLE job_meta (job_id TEXT PRIMARY KEY);
        CREATE TABLE blocks (block_id TEXT NOT NULL, job_id TEXT NOT NULL,
            flow_id TEXT NOT NULL, spine_index INTEGER NOT NULL, block_type TEXT,
            bbox_json TEXT, style_json TEXT, source_text TEXT NOT NULL,
            draft_text TEXT, target_text TEXT, status TEXT NOT NULL,
            skip_translate INTEGER, glossary_hits_json TEXT, mtqe_score REAL,
            repair_rounds INTEGER, error_flags_json TEXT, tm_hit INTEGER,
            mqm_severity TEXT, mqm_spans_json TEXT, layout_role TEXT,
            policy_translate INTEGER, policy_reason TEXT, provenance_json TEXT,
            level INTEGER, marker TEXT, confidence TEXT, decorative INTEGER,
            span_chars_json TEXT, PRIMARY KEY (job_id, block_id));
        CREATE TABLE page_geometry (job_id TEXT NOT NULL, page_no INTEGER NOT NULL);
        PRAGMA user_version = {SCHEMA_VERSION};
    """)
    conn.commit()
    conn.close()

    from ubt.core.exceptions import LedgerError

    with pytest.raises(LedgerError) as excinfo:
        SQLiteJobLedger(db_file)
    assert "page_geometry" in str(excinfo.value)
    assert "missing_columns" in excinfo.value.details


def test_fresh_drops_stale_geometry(tmp_path: Path) -> None:
    """``--fresh`` means "re-measure from the source", so rows describing the
    discarded job's pages must not survive it."""
    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(db_file)
    conn.executescript("CREATE TABLE job_meta (job_id TEXT PRIMARY KEY); PRAGMA user_version = 10;")
    conn.commit()
    conn.close()

    reset = SQLiteJobLedger(db_file, fresh=True)
    with reset._get_conn() as c:
        assert c.execute("SELECT COUNT(*) FROM page_geometry").fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
def test_record_and_retrieve_round_trips_every_column(ledger: SQLiteJobLedger) -> None:
    job = _job(ledger)
    assert ledger.record_page_geometry(job, [_page(1), _page(2, tier="L2", fits=False)]) == 2

    stored = ledger.retrieve_page_geometry(job)
    assert set(stored) == {1, 2}
    assert stored[1]["measured_size_pt"] == 9.0
    assert stored[1]["tier"] == "L1"
    assert stored[1]["fits"] is True
    assert stored[2]["tier"] == "L2"
    assert stored[2]["fits"] is False


def test_a_rerender_replaces_the_previous_answer(ledger: SQLiteJobLedger) -> None:
    """Two contradictory answers to "how big was page 12" is the failure this
    table exists to prevent, so the second render must win."""
    job = _job(ledger)
    ledger.record_page_geometry(job, [_page(12, delivered_size_pt=9.0)])
    ledger.record_page_geometry(job, [_page(12, delivered_size_pt=7.5, tier="L2")])

    stored = ledger.retrieve_page_geometry(job)
    assert len(stored) == 1
    assert stored[12]["delivered_size_pt"] == 7.5
    assert stored[12]["tier"] == "L2"


def test_geometry_is_scoped_to_its_job(ledger: SQLiteJobLedger) -> None:
    first = _job(ledger, "job-a")
    second = _job(ledger, "job-b")
    ledger.record_page_geometry(first, [_page(1, delivered_size_pt=9.0)])
    ledger.record_page_geometry(second, [_page(1, delivered_size_pt=8.0)])

    assert ledger.retrieve_page_geometry(first)[1]["delivered_size_pt"] == 9.0
    assert ledger.retrieve_page_geometry(second)[1]["delivered_size_pt"] == 8.0


def test_an_unrendered_job_reads_empty_not_error(ledger: SQLiteJobLedger) -> None:
    """A ledger from a run that never wrote geometry must read as "not
    measured", which is different from a failure."""
    job = _job(ledger)
    assert ledger.retrieve_page_geometry(job) == {}


def test_a_row_without_a_usable_page_number_is_skipped(ledger: SQLiteJobLedger) -> None:
    """Geometry is an audit record: losing one page must not lose the batch."""
    job = _job(ledger)
    written = ledger.record_page_geometry(
        job,
        [
            {"page_no": "not-a-number", "tier": "L1"},
            {"tier": "L1"},
            _page(3),
        ],
    )
    assert written == 1
    assert set(ledger.retrieve_page_geometry(job)) == {3}


def test_no_rows_writes_nothing(ledger: SQLiteJobLedger) -> None:
    job = _job(ledger)
    assert ledger.record_page_geometry(job, []) == 0
    assert ledger.retrieve_page_geometry(job) == {}


def test_null_fits_stays_null_rather_than_becoming_false(ledger: SQLiteJobLedger) -> None:
    """``None`` means "not solved"; reading it back as ``False`` would report a
    page as overflowing when it was never measured."""
    job = _job(ledger)
    ledger.record_page_geometry(job, [_page(1, fits=None)])
    assert ledger.retrieve_page_geometry(job)[1]["fits"] is None


def test_the_ddl_is_idempotent(ledger: SQLiteJobLedger) -> None:
    """It runs on every open, so a second application must be a no-op."""
    with ledger._get_conn() as conn:
        conn.executescript(_PAGE_GEOMETRY_DDL)
        conn.executescript(_PAGE_GEOMETRY_DDL)
        assert conn.execute("SELECT COUNT(*) FROM page_geometry").fetchone()[0] == 0


def test_metadata_json_is_untouched_by_geometry_writes(ledger: SQLiteJobLedger) -> None:
    """Geometry lives in its own table, not in ``job_meta.metadata_json``: that
    blob is rewritten by many writers, and 400 pages of rows would inflate it."""
    job = _job(ledger)
    ledger.set_job_metadata_value(job, "keep", {"a": 1})
    ledger.record_page_geometry(job, [_page(1)])
    assert ledger.get_job_metadata(job)["keep"] == {"a": 1}
