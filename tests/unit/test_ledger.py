"""Unit and performance tests for SQLiteJobLedger."""

import concurrent.futures
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    DocumentIR,
    FlowID,
    IRBlock,
    LayoutRole,
    SemanticRole,
    StructureRole,
)


def _manifest_stub() -> BookManifest:
    return BookManifest(doc_id="v5b", title="t", source_path="/tmp/v5b.pdf")


@pytest.fixture
def sample_doc_ir() -> DocumentIR:
    """Fixture providing a sample DocumentIR with 5 blocks across flows."""
    blocks = [
        IRBlock(
            id=f"ch01#b{i:03d}",
            flow_id=FlowID.MAIN_STORY if i % 2 == 0 else FlowID.SIDEBAR_ASIDE,
            spine_index=i,
            source_text=f"Paragraph {i} source content for testing.",
        )
        for i in range(1, 6)
    ]
    return DocumentIR(
        doc_id="test_doc_sha256",
        source_path="/tmp/test_book.epub",
        format_type="epub",
        metadata={"title": "Test Book"},
        blocks=blocks,
    )


def test_ledger_init_and_idempotence(tmp_path: Path, sample_doc_ir: DocumentIR) -> None:
    """Test ledger schema initialization and idempotent re-initialization."""
    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)

    ledger.init_job("job_001", sample_doc_ir, target_lang="zh")
    blocks = ledger.get_all_blocks("job_001")
    assert len(blocks) == 5

    # Re-initialization with same job_id must be completely idempotent and not crash
    ledger.init_job("job_001", sample_doc_ir, target_lang="zh")
    blocks_after = ledger.get_all_blocks("job_001")
    assert len(blocks_after) == 5


def test_single_checkpoint_and_resume(tmp_path: Path, sample_doc_ir: DocumentIR) -> None:
    """Test single block checkpoint update and resume filtering.

    Resume semantics follow the production primitive ``fetch_pending_blocks``:
    only pending, translatable blocks are re-drafted on a continued run.
    """
    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_002"

    ledger.init_job(job_id, sample_doc_ir, target_lang="zh")

    # Initial state: all 5 blocks should be pending for drafting
    resume_initial = ledger.fetch_pending_blocks(job_id, limit=50)
    assert len(resume_initial) == 5

    # Checkpoint block 1 to drafted
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.DRAFTED,
        draft_text="草翻段落 1",
    )
    b1 = ledger.get_block("ch01#b001")
    assert b1 is not None
    assert b1.status == BlockStatus.DRAFTED
    assert b1.draft_text == "草翻段落 1"

    # A drafted block left mid-flight by a crash is not re-drafted by the
    # keyset loop; the export-time non-terminal guard owns it.
    assert len(ledger.fetch_pending_blocks(job_id, limit=50)) == 4

    # Checkpoint block 1 to MTQE_PASSED (terminal)
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.MTQE_PASSED,
        target_text="终稿段落 1",
        mtqe_score=0.92,
    )
    b1_final = ledger.get_block("ch01#b001")
    assert b1_final is not None
    assert b1_final.status == BlockStatus.MTQE_PASSED
    assert b1_final.target_text == "终稿段落 1"
    assert b1_final.mtqe_score == 0.92

    # Block 1 is now finalized, resume blocks must now be 4
    resume_after = ledger.fetch_pending_blocks(job_id, limit=50)
    assert len(resume_after) == 4
    assert all(b.id != "ch01#b001" for b in resume_after)


def test_batch_checkpoint_and_reassemble(tmp_path: Path, sample_doc_ir: DocumentIR) -> None:
    """Test batch checkpoints persist and read back with full state."""
    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_003"
    ledger.init_job(job_id, sample_doc_ir, target_lang="zh")

    updates = [
        {
            "block_id": "ch01#b001",
            "status": BlockStatus.MTQE_PASSED,
            "target_text": "译文 1",
            "mtqe_score": 0.95,
        },
        {
            "block_id": "ch01#b002",
            "status": BlockStatus.REPAIRED,
            "target_text": "重修译文 2",
            "mtqe_score": 0.86,
            "repair_rounds": 1,
        },
    ]
    ledger.save_checkpoints_batch(updates)

    blocks = {b.id: b for b in ledger.get_all_blocks(job_id)}
    assert blocks["ch01#b001"].target_text == "译文 1"
    assert blocks["ch01#b001"].status == BlockStatus.MTQE_PASSED
    assert blocks["ch01#b002"].target_text == "重修译文 2"
    assert blocks["ch01#b002"].repair_rounds == 1

    stats = ledger.get_job_stats(job_id)
    assert stats["total"] == 5
    assert stats["completed"] == 2
    assert stats["repaired"] == 1


def test_save_checkpoints_batch_clears_verdicts_in_the_same_call(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """Human PE import needs the new text and the verdict reset together (X17).

    ``save_checkpoints_batch`` treats ``None`` as "leave alone", so it cannot
    clear a stale ``mtqe_score``/``mqm_severity``. Two separate calls left a
    window where a crash kept the human text next to a superseded "critical"
    severity — the exact state the clear exists to prevent.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger_verdict.db")
    job_id = "job_verdict"
    try:
        ledger.init_job(job_id, sample_doc_ir, target_lang="zh")
        ledger.save_checkpoints_batch(
            [
                {
                    "block_id": "ch01#b001",
                    "status": BlockStatus.REPAIRED,
                    "target_text": "旧机翻",
                    "mtqe_score": 0.12,
                    "mqm_severity": "critical",
                }
            ]
        )
        seeded = {b.id: b for b in ledger.get_all_blocks(job_id)}["ch01#b001"]
        assert seeded.mqm_severity == "critical"

        ledger.save_checkpoints_batch(
            [
                {
                    "block_id": "ch01#b001",
                    "status": BlockStatus.REPAIRED,
                    "target_text": "人工修订",
                }
            ],
            clear_verdict_for=["ch01#b001"],
        )

        block = {b.id: b for b in ledger.get_all_blocks(job_id)}["ch01#b001"]
        assert block.target_text == "人工修订"
        assert block.mtqe_score is None
        assert block.mqm_severity is None
    finally:
        ledger.close()


def test_reset_transient_failures_requeues_only_transient(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """Resume recovery: transient drafting/repair failures are
    re-queued; quality escalations stay terminal.

    An API outage used to strand blocks in FAILED/NEEDS_HUMAN — terminal
    states every resume query skips — turning a temporary provider error
    into a manual work order. Only blocks flagged by the retry paths
    (``Drafting error:`` / ``Repair error:``) may loop back to pending.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_recover", sample_doc_ir, target_lang="zh")

    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.FAILED,
        error_flags=["Drafting error: ConnectionResetError"],
    )
    ledger.save_checkpoint(
        block_id="ch01#b002",
        status=BlockStatus.NEEDS_HUMAN,
        error_flags=["Repair error: retries exhausted"],
    )
    # Quality escalation: no transient markers -> must stay terminal.
    ledger.save_checkpoint(
        block_id="ch01#b003",
        status=BlockStatus.NEEDS_HUMAN,
        error_flags=["needs_human_review", "Numeric fidelity"],
        mqm_severity="major",
    )
    # Plain FAILED (export-time crash guard): stays terminal.
    ledger.save_checkpoint(
        block_id="ch01#b004",
        status=BlockStatus.FAILED,
        error_flags=["Non-terminal at export"],
    )

    reset_ids = ledger.reset_transient_failures("job_recover")
    assert reset_ids == ["ch01#b001", "ch01#b002"]

    b1 = ledger.get_block("ch01#b001")
    assert b1 is not None
    assert b1.status == BlockStatus.PENDING
    assert b1.target_text is None
    assert b1.error_flags == []
    b2 = ledger.get_block("ch01#b002")
    assert b2 is not None
    assert b2.status == BlockStatus.PENDING
    b3 = ledger.get_block("ch01#b003")
    assert b3 is not None
    assert b3.status == BlockStatus.NEEDS_HUMAN
    b4 = ledger.get_block("ch01#b004")
    assert b4 is not None
    assert b4.status == BlockStatus.FAILED


def test_resume_keeps_the_paid_draft_of_a_repair_only_failure(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """A repair-stage outage must not throw away the draft it already paid for.

    ``stages/repair.py`` persists ``target_text`` deliberately, but the old
    blanket PENDING reset NULLed it on resume, so the next run re-drafted the
    same paragraph and billed it a second time. Repair-only markers now go back
    to ``REPAIR_PENDING`` with their text; a block that never produced usable
    text (drafting marker, or a repair marker on a textless row) still gets the
    full reset.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_repair_requeue", sample_doc_ir, target_lang="zh")

    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.FAILED,
        target_text="already paid for",
        error_flags=["Repair error: provider 5xx"],
    )
    ledger.save_checkpoint(
        block_id="ch01#b002",
        status=BlockStatus.FAILED,
        target_text="drafted before the outage",
        error_flags=["Drafting error: ConnectionResetError", "Repair error: provider 5xx"],
    )
    # Repair marker but no text: nothing to preserve, so it re-drafts.
    ledger.save_checkpoint(
        block_id="ch01#b003",
        status=BlockStatus.NEEDS_HUMAN,
        error_flags=["Repair error: retries exhausted"],
    )

    reset_ids = ledger.reset_transient_failures("job_repair_requeue")
    assert reset_ids == ["ch01#b001", "ch01#b002", "ch01#b003"]

    b1 = ledger.get_block("ch01#b001")
    assert b1 is not None
    assert b1.status == BlockStatus.REPAIR_PENDING
    assert b1.target_text == "already paid for"
    assert b1.error_flags == []

    b2 = ledger.get_block("ch01#b002")
    assert b2 is not None
    assert b2.status == BlockStatus.PENDING
    assert b2.target_text is None

    b3 = ledger.get_block("ch01#b003")
    assert b3 is not None
    assert b3.status == BlockStatus.PENDING

    # Second run is a no-op (nothing transient left).
    assert ledger.reset_transient_failures("job_recover") == []


def test_resume_keeps_the_paid_draft_of_an_untranslated_sweep(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """Export's ``untranslated:`` sweep marks still-nonterminal DRAFTED rows
    FAILED *without* clearing their paid text.

    Resume used to treat that marker like a drafting failure and route the row
    through the PENDING reset, which NULLs ``target_text`` and re-bills the same
    draft on the next run (review 2026-09 P0-2). The row carries a paid draft
    and no drafting marker, so it must go back to repair with its text intact.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_untranslated", sample_doc_ir, target_lang="zh")

    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.FAILED,
        target_text="paid draft the sweep left in place",
        error_flags=["untranslated: stage crashed before this block was terminal"],
    )

    assert ledger.reset_transient_failures("job_untranslated") == ["ch01#b001"]

    kept = ledger.get_block("ch01#b001")
    assert kept is not None
    assert kept.status == BlockStatus.REPAIR_PENDING
    assert kept.target_text == "paid draft the sweep left in place"
    ledger.close()


def test_reset_transient_failures_keeps_non_retryable_draft_failures(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """A 401/402/400 recurs on every resume, so it must not be re-queued.

    Re-queueing it would re-bill the same block on every resume for an error no
    retry can fix; only genuinely transient failures loop back to pending.
    """
    from ubt.core.qe.defect_taxonomy import NON_RETRYABLE_DRAFT_PREFIX

    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_nonretry", sample_doc_ir, target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.FAILED,
        error_flags=[f"{NON_RETRYABLE_DRAFT_PREFIX} HTTP 401 Unauthorized"],
    )
    ledger.save_checkpoint(
        block_id="ch01#b002",
        status=BlockStatus.FAILED,
        error_flags=["Drafting error: ConnectionResetError"],
    )

    assert ledger.reset_transient_failures("job_nonretry") == ["ch01#b002"]
    b1 = ledger.get_block("ch01#b001")
    assert b1 is not None
    assert b1.status == BlockStatus.FAILED


def test_reset_transient_failures_exempts_triaged_blocks(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """E2 (option B): triage keeps the original transient marker when
    upgrading to NEEDS_HUMAN, so the prefix alone cannot tell "never
    reviewed" from "reviewed, awaiting a human". Rows carrying a triage
    verdict stay terminal — the PE queue must not lose members on resume."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_pe", sample_doc_ir, target_lang="zh")

    # Triaged Major: transient history + verdict + severity -> stays.
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.NEEDS_HUMAN,
        error_flags=["Repair error: retries exhausted", "needs_human_review"],
        mqm_severity="major",
    )
    # Never triaged: transient marker only -> still re-queued.
    ledger.save_checkpoint(
        block_id="ch01#b002",
        status=BlockStatus.FAILED,
        error_flags=["Repair error: retries exhausted"],
    )

    assert ledger.reset_transient_failures("job_pe") == ["ch01#b002"]
    b1 = ledger.get_block("ch01#b001")
    assert b1 is not None
    assert b1.status == BlockStatus.NEEDS_HUMAN
    assert "needs_human_review" in b1.error_flags


def test_job_metadata_roundtrip_and_cache_invalidation(tmp_path: Path) -> None:
    """Generic metadata API + --fresh invalidation of derived caches."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job_from_manifest(
        "job_meta",
        BookManifest(
            doc_id="meta_doc_sha256",
            title="Meta",
            source_path="/tmp/meta.epub",
            target_lang="de",
        ),
    )

    assert ledger.get_job_target_lang("job_meta") == "de"
    assert ledger.get_job_metadata_value("job_meta", "bible_cache") is None

    ledger.set_job_metadata_value("job_meta", "bible_cache", {"version": "v1", "n": 1})
    ledger.set_job_metadata_value("job_meta", "memory_state", {"snapshots": [1, 2]})
    # The ingest fingerprint must coexist with the cache keys (read-modify-write).
    ledger.set_job_fingerprint("job_meta", "fp123")
    assert ledger.get_job_metadata_value("job_meta", "bible_cache") == {"version": "v1", "n": 1}
    assert ledger.get_job_fingerprint("job_meta") == "fp123"

    removed = ledger.clear_job_blocks("job_meta")
    assert removed == 0
    assert ledger.get_job_metadata_value("job_meta", "bible_cache") is None
    assert ledger.get_job_metadata_value("job_meta", "memory_state") is None
    # The fingerprint goes too: it is the proof that a previous ingest ran to
    # completion, and a fresh re-ingest has just deleted what it proves.
    assert ledger.get_job_fingerprint("job_meta") is None


def test_batch_update_preserves_error_flags_when_key_omitted(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """Omitting ``error_flags`` must preserve stored flags.

    ``save_checkpoint`` (single) skips the column when the value is ``None``.
    The batch path used to default the key to ``[]``, which serialized to
    ``"[]"`` and silently wiped defect markers through ``COALESCE`` — so
    triage and the quality report lost the history for every caller that
    omits the key (ctext, export, quality_gate).
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    job_id = "job_flags"
    ledger.init_job(job_id, sample_doc_ir, target_lang="zh")

    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.REPAIR_PENDING,
        error_flags=["Numeric fidelity", "HTML drift"],
    )
    seeded = next(b for b in ledger.get_all_blocks(job_id) if b.id == "ch01#b001")
    assert seeded.error_flags == ["Numeric fidelity", "HTML drift"]

    # Batch update omitting error_flags, as ctext/export/quality_gate do.
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.MTQE_PASSED,
                "target_text": "译文 1",
                "mtqe_score": 0.91,
            }
        ]
    )
    after = next(b for b in ledger.get_all_blocks(job_id) if b.id == "ch01#b001")
    assert after.status == BlockStatus.MTQE_PASSED
    assert after.error_flags == ["Numeric fidelity", "HTML drift"]

    # An explicit empty list is still honoured as "clear the flags".
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.MTQE_PASSED,
                "error_flags": [],
            }
        ]
    )
    cleared = next(b for b in ledger.get_all_blocks(job_id) if b.id == "ch01#b001")
    assert cleared.error_flags == []


def test_high_throughput_batch_writes(tmp_path: Path) -> None:
    """5000-block batch write stays correct; the TPS figure is reported, not asserted.

    This used to assert ``update_tps > 5000``, a wall-clock measurement of the
    machine it ran on: it failed on a loaded CI runner and passed on a fast one
    even if the batch path had gone quadratic. The throughput is still printed
    for a human reading the log; the assertion below is the part that tests code
    (review-2 X47).
    """
    db_path = tmp_path / "perf_ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_perf"

    num_blocks = 5000
    perf_blocks = [
        IRBlock(
            id=f"perf#b{i:05d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"High performance benchmark source sentence number {i}.",
        )
        for i in range(num_blocks)
    ]
    doc = DocumentIR(
        doc_id="perf_sha256",
        source_path="perf.txt",
        format_type="txt",
        blocks=perf_blocks,
    )

    t0 = time.perf_counter()
    ledger.init_job(job_id, doc, target_lang="zh")
    init_duration = time.perf_counter() - t0
    init_tps = num_blocks / init_duration

    # Prepare batch updates for 5000 blocks
    batch_updates = [
        {
            "block_id": f"perf#b{i:05d}",
            "status": BlockStatus.MTQE_PASSED,
            "target_text": f"高吞吐压力测试中文译文第 {i} 条记录。",
            "mtqe_score": 0.91,
        }
        for i in range(num_blocks)
    ]

    t1 = time.perf_counter()
    ledger.save_checkpoints_batch(batch_updates)
    update_duration = time.perf_counter() - t1
    update_tps = num_blocks / update_duration

    print(f"\n[Perf] Init {num_blocks} blocks: {init_duration:.3f}s ({init_tps:.0f} TPS)")
    print(f"[Perf] Batch Update {num_blocks} blocks: {update_duration:.3f}s ({update_tps:.0f} TPS)")

    # Correctness, not throughput: every row must have landed with its text.
    stats = ledger.get_job_stats(job_id)
    assert stats["completed"] == num_blocks
    landed = ledger.get_block("perf#b04999")
    assert landed is not None
    assert landed.target_text == "高吞吐压力测试中文译文第 4999 条记录。"


def test_concurrent_reads_and_writes_no_lock(tmp_path: Path, sample_doc_ir: DocumentIR) -> None:
    """Validate thread-safe concurrent reads and non-conflicting writes in WAL mode."""
    db_path = tmp_path / "concurrent_ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_concurrent"
    ledger.init_job(job_id, sample_doc_ir, target_lang="zh")

    errors: list[Exception] = []

    def reader(worker_id: int) -> None:
        for _ in range(50):
            try:
                blocks = ledger.get_all_blocks(job_id)
                assert len(blocks) == 5
            except Exception as e:
                errors.append(e)

    def writer(block_id: str, score: float) -> None:
        try:
            ledger.save_checkpoint(
                block_id=block_id,
                status=BlockStatus.DRAFTED,
                draft_text=f"Draft from writer with score {score}",
                mtqe_score=score,
            )
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for w in range(4):
            futures.append(executor.submit(reader, w))
        for i in range(1, 6):
            futures.append(executor.submit(writer, f"ch01#b00{i}", 0.8 + i * 0.02))

        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
    ledger.close()


def test_schema_migration_versioning_and_contract_columns(tmp_path: Path) -> None:
    """Validate PRAGMA user_version schema migration mechanism and column presence."""
    db_path = tmp_path / "migration_test.db"
    with SQLiteJobLedger(db_path) as ledger, ledger._get_conn() as conn:
        cursor = conn.execute("PRAGMA user_version;")
        version = cursor.fetchone()[0]
        assert version == 10
        # v9: the draft keyset-pagination composite index must exist on a fresh DB.
        idx = {str(row["name"]) for row in conn.execute("PRAGMA index_list(blocks);").fetchall()}
        assert "idx_blocks_pending" in idx
        # v10: the get_job_stats covering index must exist on a fresh DB.
        assert "idx_blocks_rollup" in idx

        # v7 retired the worker-lease columns; the surviving contract columns
        # (tm_hit, MQM triage, document-v1) must still be present.
        cols_cursor = conn.execute("PRAGMA table_info(blocks);")
        cols = {row["name"] for row in cols_cursor.fetchall()}
        assert "owner_id" not in cols
        assert "lease_expires_at" not in cols
        assert "tm_hit" in cols
        assert "mqm_severity" in cols
        assert "mqm_spans_json" in cols
        # V5 document-v1 contract columns
        assert "layout_role" in cols
        assert "semantic_role" in cols
        assert "structure_role" in cols
        assert "policy_translate" in cols
        assert "policy_reason" in cols
        assert "provenance_json" in cols


def test_v2_to_v3_migration_adds_tm_hit_and_defaults_existing_rows(tmp_path: Path) -> None:
    """The v2→v3 ``ALTER TABLE`` must actually land (review-2 X46).

    The versioning test used to admit it could not simulate a legacy database
    and then assert ``SELECT tm_hit FROM blocks`` on an *empty* table — true no
    matter what the schema said. A broken migration would have shipped silently
    and surfaced as wrong resume behaviour.
    """
    db_path = tmp_path / "legacy_v2.db"
    legacy = sqlite3.connect(db_path)
    # The v2 shape: lease columns present; tm_hit and every later column absent.
    legacy.executescript(
        """
        CREATE TABLE job_meta (
            job_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            total_blocks INTEGER NOT NULL,
            status TEXT NOT NULL,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE blocks (
            block_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            flow_id TEXT NOT NULL,
            spine_index INTEGER NOT NULL,
            block_type TEXT NOT NULL DEFAULT 'narrative',
            bbox_json TEXT,
            style_json TEXT,
            source_text TEXT NOT NULL,
            draft_text TEXT,
            target_text TEXT,
            status TEXT NOT NULL,
            skip_translate INTEGER DEFAULT 0,
            glossary_hits_json TEXT,
            mtqe_score REAL,
            repair_rounds INTEGER DEFAULT 0,
            error_flags_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            owner_id TEXT DEFAULT NULL,
            lease_expires_at REAL DEFAULT NULL
        );
        INSERT INTO blocks (block_id, job_id, flow_id, spine_index, source_text, status)
        VALUES ('ch01#b001', 'job_v2', 'main_story', 1, 'Legacy row', 'pending');
        PRAGMA user_version = 2;
        """
    )
    legacy.commit()
    legacy.close()

    with SQLiteJobLedger(db_path) as ledger, ledger._get_conn() as conn:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == 10
        row = conn.execute("SELECT tm_hit FROM blocks WHERE block_id = 'ch01#b001'").fetchone()
    # The pre-existing row survived the migration and took the column default.
    assert row is not None
    assert row["tm_hit"] == 0


def test_upsert_non_destructive_semantics(tmp_path: Path, sample_doc_ir: DocumentIR) -> None:
    """Validate that ON CONFLICT DO UPDATE SET avoids DELETE+INSERT destruction of custom fields."""
    db_path = tmp_path / "upsert_test.db"
    with SQLiteJobLedger(db_path) as ledger:
        job_id = "job_upsert"
        ledger.init_job(job_id, sample_doc_ir, target_lang="zh")

        # Simulate checkpointing block 1 with drafted content and score
        ledger.save_checkpoint(
            block_id="ch01#b001",
            status=BlockStatus.DRAFTED,
            draft_text="Draft preserved",
            mtqe_score=0.88,
        )

        # Re-initialize or append again with same block_id
        ledger.init_job(job_id, sample_doc_ir, target_lang="zh")

        # Drafted text and score must NOT be wiped out by destructive REPLACE!
        b1 = ledger.get_block("ch01#b001")
        assert b1 is not None
        assert b1.draft_text == "Draft preserved"
        assert b1.mtqe_score == 0.88


def test_skip_flip_discards_stale_translation(tmp_path: Path) -> None:
    """A block that newly becomes verbatim must not keep an old translation."""
    db_path = tmp_path / "skip_flip.db"
    entry = (
        "Y. Taur, An analytical solution to a double-gate MOSFET with undoped "
        "body, IEEE Electron Device Lett. 21 (5) (2000) 245 - 247."
    )

    def _chapter(*, skip: bool) -> ChapterIR:
        block = IRBlock(
            id="pdf_main#b0207",
            spine_index=7,
            source_text=entry,
            skip_translate=skip,
            target_text=entry if skip else "Y. Taur，双栅 MOSFET 未掺杂体的解析解。",
        )
        return ChapterIR(
            doc_id="skipflip",
            chapter_id="pdf_main",
            title="t",
            spine_index=1,
            blocks=[block],
        )

    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest("job_skipflip", _manifest_stub())
        ledger.append_chapter("job_skipflip", _chapter(skip=False))
        assert ledger.get_block("pdf_main#b0207") is not None
        ledger.append_chapter("job_skipflip", _chapter(skip=True))
        got = ledger.get_block("pdf_main#b0207")
    assert got is not None
    assert got.skip_translate is True
    assert got.target_text == entry


def test_skip_stable_resume_keeps_existing_target(tmp_path: Path) -> None:
    """A block already marked verbatim keeps its stored target on re-ingest."""
    db_path = tmp_path / "skip_stable.db"
    chapter = ChapterIR(
        doc_id="skipstable",
        chapter_id="pdf_main",
        title="t",
        spine_index=1,
        blocks=[
            IRBlock(
                id="pdf_main#b0210",
                spine_index=1,
                source_text="M. Author, A title, Journal 1 (2) 2001.",
                skip_translate=True,
                target_text="M. Author, A title, Journal 1 (2) 2001.",
            )
        ],
    )
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest("job_skipstable", _manifest_stub())
        ledger.append_chapter("job_skipstable", chapter)
        ledger.save_checkpoint(
            block_id="pdf_main#b0210",
            status=BlockStatus.MTQE_PASSED,
            target_text="Manual curation kept this text",
        )
        ledger.append_chapter("job_skipstable", chapter)
        got = ledger.get_block("pdf_main#b0210")
    assert got is not None
    assert got.target_text == "Manual curation kept this text"


def test_reset_blocks_to_pending_clears_mqm_triage(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """Re-queued blocks must not keep the stale MQM severity/spans of the
    discarded draft (regression: reset cleared text/flags but left triage)."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_mqm", sample_doc_ir, target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.NEEDS_HUMAN,
        target_text="bad translation",
        mtqe_score=0.4,
        mqm_severity="major",
        mqm_spans=[{"start": 0, "end": 3, "category": "accuracy"}],
    )
    seeded = ledger.get_block("ch01#b001")
    assert seeded is not None
    assert seeded.mqm_severity == "major"
    assert seeded.mqm_spans == [{"start": 0, "end": 3, "category": "accuracy"}]

    assert ledger.reset_blocks_to_pending(["ch01#b001"]) == 1

    reset = ledger.get_block("ch01#b001")
    assert reset is not None
    assert reset.status == BlockStatus.PENDING
    assert reset.mqm_severity is None
    assert reset.mqm_spans == []


def test_keyset_pagination_never_skips_duplicate_spine_indices(tmp_path: Path) -> None:
    """Duplicate spine indices are legal; a keyset cursor (spine_index, block_id)
    must traverse all blocks across batches without skipping or exceeding limit."""
    ledger = SQLiteJobLedger(tmp_path / "dup_keyset.db")
    blocks = [
        IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="a"),
        IRBlock(id="b5a", flow_id=FlowID.MAIN_STORY, spine_index=5, source_text="b"),
        IRBlock(id="b5b", flow_id=FlowID.MAIN_STORY, spine_index=5, source_text="c"),
        IRBlock(id="b9", flow_id=FlowID.MAIN_STORY, spine_index=9, source_text="d"),
    ]
    doc = DocumentIR(doc_id="dup", source_path="t.txt", format_type="txt", blocks=blocks)
    ledger.init_job("job_dup", doc, target_lang="zh")

    # Batch 1 with limit=2: returns exactly 2 blocks ("b1", "b5a"), strictly honoring limit
    batch1 = ledger.fetch_pending_blocks("job_dup", limit=2)
    assert [b.id for b in batch1] == ["b1", "b5a"]
    for b in batch1:
        ledger.save_checkpoint(b.id, status=BlockStatus.DRAFTED)

    # Batch 2 with keyset cursor (after_spine_index=5, after_block_id="b5a"):
    # resumes from b5b without skipping it or jumping directly to b9
    batch2 = ledger.fetch_pending_blocks(
        "job_dup",
        limit=2,
        after_spine_index=batch1[-1].spine_index,
        after_block_id=batch1[-1].id,
    )
    assert [b.id for b in batch2] == ["b5b", "b9"]
    for b in batch2:
        ledger.save_checkpoint(b.id, status=BlockStatus.DRAFTED)

    # Batch 3: fully drained
    batch3 = ledger.fetch_pending_blocks(
        "job_dup",
        limit=2,
        after_spine_index=batch2[-1].spine_index,
        after_block_id=batch2[-1].id,
    )
    assert batch3 == []


def test_keyset_pagination_prevents_offset_drift(tmp_path: Path) -> None:
    """Validate that Keyset Pagination (WHERE spine_index > ?) never skips blocks when statuses change."""
    db_path = tmp_path / "keyset_test.db"
    with SQLiteJobLedger(db_path) as ledger:
        job_id = "job_keyset"
        blocks = [
            IRBlock(
                id=f"b{i:02d}",
                flow_id=FlowID.MAIN_STORY,
                spine_index=i,
                source_text=f"Text {i}",
            )
            for i in range(1, 11)
        ]
        doc = DocumentIR(
            doc_id="keyset_doc",
            source_path="test.txt",
            format_type="txt",
            blocks=blocks,
        )
        ledger.init_job(job_id, doc, target_lang="zh")

        # Pull batch 1 of 3 blocks
        batch1 = ledger.fetch_pending_blocks(job_id, limit=3)
        assert [b.id for b in batch1] == ["b01", "b02", "b03"]

        # Worker processes batch 1 and updates their status to DRAFTED (removing them from pending pool)
        for b in batch1:
            ledger.save_checkpoint(b.id, status=BlockStatus.DRAFTED)

        # With traditional OFFSET=3, pulling pending would skip b04, b05, b06!
        # But with Keyset Pagination (after_spine_index=3), it retrieves b04, b05, b06 accurately:
        batch2 = ledger.fetch_pending_blocks(
            job_id, limit=3, after_spine_index=batch1[-1].spine_index
        )
        assert [b.id for b in batch2] == ["b04", "b05", "b06"]


def test_v7_migration_retires_lease_columns_and_releases_claimed_rows(tmp_path: Path) -> None:
    """A v6 database with lease columns migrates to v7: columns/index dropped
    and rows stranded in 'claimed' return to 'pending' for re-drafting."""
    import sqlite3

    db_path = tmp_path / "v6_legacy.db"
    legacy = sqlite3.connect(str(db_path))
    # A real v6 table: the current columns plus the two block-lease columns v7
    # retires. (A stripped-down stand-in is not a v6 database -- the open-time
    # column check now refuses one, which is the point of that check.)
    legacy.executescript("""
        CREATE TABLE blocks (
            block_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            flow_id TEXT NOT NULL,
            spine_index INTEGER NOT NULL DEFAULT 0,
            block_type TEXT NOT NULL DEFAULT 'narrative',
            bbox_json TEXT,
            style_json TEXT,
            source_text TEXT NOT NULL DEFAULT '',
            draft_text TEXT,
            target_text TEXT,
            status TEXT NOT NULL,
            skip_translate INTEGER DEFAULT 0,
            glossary_hits_json TEXT,
            mtqe_score REAL,
            repair_rounds INTEGER DEFAULT 0,
            error_flags_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tm_hit INTEGER DEFAULT 0,
            mqm_severity TEXT DEFAULT NULL,
            mqm_spans_json TEXT DEFAULT NULL,
            layout_role TEXT DEFAULT NULL,
            semantic_role TEXT DEFAULT NULL,
            structure_role TEXT DEFAULT NULL,
            policy_translate INTEGER DEFAULT NULL,
            policy_reason TEXT DEFAULT NULL,
            provenance_json TEXT DEFAULT NULL,
            owner_id TEXT DEFAULT NULL,
            lease_expires_at REAL DEFAULT NULL
        );
        CREATE INDEX idx_block_claim ON blocks(job_id, status, lease_expires_at, spine_index);
        INSERT INTO blocks (block_id, job_id, flow_id, status, source_text, owner_id, lease_expires_at)
        VALUES ('b1', 'job1', 'main_story', 'claimed', 'Hello.', 'worker-a', 123.0);
        PRAGMA user_version = 6;
    """)
    legacy.commit()
    legacy.close()

    with SQLiteJobLedger(db_path) as ledger, ledger._get_conn() as conn:
        version = conn.execute("PRAGMA user_version;").fetchone()[0]
        assert version == 10
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(blocks);").fetchall()}
        assert "owner_id" not in cols
        assert "lease_expires_at" not in cols
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(blocks);").fetchall()}
        assert "idx_block_claim" not in indexes
        status = conn.execute("SELECT status FROM blocks WHERE block_id = 'b1'").fetchone()[0]
        assert status == "pending"


def test_transaction_rollback_does_not_poison_connection(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """Regression for.

    A non-SQLite exception raised inside a manual transaction (here a malformed
    batch missing ``block_id``) must roll back so the shared connection is not left
    "within a transaction". Otherwise every subsequent BEGIN IMMEDIATE would fail with
    "cannot start a transaction within a transaction" and the job becomes permanently
    unwritable.
    """
    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_rollback"
    ledger.init_job(job_id, sample_doc_ir, target_lang="zh")

    malformed = [{"status": BlockStatus.DRAFTED, "target_text": "x"}]  # missing "block_id"
    with pytest.raises(KeyError):
        ledger.save_checkpoints_batch(malformed)

    # The connection must still be usable for a fresh transaction afterwards.
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.DRAFTED,
        target_text="recovery OK",
    )
    block = ledger.get_block("ch01#b001")
    assert block is not None
    assert block.target_text == "recovery OK"


def test_v5_contract_columns_round_trip(tmp_path: Path) -> None:
    """Roles + policy verdict + provenance survive ledger persist/resume."""
    db_path = tmp_path / "ledger_v5.db"
    block = IRBlock(
        id="ch01#b001",
        flow_id=FlowID.MAIN_STORY,
        spine_index=0,
        source_text="Body prose.",
        layout_role=LayoutRole.BODY,
        semantic_role=SemanticRole.MAIN_TEXT,
        structure_role=StructureRole.PARAGRAPH,
        policy_translate=False,
        policy_reason="verdict:short_label",
        provenance={"page_kind": "mixed_complex", "continuation": {"group_id": "g1"}},
    )
    doc = DocumentIR(
        doc_id="v5",
        source_path="/tmp/v5.pdf",
        format_type="pdf",
        metadata={},
        blocks=[block],
    )
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job("job_v5", doc, target_lang="zh")
        got = ledger.get_block("ch01#b001")
    assert got is not None
    assert got.layout_role == LayoutRole.BODY
    assert got.semantic_role == SemanticRole.MAIN_TEXT
    assert got.structure_role == StructureRole.PARAGRAPH
    assert got.policy_translate is False
    assert got.policy_reason == "verdict:short_label"
    assert got.provenance["page_kind"] == "mixed_complex"
    assert got.effective_should_translate() is False
    assert got.validate_contract() == []


def test_v5_append_chapter_carries_contract(tmp_path: Path) -> None:
    """Append_chapter (ingest path) persists contract annotations."""
    db_path = tmp_path / "ledger_v5b.db"
    block = IRBlock(
        id="ch01#b002",
        flow_id=FlowID.CAPTION,
        spine_index=1,
        source_text="FIG. 1 caption.",
    )
    block.derive_roles()
    chapter = ChapterIR(
        doc_id="v5b",
        chapter_id="pdf_main",
        title="t",
        spine_index=1,
        blocks=[block],
    )
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest("job_v5b", _manifest_stub())
        ledger.append_chapter("job_v5b", chapter)
        got = ledger.get_block("ch01#b002")
    assert got is not None
    assert got.layout_role == LayoutRole.CAPTION
    assert got.structure_role == StructureRole.PARAGRAPH


def test_job_fingerprint_roundtrip(tmp_path: Path) -> None:
    """Fingerprint is None for new jobs and round-trips once stored."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job_from_manifest("job_fp", _manifest_stub())
    assert ledger.get_job_fingerprint("job_fp") is None
    assert ledger.get_job_fingerprint("job_missing") is None
    ledger.set_job_fingerprint("job_fp", "abc123")
    assert ledger.get_job_fingerprint("job_fp") == "abc123"


def test_job_fingerprint_corrupt_row_raises_instead_of_none(tmp_path: Path) -> None:
    """None on the resume path means "clear every block" — so a read that
    cannot parse metadata_json must raise, never masquerade as absence."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job_from_manifest("job_fp", _manifest_stub())
    ledger.set_job_fingerprint("job_fp", "abc123")
    with sqlite3.connect(ledger.db_path) as raw:
        raw.execute("UPDATE job_meta SET metadata_json = '{corrupt'")
    with pytest.raises(ValueError):
        ledger.get_job_fingerprint("job_fp")


def test_clear_job_blocks_resets_for_fresh_ingest(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """clear_job_blocks removes blocks but keeps the job row resumable."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_001", sample_doc_ir, target_lang="zh")
    assert len(ledger.get_all_blocks("job_001")) == 5
    removed = ledger.clear_job_blocks("job_001")
    assert removed == 5
    assert ledger.get_all_blocks("job_001") == []
    assert ledger.get_job_stats("job_001")["total"] == 0
    assert ledger.clear_job_blocks("job_missing") == 0


def test_fresh_reingest_interrupted_still_forces_a_full_reparse(
    tmp_path: Path, sample_doc_ir: DocumentIR
) -> None:
    """The resume guard must not mistake a half-loaded ``--fresh`` job for done.

    ``run_ingest_stage`` chooses between "trust these blocks" and "clear and
    re-parse" from a single signal — whether a fingerprint is on the row. So a
    ``--fresh`` run that dies partway through re-parsing has to leave that
    signal absent: with the completed run's hash still stored, the next resume
    finds nothing to complain about (same file), skips the incomplete-ingest
    clear (fingerprint present), and skips parsing (total != 0) — exporting and
    finalizing a book whose later chapters were never loaded.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger_fresh.db")
    ledger.init_job("job_fresh", sample_doc_ir, target_lang="zh")
    ledger.set_job_fingerprint("job_fresh", "sha256-of-the-real-file")
    ledger.clear_job_blocks("job_fresh")  # --fresh begins
    # ...and dies here, one chapter into the re-parse.
    ledger.append_chapter(
        "job_fresh",
        ChapterIR(
            doc_id=sample_doc_ir.doc_id,
            chapter_id="ch01",
            title="Chapter 1",
            spine_index=1,
            blocks=sample_doc_ir.blocks[:2],
        ),
    )

    assert ledger.get_job_stats("job_fresh")["total"] > 0, "resume would skip parsing"
    assert ledger.get_job_fingerprint("job_fresh") is None, (
        "a stale fingerprint on a cleared job hides the incomplete ingest from the resume guard"
    )


def test_legacy_table_without_current_columns_fails_at_open(tmp_path: Path) -> None:
    """A pre-versioning ledger must be refused at open, not at first read.

    ``user_version`` 0 takes the initialize branch, whose CREATE is
    ``IF NOT EXISTS``: an existing legacy ``blocks`` table is left alone and
    stamped version 8, and because ``current_version`` is then set to the target,
    every ``if current_version < N`` migration is skipped. The missing columns
    used to surface as ``IndexError: No item with that key`` from
    ``_row_to_block`` -- after parsing, after the bible stage, and after a paid
    batch had been submitted.
    """
    import sqlite3

    from ubt.core.exceptions import LedgerError

    db_path = tmp_path / "v0_legacy.db"
    legacy = sqlite3.connect(str(db_path))
    # A real early-schema table: everything except the provenance column added
    # by migration 5, and no user_version (so it reads back as 0).
    legacy.executescript("""
        CREATE TABLE blocks (
            block_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            flow_id TEXT NOT NULL,
            spine_index INTEGER NOT NULL DEFAULT 0,
            block_type TEXT NOT NULL DEFAULT 'narrative',
            bbox_json TEXT,
            style_json TEXT,
            source_text TEXT NOT NULL DEFAULT '',
            draft_text TEXT,
            target_text TEXT,
            status TEXT NOT NULL,
            skip_translate INTEGER DEFAULT 0,
            glossary_hits_json TEXT,
            mtqe_score REAL,
            repair_rounds INTEGER DEFAULT 0,
            error_flags_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tm_hit INTEGER DEFAULT 0,
            mqm_severity TEXT DEFAULT NULL,
            mqm_spans_json TEXT DEFAULT NULL,
            layout_role TEXT DEFAULT NULL,
            semantic_role TEXT DEFAULT NULL,
            structure_role TEXT DEFAULT NULL,
            policy_translate INTEGER DEFAULT NULL,
            policy_reason TEXT DEFAULT NULL
        );
    """)
    legacy.commit()
    legacy.close()

    with pytest.raises(LedgerError) as exc:
        SQLiteJobLedger(db_path)
    message = str(exc.value)
    assert "provenance_json" in message, message
    assert "predates this schema" in message, message


def test_job_usage_totals_survive_a_restart(tmp_path: Path) -> None:
    """A resumed job's bill belongs to the ledger, not to provider memory."""
    ledger = SQLiteJobLedger(tmp_path / "usage.sqlite")
    job_id = "job_usage"
    try:
        ledger.init_job_from_manifest(job_id, _manifest_stub())
        ledger.record_job_usage(job_id, {"m": {"prompt_tokens": 1000, "completion_tokens": 10}})
        reported: Any = {
            "m": {"prompt_tokens": 1500, "completion_tokens": 20},
            "bad": {"x": "nan"},
        }
        ledger.record_job_usage(job_id, reported)

        # Absolute, not incremental: the write is the new lifetime total, and a
        # non-numeric counter is dropped rather than guessed at.
        assert ledger.get_job_usage(job_id) == {
            "m": {"prompt_tokens": 1500, "completion_tokens": 20}
        }
        # The write shares its JSON blob with resume-critical keys; it must not
        # disturb them (that exact rewrite is what used to lose the fingerprint).
        ledger.set_job_metadata_value(job_id, "source_fingerprint", "abc123")
        ledger.record_job_usage(job_id, {"m": {"prompt_tokens": 1501, "completion_tokens": 20}})
        assert ledger.get_job_fingerprint(job_id) == "abc123"
    finally:
        ledger.close()


def test_merge_usage_totals_sums_per_model_without_touching_inputs() -> None:
    """The pipeline merges its start-of-run bill with the run delta per event."""
    from ubt.core.engine.ledger import merge_usage_totals

    lifetime = {"a": {"prompt_tokens": 5}}
    delta = {"a": {"prompt_tokens": 7, "cached_tokens": 2}, "b": {"prompt_tokens": 1}}

    merged = merge_usage_totals(lifetime, delta)

    assert merged == {"a": {"prompt_tokens": 12, "cached_tokens": 2}, "b": {"prompt_tokens": 1}}
    assert lifetime == {"a": {"prompt_tokens": 5}}
    assert delta == {"a": {"prompt_tokens": 7, "cached_tokens": 2}, "b": {"prompt_tokens": 1}}


def test_ledger_resolves_doc_id_alias(tmp_path: Path) -> None:
    """Fix 2: Verify job lookups resolve when called with doc_id instead of job_id."""
    db_file = tmp_path / "test_ledger.sqlite"
    ledger = SQLiteJobLedger(db_file)

    manifest = BookManifest(
        doc_id="sha256_abcdef1234567890",
        title="Test Book",
        source_path="/path/to/book.epub",
        chapters=[ChapterMeta(chapter_id="ch01", title="Ch 1", spine_index=1)],
    )

    actual_job_id = "job_custom_9999"
    ledger.init_job_from_manifest(actual_job_id, manifest)

    # Calling get_job_stats with doc_id directly resolves to the latest job
    # for that doc (the same aliasing the former assemble_document_ir used).
    stats = ledger.get_job_stats(manifest.doc_id)
    assert "total" in stats
    assert ledger.get_job_target_lang(manifest.doc_id) == manifest.target_lang


def test_set_job_metadata_value_raises_on_unknown_job(tmp_path: Path) -> None:
    """P1-12: a metadata write for a job with no job_meta row must surface,
    not silently return. The write side of ``source_fingerprint`` depends on
    this: a swallowed failure leaves the fingerprint absent, which the next
    resume reads as "ingest never finished" and clears every block.
    """
    from ubt.core.exceptions import LedgerError

    ledger = SQLiteJobLedger(tmp_path / "meta_raise.sqlite")
    with pytest.raises(LedgerError):
        ledger.set_job_metadata_value("no_such_job", "source_fingerprint", "sha256:deadbeef")
    # set_job_fingerprint routes through the same writer, so it propagates too.
    with pytest.raises(LedgerError):
        ledger.set_job_fingerprint("no_such_job", "sha256:deadbeef")
    ledger.close()


def test_blocks_seq_tracks_block_writes_only(tmp_path: Path, sample_doc_ir: DocumentIR) -> None:
    """The revision StageContext caches against moves when block rows change.

    Metadata writes must not move it: they happen on every progress event, and
    invalidating on those would cost a full re-read of the book per event while
    proving nothing about the blocks.
    """
    ledger = SQLiteJobLedger(tmp_path / "seq.sqlite")
    ledger.init_job("seq_job", sample_doc_ir, target_lang="zh")
    after_init = ledger.blocks_seq
    assert after_init > 0, "ingest wrote every block, so the revision must have moved"

    ledger.set_job_metadata_value("seq_job", "some_key", {"a": 1})
    assert ledger.blocks_seq == after_init, "a job_meta write is not a block write"

    assert ledger.save_checkpoints_batch(
        [{"block_id": sample_doc_ir.blocks[0].id, "status": BlockStatus.DRAFTED}]
    )
    assert ledger.blocks_seq > after_init

    before_clear = ledger.blocks_seq
    assert ledger.clear_job_blocks("seq_job") == len(sample_doc_ir.blocks)
    assert ledger.blocks_seq > before_clear


def test_job_id_resolution_logs_sqlite_errors_and_stays_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing/locked job_meta table reads as "no such job" — but audibly.

    The narrow ``sqlite3.Error`` catch keeps the documented None contract; the
    log is what separates a real DB fault from a genuinely absent job.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")

    def _boom(_conn: sqlite3.Connection, _job_id: str) -> str:
        raise sqlite3.OperationalError("no such table: job_meta")

    monkeypatch.setattr(ledger, "_resolve_actual_job_id", _boom)
    with caplog.at_level(logging.WARNING, logger="ubt.core.engine.ledger"):
        assert ledger.get_job_status("ghost") is None
        assert ledger.get_job_target_lang("ghost") is None
        assert ledger.get_job_metadata_value("ghost", "k") is None
        assert ledger.clear_job_blocks("ghost") == 0
    assert sum("resolution failed" in rec.message for rec in caplog.records) == 4


def test_job_id_resolution_propagates_non_sqlite_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrowing from ``Exception`` to ``sqlite3.Error`` must not swallow bugs."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")

    def _boom(_conn: sqlite3.Connection, _job_id: str) -> str:
        raise RuntimeError("programming error, not a DB fault")

    monkeypatch.setattr(ledger, "_resolve_actual_job_id", _boom)
    with pytest.raises(RuntimeError):
        ledger.get_job_status("ghost")


def test_reset_transient_failures_chapter_scope(tmp_path: Path) -> None:
    """Chapter-streaming runs the draft stage once per chapter; a job-wide
    transient reset from a later chapter would NULL the preserved paid
    target_text of earlier-chapter blocks whose draft pass is over, and no
    stage re-drafts them this run. Chapter-scoped resets must leave other
    chapters untouched."""
    doc = DocumentIR(
        doc_id="test_doc_sha256",
        source_path="/tmp/test_book.epub",
        format_type="epub",
        metadata={"title": "Test Book"},
        blocks=[
            IRBlock(
                id="ch01#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                source_text="Paragraph 1 source content for testing.",
            ),
            IRBlock(
                id="ch02#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=2,
                source_text="Paragraph 2 source content for testing.",
            ),
        ],
    )
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    ledger.init_job("job_scope", doc, target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.FAILED,
        target_text="已付费译文",
        error_flags=["Repair error: 503 Service Unavailable"],
    )
    ledger.save_checkpoint(
        block_id="ch02#b001",
        status=BlockStatus.FAILED,
        target_text="另一章",
        error_flags=["Drafting error: ConnectionResetError"],
    )

    reset_ids = ledger.reset_transient_failures("job_scope", chapter_id="ch02")
    assert reset_ids == ["ch02#b001"]

    untouched = ledger.get_block("ch01#b001")
    assert untouched is not None
    assert untouched.status == BlockStatus.FAILED
    assert untouched.target_text == "已付费译文"
    requeued = ledger.get_block("ch02#b001")
    assert requeued is not None
    assert requeued.status == BlockStatus.PENDING

    # Job-wide default (non-streaming resume) still resets everything transient.
    assert ledger.reset_transient_failures("job_scope") == ["ch01#b001"]
    ledger.close()


@pytest.mark.fast
def test_ledger_following_text_head_handles_colon_delimiter(tmp_path: Path) -> None:
    db_path = tmp_path / "test_ledger.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_test"

    blocks = [
        IRBlock(
            id="ch01:p0001",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="First sentence.",
        ),
        IRBlock(
            id="ch01:p0002",
            spine_index=2,
            flow_id=FlowID.MAIN_STORY,
            source_text="Following sentence in same chapter.",
        ),
    ]
    doc_ir = DocumentIR(
        doc_id="doc1",
        source_path="/tmp/test.md",
        format_type="md",
        blocks=blocks,
    )
    ledger.init_job(job_id, doc_ir, target_lang="zh")

    head = ledger.get_following_text_head(
        job_id=job_id,
        flow_id=FlowID.MAIN_STORY,
        after_spine_index=1,
        chapter_id="ch01",
    )
    assert "Following sentence" in head, (
        f"Failed to fetch following text head with colon delimiter: '{head}'"
    )
    ledger.close()


def test_chapter_filter_does_not_over_match_like_wildcards(tmp_path: Path) -> None:
    """A chapter id containing ``_`` must not match a different chapter.

    Chapter ids are parser-derived and routinely contain ``_`` (``ch_001``,
    ``docx_main``); the filter interpolated them into ``LIKE`` unescaped, so
    ``ch_001_a_b`` also matched ``ch_001_aXb`` and a chapter-scoped read or
    reset hit foreign rows (and could re-bill another chapter's paid drafts).
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    blocks = [
        IRBlock(id="ch_001_a_b#b1", spine_index=0, source_text="x"),
        IRBlock(id="ch_001_aXb#b1", spine_index=1, source_text="y"),
        IRBlock(id="ch_101#b1", spine_index=2, source_text="z"),
    ]
    doc = DocumentIR(doc_id="d", source_path="/tmp/d.pdf", format_type="pdf", blocks=blocks)
    ledger.init_job("j1", doc, target_lang="zh")

    assert sorted(b.id for b in ledger.fetch_pending_blocks("j1", chapter_id="ch_001_a_b")) == [
        "ch_001_a_b#b1"
    ]
    assert sorted(b.id for b in ledger.get_blocks_by_chapter("j1", "ch_001_a_b")) == [
        "ch_001_a_b#b1"
    ]
    assert sorted(b.id for b in ledger.fetch_pending_blocks("j1", chapter_id="ch_101")) == [
        "ch_101#b1"
    ]
    ledger.close()
