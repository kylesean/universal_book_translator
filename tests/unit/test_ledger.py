"""The SQLite job ledger: schema, job lifecycle, block checkpoints, batch jobs.

The ledger is where every block's translation state survives a restart, so its
contracts are load-bearing:

- the schema lands exactly on ``TARGET_SCHEMA_VERSION`` and every column
  ``_row_to_block`` reads is present (a drift here corrupts resume silently);
- block ids are content-derived and repeat across jobs, so a block write must be
  scoped by ``job_id`` (the composite-primary-key contract);
- a checkpoint records state, but a *non-terminal* batch write must not
  resurrect a terminal row, and ``reset_blocks_to_pending`` must clear the whole
  verdict so a re-queued block is genuinely re-derivable;
- the batch create-reservation is the only thing preventing a crash from
  re-submitting (and re-billing) the same provider batch.

All time-dependent behaviour is driven by explicit values or by back-dating a
row, never by sleeping.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.ledger import (
    TARGET_SCHEMA_VERSION,
    SQLiteJobLedger,
    _as_usage_totals,
    merge_usage_totals,
)
from ubt.core.engine.ledger_base import _REQUIRED_BLOCK_COLUMNS
from ubt.core.exceptions import LedgerError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.model.ast import Confidence, Heading, ListItem

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    *,
    spine_index: int = 0,
    text: str = "source",
    block_type: BlockType = BlockType.NARRATIVE,
    flow_id: FlowID = FlowID.MAIN_STORY,
    status: BlockStatus = BlockStatus.PENDING,
    skip_translate: bool = False,
    **state: Any,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=spine_index,
        block_type=block_type,
        flow_id=flow_id,
        source_text=text,
        skip_translate=skip_translate,
    )
    return IRBlock(element=element, status=status, **state)


def _chapter(chapter_id: str, blocks: list[IRBlock], *, spine_index: int = 0) -> ChapterIR:
    return ChapterIR(
        doc_id="doc",
        chapter_id=chapter_id,
        title=chapter_id,
        spine_index=spine_index,
        blocks=blocks,
    )


def _init(ledger: SQLiteJobLedger, job_id: str = "job", doc_id: str = "doc", **kwargs: Any) -> None:
    manifest = BookManifest(doc_id=doc_id, title="Title", source_path="book.pdf", **kwargs)
    ledger.init_job_from_manifest(job_id, manifest)


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[SQLiteJobLedger]:
    with SQLiteJobLedger(tmp_path / "ledger.db") as db:
        yield db


# --------------------------------------------------------------------------- #
# Foundation: schema, connection lifecycle.
# --------------------------------------------------------------------------- #


def test_fresh_ledger_lands_on_the_target_schema_version(ledger: SQLiteJobLedger) -> None:
    with ledger._get_conn() as conn:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == TARGET_SCHEMA_VERSION
        columns = {str(r["name"]) for r in conn.execute("PRAGMA table_info(blocks);").fetchall()}
    assert columns.issuperset(_REQUIRED_BLOCK_COLUMNS)


def test_reopening_a_ledger_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    with SQLiteJobLedger(db_path) as first:
        _init(first)
    with SQLiteJobLedger(db_path) as second:
        assert second.get_job_status("job") == "initialized"
        with second._get_conn() as conn:
            assert conn.execute("PRAGMA user_version;").fetchone()[0] == TARGET_SCHEMA_VERSION


def test_read_only_ledger_on_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(LedgerError):
        SQLiteJobLedger(tmp_path / "missing.db", read_only=True)


def test_read_only_ledger_reads_an_existing_file(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    with SQLiteJobLedger(db_path) as writable:
        _init(writable)
    with SQLiteJobLedger(db_path, read_only=True) as reader:
        assert reader.get_job_status("job") == "initialized"


def test_close_is_idempotent_and_reopens_lazily(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.close()
    ledger.close()  # must never raise
    assert ledger.get_job_status("job") == "initialized"  # read re-initializes


def test_blocks_seq_advances_on_writes_only(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    before = ledger.blocks_seq
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    after_write = ledger.blocks_seq
    assert after_write > before
    ledger.get_all_blocks("job")  # a pure read must not bump the revision
    assert ledger.blocks_seq == after_write


# --------------------------------------------------------------------------- #
# Job lifecycle: metadata, resolution, usage, finalize.
# --------------------------------------------------------------------------- #


def test_init_job_from_manifest_persists_metadata(ledger: SQLiteJobLedger) -> None:
    _init(ledger, target_lang="ja")
    snapshot = ledger.get_job_snapshot("job")
    assert snapshot is not None
    assert snapshot["doc_id"] == "doc"
    assert snapshot["source_path"] == "book.pdf"
    assert snapshot["target_lang"] == "ja"
    assert snapshot["status"] == "initialized"
    assert snapshot["total"] == 0
    assert ledger.get_job_target_lang("job") == "ja"


def test_init_job_from_manifest_is_idempotent(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.set_job_metadata_value("job", "marker", "kept")
    _init(ledger)  # second init must not reset the row
    assert ledger.get_job_metadata_value("job", "marker") == "kept"


def test_get_job_snapshot_is_none_for_an_unknown_job(ledger: SQLiteJobLedger) -> None:
    assert ledger.get_job_snapshot("ghost") is None
    assert ledger.get_job_status("ghost") is None
    assert ledger.get_job_target_lang("ghost") is None


def test_resolve_job_id_by_exact_id_and_doc_id(ledger: SQLiteJobLedger) -> None:
    _init(ledger, job_id="job-1", doc_id="doc-1")
    assert ledger.resolve_job_id("job-1") == "job-1"
    assert ledger.resolve_job_id("doc-1") == "job-1"
    assert ledger.resolve_job_id("unknown") is None


def test_fingerprint_roundtrip(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    assert ledger.get_job_fingerprint("job") is None
    ledger.set_job_fingerprint("job", "sha256:abc")
    assert ledger.get_job_fingerprint("job") == "sha256:abc"


def test_metadata_value_roundtrip_and_missing_job(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    assert ledger.get_job_metadata_value("job", "absent") is None
    ledger.set_job_metadata_value("job", "k", {"nested": [1, 2]})
    assert ledger.get_job_metadata_value("job", "k") == {"nested": [1, 2]}
    with pytest.raises(LedgerError):
        ledger.set_job_metadata_value("ghost", "k", 1)


def test_usage_totals_roundtrip(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    assert ledger.get_job_usage("job") == {}
    ledger.record_job_usage("job", {"model-a": {"in": 10, "out": 5}})
    assert ledger.get_job_usage("job") == {"model-a": {"in": 10, "out": 5}}


def test_usage_helpers_are_pure_and_drop_non_numeric() -> None:
    assert merge_usage_totals({"a": {"in": 1}}, {"a": {"in": 2, "out": 3}, "b": {"x": 1}}) == {
        "a": {"in": 3, "out": 3},
        "b": {"x": 1},
    }
    assert _as_usage_totals({"m": {"ok": 1, "bad": "x", "flag": True, "f": 1.5}}) == {
        "m": {"ok": 1, "f": 1}
    }


def test_finalize_completed_refuses_with_non_terminal_blocks(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1", status=BlockStatus.PENDING)]))
    with pytest.raises(LedgerError):
        ledger.finalize_job("job", "completed")
    ledger.fail_non_terminal_blocks("job", "stage crashed")
    ledger.finalize_job("job", "completed")
    assert ledger.get_job_status("job") == "completed"


def test_finalize_failed_is_permissive(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.finalize_job("job", "failed")  # abort path must land mid-flight
    assert ledger.get_job_status("job") == "failed"


def test_finalize_failed_does_not_overwrite_a_cancellation(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.finalize_job("job", "cancelled")
    ledger.finalize_job("job", "failed")
    assert ledger.get_job_status("job") == "cancelled"


def test_finalize_unknown_job_raises(ledger: SQLiteJobLedger) -> None:
    with pytest.raises(LedgerError):
        ledger.finalize_job("ghost", "failed")


# --------------------------------------------------------------------------- #
# Blocks: scoping, upsert, checkpoints, pagination, stats.
# --------------------------------------------------------------------------- #


def test_blocks_of_two_jobs_with_the_same_block_id_do_not_collide(
    ledger: SQLiteJobLedger,
) -> None:
    # block ids are content-derived (``ch1#1``), so two jobs over the same
    # document produce identical ids; the composite PK must keep them apart.
    _init(ledger, job_id="job-a", doc_id="doc-a")
    _init(ledger, job_id="job-b", doc_id="doc-b")
    ledger.append_chapter("job-a", _chapter("ch1", [_block("ch1#1", text="A")]))
    ledger.append_chapter("job-b", _chapter("ch1", [_block("ch1#1", text="B")]))
    assert ledger.get_block("ch1#1", job_id="job-a").source_text == "A"  # type: ignore[union-attr]
    assert ledger.get_block("ch1#1", job_id="job-b").source_text == "B"  # type: ignore[union-attr]


def test_append_chapter_roundtrips_block_fields(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    block = _block(
        "ch1#1",
        spine_index=3,
        text="hello",
        block_type=BlockType.DIALOGUE,
        flow_id=FlowID.FOOTNOTE,
        status=BlockStatus.DRAFTED,
    )
    ledger.append_chapter("job", _chapter("ch1", [block]))
    stored = ledger.get_block("ch1#1", job_id="job")
    assert stored is not None
    assert stored.id == "ch1#1"
    assert stored.spine_index == 3
    assert stored.block_type is BlockType.DIALOGUE
    assert stored.flow_id is FlowID.FOOTNOTE
    assert stored.source_text == "hello"
    assert stored.status is BlockStatus.DRAFTED


def test_append_chapter_updates_total_blocks_and_upserts(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1", text="v1")]))
    assert ledger.get_total_blocks("job") == 1
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1", text="v2")]))
    assert ledger.get_total_blocks("job") == 1  # upsert, not duplicate
    assert ledger.get_block("ch1#1", job_id="job").source_text == "v2"  # type: ignore[union-attr]


def test_append_chapter_roundtrips_structural_element_fields(ledger: SQLiteJobLedger) -> None:
    # A resumed job re-materializes every block through ``_row_to_block``; the
    # element's level/marker/confidence/decorative must survive, or a
    # ``### Section`` comes back as ``# Section`` and lists lose their bullet.
    _init(ledger)
    heading = IRBlock(
        element=make_element(
            id="ch1#h",
            spine_index=0,
            block_type=BlockType.HEADING,
            source_text="Section",
            level=3,
            confidence=Confidence.UNKNOWN,
            decorative=True,
        ),
        status=BlockStatus.DRAFTED,
    )
    item = IRBlock(
        element=make_element(
            id="ch1#l",
            spine_index=1,
            block_type=BlockType.LIST_ITEM,
            source_text="first",
            marker="1.",
        ),
        status=BlockStatus.DRAFTED,
    )
    ledger.append_chapter("job", _chapter("ch1", [heading, item]))

    stored_heading = ledger.get_block("ch1#h", job_id="job")
    assert stored_heading is not None
    assert isinstance(stored_heading.element, Heading)
    assert stored_heading.element.level == 3
    assert stored_heading.element.confidence is Confidence.UNKNOWN
    assert stored_heading.element.decorative is True

    stored_item = ledger.get_block("ch1#l", job_id="job")
    assert stored_item is not None
    assert isinstance(stored_item.element, ListItem)
    assert stored_item.element.marker == "1."
    assert stored_item.element.confidence is Confidence.INFERRED


def test_save_checkpoint_records_state(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ok = ledger.save_checkpoint(
        "ch1#1",
        BlockStatus.DRAFTED,
        target_text="t1",
        draft_text="d1",
        mtqe_score=0.5,
        repair_rounds=1,
        error_flags=["flag"],
        tm_hit=True,
        mqm_severity="minor",
        mqm_spans=[{"span": [0, 1]}],
        job_id="job",
    )
    assert ok is True
    block = ledger.get_block("ch1#1", job_id="job")
    assert block is not None
    assert block.status is BlockStatus.DRAFTED
    assert block.target_text == "t1"
    assert block.draft_text == "d1"
    assert block.mtqe_score == 0.5
    assert block.repair_rounds == 1
    assert block.error_flags == ["flag"]
    assert block.tm_hit is True
    assert block.mqm_severity == "minor"
    assert block.mqm_spans == [{"span": [0, 1]}]


def test_save_checkpoint_returns_false_for_an_unknown_block(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    assert ledger.save_checkpoint("ghost", BlockStatus.DRAFTED, job_id="job") is False


def test_save_checkpoints_batch_counts_and_preserves_omitted_flags(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1"), _block("ch1#2", spine_index=1)]))
    ledger.save_checkpoint("ch1#1", BlockStatus.DRAFTED, error_flags=["keep"], job_id="job")
    updated = ledger.save_checkpoints_batch(
        [
            {"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "target_text": "x1"},
            {"block_id": "ch1#2", "status": BlockStatus.DRAFTED, "target_text": "x2"},
        ],
        job_id="job",
    )
    assert updated == 2
    first = ledger.get_block("ch1#1", job_id="job")
    assert first is not None
    assert first.target_text == "x1"
    assert first.error_flags == ["keep"]  # omitted key -> COALESCE preserves


def test_save_checkpoints_batch_explicit_empty_list_clears_flags(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.save_checkpoint("ch1#1", BlockStatus.DRAFTED, error_flags=["drop"], job_id="job")
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "error_flags": []}], job_id="job"
    )
    assert ledger.get_block("ch1#1", job_id="job").error_flags == []  # type: ignore[union-attr]


def test_save_checkpoints_batch_does_not_resurrect_a_terminal_row(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.save_checkpoint("ch1#1", BlockStatus.MTQE_PASSED, target_text="done", job_id="job")
    updated = ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "target_text": "resurrect"}],
        job_id="job",
    )
    assert updated == 0
    assert ledger.get_block("ch1#1", job_id="job").status is BlockStatus.MTQE_PASSED  # type: ignore[union-attr]


def test_save_checkpoints_batch_can_override_a_terminal_row_when_asked(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.save_checkpoint("ch1#1", BlockStatus.FAILED, job_id="job")
    updated = ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.REPAIR_PENDING}],
        job_id="job",
        allow_terminal_override=True,
    )
    assert updated == 1
    assert ledger.get_block("ch1#1", job_id="job").status is BlockStatus.REPAIR_PENDING  # type: ignore[union-attr]


def test_fetch_pending_blocks_paginates_by_keyset(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job", _chapter("ch1", [_block(f"ch1#{i}", spine_index=i) for i in range(1, 6)])
    )
    first = ledger.fetch_pending_blocks("job", limit=2)
    assert [b.id for b in first] == ["ch1#1", "ch1#2"]
    second = ledger.fetch_pending_blocks(
        "job",
        limit=2,
        after_spine_index=first[-1].spine_index,
        after_block_id=first[-1].id,
    )
    assert [b.id for b in second] == ["ch1#3", "ch1#4"]


def test_fetch_pending_blocks_excludes_skip_translate(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.append_chapter(
        "job", _chapter("ch2", [_block("ch2#1", spine_index=1, skip_translate=True)])
    )
    assert [b.id for b in ledger.fetch_pending_blocks("job", limit=99)] == ["ch1#1"]


def test_fail_non_terminal_blocks_marks_and_flags(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job",
        _chapter(
            "ch1",
            [
                _block("ch1#1", status=BlockStatus.PENDING),
                _block("ch1#2", spine_index=1, status=BlockStatus.MTQE_PASSED),
            ],
        ),
    )
    failed = ledger.fail_non_terminal_blocks("job", "stage crashed")
    assert failed == ["ch1#1"]
    block = ledger.get_block("ch1#1", job_id="job")
    assert block is not None
    assert block.status is BlockStatus.FAILED
    assert "stage crashed" in block.error_flags
    # A terminal block is untouched.
    assert ledger.get_block("ch1#2", job_id="job").status is BlockStatus.MTQE_PASSED  # type: ignore[union-attr]


def test_reset_blocks_to_pending_clears_the_whole_verdict(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.save_checkpoint(
        "ch1#1",
        BlockStatus.MTQE_PASSED,
        target_text="t",
        draft_text="d",
        mtqe_score=0.9,
        repair_rounds=2,
        error_flags=["e"],
        tm_hit=True,
        mqm_severity="critical",
        job_id="job",
    )
    assert ledger.reset_blocks_to_pending(["ch1#1"], job_id="job") == 1
    block = ledger.get_block("ch1#1", job_id="job")
    assert block is not None
    assert block.status is BlockStatus.PENDING
    assert block.target_text is None
    assert block.draft_text is None
    assert block.mtqe_score is None
    assert block.repair_rounds == 0
    assert block.error_flags == []
    assert block.tm_hit is False
    assert block.mqm_severity is None


def test_get_job_stats_counts_by_status(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    statuses = [
        BlockStatus.PENDING,
        BlockStatus.DRAFTED,
        BlockStatus.MTQE_PASSED,
        BlockStatus.REPAIRED,
        BlockStatus.FAILED,
        BlockStatus.BLOCKED_HUMAN,
    ]
    ledger.append_chapter(
        "job",
        _chapter(
            "ch1",
            [_block(f"ch1#{i}", spine_index=i, status=s) for i, s in enumerate(statuses)],
        ),
    )
    stats = ledger.get_job_stats("job")
    assert stats["total"] == 6
    assert stats["completed"] == 2  # mtqe_passed + repaired
    assert stats["drafted"] == 1
    assert stats["failed"] == 1
    assert stats["blocked_human"] == 1
    assert stats["needs_human"] == 0


def test_preceding_text_tail_prefers_target_over_source(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job",
        _chapter(
            "ch1",
            [
                _block("ch1#1", spine_index=1, text="s1"),
                _block("ch1#2", spine_index=2, text="s2"),
            ],
        ),
    )
    ledger.save_checkpoint("ch1#2", BlockStatus.MTQE_PASSED, target_text="T2", job_id="job")
    context = ledger.get_preceding_text_tail("job", FlowID.MAIN_STORY, 3, max_chars=50)
    assert "T2" in context  # translated prose is what the model continues
    assert "s2" not in context


def test_following_text_head_stays_source_side(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job", _chapter("ch1", [_block(f"ch1#{i}", spine_index=i, text=f"s{i}") for i in (1, 2, 3)])
    )
    head = ledger.get_following_text_head("job", FlowID.MAIN_STORY, 0, max_chars=50)
    assert head == "s1 s2 s3"


# --------------------------------------------------------------------------- #
# Batch API: persistence, liveness, and the create-reservation.
# --------------------------------------------------------------------------- #


def test_batch_liveness_and_idempotency_lookup(ledger: SQLiteJobLedger) -> None:
    ledger.register_batch_job("batch-1", "job", "key-1", status="submitted")
    assert ledger.find_live_batch_by_idempotency_key("key-1") == "batch-1"
    assert ledger.is_batch_live("batch-1") is True
    ledger.update_batch_job_status("batch-1", "consumed")
    assert ledger.find_live_batch_by_idempotency_key("key-1") is None
    assert ledger.is_batch_live("batch-1") is False


def test_completed_but_unconsumed_batch_is_still_resumable(ledger: SQLiteJobLedger) -> None:
    # ``completed`` means the provider finished, not that results were reaped;
    # re-creating here would re-bill the same payload.
    ledger.register_batch_job("batch-1", "job", "key-1", status="completed")
    assert ledger.find_live_batch_by_idempotency_key("key-1") == "batch-1"


def test_is_batch_live_is_true_for_a_missing_row(ledger: SQLiteJobLedger) -> None:
    # With no recorded status the safe action is to cancel (it may be billing).
    assert ledger.is_batch_live("never-seen") is True


def test_reserve_batch_job_create_then_pending_then_resume(ledger: SQLiteJobLedger) -> None:
    assert ledger.reserve_batch_job("key-1", "job") == ("create", None)
    # A fresh sentinel means another worker is submitting right now.
    assert ledger.reserve_batch_job("key-1", "job") == ("pending", None)
    ledger.finalize_batch_job("key-1", "batch-real", status="submitted")
    assert ledger.reserve_batch_job("key-1", "job") == ("resume", "batch-real")


def test_reserve_batch_job_reclaims_a_stale_sentinel(ledger: SQLiteJobLedger) -> None:
    ledger.reserve_batch_job("key-1", "job")
    with ledger._get_conn() as conn:
        conn.execute(
            "UPDATE batch_jobs SET updated_at = datetime('now', '-1 hour') WHERE batch_id = ?",
            ("creating:key-1",),
        )
    # The worker that held the reservation died mid-create; a new caller owns it.
    assert ledger.reserve_batch_job("key-1", "job") == ("create", None)
