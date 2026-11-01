from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.ledger import (
    SQLiteJobLedger,
    _as_usage_totals,
    merge_usage_totals,
)
from ubt.core.engine.ledger_base import _REQUIRED_BLOCK_COLUMNS, SCHEMA_VERSION
from ubt.core.exceptions import LedgerError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    FlowID,
    InlineRun,
    IRBlock,
    StyleMeta,
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


def test_fresh_ledger_lands_on_the_target_schema_version(ledger: SQLiteJobLedger) -> None:
    with ledger._get_conn() as conn:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
        columns = {str(r["name"]) for r in conn.execute("PRAGMA table_info(blocks);").fetchall()}
    assert columns.issuperset(_REQUIRED_BLOCK_COLUMNS)


def test_reopening_a_ledger_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    with SQLiteJobLedger(db_path) as first:
        _init(first)
    with SQLiteJobLedger(db_path) as second:
        assert second.get_job_status("job") == "initialized"
        with second._get_conn() as conn:
            assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION


def test_read_only_ledger_on_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(LedgerError):
        SQLiteJobLedger(tmp_path / "missing.db", read_only=True)


def test_read_only_ledger_reads_an_existing_file(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    with SQLiteJobLedger(db_path) as writable:
        _init(writable)
    with SQLiteJobLedger(db_path, read_only=True) as reader:
        assert reader.get_job_status("job") == "initialized"


def test_close_is_idempotent_and_refuses_to_reopen(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.close()
    ledger.close()
    with pytest.raises(LedgerError):
        ledger.get_job_status("job")


def test_blocks_seq_advances_on_writes_only(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    before = ledger.blocks_seq
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    after_write = ledger.blocks_seq
    assert after_write > before
    ledger.get_all_blocks("job")
    assert ledger.blocks_seq == after_write


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
    _init(ledger)
    assert ledger.get_job_metadata_value("job", "marker") == "kept"


def test_a_manifest_replaces_an_accepted_placeholder(ledger: SQLiteJobLedger) -> None:
    ledger.init_job_placeholder(
        "job",
        "/uploads/book.pdf",
        "zh",
        title="Accepted",
        submitted_request={"input_path": "/uploads/book.pdf", "job_id": "job"},
    )
    assert ledger.get_job_metadata_value("job", "accepted_placeholder") is True
    assert ledger.get_job_metadata_value("job", "submitted_request") == {
        "input_path": "/uploads/book.pdf",
        "job_id": "job",
    }
    ledger.append_chapter(
        "job",
        _chapter("ch1", [_block("ch1#1", status=BlockStatus.MTQE_PASSED, target_text="译")]),
    )

    manifest = BookManifest(
        doc_id="doc-real",
        title="Real title",
        source_path="/uploads/book.pdf",
        target_lang="ja",
    )
    ledger.init_job_from_manifest("job", manifest)

    snapshot = ledger.get_job_snapshot("job")
    assert snapshot is not None
    assert snapshot["doc_id"] == "doc-real"
    assert snapshot["target_lang"] == "ja"
    assert ledger.get_job_metadata_value("job", "title") == "Real title"
    assert ledger.get_job_metadata_value("job", "accepted_placeholder") is None
    assert ledger.get_job_metadata_value("job", "submitted_request") == {
        "input_path": "/uploads/book.pdf",
        "job_id": "job",
    }
    assert snapshot["total"] == 1
    assert snapshot["completed"] == 1


def test_a_manifest_does_not_replace_a_real_job_row(ledger: SQLiteJobLedger) -> None:
    _init(ledger, doc_id="real-doc", target_lang="ja")
    ledger.set_job_metadata_value("job", "marker", "kept")
    _init(ledger, doc_id="other-doc", target_lang="fr")

    snapshot = ledger.get_job_snapshot("job")
    assert snapshot is not None
    assert snapshot["doc_id"] == "real-doc"
    assert snapshot["target_lang"] == "ja"
    assert ledger.get_job_metadata_value("job", "marker") == "kept"


def test_a_placeholder_round_trips_the_submitted_request(ledger: SQLiteJobLedger) -> None:
    request = {
        "input_path": "/uploads/book.pdf",
        "output_path": "/outputs/book_zh.pdf",
        "target_lang": "zh",
        "profile": "textbook",
        "job_id": "job",
    }

    ledger.init_job_placeholder(
        "job", "/uploads/book.pdf", "zh", title="Accepted", submitted_request=request
    )
    ledger.init_job_placeholder(
        "job", "/uploads/book.pdf", "zh", title="Accepted", submitted_request=request
    )

    assert ledger.get_job_metadata_value("job", "submitted_request") == request
    assert ledger.get_job_metadata_value("job", "title") == "Accepted"


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


def test_finalize_writes_artifact_pointers_with_the_status(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.finalize_job(
        "job",
        "completed",
        artifacts={"output_file": "/out/book.pdf", "report_file": "/out/report.json"},
    )
    assert ledger.get_job_metadata_value("job", "output_file") == "/out/book.pdf"
    assert ledger.get_job_metadata_value("job", "report_file") == "/out/report.json"


def test_finalize_failed_is_permissive(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.finalize_job("job", "failed")
    assert ledger.get_job_status("job") == "failed"


def test_finalize_failed_does_not_overwrite_a_cancellation(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.finalize_job("job", "cancelled")
    ledger.finalize_job("job", "failed")
    assert ledger.get_job_status("job") == "cancelled"


def test_finalize_unknown_job_raises(ledger: SQLiteJobLedger) -> None:
    with pytest.raises(LedgerError):
        ledger.finalize_job("ghost", "failed")


def test_blocks_of_two_jobs_with_the_same_block_id_do_not_collide(
    ledger: SQLiteJobLedger,
) -> None:
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


def test_append_chapter_upserts_a_repeated_block_id(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1", text="v1")]))
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1", text="v2")]))
    assert ledger.get_block("ch1#1", job_id="job").source_text == "v2"  # type: ignore[union-attr]


def test_append_chapter_roundtrips_structural_element_fields(ledger: SQLiteJobLedger) -> None:
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


def test_span_chars_survive_the_ledger_round_trip(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    block = IRBlock(
        element=make_element(
            id="ch1#c",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="hello world",
            chars=(12, 23),
        ),
    )
    ledger.append_chapter("job", _chapter("ch1", [block]))

    stored = ledger.get_block("ch1#c", job_id="job")
    assert stored is not None
    assert stored.element.span.chars == (12, 23)


def test_a_block_without_span_chars_round_trips_as_none(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#n")]))
    stored = ledger.get_block("ch1#n", job_id="job")
    assert stored is not None
    assert stored.element.span.chars is None


def test_save_checkpoints_batch_persists_style_and_preserves_when_omitted(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    style = StyleMeta(target_runs=(InlineRun(text="重点", bold=True),))
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "target_text": "x", "style": style}],
        job_id="job",
    )
    block = ledger.get_block("ch1#1", job_id="job")
    assert block is not None and block.style is not None
    assert block.style.target_runs == style.target_runs

    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "target_text": "x repaired"}],
        job_id="job",
    )
    block = ledger.get_block("ch1#1", job_id="job")
    assert block is not None and block.style is not None
    assert block.style.target_runs == style.target_runs


def test_save_checkpoints_batch_counts_and_preserves_omitted_flags(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1"), _block("ch1#2", spine_index=1)]))
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "error_flags": ["keep"]}],
        job_id="job",
    )
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
    assert first.error_flags == ["keep"]


def test_save_checkpoints_batch_explicit_empty_list_clears_flags(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "error_flags": ["drop"]}],
        job_id="job",
    )
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "error_flags": []}], job_id="job"
    )
    assert ledger.get_block("ch1#1", job_id="job").error_flags == []  # type: ignore[union-attr]


def test_save_checkpoints_batch_does_not_resurrect_a_terminal_row(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.MTQE_PASSED, "target_text": "done"}],
        job_id="job",
    )
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
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.FAILED}], job_id="job"
    )
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
    assert ledger.get_block("ch1#2", job_id="job").status is BlockStatus.MTQE_PASSED  # type: ignore[union-attr]


def test_reset_blocks_to_pending_clears_the_whole_verdict(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter("job", _chapter("ch1", [_block("ch1#1")]))
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch1#1",
                "status": BlockStatus.MTQE_PASSED,
                "target_text": "t",
                "draft_text": "d",
                "mtqe_score": 0.9,
                "repair_rounds": 2,
                "error_flags": ["e"],
                "tm_hit": True,
                "mqm_severity": "critical",
            }
        ],
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


def test_resume_requeues_transient_failures_but_keeps_triage_verdicts(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job",
        _chapter(
            "ch1",
            [
                _block("ch1#transient", status=BlockStatus.FAILED),
                _block("ch1#review", status=BlockStatus.NEEDS_HUMAN),
                _block("ch1#blocked", status=BlockStatus.NEEDS_HUMAN),
            ],
        ),
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch1#transient",
                "status": BlockStatus.FAILED,
                "error_flags": ["Drafting error: provider 503"],
            },
            {
                "block_id": "ch1#review",
                "status": BlockStatus.NEEDS_HUMAN,
                "error_flags": ["Drafting error: provider 503", "needs_human_review"],
            },
            {
                "block_id": "ch1#blocked",
                "status": BlockStatus.NEEDS_HUMAN,
                "error_flags": ["Drafting error: provider 503", "mqm_critical_blocked"],
            },
        ],
        job_id="job",
    )

    assert ledger.reset_transient_failures("job") == ["ch1#transient"]
    assert ledger.get_block("ch1#transient", job_id="job").status is BlockStatus.PENDING  # type: ignore[union-attr]
    assert ledger.get_block("ch1#review", job_id="job").status is BlockStatus.NEEDS_HUMAN  # type: ignore[union-attr]
    assert ledger.get_block("ch1#blocked", job_id="job").status is BlockStatus.NEEDS_HUMAN  # type: ignore[union-attr]


def test_resume_recovers_legacy_configuration_failures_only(
    ledger: SQLiteJobLedger,
) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job",
        _chapter(
            "ch1",
            [
                _block("ch1#route", status=BlockStatus.FAILED),
                _block("ch1#source", status=BlockStatus.FAILED),
            ],
        ),
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch1#route",
                "status": BlockStatus.FAILED,
                "error_flags": [
                    'Drafting unrecoverable: Model API error (404): {"error":'
                    '{"message":"no router for requested model"}}'
                ],
            },
            {
                "block_id": "ch1#source",
                "status": BlockStatus.FAILED,
                "error_flags": ["Drafting unrecoverable: Model API error (422): malformed source"],
            },
        ],
        job_id="job",
    )

    assert ledger.reset_transient_failures("job") == ["ch1#route"]
    route = ledger.get_block("ch1#route", job_id="job")
    source = ledger.get_block("ch1#source", job_id="job")
    assert route is not None and route.status is BlockStatus.PENDING
    assert source is not None and source.status is BlockStatus.FAILED


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
    assert stats["completed"] == 2
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
    ledger.save_checkpoints_batch(
        [{"block_id": "ch1#2", "status": BlockStatus.MTQE_PASSED, "target_text": "T2"}],
        job_id="job",
    )
    context = ledger.get_preceding_text_tail("job", FlowID.MAIN_STORY, 3, max_chars=50)
    assert "T2" in context
    assert "s2" not in context


def test_following_text_head_stays_source_side(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job", _chapter("ch1", [_block(f"ch1#{i}", spine_index=i, text=f"s{i}") for i in (1, 2, 3)])
    )
    head = ledger.get_following_text_head("job", FlowID.MAIN_STORY, 0, max_chars=50)
    assert head == "s1 s2 s3"


def test_batch_liveness_transitions(ledger: SQLiteJobLedger) -> None:
    ledger.reserve_batch_job("key-1", "job")
    ledger.finalize_batch_job("key-1", "batch-1", status="submitted")
    assert ledger.is_batch_live("batch-1") is True
    ledger.update_batch_job_status("batch-1", "consumed")
    assert ledger.is_batch_live("batch-1") is False


def test_completed_but_unconsumed_batch_is_still_resumable(ledger: SQLiteJobLedger) -> None:
    ledger.reserve_batch_job("key-1", "job")
    ledger.finalize_batch_job("key-1", "batch-1", status="completed")
    assert ledger.find_live_batch_for_job("job", exclude_key="key-2") == "batch-1"
    assert ledger.find_live_batch_for_job("job", exclude_key="key-1") is None


def test_is_batch_live_is_true_for_a_missing_row(ledger: SQLiteJobLedger) -> None:
    assert ledger.is_batch_live("never-seen") is True


def test_reserve_batch_job_create_then_pending_then_resume(ledger: SQLiteJobLedger) -> None:
    assert ledger.reserve_batch_job("key-1", "job") == ("create", None)
    assert ledger.reserve_batch_job("key-1", "job") == ("pending", None)
    ledger.finalize_batch_job("key-1", "batch-real", status="submitted")
    assert ledger.reserve_batch_job("key-1", "job") == ("resume", "batch-real")


def test_reserve_batch_job_reclaims_a_stale_sentinel(
    ledger: SQLiteJobLedger, caplog: pytest.LogCaptureFixture
) -> None:
    ledger.reserve_batch_job("key-1", "job")
    with ledger._get_conn() as conn:
        conn.execute(
            "UPDATE batch_jobs SET updated_at = datetime('now', '-1 hour') WHERE batch_id = ?",
            ("creating:key-1",),
        )
    with caplog.at_level(logging.WARNING):
        assert ledger.reserve_batch_job("key-1", "job") == ("create", None)
    assert "died mid-create" in caplog.text
    assert "may remain" in caplog.text


def test_finalize_batch_job_records_a_paid_batch_whose_sentinel_was_reclaimed(
    ledger: SQLiteJobLedger,
) -> None:
    ledger.reserve_batch_job("key-0", "job")
    ledger.finalize_batch_job("key-0", "batch-a", status="submitted")
    ledger.finalize_batch_job("key-1", "batch-b", status="submitted")
    with ledger._get_conn() as conn:
        rows = {
            str(row["batch_id"]): str(row["status"])
            for row in conn.execute("SELECT batch_id, status FROM batch_jobs").fetchall()
        }
    assert rows == {"batch-a": "submitted", "batch-b": "submitted"}


def test_a_legacy_ledger_is_refused_not_silently_migrated(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_file)
    conn.executescript("""
        CREATE TABLE job_meta (job_id TEXT PRIMARY KEY);
        CREATE TABLE blocks (block_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
            source_text TEXT NOT NULL, status TEXT NOT NULL);
        PRAGMA user_version = 10;
    """)
    conn.commit()
    conn.close()

    with pytest.raises(LedgerError) as excinfo:
        SQLiteJobLedger(db_file)
    assert "flattened baseline" in str(excinfo.value)
    assert excinfo.value.details["db_version"] == 10
    assert excinfo.value.details["baseline_version"] == SCHEMA_VERSION
    conn = sqlite3.connect(db_file)
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 10
    conn.close()


def test_a_fresh_ledger_lands_on_the_flat_baseline_version(ledger: SQLiteJobLedger) -> None:
    with ledger._get_conn() as conn:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION


def test_a_file_missing_a_required_column_is_refused(tmp_path: Path) -> None:
    db_file = tmp_path / "broken.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(f"""
        CREATE TABLE job_meta (job_id TEXT PRIMARY KEY);
        CREATE TABLE blocks (block_id TEXT PRIMARY KEY, job_id TEXT NOT NULL);
        PRAGMA user_version = {SCHEMA_VERSION};
    """)
    conn.commit()
    conn.close()

    with pytest.raises(LedgerError) as excinfo:
        SQLiteJobLedger(db_file)
    assert "missing required blocks columns" in str(excinfo.value)


def test_the_baseline_is_idempotent_on_reopen(tmp_path: Path) -> None:
    db_file = tmp_path / "baseline.db"
    ledger = SQLiteJobLedger(db_file)
    ledger.close()
    ledger = SQLiteJobLedger(db_file)
    try:
        assert ledger.get_job_status("nonexistent") is None
    finally:
        ledger.close()


def test_a_fresh_baseline_has_the_composite_primary_key_and_no_dead_columns(
    ledger: SQLiteJobLedger,
) -> None:
    with ledger._get_conn() as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='blocks';"
        ).fetchone()[0]
        cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(blocks);").fetchall()}
    assert "PRIMARY KEY (job_id, block_id)" in sql
    assert "semantic_role" not in cols
    assert "structure_role" not in cols
    assert "owner_id" not in cols
    assert "lease_expires_at" not in cols
    assert {"span_chars_json", "level", "marker", "provenance_json"} <= cols


def test_get_job_stats_reports_whether_qe_measured_anything(ledger: SQLiteJobLedger) -> None:
    _init(ledger)
    ledger.append_chapter(
        "job",
        _chapter(
            "ch1",
            [
                _block("ch1#1", spine_index=1),
                _block("ch1#2", spine_index=2, skip_translate=True),
            ],
        ),
    )
    stats = ledger.get_job_stats("job")
    assert stats["qe_measured"] is False
    assert stats["avg_qe_score"] == 0.0

    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch1#1",
                "status": BlockStatus.MTQE_PASSED,
                "target_text": "T",
                "mtqe_score": 0.87,
            }
        ],
        job_id="job",
    )
    stats = ledger.get_job_stats("job")
    assert stats["qe_measured"] is True
    assert stats["avg_qe_score"] == 0.87


def test_ledger_staleness_helper(tmp_path: Path) -> None:
    from ubt.mcp.server import _LEDGER_STALENESS_S, _ledger_is_stale

    ledger = tmp_path / "job_x.sqlite"
    ledger.write_bytes(b"x")
    assert _ledger_is_stale(ledger) is False

    old = time.time() - _LEDGER_STALENESS_S - 10.0
    import os

    os.utime(ledger, (old, old))
    assert _ledger_is_stale(ledger) is True


def test_fresh_resets_a_legacy_ledger_instead_of_refusing(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_file)
    conn.executescript("""
        CREATE TABLE job_meta (job_id TEXT PRIMARY KEY);
        CREATE TABLE blocks (block_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
            source_text TEXT NOT NULL, status TEXT NOT NULL);
        PRAGMA user_version = 14;
    """)
    conn.commit()
    conn.close()

    with pytest.raises(LedgerError):
        SQLiteJobLedger(db_file)

    ledger = SQLiteJobLedger(db_file, fresh=True)
    try:
        with ledger._get_conn() as c:
            assert c.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
        assert ledger.get_job_status("nope") is None
    finally:
        ledger.close()
