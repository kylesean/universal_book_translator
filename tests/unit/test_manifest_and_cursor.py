"""Unit tests for BookManifest, streaming ChapterIR partitions, and cursor pagination."""

from pathlib import Path

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)


def test_manifest_and_streaming_chapters(tmp_path: Path) -> None:
    """Validate streaming chapter ingestion without in-memory full book accumulation."""
    db_path = tmp_path / "stream_ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_manifest_test"

    manifest = BookManifest(
        doc_id="manifest_sha256",
        title="Thinking, Fast and Slow",
        source_path="thinking.epub",
        chapters=[
            ChapterMeta(chapter_id="ch01", title="Two Systems", spine_index=1),
            ChapterMeta(chapter_id="ch02", title="Attention and Effort", spine_index=2),
        ],
    )

    # 1. Initialize metadata from manifest (0 blocks in memory)
    ledger.init_job_from_manifest(job_id, manifest)
    stats = ledger.get_job_stats(job_id)
    assert stats["total"] == 0

    # 2. Stream and append Chapter 1 (e.g. 10 blocks)
    ch1_blocks = [
        IRBlock(
            id=f"ch01#b{i:02d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"Chapter 1 sentence {i}.",
        )
        for i in range(1, 11)
    ]
    ch1 = ChapterIR(
        doc_id="manifest_sha256",
        chapter_id="ch01",
        title="Two Systems",
        spine_index=1,
        blocks=ch1_blocks,
    )
    ledger.append_chapter(job_id, ch1)

    # Blocks dynamically incremented to 10
    stats1 = ledger.get_job_stats(job_id)
    assert stats1["total"] == 10

    # 3. Stream and append Chapter 2 (e.g. 5 blocks)
    ch2_blocks = [
        IRBlock(
            id=f"ch02#b{i:02d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=10 + i,
            source_text=f"Chapter 2 sentence {i}.",
        )
        for i in range(1, 6)
    ]
    ch2 = ChapterIR(
        doc_id="manifest_sha256",
        chapter_id="ch02",
        title="Attention and Effort",
        spine_index=2,
        blocks=ch2_blocks,
    )
    ledger.append_chapter(job_id, ch2)

    # Total blocks now 15
    stats2 = ledger.get_job_stats(job_id)
    assert stats2["total"] == 15


def test_paginated_cursor_and_cross_chapter_bridge(tmp_path: Path) -> None:
    """Validate cursor batch pagination and seamless cross-chapter sliding context."""
    db_path = tmp_path / "cursor_ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_cursor_test"

    manifest = BookManifest(
        doc_id="doc_cursor",
        title="The Hobbit",
        source_path="hobbit.epub",
    )
    ledger.init_job_from_manifest(job_id, manifest)

    # Append Chapter 1 ending with a key sentence
    ch1_blocks = [
        IRBlock(
            id=f"ch01#p{i}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"Sentence {i} of chapter one.",
        )
        for i in range(1, 6)
    ]
    ch1_blocks[
        -1
    ].source_text = "Bilbo Baggins sat down to enjoy his second breakfast in the Shire."
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id="doc_cursor",
            chapter_id="ch01",
            title="An Unexpected Party",
            spine_index=1,
            blocks=ch1_blocks,
        ),
    )

    # Append Chapter 2 starting at spine_index 6
    ch2_blocks = [
        IRBlock(
            id=f"ch02#p{i}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=5 + i,
            source_text=f"Sentence {i} of chapter two.",
        )
        for i in range(1, 6)
    ]
    ch2_blocks[0].source_text = "He wondered if Gandalf would return before afternoon tea."
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id="doc_cursor",
            chapter_id="ch02",
            title="Roast Mutton",
            spine_index=2,
            blocks=ch2_blocks,
        ),
    )

    # 1. Test paginated keyset cursor consumption: fetch batch of 4 blocks
    batch1 = ledger.fetch_pending_blocks(job_id, limit=4)
    assert len(batch1) == 4
    assert batch1[0].id == "ch01#p1"
    assert batch1[3].id == "ch01#p4"

    batch2 = ledger.fetch_pending_blocks(job_id, limit=4, after_spine_index=batch1[-1].spine_index)
    assert len(batch2) == 4
    assert batch2[0].id == "ch01#p5"
    assert batch2[1].id == "ch02#p1"  # Seamless transition to Chapter 2

    # 2. Test cross-chapter context bridge:
    # First block of Chapter 2 (spine_index = 6) looks back into Chapter 1
    preceding_context = ledger.get_preceding_text_tail(
        job_id=job_id,
        flow_id=FlowID.MAIN_STORY,
        before_spine_index=6,
        max_chars=150,
    )
    assert "Bilbo Baggins sat down to enjoy his second breakfast" in preceding_context


def test_preceding_text_tail_resolves_doc_id_alias(tmp_path: Path) -> None:
    """``get_preceding_text_tail`` must resolve a doc_id alias like every other read.

    The ledger keys rows by the real job_id but also indexes ``job_meta`` by
    ``doc_id``; a caller passing the doc_id used to hit zero rows silently
    (empty context) instead of the latest job's prose — the one read method
    that skipped ``_resolve_actual_job_id``. Cross-job contamination risk and
    a silent prompt-quality bug in one.
    """
    ledger = SQLiteJobLedger(tmp_path / "alias_tail.db")
    job_id = "job_alias_tail"
    manifest = BookManifest(doc_id="doc_alias_tail", title="Alias Book", source_path="x.epub")
    ledger.init_job_from_manifest(job_id, manifest)
    blocks = [
        IRBlock(
            id="ch01#b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text="The lighthouse keeper lit the lamp.",
            target_text="守灯塔人点亮了灯。",
        ),
        IRBlock(
            id="ch01#b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            source_text="Nobody noticed.",
        ),
    ]
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id="doc_alias_tail", chapter_id="ch01", title="c", spine_index=1, blocks=blocks
        ),
    )

    # Exact job_id still works…
    direct = ledger.get_preceding_text_tail(
        job_id=job_id, flow_id=FlowID.MAIN_STORY, before_spine_index=2
    )
    assert "守灯塔人点亮了灯" in direct

    # …and the doc_id alias now resolves to the same latest job's prose.
    via_alias = ledger.get_preceding_text_tail(
        job_id="doc_alias_tail", flow_id=FlowID.MAIN_STORY, before_spine_index=2
    )
    assert via_alias == direct
    assert "守灯塔人点亮了灯" in via_alias
    ledger.close()
