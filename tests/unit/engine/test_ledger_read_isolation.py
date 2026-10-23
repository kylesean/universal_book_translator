"""Reads must not queue behind the writer's lock.

Every read used to share the writer's single connection and its process-wide
``RLock``, so a full-table scan (346 ms on a 20k-block book, 88 % of it
``IRBlock`` construction) stalled every other reader. The progress event's
``get_job_stats`` — 666 of them on a 20k-block run at the default
``batch_limit`` — went from 3 ms uncontended to 384 ms p95 under that loop, and
the chapter-streaming pipeline runs a draft worker and a QE worker at once.

SQLite's WAL mode already allows readers concurrent with the single writer, so
the ledger gives reads a connection of their own (``_get_read_conn``) and
converts results after releasing it. What is pinned here:

* a read sees committed data and is not blocked by an in-flight writer;
* the writer still serialises (one connection, ``BEGIN IMMEDIATE``);
* reads keep working after a thread's connection is opened, and ``close``
  releases them;
* a ``read_only`` ledger keeps its historical single-connection path.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import LedgerError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast


def _block(index: int) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=f"ch001#b{index:04d}",
            spine_index=index,
            block_type=BlockType.NARRATIVE,
            source_text=f"Paragraph {index} of the fixture book.",
        ),
        status=BlockStatus.PENDING,
    )


def _seeded(tmp_path: Path, count: int = 40) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job_from_manifest("job", BookManifest(doc_id="doc", title="t", source_path="s.pdf"))
    ledger.append_chapter(
        "job",
        ChapterIR(
            doc_id="doc",
            chapter_id="ch001",
            title="C1",
            spine_index=0,
            blocks=[_block(i) for i in range(count)],
        ),
    )
    return ledger


def test_a_read_connection_is_separate_from_the_writer(tmp_path: Path) -> None:
    ledger = _seeded(tmp_path)
    try:
        with ledger._get_conn() as writer:
            writer_id = id(writer)
        with ledger._get_read_conn() as reader:
            assert id(reader) != writer_id, "reads must not reuse the writer connection"
            assert reader.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 40
        # Cached per thread, so a second read does not pay for another open.
        with ledger._get_read_conn() as again:
            assert id(again) == id(reader)
    finally:
        ledger.close()


def test_reads_see_committed_writes(tmp_path: Path) -> None:
    ledger = _seeded(tmp_path)
    try:
        assert ledger.get_job_stats("job")["drafted"] == 0
        ledger.save_checkpoints_batch(
            [
                {
                    "block_id": "ch001#b0000",
                    "target_text": "译文",
                    "status": BlockStatus.DRAFTED,
                }
            ]
        )
        stats = ledger.get_job_stats("job")
        assert stats["drafted"] == 1
        assert stats["total"] == 40
        assert len(ledger.get_all_blocks("job")) == 40
        assert len(ledger.fetch_source_texts("job")) == 40
    finally:
        ledger.close()


def test_a_progress_read_is_not_blocked_by_an_in_flight_scan(tmp_path: Path) -> None:
    """The regression this exists for: 384 ms progress events behind one scan.

    A scan loop holds a read connection busy; progress queries on another thread
    must stay fast. The threshold is deliberately loose (a scan of this fixture
    is sub-millisecond) — it is the *shape* that is pinned: if progress queries
    are once again serialised behind the scan, the median jumps to the scan's
    own duration plus the lock wait, which is orders of magnitude larger.
    """
    ledger = _seeded(tmp_path, count=400)
    stop = threading.Event()
    latencies: list[float] = []
    try:

        def scan_loop() -> None:
            while not stop.is_set():
                ledger.get_all_blocks("job")

        def progress_loop() -> None:
            while not stop.is_set():
                started = time.perf_counter()
                ledger.get_job_stats("job")
                latencies.append(time.perf_counter() - started)
                time.sleep(0.001)

        scanner = threading.Thread(target=scan_loop, daemon=True)
        prober = threading.Thread(target=progress_loop, daemon=True)
        scanner.start()
        prober.start()
        time.sleep(1.2)
        stop.set()
        scanner.join(timeout=5)
        prober.join(timeout=5)

        assert latencies, "the progress loop never ran"
        latencies.sort()
        median = latencies[len(latencies) // 2]
        # Contended-before was ~375 ms; isolated reads are single-digit ms.
        assert median < 0.05, (
            f"progress queries stalled behind the scan (median {median * 1e3:.1f} ms); "
            "reads are sharing the writer's lock again"
        )
    finally:
        stop.set()
        ledger.close()


def test_the_writer_still_serialises_its_own_writes(tmp_path: Path) -> None:
    """Read isolation must not weaken the single-writer guarantee."""
    ledger = _seeded(tmp_path, count=5)
    try:
        ledger.save_checkpoints_batch(
            [
                {
                    "block_id": f"ch001#b{i:04d}",
                    "target_text": "x",
                    "status": BlockStatus.DRAFTED,
                }
                for i in range(5)
            ]
        )
        assert ledger.get_job_stats("job")["drafted"] == 5
        # A batch write is one transaction: each row is applied exactly once and
        # the revision counter advances once per commit (see BlockReader).
        before = ledger.blocks_seq
        ledger.save_checkpoints_batch(
            [{"block_id": "ch001#b0000", "target_text": "y", "status": BlockStatus.REPAIRED}]
        )
        assert ledger.blocks_seq == before + 1
    finally:
        ledger.close()


def test_close_releases_the_read_connections(tmp_path: Path) -> None:
    ledger = _seeded(tmp_path)
    ledger.get_all_blocks("job")  # opens this thread's read connection
    assert ledger._read_conns
    ledger.close()
    assert not ledger._read_conns
    with pytest.raises(LedgerError):
        ledger.get_all_blocks("job")


def test_a_read_only_ledger_keeps_its_single_connection_path(tmp_path: Path) -> None:
    writer = _seeded(tmp_path)
    writer.close()
    reader = SQLiteJobLedger(tmp_path / "job.sqlite", read_only=True)
    try:
        assert len(reader.get_all_blocks("job")) == 40
        assert reader.get_job_stats("job")["total"] == 40
        # Nothing was opened on the side: the only connection is already a reader.
        assert not reader._read_conns
    finally:
        reader.close()
