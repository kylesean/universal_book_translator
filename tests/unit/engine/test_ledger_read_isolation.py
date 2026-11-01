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
        assert median < 0.05, (
            f"progress queries stalled behind the scan (median {median * 1e3:.1f} ms); "
            "reads are sharing the writer's lock again"
        )
    finally:
        stop.set()
        ledger.close()


def test_the_writer_still_serialises_its_own_writes(tmp_path: Path) -> None:
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
        before = ledger.blocks_seq
        ledger.save_checkpoints_batch(
            [{"block_id": "ch001#b0000", "target_text": "y", "status": BlockStatus.REPAIRED}]
        )
        assert ledger.blocks_seq == before + 1
    finally:
        ledger.close()


def test_close_releases_the_read_connections(tmp_path: Path) -> None:
    ledger = _seeded(tmp_path)
    ledger.get_all_blocks("job")
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
        assert not reader._read_conns
    finally:
        reader.close()
