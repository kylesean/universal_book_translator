"""Unit-tier long-form stress tests: sliding epoch memory, process pool cleanup, and state invariants.

This module validates that:
1. A 45-chapter book (>40 chapter cliff) runs end-to-end with sliding epoch memory
   (memory_sliding_window=True, memory_max_epochs=3), constraining memory growth
   and keeping RSS memory strictly bounded.
2. Companion artifacts (XLIFF, HTML, EPUB, Attestation) comply with typed contracts,
   and non-PDF formats gracefully skip PDF-specific attestation checks.
3. DoclingWorkerPool terminates isolated child worker processes upon close without
   leaving orphan/zombie processes behind.
4. Concurrent jobs execute without ledger locks or state transition violations.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import psutil
import pytest

from ubt.adapters.pdf.docling_worker import close_docling_worker_pool, get_docling_worker_pool
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator, rehearsal_db_dir
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import TERMINAL_STATUSES

pytestmark = pytest.mark.fast


def _build_45_chapter_markdown(dest: Path) -> Path:
    """Build a synthetic 45-chapter markdown manuscript to cross the memory cliff."""
    lines: list[str] = []
    for i in range(1, 46):
        lines.append(f"# Chapter {i}: The Principles of Analysis {i}")
        lines.append("")
        lines.append(
            f"In this discourse {i}, Archimedes and Gauss examine Riemannian manifolds of dimension {i}. "
            f"The continuous geometry demonstrates canonical properties across all boundary domains."
        )
        lines.append("")
        lines.append(
            f"Following theorem {i}, the geodesic curves maintain stability regardless of perturbation."
        )
        lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


async def test_45_chapter_sliding_epoch_stress_markdown(tmp_path: Path) -> None:
    """Validate 45-chapter long-form run with sliding epoch memory and companion exports."""
    book_md = _build_45_chapter_markdown(tmp_path / "book_45.md")
    output_md = tmp_path / "book_45_zh.md"
    job_id = "longform-stress-45"

    cfg = UBTConfig(
        db_dir=tmp_path / "ledgers",
        enable_rolling_summary=True,
        memory_sliding_window=True,
        memory_max_epochs=3,
        step_chars=250,
        max_concurrency=4,
        emit_xliff_companion=True,
        emit_html_companion=True,
        emit_epub_companion=True,
        emit_attestation_shadow=True,
    )

    proc = psutil.Process()
    rss_start = proc.memory_info().rss

    orchestrator = create_dry_run_orchestrator(cfg)
    events = []
    async for event in orchestrator.run(book_md, output_md, target_lang="zh", job_id=job_id):
        events.append(event)

    rss_end = proc.memory_info().rss
    rss_growth_mb = (rss_end - rss_start) / (1024 * 1024)

    # 1. Pipeline reached clean export completion
    assert len(events) > 0
    assert events[0].event_type is EventType.JOB_STARTED
    assert events[-1].event_type is EventType.EXPORT_COMPLETED
    assert not any(e.event_type is EventType.PIPELINE_FAILED for e in events)

    # 2. Delivered manuscript integrity
    assert output_md.exists()
    delivered_text = output_md.read_text(encoding="utf-8")
    for i in (1, 15, 30, 45):
        assert f"Chapter {i}: The Principles of Analysis {i}" in delivered_text

    # 3. RSS memory growth is strictly bounded (< 150 MB delta for 45 chapters)
    assert rss_growth_mb < 150.0, f"Excessive RSS growth: {rss_growth_mb:.2f} MB"

    # 4. Ledger state machine invariants and block counts
    mock_db = rehearsal_db_dir(tmp_path / "ledgers") / f"{job_id}.sqlite"
    ledger = SQLiteJobLedger(mock_db)
    try:
        assert ledger.get_job_status(job_id) == "completed"
        blocks = ledger.get_all_blocks(job_id)
        assert len(blocks) >= 45, f"Expected at least 45 blocks, got {len(blocks)}"
        assert all(b.status in TERMINAL_STATUSES for b in blocks)
        failed_blocks = [b.id for b in blocks if b.status.value in ("failed", "needs_human")]
        assert failed_blocks == []

        # 5. Companion artifact contract validation in ledger metadata
        meta = ledger.get_job_metadata(job_id)
        companion_summary = meta.get("companion_artifacts")
        assert companion_summary is not None
        results = {r["kind"]: r for r in companion_summary["results"]}

        # XLIFF, HTML, EPUB generated with non-zero bytes and durations
        for kind in ("xliff", "html", "epub"):
            assert kind in results
            assert results[kind]["status"] == "generated"
            assert results[kind]["size_bytes"] is not None and results[kind]["size_bytes"] > 0
            assert results[kind]["duration_ms"] is not None and results[kind]["duration_ms"] >= 0

        # Attestation gracefully skipped for non-PDF output
        assert "attestation" in results
        assert results["attestation"]["status"] == "skipped"
        assert "only applies to PDF" in (results["attestation"]["reason"] or "")
    finally:
        ledger.close()


def test_docling_worker_pool_process_isolation_and_cleanup() -> None:
    """Verify that DoclingWorkerPool spawns worker in a separate process and terminates it on close."""
    pool = get_docling_worker_pool()
    executor = pool._ensure_executor()

    # Submit a minimal lambda task to ensure the worker process is initialized
    future = executor.submit(os.getpid)
    worker_pid = future.result(timeout=5.0)

    # Worker process must have a distinct PID from main process
    current_pid = os.getpid()
    assert worker_pid != current_pid, "Worker process must run in an isolated subprocess"

    # Worker must be visible in child processes
    proc = psutil.Process()
    child_pids = {c.pid for c in proc.children(recursive=True)}
    assert worker_pid in child_pids, "Spawned worker PID not found among child processes"

    # Close worker pool and verify clean termination
    close_docling_worker_pool()
    time.sleep(0.1)

    alive_child_pids = {c.pid for c in proc.children(recursive=True)}
    assert worker_pid not in alive_child_pids, "Worker process was not terminated after close()"


async def test_concurrent_multi_job_stress_and_state_machine_invariants(tmp_path: Path) -> None:
    """Verify concurrent multi-job execution without ledger locking or state machine errors."""
    jobs = ["concurrent-stress-1", "concurrent-stress-2", "concurrent-stress-3"]
    tasks = []

    for jid in jobs:
        manuscript = _build_45_chapter_markdown(tmp_path / f"book_{jid}.md")
        out = tmp_path / f"out_{jid}.md"
        cfg = UBTConfig(
            db_dir=tmp_path / "ledgers",
            enable_rolling_summary=True,
            memory_sliding_window=True,
            memory_max_epochs=2,
            step_chars=200,
            max_concurrency=4,
        )
        orchestrator = create_dry_run_orchestrator(cfg)

        async def _run_job(
            orch: object, src: Path, dest: Path, job_name: str
        ) -> list[TranslationProgressEvent]:
            results: list[TranslationProgressEvent] = []
            async for ev in orch.run(src, dest, target_lang="zh", job_id=job_name):  # type: ignore[attr-defined]
                results.append(ev)
            return results

        tasks.append(_run_job(orchestrator, manuscript, out, jid))

    # Run all 3 multi-chapter jobs concurrently
    all_results = await asyncio.gather(*tasks)

    for jid, events in zip(jobs, all_results, strict=True):
        assert len(events) > 0
        assert events[-1].event_type is EventType.EXPORT_COMPLETED

        mock_db = rehearsal_db_dir(tmp_path / "ledgers") / f"{jid}.sqlite"
        ledger = SQLiteJobLedger(mock_db)
        try:
            assert ledger.get_job_status(jid) == "completed"
            blocks = ledger.get_all_blocks(jid)
            assert len(blocks) >= 45
            assert all(b.status in TERMINAL_STATUSES for b in blocks)
        finally:
            ledger.close()
