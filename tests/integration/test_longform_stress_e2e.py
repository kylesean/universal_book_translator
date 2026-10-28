"""End-to-end long-form PDF stress and memory benchmark: 45-page synthetic book.

This module validates that:
1. A 45-page synthetic PDF book runs end-to-end through PDFium Ingest,
   Sliding Epoch Memory, MTQE, Consistency, and Typst compilation.
2. Typst renderer produces valid PDF output (%PDF-) and delivers all chapters.
3. Companion artifacts (XLIFF, HTML, EPUB, Attestation) are 100% generated
   with non-zero sizes and valid durations.
4. Process RSS memory remains strictly bounded without runaway leaks across
   multi-page Typst compiles and long-form translation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import psutil
import pytest
from pdf_builders import write_text_pdf

import ubt.adapters.pdf.oxide_render as oxide_render
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator, rehearsal_db_dir
from ubt.core.engine.events import EventType
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import TERMINAL_STATUSES
from ubt.core.job_options import companion_path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the typst compiler is not installed; the render path cannot run",
    ),
]


def _build_45_page_pdf(path: Path) -> Path:
    """Build a synthetic 45-page PDF book where each page is an independent chapter."""
    pages = []
    for i in range(1, 46):
        title = f"Chapter {i}: Foundations of Geometry {i}"
        prose = (
            f"In chapter {i}, mathematician Gauss and Archimedes formalize the curvature tensor. "
            f"The continuous manifold exhibits Riemannian metric characteristics across section {i}. "
            f"Consequently, geodesic deviation equation converges rapidly."
        )
        pages.append((title, prose))
    return write_text_pdf(path, pages)


async def test_45_page_pdf_e2e_stress_and_companion_attestation(tmp_path: Path) -> None:
    """Run full 45-page synthetic PDF book through the entire pipeline and verify contracts."""
    book_pdf = _build_45_page_pdf(tmp_path / "book_45.pdf")
    output_pdf = tmp_path / "book_45_es.pdf"
    job_id = "longform-pdf-stress-45"

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
        pdf_engine="pdfium",
    )

    proc = psutil.Process()
    rss_start = proc.memory_info().rss

    orchestrator = create_dry_run_orchestrator(cfg)
    events = []
    async for event in orchestrator.run(book_pdf, output_pdf, target_lang="es", job_id=job_id):
        events.append(event)

    rss_end = proc.memory_info().rss
    rss_growth_mb = (rss_end - rss_start) / (1024 * 1024)

    # 1. Pipeline reaches export completion
    assert len(events) > 0
    assert events[0].event_type is EventType.JOB_STARTED
    assert events[-1].event_type is EventType.EXPORT_COMPLETED
    assert not any(e.event_type is EventType.PIPELINE_FAILED for e in events)

    # 2. Delivered PDF artifact validity
    assert output_pdf.exists()
    assert output_pdf.read_bytes()[:5] == b"%PDF-", "Delivered artifact is not a valid PDF"

    # 3. Delivered text verification across sample pages
    delivered = " ".join(oxide_render.extract_page_texts(output_pdf))
    for i in (1, 10, 25, 45):
        assert f"In chapter {i}, mathematician Gauss and Archimedes" in delivered
        assert f"[模拟翻译] In chapter {i}" in delivered

    # 4. Memory stability (< 250 MB RSS growth across 45-page Typst render)
    assert rss_growth_mb < 250.0, f"Excessive RSS growth during PDF run: {rss_growth_mb:.2f} MB"

    # 5. Ledger invariants: all 45 blocks terminal
    mock_db = rehearsal_db_dir(tmp_path / "ledgers") / f"{job_id}.sqlite"
    ledger = SQLiteJobLedger(mock_db)
    try:
        assert ledger.get_job_status(job_id) == "completed"
        blocks = ledger.get_all_blocks(job_id)
        assert len(blocks) >= 45
        assert all(b.status in TERMINAL_STATUSES for b in blocks)
        failed_blocks = [b.id for b in blocks if b.status.value in ("failed", "needs_human")]
        assert failed_blocks == []

        # 6. Companion artifacts: all 4 kinds (including attestation) generated
        meta = ledger.get_job_metadata(job_id)
        companion_summary = meta.get("companion_artifacts")
        assert companion_summary is not None
        results = {r["kind"]: r for r in companion_summary["results"]}

        for kind in ("xliff", "html", "epub", "attestation"):
            assert kind in results
            assert results[kind]["status"] == "generated", f"{kind} failed: {results[kind]}"
            assert results[kind]["size_bytes"] is not None and results[kind]["size_bytes"] > 0
            assert results[kind]["duration_ms"] is not None and results[kind]["duration_ms"] >= 0

        # Physical attestation shadow file exists
        shadow = companion_path(output_pdf, "_attestations.json")
        assert shadow.exists()
    finally:
        ledger.close()
