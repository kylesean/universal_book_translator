"""Unit tests for the human PE queue exporters: CSV + XLIFF 2.1."""
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pe_queue import (
    CSV_COLUMNS,
    export_pe_queue,
    export_pe_queue_csv,
    export_pe_queue_xliff,
    select_pe_queue_blocks,
)
from ubt.core.engine.reporter import build_quality_report
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterMeta,
    FlowID,
    IRBlock,
)

_XLIFF_NS = "urn:oasis:names:tc:xliff:document:2.1"


def _queue_block(
    block_id: str,
    status: BlockStatus,
    *,
    severity: str | None = "major",
    spine_index: int = 1,
    score: float | None = 0.55,
    spans: list[dict[str, Any]] | None = None,
) -> IRBlock:
    return IRBlock(
        id=block_id,
        flow_id=FlowID.MAIN_STORY,
        spine_index=spine_index,
        source_text=f"Source text for {block_id}.",
        draft_text=f"Draft for {block_id}.",
        target_text=f"Target for {block_id}.",
        status=status,
        mtqe_score=score,
        mqm_severity=severity,
        mqm_spans=spans or [],
    )


def _manifest() -> BookManifest:
    return BookManifest(
        doc_id="test_doc_sha256",
        title="Test Book",
        source_path="/tmp/test_book.epub",
        source_lang="en",
        target_lang="zh",
        chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Selection + CSV
# ---------------------------------------------------------------------------


def test_select_pe_queue_blocks_filters_and_orders() -> None:
    blocks = [
        _queue_block("b-pass", BlockStatus.MTQE_PASSED, spine_index=0),
        _queue_block("b-blocked", BlockStatus.BLOCKED_HUMAN, spine_index=1),
        _queue_block("b-needs", BlockStatus.NEEDS_HUMAN, spine_index=2),
        _queue_block("b-failed", BlockStatus.FAILED, spine_index=3),
    ]
    selected = select_pe_queue_blocks(blocks)
    assert [b.id for b in selected] == ["b-blocked", "b-needs"]


def test_export_pe_queue_csv_round_trip(tmp_path: Path) -> None:
    spans = [
        {
            "id": "1",
            "error_type": "numeric",
            "reason": "Mismatched number '84', expected '42'",
            "expected": "42",
            "start_pos": 0,
            "end_pos": 2,
            "erroneous_text": "84",
            "severity": "critical",
        }
    ]
    blocks = [
        _queue_block(
            "ch01#b001",
            BlockStatus.BLOCKED_HUMAN,
            severity="critical",
            spine_index=1,
            score=0.42,
            spans=spans,
        ),
        _queue_block("ch01#b002", BlockStatus.NEEDS_HUMAN, spine_index=2, score=0.55),
    ]
    path = tmp_path / "book_pe_queue.csv"
    result = export_pe_queue_csv(blocks, path, job_id="job_pe")

    assert result.segment_count == 2
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(CSV_COLUMNS)
    first = rows[0]
    assert first["block_id"] == "ch01#b001"
    assert first["job_id"] == "job_pe"  # re-import binding
    assert first["status"] == "blocked_human"
    assert first["mqm_severity"] == "critical"
    assert first["source_text"] == "Source text for ch01#b001."
    assert first["draft_text"] == "Draft for ch01#b001."
    assert json.loads(first["mqm_spans"])[0]["expected"] == "42"
    assert "84' -> '42" in first["suggested_correction"]
    # The human-editable column ships empty.
    assert first["revised_translation"] == ""


# ---------------------------------------------------------------------------
# XLIFF 2.1
# ---------------------------------------------------------------------------


def test_export_pe_queue_xliff_structure_round_trip(tmp_path: Path) -> None:
    """Structural validation (ElementTree round-trip): namespace, units, states."""
    blocks = [
        _queue_block(
            "ch01#b001",
            BlockStatus.BLOCKED_HUMAN,
            severity="critical",
            spine_index=1,
            score=0.42,
        ),
        _queue_block("ch01#b002", BlockStatus.NEEDS_HUMAN, severity="major", spine_index=2),
    ]
    path = tmp_path / "book_pe_queue.xliff"
    result = export_pe_queue_xliff(
        blocks,
        path,
        source_lang="en",
        target_lang="zh",
        job_id="job_pe",
        original_name="book_bilingual.md",
    )
    assert result.segment_count == 2

    # Round-trip: the file must parse as valid XML with the XLIFF 2.1 namespace.
    root = ET.parse(path).getroot()
    assert root.tag == f"{{{_XLIFF_NS}}}xliff"
    assert root.get("version") == "2.1"
    assert root.get("srcLang") == "en"
    assert root.get("trgLang") == "zh"

    file_el = root.find(f"{{{_XLIFF_NS}}}file")
    assert file_el is not None
    assert file_el.get("original") == "book_bilingual.md"

    units = root.findall(f".//{{{_XLIFF_NS}}}unit")
    assert [u.get("id") for u in units] == ["ch01#b001", "ch01#b002"]

    segments = root.findall(f".//{{{_XLIFF_NS}}}segment")
    assert segments[0].get("state") == "initial"  # Critical quarantine
    assert segments[1].get("state") == "translated"  # Major review

    sources = root.findall(f".//{{{_XLIFF_NS}}}source")
    targets = root.findall(f".//{{{_XLIFF_NS}}}target")
    assert sources[0].text == "Source text for ch01#b001."
    assert targets[0].text == "Target for ch01#b001."

    notes = {
        n.get("category"): n.text
        for n in units[0].findall(f"{{{_XLIFF_NS}}}notes/{{{_XLIFF_NS}}}note")
    }
    assert notes["ubt-job-id"] == "job_pe"  # re-import binding
    assert notes["ubt-status"] == "blocked_human"
    assert notes["ubt-mqm-severity"] == "critical"
    assert notes["ubt-qe-score"] == "0.4200"


# ---------------------------------------------------------------------------
# Dispatcher + report/stat surfacing
# ---------------------------------------------------------------------------


def test_export_pe_queue_dispatcher(tmp_path: Path) -> None:
    blocks = [
        _queue_block("b1", BlockStatus.NEEDS_HUMAN, spine_index=1),
        _queue_block("b2", BlockStatus.BLOCKED_HUMAN, spine_index=2),
    ]

    # none -> disabled
    assert export_pe_queue(blocks, tmp_path / "out.md", "none", "en", "zh", "job_pe") is None
    # empty queue -> nothing to do
    assert (
        export_pe_queue(
            [_queue_block("b", BlockStatus.REPAIRED)],
            tmp_path / "out.md",
            "csv",
            "en",
            "zh",
            "job_pe",
        )
        is None
    )
    # unknown format falls back to csv
    res = export_pe_queue(blocks, tmp_path / "out.md", "parquet", "en", "zh", "job_pe")
    assert res is not None and res.fmt == "csv"
    assert res.path.name == "out_pe_queue.csv"
    assert res.path.exists()
    # xliff naming
    res = export_pe_queue(blocks, tmp_path / "out.md", "xliff", "en", "zh", "job_pe")
    assert res is not None and res.fmt == "xliff"
    assert res.path.name == "out_pe_queue.xliff"
    assert res.path.exists()


def test_job_stats_and_report_surface_human_queue_counts(tmp_path: Path) -> None:
    blocks = [
        _queue_block("b1", BlockStatus.MTQE_PASSED, spine_index=1),
        _queue_block("b2", BlockStatus.NEEDS_HUMAN, spine_index=2),
        _queue_block("b3", BlockStatus.BLOCKED_HUMAN, severity="critical", spine_index=3),
    ]
    doc_ir = SeedDoc(
        doc_id="test_doc_sha256",
        source_path="/tmp/test_book.epub",
        format_type="epub",
        metadata={"title": "Test Book"},
        blocks=blocks,
    )
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_pe", doc_ir, target_lang="zh")
    ledger.save_checkpoints_batch(
        [
            {"block_id": "b1", "status": BlockStatus.MTQE_PASSED, "mtqe_score": 0.9},
            {"block_id": "b2", "status": BlockStatus.NEEDS_HUMAN, "mtqe_score": 0.55},
            {"block_id": "b3", "status": BlockStatus.BLOCKED_HUMAN, "mtqe_score": 0.42},
        ]
    )

    stats = ledger.get_job_stats("job_pe")
    assert stats["needs_human"] == 1
    assert stats["blocked_human"] == 1
    assert stats["completed"] == 1  # human-queue states do not count as shipped

    report = build_quality_report(
        ledger=ledger,
        job_id="job_pe",
        manifest=_manifest(),
        output_path=tmp_path / "out.md",
        token_cost_usd=None,
    )
    assert report.summary.needs_human_blocks == 1
    assert report.summary.blocked_human_blocks == 1
    ledger.close()
