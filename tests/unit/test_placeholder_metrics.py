"""Unit tests for placeholder retention metrics (math/code spans)."""

from pathlib import Path

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.reporter import (
    build_quality_report,
    compute_placeholder_metrics,
    render_kdp_audit_markdown,
)
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)


def _math_block(block_id: str, source: str, corrupt: bool = False) -> IRBlock:
    return IRBlock(
        id=block_id,
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        target_text="译文",
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=0.95,
        repair_rounds=0,
        error_flags=(
            # Token indices are 1-based (maskers count from 1); [0] is unproducible.
            ["math_token_corrupt missing=[1] mismatched=[] mutated=[]"] if corrupt else []
        ),
    )


def test_full_retention_clean_math_book() -> None:
    blocks = [
        _math_block("b1", "Energy is $E = mc^2$ here."),
        _math_block("b2", "Also $a^2 + b^2 = c^2$ there."),
    ]
    metrics = compute_placeholder_metrics(blocks)
    assert metrics.masked_spans == 2
    assert metrics.corrupt_spans == 0
    assert metrics.retention_rate == 1.0
    assert metrics.masked_blocks == 2
    assert metrics.corrupt_blocks == 0


def test_corrupt_flag_lowers_retention() -> None:
    blocks = [
        _math_block("b1", "Energy is $E = mc^2$ here.", corrupt=True),
        _math_block("b2", "Also $a^2 + b^2 = c^2$ there."),
    ]
    metrics = compute_placeholder_metrics(blocks)
    assert metrics.masked_spans == 2
    assert metrics.corrupt_spans == 1
    assert metrics.retention_rate == 0.5
    assert metrics.corrupt_blocks == 1


def test_no_math_book_not_applicable(tmp_path: Path) -> None:
    blocks = [_math_block("b1", "Plain prose, no formulas.")]
    metrics = compute_placeholder_metrics(blocks)
    assert metrics.masked_spans == 0
    assert metrics.retention_rate == 1.0

    db_path = tmp_path / "test_placeholder.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_placeholder_001"
    manifest = BookManifest(
        doc_id="doc_placeholder",
        title="Placeholder Test Book",
        source_path="source.md",
        chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
    )
    ledger.init_job_from_manifest(job_id, manifest)
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id="ch01",
            title="Chapter 1",
            spine_index=1,
            blocks=blocks,
        ),
    )
    for b in blocks:
        ledger.save_checkpoint(
            block_id=b.id,
            target_text=b.target_text or "",
            status=b.status,
            mtqe_score=b.mtqe_score,
            repair_rounds=b.repair_rounds,
            error_flags=b.error_flags,
        )
    report = build_quality_report(ledger, job_id, manifest, tmp_path / "r.md")
    assert report.placeholder.masked_spans == 0
    markdown = render_kdp_audit_markdown(report)
    assert "not applicable" in markdown


def test_code_and_citation_corruption_counts_toward_retention() -> None:
    """Retention must count code/cite corruption, not only math.

    ``compute_placeholder_metrics`` took ``len(math_map)`` and parsed only
    ``math_token_corrupt``, so code/citation corruption (MQM-Critical) left
    retention at 1.0.
    """
    block = IRBlock(
        id="b_code",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="```python\nx = 1\n```\nSee [1] and $y$.",
        target_text="译文",
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=0.95,
        repair_rounds=0,
        error_flags=["code_token_corrupt missing=[1] mismatched=[] mutated=[]"],
    )
    metrics = compute_placeholder_metrics([block])
    assert metrics.masked_spans >= 2
    assert metrics.corrupt_spans >= 1
    assert metrics.retention_rate < 1.0


def test_placeholder_metrics_count_soup_spans() -> None:
    """Soup spans are part of the draft mask order; the recount must match.

    The reporter's recount skipped the soup masker entirely, so delimiter-free
    unicode math (F_th,SI / β2) did not count toward retention and a corrupt
    soup restore read as a perfect 1.0.
    """
    block = _math_block("b1", "由F_th,SI得到β2项 [2]")
    metrics = compute_placeholder_metrics([block])
    assert metrics.masked_spans >= 2
