"""Unit tests for Gate 4: FastPass math-span hard gate + verified Typst emission.

Closed-loop property under test: inline math masked before drafting must be
restored verbatim after unmasking. A dropped ⟦MATH_MASK_*⟧ token surfaces as
a span multiset mismatch → REPAIR_PENDING (fatal: QE score cannot auto-pass
it) → repair re-masks from source (self-healing).
"""
import asyncio
from pathlib import Path

from tests.stage_ctx_factory import build_stage_ctx, drain, inert_event
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.adapters.pdf.typst_reconstructor import (
    _balanced_delimiters,
    _emit_formula_math,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.core.qe.fast_pass import FastPassFilter


def _fp() -> FastPassFilter:
    return FastPassFilter(source_lang="en", target_lang="zh")


def test_math_spans_intact_pass() -> None:
    d = _fp().evaluate("When $T$ grows, cache $M_{KV}$ too.", "当 $T$ 增长时,缓存 $M_{KV}$。")
    assert d.passed, d.reason


def test_dropped_math_token_fails_fatal() -> None:
    d = _fp().evaluate("When $T$ grows fast.", "当增长很快。")
    assert not d.passed
    assert "Math span mismatch" in d.reason


def test_currency_never_trips_math_gate() -> None:
    d = _fp().evaluate("It costs $5 today.", "今天花费 $5。")
    assert d.passed, d.reason


def test_no_math_source_unaffected() -> None:
    d = _fp().evaluate("The cache stores states.", "缓存存储状态。")
    assert d.passed, d.reason


def test_balanced_delimiters() -> None:
    assert _balanced_delimiters("frac(q _(t), sqrt(d))")
    assert not _balanced_delimiters("frac(q _(t), sqrt(d)")
    assert not _balanced_delimiters("frac)q(")
    assert _balanced_delimiters("no delimiters at all")


def test_emit_formula_math_clean_passthrough() -> None:
    line = _emit_formula_math("K _ { 1 } = [ k _ { t } ]", "b1")
    assert line.startswith("$") and line.endswith("$")
    assert "K" in line


def test_emit_formula_math_unbalanced_falls_back_verbatim() -> None:
    line = _emit_formula_math("frac(a, (b)", "b2")
    assert line.startswith("`") and "frac(a, (b)" in line


def test_emit_formula_math_empty_falls_back_loud() -> None:
    line = _emit_formula_math("   ", "b3")
    assert "b3" in line  # block id preserved for traceability


def _make_doc() -> SeedDoc:
    return SeedDoc(
        doc_id="gate4",
        source_path="/tmp/gate4.epub",
        format_type="epub",
        metadata={},
        blocks=[
            IRBlock(
                id="ch01#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                source_text="When $T$ grows, cache grows.",
            ),
            IRBlock(
                id="ch01#b002",
                flow_id=FlowID.MAIN_STORY,
                spine_index=2,
                source_text="Plain prose without math.",
            ),
        ],
    )


def test_quality_gate_math_mismatch_is_repair_pending(tmp_path: Path) -> None:
    """End of the closed loop: dropped math cannot QE-pass into release."""
    ledger = SQLiteJobLedger(tmp_path / "gate4.sqlite")
    seed_job(ledger, "job_gate4", _make_doc(), target_lang="zh")
    manifest = BookManifest(
        doc_id="gate4",
        title="Gate4",
        source_path="/tmp/gate4.epub",
        chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
        metadata={},
    )
    assert manifest is not None
    # b001 dropped its math token; b002 is a clean translation.
    ledger.save_checkpoint(
        block_id="ch01#b001",
        target_text="当增长时,缓存增长。",
        draft_text="当增长时,缓存增长。",
        status=BlockStatus.DRAFTED,
    )
    ledger.save_checkpoint(
        block_id="ch01#b002",
        target_text="没有数学的普通散文。",
        draft_text="没有数学的普通散文。",
        status=BlockStatus.DRAFTED,
    )
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_gate4",
        fast_pass=_fp(),
        qe_runner=HeuristicQERunner(),
        create_event=inert_event,
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))
    by_id = {b.id: b for b in ledger.get_all_blocks("job_gate4")}
    assert by_id["ch01#b001"].status is BlockStatus.REPAIR_PENDING
    assert any("Math span mismatch" in f for f in by_id["ch01#b001"].error_flags)
    assert by_id["ch01#b002"].status is BlockStatus.MTQE_PASSED


def test_quality_gate_rejects_mismatched_score_count(tmp_path: Path) -> None:
    """Regression: a runner violating the one-score-per-pair contract used to
    be silently truncated by ``zip(strict=False)``; it must fail loudly."""
    import pytest

    from ubt.core.exceptions import MTQEEvaluationError
    from ubt.core.qe.base import BaseQERunner

    class _ShortRunner(BaseQERunner):
        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return []

    ledger = SQLiteJobLedger(tmp_path / "short_scores.sqlite")
    seed_job(ledger, "job_short_scores", _make_doc(), target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001",
        target_text="当增长时,缓存增长。",
        draft_text="当增长时,缓存增长。",
        status=BlockStatus.DRAFTED,
    )

    with pytest.raises(MTQEEvaluationError):
        ctx = build_stage_ctx(
            tmp_path,
            ledger=ledger,
            job_id="job_short_scores",
            fast_pass=_fp(),
            qe_runner=_ShortRunner(),
            create_event=inert_event,
        )
        asyncio.run(drain(run_quality_gate_stage(ctx)))


def test_quality_gate_numeric_defect_never_auto_passes(tmp_path: Path) -> None:
    """Regression: 'Numeric fidelity' was missing from the QE fatal-marker
    table, so a high neural score could clear the flag and ship a wrong
    number; the shared taxonomy makes it fatal in QE and triage alike."""
    from ubt.core.qe.base import BaseQERunner

    class _HighScoreRunner(BaseQERunner):
        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [0.95] * len(pairs)

    doc = SeedDoc(
        doc_id="gate4num",
        source_path="/tmp/gate4num.epub",
        format_type="epub",
        metadata={},
        blocks=[
            IRBlock(
                id="ch01#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                source_text="The cache stores 42 states.",
            )
        ],
    )
    ledger = SQLiteJobLedger(tmp_path / "numeric.sqlite")
    seed_job(ledger, "job_numeric", doc, target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001",
        target_text="缓存存储 43 个状态。",
        draft_text="缓存存储 43 个状态。",
        status=BlockStatus.DRAFTED,
    )

    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_numeric",
        fast_pass=_fp(),
        qe_runner=_HighScoreRunner(),
        create_event=inert_event,
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))
    block = ledger.get_block("ch01#b001")
    assert block is not None
    assert block.status is BlockStatus.REPAIR_PENDING
    assert any("Numeric fidelity" in f for f in block.error_flags)
