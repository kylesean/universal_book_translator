"""Unit tests for fail-closed render skip pass-through to ledger + audit report."""

from types import SimpleNamespace

from ubt.core.engine.reporter import (
    QualityReport,
    ReportRepairBreakdown,
    ReportScoreMetrics,
    ReportSummary,
    render_kdp_audit_markdown,
    summarize_render_skips,
)
from ubt.core.engine.stages.export import apply_length_policy_flags, apply_render_skip_flags
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    FlowID,
    IRBlock,
)
from ubt.core.ports import get_last_render_skips


def _block(block_id: str) -> IRBlock:
    return IRBlock(
        id=block_id,
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="source",
        target_text="译文",
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=0.9,
        repair_rounds=0,
    )


def _report(defect_flags: dict[str, int]) -> QualityReport:
    return QualityReport(
        job_id="job_a",
        doc_id="doc_a",
        book_title="Skip Test Book",
        source_path="source.pdf",
        output_path="out.pdf",
        target_lang="zh",
        summary=ReportSummary(
            total_blocks=2,
            completed_blocks=2,
            repaired_blocks=0,
            failed_blocks=0,
            pass_rate=1.0,
            estimated_cost_usd=0.0,
            needs_human_blocks=0,
            blocked_human_blocks=0,
        ),
        score_metrics=ReportScoreMetrics(
            avg_qe=0.9,
            min_qe=0.9,
            max_qe=0.9,
            p10_qe=0.9,
            p50_qe=0.9,
            p90_qe=0.9,
            bottom_15_avg_qe=0.9,
        ),
        repair_breakdown=ReportRepairBreakdown(
            direct_pass_count=2,
            round_1_repaired_count=0,
            round_2_repaired_count=0,
            exhausted_count=0,
        ),
        defect_flags=defect_flags,
    )


def test_ports_helper_passes_through_valid_skips() -> None:
    adapter = SimpleNamespace(last_render_skips=[("b1", "spill"), ("b2", "no_zone")])
    assert get_last_render_skips(adapter) == [
        ("b1", "spill"),
        ("b2", "no_zone"),
    ]


def test_ports_helper_never_raises() -> None:
    assert get_last_render_skips(SimpleNamespace()) == []
    assert get_last_render_skips(SimpleNamespace(last_render_skips=None)) == []
    assert get_last_render_skips(
        SimpleNamespace(last_render_skips=[("b1", "spill"), ("bad",), "nope", (1, 2)])
    ) == [("b1", "spill")]


def test_apply_skip_flags_maps_and_dedupes() -> None:
    blocks = [_block("b1"), _block("b2")]
    first = apply_render_skip_flags(blocks, [("b1", "spill"), ("b9", "no_zone")])
    assert len(first) == 1
    assert blocks[0].error_flags == ["render_skip:spill"]
    assert blocks[1].error_flags == []
    # Re-export is idempotent: no duplicate flags, no new checkpoints.
    assert apply_render_skip_flags(blocks, [("b1", "spill")]) == []


def _length_block(block_id: str, page_kind: str) -> IRBlock:
    block = _block(block_id)
    block.error_flags.append("render_skip:overflow(base=10.0)")
    block.provenance = {"page_kind": page_kind}
    return block


def test_length_policy_flips_resume_poster_to_human() -> None:
    """Overflow on length-policy pages reaches the human queue."""
    resume = _length_block("r1", "resume_dense")
    poster = _length_block("p1", "poster_fixed")
    prose = _length_block("t1", "editable_text")
    assert apply_length_policy_flags([resume, poster, prose], {"r1", "p1", "t1"}) == 2
    assert resume.status == BlockStatus.NEEDS_HUMAN
    assert poster.status == BlockStatus.NEEDS_HUMAN
    assert "length_overflow:overflow(base=10.0)" in resume.error_flags
    # Non-policy pages keep flag-only P7 behaviour.
    assert prose.status == BlockStatus.MTQE_PASSED
    assert all(not f.startswith("length_overflow:") for f in prose.error_flags)
    # Unknown ids ignored; statuses stay terminal.
    assert apply_length_policy_flags([resume], {"ghost"}) == 0
    assert resume.is_finalized and poster.is_finalized


def test_length_policy_reads_legacy_inplace_prefix() -> None:
    """Historical ledgers carry ``inplace_skip:``; the length policy still reads it."""
    block = _block("r1")
    block.error_flags.append("inplace_skip:overflow(base=10.0)")
    block.provenance = {"page_kind": "resume_dense"}
    assert apply_length_policy_flags([block], {"r1"}) == 1
    assert "length_overflow:overflow(base=10.0)" in block.error_flags


def test_length_policy_ignores_non_prose_and_policy_skips() -> None:
    """Code/table blocks (non_prose) and policy-skipped blocks on resume_dense pages are not length overflows."""
    code_blk = _block("c1")
    code_blk.block_type = BlockType.CODE
    code_blk.skip_translate = True
    code_blk.error_flags.append("render_skip:non_prose")
    code_blk.provenance = {"page_kind": "resume_dense"}

    policy_blk = _block("p1")
    policy_blk.skip_translate = True
    policy_blk.error_flags.append("render_skip:policy")
    policy_blk.provenance = {"page_kind": "resume_dense"}

    assert apply_length_policy_flags([code_blk, policy_blk], {"c1", "p1"}) == 0
    assert code_blk.status == BlockStatus.MTQE_PASSED
    assert policy_blk.status == BlockStatus.MTQE_PASSED
    assert all(not f.startswith("length_overflow:") for f in code_blk.error_flags)
    assert all(not f.startswith("length_overflow:") for f in policy_blk.error_flags)


def test_summarize_groups_families() -> None:
    total, verdict = summarize_render_skips(
        {
            "render_skip:spill": 2,
            "render_skip:no_zone": 1,
            "render_skip:overlay_compile": 1,
            "html_tag_mismatch": 4,
        }
    )
    assert total == 4
    assert "spill×2" in verdict
    assert "no_zone×1" in verdict
    assert "review required" in verdict


def test_summarize_reads_legacy_inplace_prefix() -> None:
    total, verdict = summarize_render_skips(
        {"inplace_skip:overflow(base=10.0)": 2, "inplace_skip:guarded": 1}
    )
    assert total == 3
    assert "overflow×2" in verdict
    assert "guarded×1" in verdict


def test_summarize_empty_and_markdown() -> None:
    assert summarize_render_skips({}) == (0, "No fail-closed skips recorded")
    assert summarize_render_skips({"html_tag_mismatch": 1})[0] == 0

    markdown = render_kdp_audit_markdown(_report({"render_skip:spill": 2}))
    assert "Render-Skipped Segments" in markdown
    assert "spill×2" in markdown

    clean = render_kdp_audit_markdown(_report({}))
    assert "No fail-closed skips recorded" in clean


def test_apply_skip_flags_clears_stale_flags_from_previous_render() -> None:
    """A render-only rerun can render what an older artifact skipped: the
    stale flag must not survive, or the quality report describes a file
    that no longer exists (chapter-3 captions after the pairing fix)."""
    block = _block("b1")
    apply_render_skip_flags([block], [("b1", "spill")])
    assert block.error_flags == ["render_skip:spill"]
    # Same render again: no duplicates, no new checkpoints.
    assert apply_render_skip_flags([block], [("b1", "spill")]) == []
    # New render renders b1 and skips another block: stale flag drops.
    other = _block("b2")
    checkpoints = apply_render_skip_flags([block, other], [("b2", "chrome")])
    assert block.error_flags == []
    assert other.error_flags == ["render_skip:chrome"]
    assert [c["block_id"] for c in checkpoints] == ["b2"]


def test_apply_skip_flags_clears_legacy_inplace_flags() -> None:
    """Pre-rename ledgers keep the retired prefix; a rerun must clear it."""
    block = _block("b1")
    block.error_flags.append("inplace_skip:unmatched")
    assert apply_render_skip_flags([block], []) == []
    assert block.error_flags == []


def test_apply_skip_flags_matches_reasons_per_block_not_globally() -> None:
    """The same reason string on another block must not keep a stale flag:
    'unmatched' also flagged every other skipped block (chapter-3 b0018
    kept its stale unmatched flag while 63 other blocks still reported it)."""
    stale = _block("b1")
    apply_render_skip_flags([stale], [("b1", "unmatched")])
    still_skipped = _block("b2")
    apply_render_skip_flags([stale, still_skipped], [("b2", "unmatched")])
    assert stale.error_flags == []
    assert still_skipped.error_flags == ["render_skip:unmatched"]
