"""Unit tests for Gate 1/3 math-debris detection + Gate 3 export invariant.

Positive samples are verbatim ``source_text`` from the KV-Cache handbook
pdfium ledger (job_fef00b35, page 5) — the exact debris that shipped as
hallucinated drafts. Negatives are clean prose/headings from the same pages
plus CJK headings that must never flip (conservative classifier: a miss
still goes through normal QE, a false positive skips translation).
"""

from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.validators.math_guard import (
    apply_math_guards,
    formula_target_intact,
    looks_like_math_debris,
    normalize_math,
)

# Verbatim debris sources from job_fef00b35 (pdfium path, page 5).
DEBRIS_KT = "= [ ] = [ ] k k K , V1 v v ; ; ; ; 1: 1 1 1 V1: 1 1 1 t t t t . . . . . . . − − − −"
DEBRIS_QKV = ", k q , v t t t ,"
DEBRIS_KT2 = "= [ ] = [ ] k K K , V1 V1 v ; ; 1: 1: 1 V1: V1: 1 t t t t t t . − −"
DEBRIS_SOFTMAX = "T  K q t 1: t softmax V1 V1: t . √ d h"
DEBRIS_SHAPE = "B H D T × × × R K, V KV h ∈ ,"
DEBRIS_DOTS = ". . ."


def test_debris_positives_from_kv_ledger() -> None:
    for sample in (
        DEBRIS_KT,
        DEBRIS_QKV,
        DEBRIS_KT2,
        DEBRIS_SOFTMAX,
        DEBRIS_SHAPE,
        DEBRIS_DOTS,
    ):
        assert looks_like_math_debris(sample), sample


def test_clean_prose_and_headings_negative() -> None:
    for sample in [
        "Suppose the model has already processed positions 1, ..., t - 1.",
        "What exactly is cached?",
        "The cache does not store thoughts or symbolic facts.",
        "Research anchors: [1], [6]",
        "Abstract",
        "Uses.",
        "MHA GQA MQA",
        "Q1 Q2 Q3 Q4 Q5 Q6",
        "For a common logical layout, one layer's cache can be thought of as",
        "where B is batch size and T is cached sequence length",
        # Former false positives (real headings/captions, must translate):
        "1 - AUTOREGRESSIVE GENERATION",
        "3 - THE MECHANISM",
        "(1) (1) Layer 1: K , V",
        "(2) (2) Layer 2: K , V",
    ]:
        assert not looks_like_math_debris(sample), sample


def test_cjk_never_debris() -> None:
    for sample in [
        "1 - 自回归生成",
        "第 1 层: K^(1), V^(1)",
        "什么是缓存呢?",
        "对于新的位置 t,该层仅计算新的投影值",
    ]:
        assert not looks_like_math_debris(sample), sample


def test_single_token_and_long_prose_never_debris() -> None:
    assert not looks_like_math_debris("Hello")
    assert not looks_like_math_debris("x")
    assert not looks_like_math_debris("K, V " * 60)


def test_normalize_math_ignores_whitespace() -> None:
    assert normalize_math("K _ { 1 : t }") == normalize_math("K_{1:t}")
    assert normalize_math("") == ""


def test_subscript_flat_ignores_ordinary_callouts() -> None:
    """'Fig 3'/'Tab 1' are callouts, not flattened subscripts; prose stays prose."""
    from ubt.core.validators.math_guard import _SUBSCRIPT_FLAT_RE, target_missing_math_delimiters

    assert _SUBSCRIPT_FLAT_RE.search("N 2") is not None
    assert _SUBSCRIPT_FLAT_RE.search("Fig 3") is None
    assert _SUBSCRIPT_FLAT_RE.search("Tab 1") is None
    assert _SUBSCRIPT_FLAT_RE.search("Eq 3") is None

    src = "See Fig 3 and Tab 1 for the pattern \\d+."
    tgt = "参见 Fig 3 和 Tab 1 中的模式。"
    assert not target_missing_math_delimiters(src, tgt)


def test_undelimited_gate_ignores_spaced_capital_debris() -> None:
    """A drop-cap TOC/header run is not math: there is no candidate to redelimit.

    Regression: "C OVER T ITLE P AGE" scored high on isolated-letter density, so
    every repair round was routed to redelimit nonexistent math until the block
    was quarantined as MQM-critical (animal-farm baseline).
    """
    from ubt.core.validators.math_guard import target_missing_math_delimiters

    debris = "C OVER T ITLE P AGE C ONTENTS T HE F REEDOM OF THE P RESS"
    assert not target_missing_math_delimiters(debris, debris)
    # The true positive this gate exists for must still fire.
    assert target_missing_math_delimiters(
        "where Ag 0 , r , F , and F th,SI are defined as follows:",
        "其中 Ag0、r、F 和 Fth,SI 的定义如下：",
    )


def _block(
    bid: str,
    btype: BlockType,
    source: str,
    target: str | None,
    status: BlockStatus = BlockStatus.DRAFTED,
) -> IRBlock:
    return IRBlock(
        id=bid,
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=btype,
        source_text=source,
        target_text=target,
        draft_text=target,
        status=status,
    )


def test_formula_invariant_intact_and_violated() -> None:
    latex = "K _ { 1 \\colon t } = [ k _ { t } ]"
    assert formula_target_intact(_block("a", BlockType.FORMULA, latex, latex))
    assert (
        formula_target_intact(_block("a", BlockType.FORMULA, latex, "K _(1:t) = [k _(t)]")) is False
    )
    # Non-formula blocks always pass the formula invariant.
    assert formula_target_intact(_block("a", BlockType.NARRATIVE, "hi", "你好"))


def test_apply_guards_repairs_formula_drift_with_count() -> None:
    latex = "K _ { 1 \\colon t } = [ k _ { t } ]"
    bad = _block("f1", BlockType.FORMULA, latex, "K 的翻译", BlockStatus.MTQE_PASSED)
    checkpoints, counts = apply_math_guards([bad])
    assert counts == {
        "formula_invariant_repairs": 1,
        "math_debris_fallbacks": 0,
        "c_text_accepted": 0,
    }
    assert bad.target_text == latex
    assert any("math_invariant_repair" in f for f in bad.error_flags)
    # Status untouched: no pass-rate cosmetics.
    assert bad.status is BlockStatus.MTQE_PASSED
    assert checkpoints[0]["block_id"] == "f1"


def test_apply_guards_falls_back_failed_debris_with_count() -> None:
    blk = _block("d1", BlockType.NARRATIVE, DEBRIS_QKV, "k, q, v, t, t, t", BlockStatus.FAILED)
    checkpoints, counts = apply_math_guards([blk])
    assert counts == {
        "formula_invariant_repairs": 0,
        "math_debris_fallbacks": 1,
        "c_text_accepted": 0,
    }
    assert blk.target_text == DEBRIS_QKV
    assert any("math_debris_source_fallback" in f for f in blk.error_flags)
    assert blk.status is BlockStatus.FAILED
    assert checkpoints[0]["block_id"] == "d1"


def test_apply_guards_idempotent_across_export_reruns() -> None:
    """E3: resume/re-export re-runs export; flags and counts must not inflate."""
    latex = "K _ { 1 \\colon t } = [ k _ { t } ]"
    bad = _block("f1", BlockType.FORMULA, latex, "K 的翻译", BlockStatus.MTQE_PASSED)
    debris = _block("d1", BlockType.NARRATIVE, DEBRIS_QKV, "k, q, v, t, t, t", BlockStatus.FAILED)
    first_checkpoints, first_counts = apply_math_guards([bad, debris])
    second_checkpoints, second_counts = apply_math_guards([bad, debris])
    assert first_counts == {
        "formula_invariant_repairs": 1,
        "math_debris_fallbacks": 1,
        "c_text_accepted": 0,
    }
    # Second pass finds nothing new: zero counts, zero checkpoints, zero dup flags.
    assert second_counts == {
        "formula_invariant_repairs": 0,
        "math_debris_fallbacks": 0,
        "c_text_accepted": 0,
    }
    assert second_checkpoints == []
    assert len(bad.error_flags) == len(set(bad.error_flags)) == 1
    assert len(debris.error_flags) == len(set(debris.error_flags)) == 1


def test_apply_guards_leaves_healthy_blocks_alone() -> None:
    healthy = [
        _block("n1", BlockType.NARRATIVE, "Hello world.", "你好世界。"),
        _block(
            "d2",
            BlockType.NARRATIVE,
            DEBRIS_QKV,
            "k, q, v",
            BlockStatus.MTQE_PASSED,
        ),
        _block("f2", BlockType.FORMULA, "x^2", None, BlockStatus.DRAFTED),
    ]
    checkpoints, counts = apply_math_guards(healthy)
    assert checkpoints == []
    assert counts == {
        "formula_invariant_repairs": 0,
        "math_debris_fallbacks": 0,
        "c_text_accepted": 0,
    }
    assert healthy[1].target_text == "k, q, v"


def test_c_era_skeleton_accepts_span_only_translation() -> None:
    from ubt.core.validators.math_guard import formula_skeleton_intact

    src = "K = [k_1; \\text{by terms}] + x"
    good = _block(
        "f1",
        BlockType.FORMULA,
        src,
        "K = [k_1; \\text{按项}] + x",
        BlockStatus.MTQE_PASSED,
    )
    assert formula_skeleton_intact(good)
    checkpoints, counts = apply_math_guards([good])
    # Accepted, NOT repaired: target keeps the translation.
    assert counts.get("c_text_accepted", 0) == 1
    assert counts["formula_invariant_repairs"] == 0
    assert good.target_text is not None and "\\text{按项}" in good.target_text
    assert any("c_text_span_translation" in f for f in good.error_flags)
    assert checkpoints and checkpoints[0]["block_id"] == "f1"


def test_c_era_skeleton_still_repairs_math_drift() -> None:
    src = "K = [k_1; \\text{by terms}] + x"
    bad = _block(
        "f2",
        BlockType.FORMULA,
        src,
        "K = [k_1; \\text{按项}] + y",
        BlockStatus.MTQE_PASSED,
    )
    _, counts = apply_math_guards([bad])
    assert counts["formula_invariant_repairs"] == 1
    assert bad.target_text == src


def test_undelimited_gate_exempts_code_identifiers_and_snake_case() -> None:
    """CS/software engineering snake_case identifiers (create_task, code_change)
    must not be flagged as undelimited math subscript residue.

    Regression: b0656 (create_task) and b0784 (code_change) in 2608.25512v1 were
    flagged as undelimited math, sent to 3 repair rounds, and quarantined as
    MQM-critical (【待人工审校】) solely due to _SOUP_RESIDUE_RE matching standard
    identifier names.
    """
    from ubt.core.validators.math_guard import target_missing_math_delimiters

    # Normal programming identifiers in prose should NOT trip the gate
    src1 = "create_task schedules an async function to run concurrently."
    tgt1 = "create_task 负责调度异步函数并发运行。"
    assert not target_missing_math_delimiters(src1, tgt1)

    src2 = "migrating state through code_change/3 and recovering from faults."
    tgt2 = "经由 code_change/3 迁移状态，并通过故障中恢复。"
    assert not target_missing_math_delimiters(src2, tgt2)

    src3 = "backup = invalidate_caches(accepted, stale_entries)"
    tgt3 = "backup = invalidate_caches(accepted, stale_entries)"
    assert not target_missing_math_delimiters(src3, tgt3)

    # Actual math soup residue MUST still trip the gate
    assert target_missing_math_delimiters(
        "where psi_pert is the perturbation parameter.",
        "其中 psi_pert 是微扰参数。",
    )
    assert target_missing_math_delimiters(
        "where x_1 is the input coordinate.",
        "其中 x_1 是输入坐标。",
    )
    assert target_missing_math_delimiters(
        "where A_{g0} is constant.",
        "其中 A_{g0} 是常数。",
    )

