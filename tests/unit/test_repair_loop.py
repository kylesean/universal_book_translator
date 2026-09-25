from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import BaseModelProvider, MockModelProvider
from ubt.core.router.router import ModelRouter


class ControlledScoreQERunner(BaseQERunner):
    """Controlled score provider for repair verification."""

    def __init__(self, next_scores: list[float]) -> None:
        self.scores = list(next_scores)

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        result: list[float] = []
        for _ in pairs:
            result.append(self.scores.pop(0) if self.scores else 0.5)
        return result


class GlossaryAwareScoreQERunner(ControlledScoreQERunner):
    """A scorer that re-checks terminology itself (``is_glossary_aware``)."""

    def is_glossary_aware(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_repair_loop_selects_bottom_percentile_and_flags() -> None:
    router = ModelRouter(provider=MockModelProvider())
    qe = MockQERunner(default_score=0.85)
    repair_loop = RepairLoop(
        router=router,
        qe_runner=qe,
        max_rounds=2,
        qe_threshold=0.75,
        bottom_percentile=0.20,
    )

    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            source_text="s1",
            mtqe_score=0.92,
            status=BlockStatus.MTQE_PASSED,
        ),
        IRBlock(
            id="b2", spine_index=2, source_text="s2", mtqe_score=0.88, status=BlockStatus.DRAFTED
        ),
        IRBlock(
            id="b3", spine_index=3, source_text="s3", mtqe_score=0.82, status=BlockStatus.DRAFTED
        ),
        IRBlock(
            id="b4", spine_index=4, source_text="s4", mtqe_score=0.78, status=BlockStatus.DRAFTED
        ),
        IRBlock(
            id="b5", spine_index=5, source_text="s5", mtqe_score=0.45, status=BlockStatus.DRAFTED
        ),  # Lowest!
        IRBlock(
            id="b6",
            spine_index=6,
            source_text="s6",
            mtqe_score=0.85,
            status=BlockStatus.DRAFTED,
            error_flags=["broken_tag"],
        ),
    ]

    candidates = repair_loop.select_repair_candidates(blocks)
    candidate_ids = {c.id for c in candidates}
    # b5 is in lowest percentile (<0.75), b6 has error_flags, b1 is MTQE_PASSED
    assert "b5" in candidate_ids
    assert "b6" in candidate_ids
    assert "b1" not in candidate_ids
    assert "b2" not in candidate_ids


@pytest.mark.asyncio
async def test_repair_loop_executes_repair_and_upgrades_status() -> None:
    router = ModelRouter(provider=MockModelProvider(default_response="优秀修复译文"))
    # Return upgraded score 0.89 on repair
    qe = ControlledScoreQERunner(next_scores=[0.89])
    repair_loop = RepairLoop(router=router, qe_runner=qe, qe_threshold=0.75, max_rounds=2)

    block = IRBlock(
        id="b_low",
        spine_index=1,
        source_text="Difficult philosophical sentence.",
        draft_text="拙劣草翻",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.42,
        repair_rounds=0,
        error_flags=["grammar_error"],
    )

    repaired = await repair_loop.repair_single_block(block)
    assert repaired.target_text == "优秀修复译文"
    assert repaired.mtqe_score == 0.89
    assert repaired.status == BlockStatus.REPAIRED
    assert repaired.repair_rounds == 1
    assert repaired.error_flags == []


@pytest.mark.asyncio
async def test_repair_loop_trips_circuit_breaker_on_max_rounds() -> None:
    router = ModelRouter(provider=MockModelProvider(default_response="依然低质"))
    # Always return degraded score 0.30
    qe = ControlledScoreQERunner(next_scores=[0.30, 0.30])
    repair_loop = RepairLoop(router=router, qe_runner=qe, qe_threshold=0.75, max_rounds=2)

    block = IRBlock(
        id="b_hopeless",
        spine_index=1,
        source_text="Hopeless untranslatable string.",
        draft_text="草翻",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.35,
        repair_rounds=1,
    )

    # Round 2: Reaches max_rounds=2
    repaired_r2 = await repair_loop.repair_single_block(block)
    assert repaired_r2.repair_rounds == 2

    # Round 3 attempt: Circuit breaker trips immediately without invoking LLM
    tripped = await repair_loop.repair_single_block(repaired_r2)
    assert tripped.status == BlockStatus.FAILED
    assert tripped.repair_rounds == 2


@pytest.mark.asyncio
async def test_repair_loop_dynamic_reasoning_effort_dispatch() -> None:
    """Format-only defects receive reasoning_effort='low'; semantic/low-score defects receive router.repair_reasoning_effort ('high')."""
    mock_provider = MockModelProvider(default_response="修复完成")
    router = ModelRouter(
        provider=mock_provider,
        repair_reasoning_effort="high",
    )
    qe = ControlledScoreQERunner(next_scores=[0.85, 0.85])
    repair_loop = RepairLoop(router=router, qe_runner=qe)

    # 1. Format-only block: html_tag_mismatch with acceptable score (0.75)
    format_block = IRBlock(
        id="b_format",
        spine_index=1,
        source_text="<b>Hello</b>",
        draft_text="Hello",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.75,
        error_flags=["html_tag_mismatch"],
    )
    await repair_loop.repair_single_block(format_block)
    assert len(mock_provider.call_history) == 1
    assert mock_provider.call_history[0]["reasoning_effort"] == "low"

    # 2. Semantic defect block: glossary_mismatch with low score (0.45)
    semantic_block = IRBlock(
        id="b_semantic",
        spine_index=2,
        source_text="Neural network weights",
        draft_text="神经网络体重",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.45,
        error_flags=["glossary_mismatch"],
    )
    await repair_loop.repair_single_block(semantic_block)
    assert len(mock_provider.call_history) == 2
    assert mock_provider.call_history[1]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_repair_loop_rejects_structural_corruption() -> None:
    """If repaired text violates 0-token structural invariants (e.g. repetition hallucination),
    it must be rejected, error_flags updated, and not released as REPAIRED."""
    # Hallucinatory repetitive output
    hallucinatory_response = "很棒很棒很棒很棒很棒很棒很棒很棒"
    router = ModelRouter(provider=MockModelProvider(default_response=hallucinatory_response))
    qe = ControlledScoreQERunner(next_scores=[0.95])  # High semantic score!
    from ubt.core.qe.fast_pass import FastPassFilter

    repair_loop = RepairLoop(
        router=router,
        qe_runner=qe,
        qe_threshold=0.75,
        max_rounds=2,
        fast_pass=FastPassFilter(),
    )

    block = IRBlock(
        id="b_struct",
        spine_index=1,
        source_text="This is a great novel.",
        draft_text="这是一部不错的小说。",
        target_text="这是一部不错的小说。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.60,
        repair_rounds=0,
        error_flags=["minor_awkwardness"],
    )

    repaired = await repair_loop.repair_single_block(block)
    # Repaired text should NOT be adopted because structural evaluation failed
    assert repaired.target_text == "这是一部不错的小说。"
    assert any("repair_structural_failure" in flag for flag in repaired.error_flags)
    assert repaired.status == BlockStatus.REPAIR_PENDING
    assert repaired.repair_rounds == 1


@pytest.mark.asyncio
async def test_repair_loop_rejects_html_tag_corruption() -> None:
    """If repaired text corrupts HTML markup tags, it must be rejected even if semantic score is high."""
    # Corrupted HTML with unescaped quote in alt attribute
    corrupted_html = '<img src="cat.jpg" alt="一只可爱的"小猫""/>'
    router = ModelRouter(provider=MockModelProvider(default_response=corrupted_html))
    qe = ControlledScoreQERunner(next_scores=[0.98])
    from ubt.core.qe.fast_pass import FastPassFilter

    repair_loop = RepairLoop(
        router=router,
        qe_runner=qe,
        qe_threshold=0.75,
        max_rounds=1,
        fast_pass=FastPassFilter(),
    )

    block = IRBlock(
        id="b_html_corrupt",
        spine_index=1,
        source_text='<img src="cat.jpg" alt="A cute cat"/>',
        draft_text='<img src="cat.jpg" alt="一只可爱的小猫"/>',
        target_text='<img src="cat.jpg" alt="一只可爱的小猫"/>',
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.65,
        repair_rounds=0,
        error_flags=["html_tag_mismatch"],
    )

    repaired = await repair_loop.repair_single_block(block)
    assert repaired.target_text == '<img src="cat.jpg" alt="一只可爱的小猫"/>'
    assert any("HTML delta failure" in flag for flag in repaired.error_flags)
    assert repaired.status == BlockStatus.FAILED


@pytest.mark.asyncio
async def test_repair_loop_invokes_visual_scalpel_on_eligible_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eligible block (BlockType.FORMULA or mtqe_score < 0.60) triggers visual crop."""
    from ubt.core import ports
    from ubt.core.ir.models import BlockType, BoundingBox

    captured_crops: list[str] = []

    def _mock_cropper(pdf_path: Path, block: Any, dpi: int = 150) -> str:
        captured_crops.append(f"{block.id}@{dpi}")
        return "mock_base64_crop"

    # The cropper is reached through the ports bridge, so the fake is installed
    # there — the same seam the repair loop imports at call time.
    monkeypatch.setattr(ports, "crop_block_image", _mock_cropper)

    provider = MockModelProvider(default_response="视觉修复公式")
    router = ModelRouter(provider=provider)
    qe = ControlledScoreQERunner(next_scores=[0.95])
    repair_loop = RepairLoop(router=router, qe_runner=qe, qe_threshold=0.75, max_rounds=1)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    block = IRBlock(
        id="b_formula",
        spine_index=1,
        block_type=BlockType.FORMULA,
        source_text=r"\nabla^2 \psi = 0",
        draft_text="nabla psi = 0",
        target_text="nabla psi = 0",
        status=BlockStatus.REPAIR_PENDING,
        bbox=BoundingBox(page=1, x0=10, y0=20, x1=100, y1=50),
        mtqe_score=0.45,
        error_flags=["formula_defect"],
    )

    repaired = await repair_loop.repair_single_block(block, source_pdf_path=fake_pdf)
    assert len(captured_crops) == 1
    assert "b_formula" in captured_crops[0]
    assert repaired.target_text == "视觉修复公式"
    assert repaired.status == BlockStatus.REPAIRED


@pytest.mark.asyncio
async def test_repair_loop_rejects_numeric_fidelity_failure() -> None:
    """Repair that drops numbers must fail evaluation, retain error_flags, and not promote to REPAIRED."""
    # Repair model returns text without the number 42
    provider = MockModelProvider(default_response="这里有许多苹果。")
    router = ModelRouter(provider=provider)
    qe = ControlledScoreQERunner(next_scores=[0.95])  # High semantic score!
    from ubt.core.qe.fast_pass import FastPassFilter

    repair_loop = RepairLoop(
        router=router,
        qe_runner=qe,
        qe_threshold=0.75,
        max_rounds=1,
        fast_pass=FastPassFilter(),
    )

    block = IRBlock(
        id="b_numeric",
        spine_index=1,
        source_text="There are 42 apples here.",
        draft_text="这里有一些苹果。",
        target_text="这里有一些苹果。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.45,
        repair_rounds=0,
        error_flags=["Numeric fidelity failure: missing numbers [42]"],
    )

    repaired = await repair_loop.repair_single_block(block)
    # The numeric check must fail, preventing promotion to REPAIRED
    assert repaired.status == BlockStatus.FAILED
    assert any("Numeric fidelity" in flag for flag in repaired.error_flags)
    assert repaired.target_text == "这里有一些苹果。"  # Corrupted repair rejected


@pytest.mark.asyncio
async def test_tied_repair_keeps_the_defect_evidence() -> None:
    """A repair that neither improves nor reaches the pass line stays flagged.

    Every structural flag caps its band at 0.70 (``score_from_flags``), so a
    candidate that ties the draft at 0.70 proved nothing, yet the old ``>=``
    comparison adopted the text *and* cleared error_flags -- and a block with no
    flags is what the next resume's FastPass releases as MTQE_PASSED. Adoption
    itself is kept: the consistency stage repairs terminology drift that no QE
    band measures.
    """
    router = ModelRouter(provider=MockModelProvider(default_response="工作内存是核心。"))
    repair_loop = RepairLoop(
        router=router,
        qe_runner=ControlledScoreQERunner([0.70]),
        qe_threshold=0.75,
        max_rounds=2,
    )
    block = IRBlock(
        id="b1",
        spine_index=1,
        source_text="Working Memory is central.",
        target_text="工作内存是核心。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.70,
        error_flags=["math_token_corrupt"],
    )

    repaired = await repair_loop.repair_single_block(block)

    assert repaired.error_flags == ["math_token_corrupt"]
    assert repaired.status is not BlockStatus.REPAIRED


@pytest.mark.asyncio
async def test_repair_that_reaches_the_pass_line_clears_the_flags() -> None:
    """The other side of the same rule: crossing the threshold is evidence."""
    router = ModelRouter(provider=MockModelProvider(default_response="工作记忆是核心。"))
    repair_loop = RepairLoop(
        router=router,
        qe_runner=ControlledScoreQERunner([0.80]),
        qe_threshold=0.75,
        max_rounds=2,
    )
    block = IRBlock(
        id="b2",
        spine_index=1,
        source_text="Working Memory is central.",
        target_text="工作内存是核心。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.70,
        error_flags=["math_token_corrupt"],
    )

    repaired = await repair_loop.repair_single_block(block)

    assert repaired.error_flags == []
    assert repaired.status is BlockStatus.REPAIRED


async def test_repair_loop_targeted_span_flow() -> None:
    """Verify that RepairLoop end-to-end annotates error spans and splices targeted corrections."""

    class TargetedRepairProvider(BaseModelProvider):
        @property
        def provider_name(self) -> str:
            return "targeted_mock"

        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.0,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            # Check if prompt contains the <error_span>
            assert "<error_span" in prompt
            assert "分散记账簿" in prompt
            # Return targeted in-place correction tag
            return '<correction id="1">分布式账本</correction>'

    router = ModelRouter(provider=TargetedRepairProvider())
    qe_runner = MockQERunner(default_score=0.92)

    repair_loop = RepairLoop(
        router=router,
        qe_runner=qe_runner,
        max_rounds=2,
        qe_threshold=0.75,
    )

    glossary = [
        {
            "source": "distributed ledger",
            "translation": "分布式账本",
            "aliases": ["分散记账簿"],
        }
    ]

    block = IRBlock(
        id="b_001",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="This network relies on a distributed ledger for state updates.",
        target_text="该网络依托于分散记账簿进行状态更新。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.60,
        error_flags=["terminology_inconsistency: found alias '分散记账簿'"],
    )

    repaired = await repair_loop.repair_single_block(
        block=block,
        glossary_table="",
        target_lang="zh",
        source_lang="en",
        glossary_entries=glossary,
    )

    assert repaired.status == BlockStatus.REPAIRED
    assert repaired.target_text == "该网络依托于分布式账本进行状态更新。"
    assert (repaired.mtqe_score or 0.0) >= 0.75
    assert repaired.error_flags == []
    assert repaired.repair_rounds == 1


@pytest.mark.asyncio
async def test_repair_loop_does_not_wash_glossary_violation_with_unaware_scorer() -> None:
    """P1-4: a scorer that does not check terminology (COMET / Tiered / LLM judge
    — modelled here by a fixed-high ControlledScoreQERunner) must not launder a
    glossary-violation flag into a clean REPAIRED. The marker survives so triage
    still routes the block to the human queue."""
    from ubt.core.qe.comet_runner import GLOSSARY_VIOLATION_MARKER

    router = ModelRouter(provider=MockModelProvider(default_response="优秀修复译文"))
    qe = ControlledScoreQERunner(next_scores=[0.89])  # high, but term-blind
    repair_loop = RepairLoop(router=router, qe_runner=qe, qe_threshold=0.75, max_rounds=2)

    block = IRBlock(
        id="b_term",
        spine_index=1,
        source_text="We use a neural network.",
        draft_text="我们用了网络模型。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.6,
        repair_rounds=0,
        error_flags=[f"{GLOSSARY_VIOLATION_MARKER}: dropped 'neural network'"],
    )
    repaired = await repair_loop.repair_single_block(block)
    assert any(GLOSSARY_VIOLATION_MARKER in f for f in repaired.error_flags)
    assert repaired.status is not BlockStatus.REPAIRED


@pytest.mark.asyncio
async def test_repair_loop_clears_glossary_marker_when_scorer_verified_the_term() -> None:
    """A glossary-aware re-score IS evidence the dropped term came back.

    The block must then be allowed to reach REPAIRED. Keeping the marker would
    make ``not block.error_flags`` unsatisfiable, so a genuinely fixed
    terminology defect would be quarantined as if it were still broken.
    """
    from ubt.core.qe.comet_runner import GLOSSARY_VIOLATION_MARKER

    router = ModelRouter(provider=MockModelProvider(default_response="我们使用神经网络。"))
    qe = GlossaryAwareScoreQERunner(next_scores=[0.89])
    repair_loop = RepairLoop(router=router, qe_runner=qe, qe_threshold=0.75, max_rounds=2)

    block = IRBlock(
        id="b_term_fixed",
        spine_index=1,
        source_text="We use a neural network.",
        draft_text="我们用了网络模型。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.6,
        repair_rounds=0,
        error_flags=[f"{GLOSSARY_VIOLATION_MARKER}: dropped 'neural network'"],
    )
    repaired = await repair_loop.repair_single_block(block)
    assert repaired.error_flags == []
    assert repaired.status is BlockStatus.REPAIRED


def test_rerank_requires_calibrated_runner() -> None:
    """best-of-n may only rank by a graded quality score.

    The isinstance(HeuristicQERunner) gate could not see a COMET subprocess
    that fell back to its own heuristic — twelve discrete bands deciding the
    "best" candidate while pretending to be CometKiwi. The calibration flag
    closes exactly that hole.
    """
    router = ModelRouter(provider=MockModelProvider(default_response="修复"))

    class _Uncalibrated(MockQERunner):
        def is_calibrated(self) -> bool:
            return False

    calibrated = RepairLoop(
        router=router, qe_runner=MockQERunner(), qe_threshold=0.75, max_rounds=1, rerank_k=3
    )
    uncalibrated = RepairLoop(
        router=router, qe_runner=_Uncalibrated(), qe_threshold=0.75, max_rounds=1, rerank_k=3
    )
    assert calibrated._rerank_enabled() is True
    assert uncalibrated._rerank_enabled() is False


def test_out_of_span_preservation_guard() -> None:
    """The wholesale-replacement guard compares character multisets outside spans."""
    from types import SimpleNamespace

    from ubt.core.engine.repair_loop import _out_of_span_content_preserved

    draft = "The quick brown fox jumps over the lazy dog, repeatedly and quite loudly."
    spans = [SimpleNamespace(start_pos=4, end_pos=9)]  # "quick"
    surgical = "The slow brown fox jumps over the lazy dog, repeatedly and quite loudly."
    assert _out_of_span_content_preserved(draft, spans, surgical)
    rewrite = "A totally different paragraph that no annotator flagged, nope, nothing shared."
    assert not _out_of_span_content_preserved(draft, spans, rewrite)
    # Short out-of-span remainder: re-rendering is allowed there.
    assert _out_of_span_content_preserved("the quick fox", spans[:1], "a red cat runs")


@pytest.mark.asyncio
async def test_repair_rejects_wholesale_final_translation_rewrite() -> None:
    """A <final_translation> that rewrites unflagged text must not be adopted.

    The over-editing guard lives in the splicer's per-span path; when the
    model ignores the protocol and ships whole-segment output, only the
    out-of-span preservation check stands between a fluent paraphrase and
    the ledger.
    """

    class WholesaleRewriteProvider(BaseModelProvider):
        @property
        def provider_name(self) -> str:
            return "wholesale_mock"

        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.0,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            return (
                "<final_translation>这是一份完全重写的译文，它抛弃了原稿里所有没有被"
                "标注出来的句子内容并且用完全不同的措辞重新组织了信息的表达方式。</final_translation>"
            )

    router = ModelRouter(provider=WholesaleRewriteProvider())
    repair_loop = RepairLoop(
        router=router, qe_runner=MockQERunner(default_score=0.92), max_rounds=1, qe_threshold=0.75
    )
    draft = (
        "该网络依托于分散记账簿进行状态更新并且所有参与方都必须按照既定协议"
        "在同一个时间窗口之内完成对账流程以保证账目的一致性。"
    )
    block = IRBlock(
        id="b_wholesale",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text=(
            "This network relies on a distributed ledger for state updates and all "
            "participants must reconcile within the same window to keep the books consistent."
        ),
        target_text=draft,
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.60,
        error_flags=["terminology_inconsistency: found alias '分散记账簿'"],
    )
    repaired = await repair_loop.repair_single_block(
        block=block,
        glossary_table="",
        target_lang="zh",
        source_lang="en",
        glossary_entries=[
            {
                "source": "distributed ledger",
                "translation": "分布式账本",
                "aliases": ["分散记账簿"],
            }
        ],
    )
    assert repaired.target_text == draft  # wholesale rewrite not adopted
    assert any("repair_overreach" in f for f in repaired.error_flags)


@pytest.mark.asyncio
async def test_rejected_overreach_keeps_the_drafts_critical_flags() -> None:
    """Rejecting a wholesale rewrite must not erase the defect it was fixing.

    ``repair_overreach`` is in neither the structural nor the critical marker
    table, so *replacing* ``error_flags`` with it alone downgraded a block that
    had lost a figure to a Minor defect: triage then resolved it to
    NEEDS_HUMAN/MTQE_PASSED instead of the Critical quarantine it earns.
    """

    class WholesaleRewriteProvider(BaseModelProvider):
        @property
        def provider_name(self) -> str:
            return "wholesale_mock"

        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.0,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            return (
                "<final_translation>这是一份完全重写的译文，它抛弃了原稿里所有没有被"
                "标注出来的句子内容并且用完全不同的措辞重新组织了信息的表达方式。</final_translation>"
            )

    router = ModelRouter(provider=WholesaleRewriteProvider())
    repair_loop = RepairLoop(
        router=router, qe_runner=MockQERunner(default_score=0.92), max_rounds=1, qe_threshold=0.75
    )
    block = IRBlock(
        id="b_critical",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text=("Revenue rose to 12.5 million in 2024 and headcount grew by 30 percent."),
        target_text=(
            "该网络依托于分散记账簿进行状态更新并且所有参与方都必须按照既定协议"
            "在同一个时间窗口之内完成对账流程以保证账目的一致性。"
        ),
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.60,
        error_flags=[
            "Numeric fidelity failure: 12.5 missing from target",
            "terminology_inconsistency: found alias '分散记账簿'",
        ],
    )
    repaired = await repair_loop.repair_single_block(
        block=block, glossary_table="", target_lang="zh", source_lang="en"
    )
    assert any("repair_overreach" in f for f in repaired.error_flags)
    # The Critical marker that sent the block to repair survives the rejection,
    # so triage still quarantines it instead of shipping the draft.
    from ubt.core.qe.defect_taxonomy import has_critical_defect

    assert has_critical_defect(repaired.error_flags)


@pytest.mark.asyncio
async def test_repair_loop_tiered_fallback_to_wholesale_when_splicing_degrades() -> None:
    """When span repair is skipped by the model in favor of a full grammatical
    rewrite that passes structural checks and scores significantly higher on QE,
    the repair loop adopts it as a tiered fallback rather than forcing a Frankenstein sentence.
    """

    class ValidWholesaleProvider(BaseModelProvider):
        @property
        def provider_name(self) -> str:
            return "valid_wholesale_mock"

        async def generate(self, *args: Any, **kwargs: Any) -> str:
            return "<final_translation>2024年总收入增长至12.5 百万，且员工总数增长了30%。</final_translation>"

    router = ModelRouter(provider=ValidWholesaleProvider())
    repair_loop = RepairLoop(
        router=router, qe_runner=MockQERunner(default_score=0.92), max_rounds=1, qe_threshold=0.75
    )
    block = IRBlock(
        id="b_grammatical",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="Revenue rose to 12.5 million in 2024 and headcount grew by 30 percent.",
        draft_text="该网络依托于分散记账簿进行状态更新并且所有参与方都必须按照既定协议在同一个时间窗口之内完成对账流程以保证账目的一致性。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.55,
        error_flags=["Numeric fidelity failure: 12.5 missing from target"],
    )
    repaired = await repair_loop.repair_single_block(
        block=block, glossary_table="", target_lang="zh", source_lang="en"
    )
    assert repaired.status in (BlockStatus.REPAIRED, BlockStatus.MTQE_PASSED)
    assert "2024年总收入增长至12.5" in (repaired.target_text or "")
    assert not any("repair_overreach" in f for f in repaired.error_flags)


@pytest.mark.asyncio
async def test_rerank_surfaces_short_score_reply() -> None:
    """The QE runner promises one score per pair; a degraded runner that
    returns fewer must fail the round explicitly, not crash the repair with
    a bare IndexError from the best-of-n indexing (quality_gate has the
    same contract check, L21)."""
    from ubt.core.exceptions import MTQEEvaluationError

    router = ModelRouter(provider=MockModelProvider(default_response="我们使用神经网络模型。"))

    class _ShortScorer(MockQERunner):
        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [0.9]  # one short of the two candidates

    loop = RepairLoop(
        router=router, qe_runner=_ShortScorer(), qe_threshold=0.75, max_rounds=1, rerank_k=2
    )
    block = IRBlock(
        id="b_short",
        spine_index=1,
        source_text="We use a neural network.",
        draft_text="我们用了网络。",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.6,
        repair_rounds=0,
        error_flags=["Numeric fidelity"],
    )
    with pytest.raises(MTQEEvaluationError, match="1 score"):
        await loop.repair_single_block(block)
