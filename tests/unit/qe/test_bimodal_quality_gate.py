"""Bi-modal Quality Gate tests: Track 1 (Structural Integrity) vs Track 2 (Semantic Quality).

Validates that:
1. Uncalibrated runners (HeuristicQERunner) do not fabricate continuous float scores.
2. In the absence of calibrated neural/judge scorers, mtqe_score remains None.
3. Repair candidate lifecycle is governed strictly by structural integrity (error_flags resolved -> REPAIRED).
4. A defect the span annotator cannot *locate* never reaches the paid repair model:
   a whole-block rewrite cannot be verified against it, so the block is closed as
   FAILED with the original draft instead of being "repaired" unverifiably.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast

#: A flag the span annotator can *locate*: the draft states 41 where the source
#: states 42, so `annotate_draft` produces a numeric span over "41". Repair
#: tests that exercise the model call need a locatable defect — that is exactly
#: the contract this module now pins.
_LOCATABLE_FLAG = "Numeric fidelity failure: Unauthorized numeric tokens in translation: ['41']"

#: A flag with no locatable surface at all (a length/format complaint).
_UNLOCATABLE_FLAG = "html_attr_mismatch"


def _make_block(block_id: str, src: str, draft: str, flags: list[str] | None = None) -> IRBlock:
    el = Paragraph(id=block_id, spine_index=0, span=Span(page=1), text=src)
    block = IRBlock(element=el)
    block.draft_text = draft
    block.target_text = draft
    block.error_flags = flags or []
    return block


def test_heuristic_runner_is_uncalibrated() -> None:
    runner = HeuristicQERunner()
    assert runner.is_calibrated() is False


@pytest.mark.asyncio
async def test_repair_loop_uncalibrated_runner_adopts_structurally_valid_candidate() -> None:
    runner = HeuristicQERunner()
    router = MagicMock()
    # The candidate must still state the source's number: the residual-defect
    # check re-runs the numeric validator on the repaired text, so a "clean"
    # candidate that dropped 42 would (correctly) keep the block REPAIR_PENDING.
    router.repair = AsyncMock(return_value="答案是 42。")

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85)

    # A *locatable* defect: the source states 42 and the draft says 41, so the
    # span annotator produces a numeric span. Only such a defect may reach the
    # model — see the no-span test below.
    block = _make_block(
        "b1",
        "The answer is 42.",
        "答案是 41。",
        flags=[_LOCATABLE_FLAG],
    )
    block.status = BlockStatus.REPAIR_PENDING

    repaired = await loop.repair_single_block(block)
    # Track 1 passes: structural integrity is sound and error_flags cleared
    assert repaired.status == BlockStatus.REPAIRED
    assert repaired.error_flags == []
    assert repaired.target_text == "答案是 42。"
    # The repaired text is measured, not cleared. This test previously asserted
    # ``mtqe_score is None``, which read as "clear the score" but did nothing:
    # the checkpoint binds the value through ``COALESCE(?, mtqe_score)``, so an
    # explicit None preserves whatever was stored. The block then kept the score
    # that had sent it to repair (measured: 0.55 on a block that now scores
    # 0.92), and a successful repair looked like it changed nothing.
    assert repaired.mtqe_score is not None


@pytest.mark.asyncio
async def test_a_defect_with_no_locatable_span_never_calls_the_model() -> None:
    """An unlocatable defect cannot be verified, so it must not be repaired.

    Format/length/omission flags name no position in the draft. The old path
    re-translated the whole block and adopted the result on structural
    soundness alone — an unverifiable replacement, paid for. Measured on arXiv
    2609.00006v1: 24 blocks ended ``repaired`` that way. The block is now closed
    as FAILED with the original draft and an explicit reason instead.
    """
    runner = HeuristicQERunner()
    router = MagicMock()
    router.repair = AsyncMock(return_value="这是干净的修复文本。")

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85)
    block = _make_block(
        "b-nospan", "This is source text.", "这是草稿文本。", flags=[_UNLOCATABLE_FLAG]
    )
    block.status = BlockStatus.REPAIR_PENDING

    result = await loop.repair_single_block(block)

    router.repair.assert_not_called()
    assert result.status == BlockStatus.FAILED
    assert result.target_text == "这是草稿文本。", "the paid draft must survive"
    assert any(flag.startswith("repair_unverifiable:") for flag in result.error_flags)
    assert result.repair_rounds == 1, "the attempt still consumes a round"


@pytest.mark.asyncio
async def test_a_deterministic_defect_with_no_span_is_still_repaired() -> None:
    """A repetition/table/echo flag names no position but IS re-checkable.

    ``evaluate()`` re-runs the same deterministic checks that raised those flags,
    so a rewrite can be confirmed or refuted by re-reading it. Treating "no span"
    as "unverifiable" left every repetition and dropped-table block terminal
    FAILED with no retry path (``repair_unverifiable`` is not a transient
    prefix, so resume never revisited them either).
    """
    runner = HeuristicQERunner()
    router = MagicMock()
    router.repair = AsyncMock(return_value="这是一段完整的、干净的译文。")

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85)
    block = _make_block(
        "b-repeat",
        "This is source text.",
        "这是草稿文本。这是草稿文本。这是草稿文本。这是草稿文本。",
        flags=["Repetitive loop hallucination: '这是草稿文本。' x4"],
    )
    block.status = BlockStatus.REPAIR_PENDING

    result = await loop.repair_single_block(block)

    router.repair.assert_called_once()
    assert not any(flag.startswith("repair_unverifiable:") for flag in result.error_flags), (
        "a re-checkable defect must not be closed as unverifiable"
    )
    assert result.target_text == "这是一段完整的、干净的译文。"


def test_only_evaluator_backed_flags_count_as_recheckable() -> None:
    """The predicate behind that rule, pinned directly.

    Lifecycle notes (provider failures) are structural in the taxonomy — that is
    what keeps them out of the auto-pass path — but no candidate can be checked
    against them, so they must not license a paid rewrite.
    """
    from ubt.core.engine.repair_loop import RepairLoop

    cases = {
        "Repetitive loop hallucination: 'x' x4": True,
        "Table dropped: table has no rows": True,
        "Numeric fidelity failure: Missing numeric tokens in translation: ['3']": True,
        "Omission suspected: identifier term(s) missing from target: ['SOTA']": True,
        "repair_unverifiable: the defect names no location in the draft": False,
        "Repair error: upstream returned 500": False,
        "some unknown lifecycle note": False,
    }
    for flag, expected in cases.items():
        assert RepairLoop._flags_are_recheckable([flag]) is expected, flag


@pytest.mark.asyncio
async def test_repair_loop_uncalibrated_runner_fails_after_max_rounds_when_structural_defect_persists() -> (
    None
):
    runner = HeuristicQERunner()
    router = MagicMock()
    # Mock router to return empty candidate (fails structural integrity)
    router.repair = AsyncMock(return_value="")

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85, max_rounds=2)

    block = _make_block(
        "b2",
        "The answer is 42.",
        "答案是 41。",
        flags=[_LOCATABLE_FLAG],
    )
    block.repair_rounds = 1  # will become 2 (rounds_cap)
    block.status = BlockStatus.REPAIR_PENDING

    repaired = await loop.repair_single_block(block)
    assert repaired.status == BlockStatus.FAILED
    assert len(repaired.error_flags) > 0
    assert repaired.mtqe_score is None


@pytest.mark.asyncio
async def test_a_raising_repair_still_consumes_a_round() -> None:
    # A repair call that raises (provider outage) must still advance the round
    # counter: the round is consumed before the call. Otherwise a persistently
    # failing repair stays at round 0, the transient-failure reset requeues it to
    # REPAIR_PENDING on every resume, and the circuit breaker (gated on
    # repair_rounds < max_rounds) never fires -- re-billing the model forever.
    runner = HeuristicQERunner()
    router = MagicMock()
    router.repair = AsyncMock(side_effect=RuntimeError("provider outage"))

    loop = RepairLoop(qe_runner=runner, router=router, qe_threshold=0.85)

    # A *locatable* defect, so the model call actually happens: the no-span path
    # returns before the provider is reached, and this test is about the round
    # counter on a raising call.
    block = _make_block("b3", "The answer is 42.", "答案是 41。", flags=[_LOCATABLE_FLAG])
    block.status = BlockStatus.REPAIR_PENDING
    assert block.repair_rounds == 0

    with pytest.raises(RuntimeError):
        await loop.repair_single_block(block)

    assert block.repair_rounds == 1, "a failed repair attempt must consume a round"


@pytest.mark.asyncio
async def test_the_runner_scores_a_table_label_with_the_document_context() -> None:
    """The gate's scanner and its scorer must agree about "omitted".

    The draft stage binds ``set_document_context`` on its own filter and the gate
    screens blocks with that filter, but ``score_pairs`` runs through the
    runner's *internal* ``fast_pass`` — which was told the glossary and never
    the document. An uppercase table column label (``MASK | SAME | RND``) is not
    an identifier when the document uses those words as ordinary prose, but the
    context-free inner gate demanded the English surface and capped the block at
    0.35 while the gate itself had accepted it.
    """
    book = (
        "We mask the input. The same mask is applied. Random noise is added to each token. "
    ) * 2
    pair = [{"src": "MASK | SAME | RND", "mt": "掩码 | 相同 | 随机"}]

    unbound = HeuristicQERunner(source_lang="en", target_lang="zh")
    assert await unbound.score_pairs(pair) == [0.35], "the bug's signature"

    bound = HeuristicQERunner(source_lang="en", target_lang="zh")
    kept = bound.with_document_context(book)
    assert kept is bound, "the gate holds this reference and rebinds it in place"
    assert await bound.score_pairs(pair) == [0.92]

    # Rebinding the glossary must not drop the context: the rebound runner
    # keeps the *same* filter object, so whatever was bound onto it — the
    # document context here — is what the rebound runner scores with. (The
    # runner-level mirror used to be re-applied onto a fresh filter; the
    # single-owner filter made that carry unnecessary.)
    rebound = bound.with_glossary([{"source": "MASK", "target": "掩码"}])
    assert rebound.fast_pass is bound.fast_pass
    assert await rebound.score_pairs(pair) == [0.92]

    # Neither must a language rebind (that path still builds a fresh filter
    # for the new pair, so it carries the context the explicit way).
    relang = bound.with_languages("en", "ja")
    assert relang._document_text == book


def test_the_tiered_runner_forwards_the_document_context() -> None:
    """The judge scores semantics; omission lives in the heuristic leg."""
    from unittest.mock import MagicMock

    from ubt.core.qe.llm_judge import TieredQERunner

    heuristic = HeuristicQERunner(source_lang="en", target_lang="zh")
    tiered = TieredQERunner(heuristic=heuristic, judge=MagicMock())
    rebound = tiered.with_document_context("The SOTA model is described here.")

    # The heuristic leg binds in place and returns itself, so the tier does too:
    # the gate already holds the runner reference and expects no replacement.
    assert rebound is tiered
    assert rebound.heuristic._document_text == "The SOTA model is described here."


@pytest.mark.asyncio
async def test_judge_calls_counts_paid_model_calls_not_blocks() -> None:
    """One block is not one call: sentence judging makes it many.

    ``judge_calls += len(subset)`` counted *blocks*, but with
    ``qe_judge_sentences`` on, each sentence is its own ``_score_whole`` call —
    so an 8-sentence block is 8 paid requests. The counter is what the CLI
    prints as "N call(s)" and what ``assess`` prices at one call per block, so
    both the ROI reading and the quote were low by the sentence count.
    """
    from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

    calls: list[int] = []

    async def _judge(**_: object) -> str:
        calls.append(1)
        return '{"score": 0.6}'

    eight = (
        "Alpha one. Beta two. Gamma three. Delta four. "
        "Epsilon five. Zeta six. Eta seven. Theta eight."
    )
    eight_mt = "阿尔法一。贝塔二。伽马三。德尔塔四。艾普西龙五。泽塔六。伊塔七。西塔八。"

    runner = TieredQERunner(
        heuristic=HeuristicQERunner(source_lang="en", target_lang="zh"),
        judge=LLMJudgeQERunner(judge_fn=_judge),
        pass_sample=1.0,
    )
    await runner.score_pairs([{"src": eight, "mt": eight_mt}])

    assert len(calls) == 8, "eight sentences are eight paid calls"
    assert runner.judge_calls == len(calls), "the counter must match the real calls"

    # A single-sentence block is still one call, so the counter did not change
    # units into something else.
    calls.clear()
    one = TieredQERunner(
        heuristic=HeuristicQERunner(source_lang="en", target_lang="zh"),
        judge=LLMJudgeQERunner(judge_fn=_judge),
        pass_sample=1.0,
    )
    await one.score_pairs([{"src": "Alpha one.", "mt": "阿尔法一。"}])
    assert one.judge_calls == 1
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_wholly_failed_judge_batch_does_not_read_as_judged() -> None:
    """``_judged_any`` means "got an opinion", not "asked for one".

    It was set True before the results were inspected, so a batch where every
    call failed still marked the run as judged. The quality gate reads
    ``judged_this_run`` to decide whether to keep the batch's scores, so it then
    stored the 12-band heuristic values as ``mtqe_score`` — blocks no judge had
    looked at entered the reported distribution as graded verdicts.
    """
    from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

    async def _dead(**_: object) -> str:
        raise RuntimeError("provider down")

    failed = TieredQERunner(
        heuristic=HeuristicQERunner(source_lang="en", target_lang="zh"),
        judge=LLMJudgeQERunner(judge_fn=_dead),
        pass_sample=1.0,
    )
    await failed.score_pairs([{"src": "Alpha one.", "mt": "阿尔法一。"}])
    assert failed.judge_errors == 1
    assert failed.judged_this_run is False, "a dead judge produced no opinion"

    async def _ok(**_: object) -> str:
        return '{"score": 0.7}'

    working = TieredQERunner(
        heuristic=HeuristicQERunner(source_lang="en", target_lang="zh"),
        judge=LLMJudgeQERunner(judge_fn=_ok),
        pass_sample=1.0,
    )
    await working.score_pairs([{"src": "Alpha one.", "mt": "阿尔法一。"}])
    assert working.judged_this_run is True
