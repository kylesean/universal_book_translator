from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ubt.core.engine.stages.quality_gate import _audit_pass_sample, _measured_indices
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast

_SRC = "The model follows a query-based paradigm."
_MT = "该模型遵循基于查询的范式。"


def _block(block_id: str, source: str = _SRC) -> IRBlock:
    element = Paragraph(id=block_id, spine_index=1, span=Span(page=1), text=source)
    return IRBlock(element=element, target_text=_MT, status=BlockStatus.DRAFTED)


def _tiered(judge_fn: Any, *, pass_sample: float = 1.0) -> TieredQERunner:
    return TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
        pass_sample=pass_sample,
    )


def test_a_dead_judge_records_no_score() -> None:

    async def dead(**_kwargs: Any) -> str:
        raise RuntimeError("provider 500")

    blocks = [_block("p1")]
    demoted, scored = asyncio.run(_audit_pass_sample(_tiered(dead), blocks, 0.5))
    assert scored == []
    assert demoted == []
    assert blocks[0].mtqe_score is None


def test_a_live_judge_records_its_opinion() -> None:
    async def good(**_kwargs: Any) -> str:
        return "score: 88"

    blocks = [_block("p1")]
    _demoted, scored = asyncio.run(_audit_pass_sample(_tiered(good), blocks, 0.5))
    assert [b.id for b in scored] == ["p1"]
    assert blocks[0].mtqe_score == pytest.approx(0.88)


def test_a_partly_failed_batch_keeps_only_the_answered_entries() -> None:
    calls = {"n": 0}

    async def partial(**_kwargs: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "score: 88"
        raise RuntimeError("timeout")

    blocks = [_block("p1"), _block("p2", source="A second sentence about models.")]
    runner = _tiered(partial)
    _demoted, scored = asyncio.run(_audit_pass_sample(runner, blocks, 0.5))

    assert [b.id for b in scored] == ["p1"]
    assert blocks[0].mtqe_score == pytest.approx(0.88)
    assert blocks[1].mtqe_score is None
    assert _measured_indices(runner, 2) == frozenset({0})


def test_a_calibrated_runner_measures_every_entry() -> None:

    class _Calibrated(HeuristicQERunner):
        pass_sample = 1.0

        def is_calibrated(self) -> bool:
            return True

        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [0.9 for _ in pairs]

    runner = _Calibrated()
    blocks = [_block("p1"), _block("p2")]
    _demoted, scored = asyncio.run(_audit_pass_sample(runner, blocks, 0.5))
    assert [b.id for b in scored] == ["p1", "p2"]
    assert _measured_indices(runner, 2) == frozenset({0, 1})


def test_a_batch_level_runner_still_works() -> None:

    class _BatchOnly(HeuristicQERunner):
        judged_this_run = True

        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [0.8 for _ in pairs]

    assert _measured_indices(_BatchOnly(), 3) == frozenset({0, 1, 2})


def test_an_uncalibrated_unjudged_runner_measures_nothing() -> None:
    assert _measured_indices(HeuristicQERunner(), 3) == frozenset()


def test_a_judged_block_below_the_bar_is_still_demoted() -> None:

    async def harsh(**_kwargs: Any) -> str:
        return "score: 20"

    blocks = [_block("p1")]
    demoted, scored = asyncio.run(_audit_pass_sample(_tiered(harsh), blocks, 0.5))
    assert [b.id for b in demoted] == ["p1"]
    assert [b.id for b in scored] == ["p1"]
