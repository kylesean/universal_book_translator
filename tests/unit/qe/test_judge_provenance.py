from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

pytestmark = pytest.mark.fast

_CLEAN = "The model follows a query-based paradigm."


def _tiered(judge_fn: Any, *, pass_sample: float = 1.0) -> TieredQERunner:
    return TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
        pass_sample=pass_sample,
    )


def _pairs(*sources: str) -> list[dict[str, str]]:
    return [{"src": s, "mt": "该模型遵循基于查询的范式。"} for s in sources]


def test_score_source_is_cumulative_across_batches() -> None:
    calls: list[int] = []

    async def judge_fn(**_kwargs: Any) -> str:
        calls.append(1)
        return '{"score": 0.80, "reason": "dropped word"}'

    runner = _tiered(judge_fn, pass_sample=1.0)

    asyncio.run(runner.score_pairs(_pairs(_CLEAN)))
    assert runner.score_source() == "llm_judge"

    src = "The table lists 42 rows."
    mt = "该表列出了 41 行。"
    asyncio.run(runner.score_pairs([{"src": src, "mt": mt}]))

    assert runner.score_source() == "llm_judge", (
        "the run did consult the judge; a batch with no judge-eligible pair "
        "must not relabel it as heuristic"
    )


def test_score_source_stays_heuristic_when_the_judge_never_answered() -> None:

    async def dead(**_kwargs: Any) -> str:
        raise RuntimeError("provider 500")

    runner = _tiered(dead, pass_sample=1.0)
    asyncio.run(runner.score_pairs(_pairs(_CLEAN)))
    assert runner.score_source() == "heuristic"


def test_judged_state_survives_a_language_rebind() -> None:
    calls: list[int] = []

    async def judge_fn(**_kwargs: Any) -> str:
        calls.append(1)
        return '{"score": 0.80, "reason": "dropped word"}'

    runner = _tiered(judge_fn, pass_sample=1.0)
    asyncio.run(runner.score_pairs(_pairs(_CLEAN)))
    rebound = runner.with_languages("en", "ja")

    assert rebound is not runner
    assert rebound.judge_calls == runner.judge_calls
    assert rebound.score_source() == "llm_judge"


def test_judged_state_survives_a_glossary_rebind() -> None:
    calls: list[int] = []

    async def judge_fn(**_kwargs: Any) -> str:
        calls.append(1)
        return '{"score": 0.80, "reason": "dropped word"}'

    runner = _tiered(judge_fn, pass_sample=1.0)
    asyncio.run(runner.score_pairs(_pairs(_CLEAN)))
    rebound = runner.with_glossary([{"source": "query", "translation": "查询"}])

    assert rebound is not runner
    assert rebound.score_source() == "llm_judge"


def test_a_judged_batch_marks_only_the_indices_it_answered() -> None:
    calls: list[int] = []
    answers = iter(['{"score": 0.80, "reason": "ok"}', "not json at all"])

    async def judge_fn(**_kwargs: Any) -> str:
        calls.append(1)
        return next(answers, "not json at all")

    runner = _tiered(judge_fn, pass_sample=1.0)
    asyncio.run(runner.score_pairs(_pairs(_CLEAN, "Another clean sentence here.")))

    judged = runner.judged_indices(2)
    assert len(judged) == 1, "exactly one pair produced a usable opinion"
    assert runner.score_source() == "llm_judge"
