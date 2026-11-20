from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ubt.core.qe.heuristic import (
    QE_SCORE_PASS,
    QE_SCORE_STRUCTURAL_OTHER,
    HeuristicQERunner,
)
from ubt.core.qe.llm_judge import (
    LLMJudgeQERunner,
    TieredQERunner,
    build_judge_prompts,
    parse_judge_score,
)

pytestmark = pytest.mark.fast


def test_parse_judge_score_formats() -> None:
    assert parse_judge_score("") is None
    assert parse_judge_score("unparsable output here") is None

    assert parse_judge_score("score: 85") == 0.85
    assert parse_judge_score("Score: 92/100") == 0.92
    assert parse_judge_score("**score**: 75") == 0.75

    assert parse_judge_score('{"score": 90}') == 0.90
    assert parse_judge_score('{"score": 0.88}') == 0.88

    assert parse_judge_score("95") == 0.95
    assert parse_judge_score("80/100") == 0.80

    assert parse_judge_score("score: 1") == 0.01
    assert parse_judge_score("score: 100") == 1.0
    assert parse_judge_score("score: 0") == 0.0


def test_build_judge_prompts() -> None:
    system, user = build_judge_prompts(
        "Hello world", "你好，世界", target_lang="zh", source_lang="en"
    )
    assert "translation quality estimator" in system
    assert "Source (en):\nHello world" in user
    assert "Translation (zh):\n你好，世界" in user


@pytest.mark.asyncio
async def test_llm_judge_qe_runner_scoring() -> None:
    mock_judge_fn = AsyncMock(return_value="score: 88")
    runner = LLMJudgeQERunner(judge_fn=mock_judge_fn, model="gpt-4o-mini")

    scores = await runner.score_pairs([{"src": "Hello", "mt": "你好"}])
    assert scores == [0.88]
    assert runner.is_calibrated() is False

    rebound = runner.with_languages("fr", "de")
    assert rebound is not runner
    assert rebound._source_lang == "fr"
    assert rebound._target_lang == "de"


@pytest.mark.asyncio
async def test_llm_judge_qe_runner_handles_failure() -> None:
    mock_judge_fn = AsyncMock(side_effect=RuntimeError("API network error"))
    runner = LLMJudgeQERunner(judge_fn=mock_judge_fn)

    scores = await runner.score_pairs([{"src": "Hello", "mt": "你好"}])
    assert scores == [-1.0]


@pytest.mark.asyncio
async def test_tiered_qe_runner_gray_zone_routing() -> None:
    heuristic = HeuristicQERunner()
    mock_judge_fn = AsyncMock(return_value="score: 85")
    judge = LLMJudgeQERunner(judge_fn=mock_judge_fn)

    tiered = TieredQERunner(heuristic=heuristic, judge=judge)

    scores = await tiered.score_pairs(
        [{"src": "This is a normal sentence.", "mt": "这是一句正常的话。"}]
    )
    assert scores == [QE_SCORE_PASS]
    assert tiered.judge_calls == 0

    class FakeHeuristic(HeuristicQERunner):
        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [QE_SCORE_STRUCTURAL_OTHER]

    tiered_mocked = TieredQERunner(heuristic=FakeHeuristic(), judge=judge)
    judged_scores = await tiered_mocked.score_pairs([{"src": "text", "mt": "text"}])
    assert tiered_mocked.judge_calls == 1
    assert judged_scores == [QE_SCORE_STRUCTURAL_OTHER]


@pytest.mark.asyncio
async def test_two_tiers_sharing_a_judge_do_not_share_the_call_counter() -> None:
    """A tier must count only its own batch, even when both use one judge.

    The old implementation installed ``judge._on_call = self._count_judge_call``
    in the constructor, so the second tier overwrote the first's callback and
    both reported the other's paid calls. Reading ``calls_made`` as a delta
    leaves the shared judge untouched.
    """
    calls: list[int] = []

    async def _judge(**_: object) -> str:
        calls.append(1)
        return '{"score": 0.6}'

    judge = LLMJudgeQERunner(judge_fn=_judge)
    first = TieredQERunner(heuristic=HeuristicQERunner(), judge=judge, pass_sample=1.0)
    second = TieredQERunner(heuristic=HeuristicQERunner(), judge=judge, pass_sample=1.0)

    await first.score_pairs([{"src": "Alpha one.", "mt": "阿尔法一。"}])
    assert first.judge_calls == 1
    assert second.judge_calls == 0
    assert calls == [1]

    await second.score_pairs([{"src": "Beta two.", "mt": "贝塔二。"}])
    assert first.judge_calls == 1
    assert second.judge_calls == 1
    assert calls == [1, 1]
