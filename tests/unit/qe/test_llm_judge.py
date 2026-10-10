"""Unit tests for L3 LLM-as-Judge QE and TieredQERunner."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ubt.core.qe.comet_runner import (
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

    # score: N / 100
    assert parse_judge_score("score: 85") == 0.85
    assert parse_judge_score("Score: 92/100") == 0.92
    assert parse_judge_score("**score**: 75") == 0.75

    # JSON response
    assert parse_judge_score('{"score": 90}') == 0.90
    assert parse_judge_score('{"score": 0.88}') == 0.88

    # Bare number
    assert parse_judge_score("95") == 0.95
    assert parse_judge_score("80/100") == 0.80

    # Boundary edge: score of 1 means 1% (0.01), not 100%
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

    # Language rebinding
    rebound = runner.with_languages("fr", "de")
    assert rebound is not runner
    assert rebound._source_lang == "fr"
    assert rebound._target_lang == "de"


@pytest.mark.asyncio
async def test_llm_judge_qe_runner_handles_failure() -> None:
    mock_judge_fn = AsyncMock(side_effect=RuntimeError("API network error"))
    runner = LLMJudgeQERunner(judge_fn=mock_judge_fn)

    scores = await runner.score_pairs([{"src": "Hello", "mt": "你好"}])
    # Returns sentinel -1.0 on failure
    assert scores == [-1.0]


@pytest.mark.asyncio
async def test_tiered_qe_runner_gray_zone_routing() -> None:
    heuristic = HeuristicQERunner()
    mock_judge_fn = AsyncMock(return_value="score: 85")
    judge = LLMJudgeQERunner(judge_fn=mock_judge_fn)

    tiered = TieredQERunner(heuristic=heuristic, judge=judge, gray_low=0.7, gray_high=0.8)

    # Clean translation that passes heuristic cleanly does not trigger judge (pass_sample=0)
    scores = await tiered.score_pairs(
        [{"src": "This is a normal sentence.", "mt": "这是一句正常的话。"}]
    )
    assert scores == [QE_SCORE_PASS]
    assert tiered.judge_calls == 0

    # Pair with structural issue (which lands in QE_SCORE_STRUCTURAL_OTHER ~ 0.78)
    class FakeHeuristic(HeuristicQERunner):
        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [QE_SCORE_STRUCTURAL_OTHER]

    tiered_mocked = TieredQERunner(
        heuristic=FakeHeuristic(),
        judge=judge,
        gray_low=0.7,
        gray_high=0.8,
        allow_upgrade=True,
    )
    judged_scores = await tiered_mocked.score_pairs([{"src": "text", "mt": "text"}])
    assert tiered_mocked.judge_calls == 1
    # Upgrade allowed: adopts judge's 0.85
    assert judged_scores == [0.85]
