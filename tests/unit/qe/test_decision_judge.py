"""Unit tests for DecisionQERunner (Gemini Decisions API / Jev)."""

from __future__ import annotations

from typing import Any

import pytest

from ubt.core.qe.decision_judge import DecisionQERunner, _extract_decision_scores
from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.qe.llm_judge import TieredQERunner

pytestmark = pytest.mark.fast


def test_extract_decision_scores_supports_scalar_probs_and_choices() -> None:
    payload = {
        "decisions": [
            {"score": 0.91},
            {"score": 82},
            {
                "probabilities": {
                    "accurate_faithful_fluent": 0.8,
                    "minor_style_or_fluency_issue": 0.2,
                }
            },
            {
                "choices": [
                    {"label": "critical_corruption_or_untranslated", "probability": 1.0},
                ]
            },
        ]
    }
    scores = _extract_decision_scores(payload, 4)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(0.91)
    assert scores[1] == pytest.approx(0.82)
    assert scores[2] == pytest.approx(0.8 * 0.95 + 0.2 * 0.78)
    assert scores[3] == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_decision_qe_runner_gemini_batches_and_counts_calls() -> None:
    calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    async def fake_post(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append((url, headers, payload))
        n = len(payload["questions"])
        return {"decisions": [{"score": 0.93} for _ in range(n)]}

    counted = 0

    def on_call() -> None:
        nonlocal counted
        counted += 1

    runner = DecisionQERunner(
        backend="gemini_decision",
        api_key="test-key",
        batch_size=2,
        post_fn=fake_post,
        on_call=on_call,
    )
    assert runner.is_calibrated() is True
    assert runner.score_source() == "decision_judge:gemini_decision"

    pairs = [
        {"src": "Hello world", "mt": "你好世界"},
        {"src": "Second sentence", "mt": "第二句话"},
        {"src": "Third sentence", "mt": "第三句话"},
    ]
    scores = await runner.score_pairs(pairs)
    assert scores == pytest.approx([0.93, 0.93, 0.93])
    assert counted == 2
    assert len(calls) == 2
    assert calls[0][0].endswith("/models/gemini-decision-1:decide")
    assert calls[0][1]["x-goog-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_decision_qe_runner_jev_and_tiered_integration() -> None:
    async def fake_post(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        assert url.endswith("/decide")
        assert headers["Authorization"] == "Bearer jev-secret"
        n = len(payload["items"])
        return {"results": [{"score": 0.88} for _ in range(n)]}

    decision_judge = DecisionQERunner(
        backend="jev",
        api_key="jev-secret",
        post_fn=fake_post,
    )
    tiered = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=decision_judge,
        pass_sample=1.0,
    )
    scores = await tiered.score_pairs(
        [{"src": "Neural networks learn representations.", "mt": "神经网络学习表征。"}]
    )
    assert len(scores) == 1
    assert scores[0] <= 0.88
    assert tiered.judge_calls == 1


@pytest.mark.asyncio
async def test_decision_qe_runner_degrades_gracefully_on_http_error() -> None:
    async def failing_post(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise RuntimeError("network timeout")

    runner = DecisionQERunner(backend="gemini_decision", post_fn=failing_post)
    scores = await runner.score_pairs([{"src": "a", "mt": "甲"}])
    assert scores == [0.5]
