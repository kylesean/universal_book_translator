"""The semantic layer: the only gate that can see a fluent compression.

Every zero-token gate scores a compressed translation as clean. Measured on
arXiv 2606.23344, two blocks shipped as ``MTQE_PASSED`` with content missing:

* b0010 — "the model weights are **publicly available** at …" became
  "模型权重可在 … **获取**" — an adjective and a past participle collapsed into one
  verb, so "you can obtain it" replaced "it is publicly available";
* b0015 — "content interference from background noise or adjacent elements"
  became "背景噪声或相邻元素的干扰" — the noun the sentence is *about* (内容) is gone.

Both read perfectly well and both are wrong. The tests below pin the two things
that let them through: that the judge is never consulted for a clean pass unless
a sample rate says so, and that the judge's rubric names the loss it must catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.qe.llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    LLMJudgeQERunner,
    TieredQERunner,
    build_judge_prompts,
)

pytestmark = pytest.mark.fast

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "measured_gate_false_positives.json"


def _semantic_defect_cases() -> list[dict[str, Any]]:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return [case for case in payload["cases"] if case.get("_defect") == "semantic-loss"]


def test_the_fixture_still_holds_the_defects_this_layer_exists_for() -> None:
    """The regression corpus must keep the two blocks that motivated the judge.

    If they disappear from the fixture, the tests below stop proving anything.
    """
    cases = _semantic_defect_cases()
    assert {case["block_id"] for case in cases} == {"pdf_main#b0010", "pdf_main#b0015"}
    # The loss is verifiable without an LLM: the target does not carry the
    # source's key content word at all.
    assert "公开" not in cases[0]["target_text"]
    assert "内容" not in cases[1]["target_text"]


@pytest.mark.asyncio
async def test_the_heuristic_cannot_see_either_defect() -> None:
    """Why a semantic layer is needed at all: the cheap gates are blind here."""
    runner = HeuristicQERunner()
    cases = _semantic_defect_cases()
    scores = await runner.score_pairs(
        [{"src": case["source_text"], "mt": case["target_text"]} for case in cases]
    )
    for case, score in zip(cases, scores, strict=True):
        assert score == pytest.approx(0.92), f"{case['block_id']} scored {score}"


def test_a_clean_pass_is_never_judged_without_a_sample_rate() -> None:
    """`pass_sample=0` means the judge never sees a 0.92 — the leak path.

    This is the behaviour that shipped the two defects: the LLM judge existed
    and was enabled, but a clean heuristic score short-circuited it.
    """
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=0.0)
    for case in _semantic_defect_cases():
        assert tier._needs_judge(0.92, case["source_text"]) is False


def test_a_clean_pass_is_judged_when_sampling_is_on() -> None:
    """`pass_sample=1.0` audits every clean pass — the setting that catches them."""
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=1.0)
    for case in _semantic_defect_cases():
        assert tier._needs_judge(0.92, case["source_text"]) is True


def test_a_hard_defect_is_not_sent_to_the_judge() -> None:
    """A judge cannot un-drop a number; those pairs cost tokens for nothing."""
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=1.0)
    assert tier._needs_judge(0.55, "any source text") is False


def test_the_rubric_names_compression_that_reads_well() -> None:
    """A generic "adequacy" rubric scored both defects as clean.

    The prompt must say, in so many words, that a fluent translation which drops
    a noun, modifier or relation is still a loss — otherwise a judge rewards the
    fluency and misses the omission.
    """
    lowered = JUDGE_SYSTEM_PROMPT.lower()
    assert "clause by clause" in lowered
    assert "compression" in lowered
    assert "drops source content" in lowered
    # The two measured shapes are named, so a future edit cannot quietly
    # generalize the instruction away.
    assert "publicly available" in JUDGE_SYSTEM_PROMPT
    assert "content interference" in JUDGE_SYSTEM_PROMPT


def test_the_judge_prompt_carries_both_sides_verbatim() -> None:
    """The judge can only catch a loss it can compare against."""
    system, user = build_judge_prompts("publicly available", "获取")
    assert "publicly available" in user
    assert "获取" in user
    assert "score:" in user
    assert system == JUDGE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_a_judge_that_sees_the_loss_lowers_the_score() -> None:
    """End to end through the real runner with a scripted judge.

    The judge is scripted (it reports what a real one should report after reading
    the rubric); what this test pins is the plumbing — the lowered score reaches
    the caller instead of being silently discarded.
    """
    case = _semantic_defect_cases()[0]

    async def judge_fn(**kwargs: Any) -> str:
        assert case["source_text"] in kwargs["user_prompt"]
        return "score: 55"

    runner = LLMJudgeQERunner(judge_fn=judge_fn).with_languages("en", "zh")
    scores = await runner.score_pairs([{"src": case["source_text"], "mt": case["target_text"]}])
    assert scores[0] == pytest.approx(0.55)
