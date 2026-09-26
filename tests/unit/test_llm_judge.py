"""Unit tests for L3 LLM-as-Judge + TieredQERunner gray-zone routing."""

import pytest

from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.core.qe.llm_judge import (
    LLMJudgeQERunner,
    TieredQERunner,
    parse_judge_score,
)


def test_parse_judge_score_formats() -> None:
    assert parse_judge_score("score: 85") == 0.85
    assert parse_judge_score("Score=42.5") == 0.425
    assert parse_judge_score('{"score": 90}') == 0.9
    assert parse_judge_score('{"score": 0.73}') == 0.73
    assert parse_judge_score("0.61") == 0.61
    assert parse_judge_score("garbage no number here!!!") is None
    assert parse_judge_score("") is None


def test_parse_judge_score_one_is_one_percent_not_perfect() -> None:
    """The judge prompt asks for 0-100, so "1" means 1/100 — not a perfect 1.0.

    Normalizing a bare 1 to 1.0 let a near-failing gray-zone block auto-pass
    under qe_judge_allow_upgrade.
    """
    assert parse_judge_score("score: 1") == 0.01
    assert parse_judge_score("1") == 0.01
    assert parse_judge_score('{"score": 1}') == 0.01
    # A fractional reply on the 0-1 scale is still read as a fraction.
    assert parse_judge_score("score: 0.95") == 0.95
    assert parse_judge_score("score: 0") == 0.0


@pytest.mark.asyncio
async def test_llm_judge_scores_and_signals_no_opinion_on_failure() -> None:
    async def fake_judge(**kwargs: object) -> str:
        user = str(kwargs.get("user_prompt", ""))
        if "good-src" in user:
            return "score: 88"
        if "bad-reply" in user:
            return "not a score at all!!!"
        raise RuntimeError("boom")

    runner = LLMJudgeQERunner(judge_fn=fake_judge)
    scores = await runner.score_pairs(
        [
            {"src": "good-src hello", "mt": "你好"},
            {"src": "bad-reply hello", "mt": "你好"},
            {"src": "explode hello", "mt": "你好"},
        ]
    )
    assert scores[0] == 0.88
    # On unparsable reply / call failure the judge returns a sentinel so the
    # Tiered runner can preserve the heuristic score instead of
    # silently adopting a 0.5 fallback that lands exactly on the pass line.
    assert scores[1] == -1.0
    assert scores[2] == -1.0


@pytest.mark.asyncio
async def test_tiered_only_gray_zone_hits_judge() -> None:
    heuristic = HeuristicQERunner()
    calls: list[str] = []

    async def fake_judge(**kwargs: object) -> str:
        calls.append(str(kwargs.get("user_prompt", "")))
        return "score: 77"

    judge = LLMJudgeQERunner(judge_fn=fake_judge)
    tiered = TieredQERunner(heuristic=heuristic, judge=judge, allow_upgrade=True)

    good_src = "Psychological research shows that sleep deprivation impairs cognition."
    good_mt = "心理学研究表明，睡眠不足会损害认知能力。"
    # Gray zone input: math span mismatch maps to the heuristic "other" band
    # (0.70), which lands inside [0.7, 0.8) — the only band the judge reviews.
    gray_src = "The total energy is $E = mc^2$ according to physics."
    gray_mt = "总能量符合物理学规律。"
    # Below the gray zone after (gray_low=0.7): the numeric band 0.55
    # is a deterministic failure and must NOT spend a judge call.
    numeric_src = "Founded in 1998 with 500 members."
    numeric_mt = "该组织成立较早，拥有许多成员。"
    # Heuristic fail with non-gray score (empty -> 0.0)
    bad_src = "Hello world, this is a test."
    bad_mt = ""

    scores = await tiered.score_pairs(
        [
            {"src": good_src, "mt": good_mt},
            {"src": gray_src, "mt": gray_mt},
            {"src": numeric_src, "mt": numeric_mt},
            {"src": bad_src, "mt": bad_mt},
        ]
    )
    assert scores[0] == 0.92
    # With allow_upgrade=True (default), the LLM Judge score applies to the gray zone.
    assert scores[1] == 0.77
    assert scores[2] == 0.55
    assert scores[3] == 0.0
    assert len(calls) == 1
    assert tiered.judge_calls == 1


@pytest.mark.asyncio
async def test_judge_cannot_raise_score_above_heuristic() -> None:
    """When allow_upgrade=False (conservative mode), judge cannot raise score above heuristic."""
    heuristic = HeuristicQERunner()

    async def over_confident_judge(**kwargs: object) -> str:
        return "score: 95"

    judge = LLMJudgeQERunner(judge_fn=over_confident_judge)
    tiered = TieredQERunner(heuristic=heuristic, judge=judge, allow_upgrade=False)

    # Gray zone: math span mismatch -> heuristic 0.70, inside [0.7, 0.8)
    gray_src = "The total energy is $E = mc^2$ according to physics."
    gray_mt = "总能量符合物理学规律。"
    scores = await tiered.score_pairs([{"src": gray_src, "mt": gray_mt}])
    # In conservative mode, min(0.70, 0.95) = 0.70.
    assert scores[0] == 0.70


@pytest.mark.asyncio
async def test_judge_can_rescue_gray_zone_block() -> None:
    """By default, a justified high judge score rescues a borderline 0.70 block."""
    heuristic = HeuristicQERunner()

    async def favorable_judge(**kwargs: object) -> str:
        return "score: 88"

    judge = LLMJudgeQERunner(judge_fn=favorable_judge)
    tiered = TieredQERunner(heuristic=heuristic, judge=judge, allow_upgrade=True)

    gray_src = "The total energy is $E = mc^2$ according to physics."
    gray_mt = "总能量符合物理学规律。"
    scores = await tiered.score_pairs([{"src": gray_src, "mt": gray_mt}])
    assert scores[0] == 0.88


@pytest.mark.asyncio
async def test_judge_one_does_not_rescue_gray_zone_block() -> None:
    """A judge reply of 1 (1%) must not upgrade a near-failing gray block to pass."""
    heuristic = HeuristicQERunner()

    async def near_failing_judge(**kwargs: object) -> str:
        return "score: 1"

    judge = LLMJudgeQERunner(judge_fn=near_failing_judge)
    tiered = TieredQERunner(heuristic=heuristic, judge=judge, allow_upgrade=True)

    gray_src = "The total energy is $E = mc^2$ according to physics."
    gray_mt = "总能量符合物理学规律。"
    scores = await tiered.score_pairs([{"src": gray_src, "mt": gray_mt}])
    assert scores[0] == 0.01
    assert scores[0] != 1.0


@pytest.mark.asyncio
async def test_judge_failure_preserves_heuristic_and_records_error() -> None:
    """Judge failure keeps the original heuristic score (no silent
    0.5 fallback) and is observable via judge_errors rather than swallowed."""
    heuristic = HeuristicQERunner()

    async def exploding_judge(**kwargs: object) -> str:
        raise RuntimeError("judge down")

    judge = LLMJudgeQERunner(judge_fn=exploding_judge)
    tiered = TieredQERunner(heuristic=heuristic, judge=judge)

    # Gray zone: math span mismatch -> heuristic 0.70, inside [0.7, 0.8)
    gray_src = "The total energy is $E = mc^2$ according to physics."
    gray_mt = "总能量符合物理学规律。"
    scores = await tiered.score_pairs([{"src": gray_src, "mt": gray_mt}])
    # heuristic 0.70 preserved; judge call still counted but error recorded
    assert scores[0] == 0.70
    assert tiered.judge_calls == 1
    assert tiered.judge_errors == 1


@pytest.mark.asyncio
async def test_tiered_without_judge_is_passthrough() -> None:
    heuristic = HeuristicQERunner()
    tiered = TieredQERunner(heuristic=heuristic, judge=None)
    scores = await tiered.score_pairs([{"src": "a", "mt": ""}])
    assert scores == [0.0]


def test_standalone_llm_judge_is_not_calibrated_for_reranking() -> None:
    async def judge(**kwargs: object) -> str:
        return "score: 90"

    assert LLMJudgeQERunner(judge_fn=judge).is_calibrated() is False


def test_pipeline_wiring_judge_enabled() -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.engine.pipeline import PipelineOrchestrator
    from ubt.core.qe.comet_runner import HeuristicQERunner
    from ubt.core.qe.llm_judge import TieredQERunner

    cfg_off = UBTConfig(qe_engine="heuristic")
    assert isinstance(PipelineOrchestrator(config=cfg_off).qe_runner, HeuristicQERunner)

    cfg_on = UBTConfig(qe_engine="heuristic")
    cfg_on.qe_judge_enabled = True
    assert isinstance(PipelineOrchestrator(config=cfg_on).qe_runner, TieredQERunner)

    cfg_tiered = UBTConfig(qe_engine="tiered")
    assert isinstance(PipelineOrchestrator(config=cfg_tiered).qe_runner, HeuristicQERunner)


def test_build_judge_prompts_labels_both_languages() -> None:
    """P1-6: the source language must not be hardcoded to 'en'."""
    from ubt.core.qe.llm_judge import build_judge_prompts

    _, user = build_judge_prompts("Bonjour", "こんにちは", target_lang="ja", source_lang="fr")
    assert "Source (fr):" in user
    assert "Translation (ja):" in user
    assert "Source (en):" not in user


@pytest.mark.asyncio
async def test_tiered_with_languages_rebinds_judge() -> None:
    """P1-6: TieredQERunner.with_languages previously rebound only the heuristic
    and passed the judge through, so a non-en/zh run judged with a mislabeled
    (en/zh) prompt. The judge must be rebound too."""
    captured: list[str] = []

    async def fake_judge(**kwargs: object) -> str:
        captured.append(str(kwargs.get("user_prompt", "")))
        return "score: 75"

    heuristic = HeuristicQERunner()
    judge = LLMJudgeQERunner(judge_fn=fake_judge)
    tiered = TieredQERunner(heuristic=heuristic, judge=judge, gray_low=0.0, gray_high=1.0)

    rebound = tiered.with_languages("fr", "ja")
    assert rebound is not tiered
    await rebound.score_pairs([{"src": "Bonjour le monde", "mt": "こんにちは世界"}])
    assert captured, "judge was not called"
    assert "Source (fr):" in captured[0]
    assert "Translation (ja):" in captured[0]


def test_tiered_with_languages_preserves_judge_counters() -> None:
    """The pipeline rebinds the runner per run to set language labels; the
    paid-judge counters must survive or the report always reads zero."""
    heuristic = HeuristicQERunner()
    tiered = TieredQERunner(heuristic=heuristic, judge=None)
    tiered.judge_calls = 7
    tiered.judge_errors = 2

    rebound = tiered.with_languages("fr", "ja")

    assert rebound is not tiered
    assert rebound.judge_calls == 7
    assert rebound.judge_errors == 2


def test_tiered_is_calibrated_delegates_to_heuristic() -> None:
    """The tier scores every pair through its heuristic leg (the judge only
    rescores gray bands and sampled passes), so inheriting BaseQERunner's
    calibrated default re-enabled best-of-n rerank on flat 0.92 bands — the
    exact fake-calibration the is_calibrated gate exists to block."""
    tiered = TieredQERunner(heuristic=HeuristicQERunner(), judge=None)
    assert tiered.is_calibrated() is False


def test_llm_judge_markdown_bold_and_fractional_scores() -> None:
    from ubt.core.qe.llm_judge import parse_judge_score

    # Markdown bold formatting from LLM replies
    assert parse_judge_score("**Score**: 90") == 0.90
    assert parse_judge_score("**Score:** 85") == 0.85
    assert parse_judge_score("**score**: 75") == 0.75

    # Fractional scores with explicit denominator (e.g. out of 10)
    assert parse_judge_score("score: 9.5/10") == 0.95
    assert parse_judge_score("score: 9/10") == 0.90
    assert parse_judge_score("score: 4/5") == 0.80


@pytest.mark.fast
def test_tiered_runner_binds_glossary_to_heuristic() -> None:
    from ubt.core.qe.comet_runner import HeuristicQERunner
    from ubt.core.qe.llm_judge import TieredQERunner

    tiered = TieredQERunner(heuristic=HeuristicQERunner())
    assert tiered.is_glossary_aware() is False

    bound = tiered.with_glossary([{"source": "term", "target": "术语"}])
    assert bound.is_glossary_aware() is True


@pytest.mark.asyncio
async def test_paid_judge_follows_defect_class_and_sampled_passes() -> None:
    from ubt.core.qe.comet_runner import (
        QE_SCORE_EMPTY,
        QE_SCORE_LEAK,
        QE_SCORE_PASS,
        QE_SCORE_STRUCTURAL_OTHER,
        HeuristicQERunner,
    )
    from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

    class _ClassedHeuristic(HeuristicQERunner):
        """Emit a chosen defect class per pair instead of deriving it from content."""

        def __init__(self, scores: list[float]) -> None:
            super().__init__()
            self._scores = scores

        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return self._scores[: len(pairs)]

    scores = [QE_SCORE_PASS, QE_SCORE_STRUCTURAL_OTHER, QE_SCORE_LEAK, QE_SCORE_EMPTY]
    pairs = [{"src": f"seg-{i}", "mt": "译文"} for i in range(len(scores))]
    judged: list[str] = []

    async def fake_judge(**kwargs: object) -> str:
        judged.append(str(kwargs.get("user_prompt", "")))
        return "score: 30"

    def _tiered(pass_sample: float) -> TieredQERunner:
        return TieredQERunner(
            heuristic=_ClassedHeuristic(scores),
            judge=LLMJudgeQERunner(judge_fn=fake_judge),
            pass_sample=pass_sample,
        )

    runner = _tiered(0.0)
    out = await runner.score_pairs(pairs)
    # Only the unclassified structural class is ambiguous enough to be worth asking.
    assert len(judged) == 1 and runner.judge_calls == 1
    assert out[0] == QE_SCORE_PASS, "a clean pass must not move without being sampled"

    judged.clear()
    runner = _tiered(1.0)
    out = await runner.score_pairs(pairs)
    # Sampling passes lets the judge lower a pass that merely broke no invariant;
    # hard defects still never reach it (a judge cannot un-drop a number).
    assert len(judged) == 2 and runner.judge_calls == 2
    assert out[0] == 0.30 and out[2] == QE_SCORE_LEAK and out[3] == QE_SCORE_EMPTY
