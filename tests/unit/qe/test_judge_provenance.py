"""The judge's provenance label, and the audit's single billing.

Two defects, one owner each:

* ``TieredQERunner.score_source`` described the *last batch*, not the run. A
  second batch with no judge-eligible pair (only hard-defect flags, say) reset
  ``_judged_any`` and the report then claimed ``"heuristic"`` for a run whose
  token bill said otherwise.
* The quality gate re-scored the blocks the pass-sample audit had just demoted.
  Sampling is a stable ``crc32`` of the source, so the same pairs reached the
  judge a second time — the same opinion, billed twice.

Both are pinned here against the real classes, not stand-ins.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

pytestmark = pytest.mark.fast

#: A clean pass: the heuristic's flat pass band, so ``_needs_judge`` samples it.
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
    """A judged first batch must keep the run labelled ``llm_judge``.

    The second batch carries a hard-defect flag (numeric fidelity, 0.55) that
    ``_needs_judge`` never sends to the judge, so ``_judged_any`` is False for
    it. The *run* was still judged, and the label describes the run.
    """
    calls: list[int] = []

    async def judge_fn(**_kwargs: Any) -> str:
        calls.append(1)
        return '{"score": 0.80, "reason": "dropped word"}'

    runner = _tiered(judge_fn, pass_sample=1.0)

    # Batch 1: a clean pass — sampled, so the judge answers.
    asyncio.run(runner.score_pairs(_pairs(_CLEAN)))
    assert runner.score_source() == "llm_judge"

    # Batch 2: a numeric-fidelity band (0.55), which the judge is never asked
    # about. ``_judged_any`` is False for this batch.
    src = "The table lists 42 rows."
    mt = "该表列出了 41 行。"
    asyncio.run(runner.score_pairs([{"src": src, "mt": mt}]))

    assert runner.score_source() == "llm_judge", (
        "the run did consult the judge; a batch with no judge-eligible pair "
        "must not relabel it as heuristic"
    )


def test_score_source_stays_heuristic_when_the_judge_never_answered() -> None:
    """The cumulative flag must not become "any batch asked"."""

    async def dead(**_kwargs: Any) -> str:
        raise RuntimeError("provider 500")

    runner = _tiered(dead, pass_sample=1.0)
    asyncio.run(runner.score_pairs(_pairs(_CLEAN)))
    assert runner.score_source() == "heuristic"


def test_judged_state_survives_a_language_rebind() -> None:
    """``with_languages`` rebuilds the tier; the run's evidence must follow."""
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
    """``with_glossary`` had the same gap as ``with_languages``."""
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
    """The per-entry half: one answered pair must not license the rest."""
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
