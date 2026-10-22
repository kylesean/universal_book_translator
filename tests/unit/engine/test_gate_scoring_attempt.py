"""P2-5: a scoring runner must actually be *used* by the gate.

A runner's ``is_calibrated()`` is False until its first batch answers (an honest
"not yet proven"). The quality gate tested ``is_calibrated()`` *before* deciding
whether to score, so the first call was never made and every block was stamped
``mtqe_score=None`` for the life of the run. The neural subprocess runner this
originally described was retired for license compliance, but the split it pins
is engine-agnostic and still correct.

The fix splits the two questions: ``may_be_calibrated()`` ("is a scoring call
worth attempting") gates the call, and ``is_calibrated()`` (the settled verdict)
decides whether the returned numbers are trusted. A runner that answers
non-calibrated still yields ``None`` scores.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock, make_element
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.heuristic import HeuristicQERunner, MockQERunner
from ubt.core.qe.llm_judge import TieredQERunner

pytestmark = pytest.mark.fast


class _SubprocessLikeRunner(BaseQERunner):
    """A calibration-protocol runner without the retired neural subprocess."""

    def __init__(self, *, neural: bool) -> None:
        self._requested_neural = neural
        self._last_engine: str | None = None
        self.score_calls = 0

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        self.score_calls += 1
        self._last_engine = "neural" if self._requested_neural else "heuristic_fallback"
        return [0.91] * len(pairs)

    def is_calibrated(self) -> bool:
        # Unproven until the first batch answers, exactly like the real runner.
        return self._last_engine == "neural"

    def may_be_calibrated(self) -> bool:
        # A call is worth attempting while the subprocess has not yet answered
        # non-neural.
        return self._last_engine is None or self._last_engine == "neural"


@dataclass
class _Ledger:
    blocks: list[IRBlock]
    batches: list[list[dict[str, Any]]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def fetch_blocks_by_status(self, _job: str, status: Any, **_kw: Any) -> list[IRBlock]:
        return [b for b in self.blocks if b.status is status]

    def save_checkpoints_batch(self, checkpoints: Any, **_kw: Any) -> int:
        self.batches.append(list(checkpoints))
        return len(checkpoints)

    def get_job_metadata_value(self, _job: str, key: str, **_kw: Any) -> Any:
        return self.metadata.get(key)

    def set_job_metadata_value(self, _job: str, key: str, value: Any, **_kw: Any) -> None:
        self.metadata[key] = value


def _block(block_id: str) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="The value is 3.5 percent.",
    )
    # An empty target fails FastPass, so the block is "suspicious" and reaches
    # the scoring branch.
    return IRBlock(element=element, status=BlockStatus.DRAFTED, target_text="")


class _Ctx:
    def __init__(self, ledger: _Ledger) -> None:
        self.ledger = ledger
        self.job_id = "job"
        self.target_lang = "zh"
        self.source_lang = "en"
        self.config = _Config()
        self.router = object()

    async def create_event(self, event_type: Any, *_a: Any, **kw: Any) -> Any:
        return _Event(event_type)


class _Config:
    qe_threshold = 0.5


@dataclass
class _Event:
    type: Any


@dataclass
class _Terminology:
    glossary_dicts: list[Any] = field(default_factory=list)
    abbreviation_entries: list[Any] = field(default_factory=list)


@dataclass
class _Services:
    qe_runner: BaseQERunner
    fast_pass: FastPassFilter
    qe_runner_explicit: bool = True

    def repair_loop_for(self, *_a: Any, **_kw: Any) -> Any:  # pragma: no cover
        return None


@dataclass
class _Scoring:
    qe_runner: BaseQERunner | None = None
    repair_loop: Any = None


async def _run_with_ledger(runner: BaseQERunner) -> tuple[list[dict[str, Any]], _Ledger]:
    ledger = _Ledger([_block("b1")])
    ctx = _Ctx(ledger)
    services = _Services(
        qe_runner=runner,
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
    )
    gen: AsyncIterator[Any] = run_quality_gate_stage(
        ctx,  # type: ignore[arg-type]
        services,  # type: ignore[arg-type]
        _Terminology(),  # type: ignore[arg-type]
        _Scoring(),  # type: ignore[arg-type]
    )
    async for _event in gen:
        pass
    return (ledger.batches[-1] if ledger.batches else []), ledger


async def _run(runner: BaseQERunner) -> list[dict[str, Any]]:
    updates, _ = await _run_with_ledger(runner)
    return updates


def test_a_neural_subprocess_runner_is_scored_on_the_first_batch() -> None:
    """The gate must attempt the first call, then trust the neural verdict."""
    runner = _SubprocessLikeRunner(neural=True)
    updates = asyncio.run(_run(runner))

    assert runner.score_calls == 1, "the gate never called the runner"
    assert updates, "no checkpoint was written"
    assert updates[0]["mtqe_score"] == 0.91


def test_a_heuristic_fallback_subprocess_runner_yields_no_pseudo_scores() -> None:
    """A non-neural answer still produces None — the honesty rule is intact."""
    runner = _SubprocessLikeRunner(neural=False)
    updates = asyncio.run(_run(runner))

    assert runner.score_calls == 1, "the gate must still make the attempt"
    assert updates[0]["mtqe_score"] is None


def test_the_gate_records_which_engine_produced_the_scores() -> None:
    """The audit artifact must name the scoring engine.

    The probe used to read ``last_engine``, an attribute only the retired
    CometKiwi subprocess set, so ``qe_score_source`` was never written and the
    report field was permanently ``None`` while the console fell back to a
    hard-coded label. Reading ``score_source()`` — part of the runner
    interface — makes the write unconditional.
    """
    runner = _SubprocessLikeRunner(neural=True)
    _, ledger = asyncio.run(_run_with_ledger(runner))

    assert ledger.metadata["qe_score_source"] == runner.score_source()


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (HeuristicQERunner(), "heuristic"),
        (MockQERunner(), "mock"),
    ],
)
def test_each_runner_names_its_own_engine(runner: BaseQERunner, expected: str) -> None:
    """The label is per-runner, not a single hard-coded string."""
    assert runner.score_source() == expected


def test_a_tiered_runner_reports_the_leg_that_answered() -> None:
    """A tier that never reached its judge must not claim a judge opinion."""
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=0.0)

    assert tier.score_source() == "heuristic"
