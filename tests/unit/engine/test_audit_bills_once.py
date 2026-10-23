"""The pass-sample audit must bill the judge once per block.

``_audit_pass_sample`` scores a sample of FastPass passes. A block the judge
lowered below the line was then re-added to ``suspicious_blocks``, which the
gate scores again — and because the sampling is a stable ``crc32`` of the
source, ``_needs_judge`` selected the *same* pair a second time. The judge
answered twice, the bill recorded two calls, and the second opinion could only
agree with the first.

The fix persists the audit's own measurement (status REPAIR_PENDING + the score
it computed) instead of re-deriving it, so the block still reaches repair and
still carries the demotion evidence.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

pytestmark = pytest.mark.fast

#: A clean pass the heuristic scores at its flat 0.92 band, so ``pass_sample``
#: samples it and the judge gets asked.
_CLEAN_SRC = "The model follows a query-based paradigm."
_CLEAN_MT = "该模型遵循基于查询的范式。"


def _block(block_id: str) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=_CLEAN_SRC,
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
        ),
        target_text=_CLEAN_MT,
        status=BlockStatus.DRAFTED,
    )


@dataclass
class _Ledger:
    blocks: list[IRBlock]
    batches: list[list[dict[str, Any]]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def get_all_blocks(self, _job_id: str) -> list[IRBlock]:
        return list(self.blocks)

    def fetch_blocks_by_status(
        self, _job_id: str, status: BlockStatus, chapter_id: str | None = None
    ) -> list[IRBlock]:
        return [b for b in self.blocks if b.status is status]

    def save_checkpoints_batch(
        self, updates: list[dict[str, Any]], allow_terminal_override: bool = False
    ) -> int:
        self.batches.append(list(updates))
        return len(updates)

    def get_job_metadata_value(self, _job_id: str, key: str) -> Any:
        return self.metadata.get(key)

    def set_job_metadata_value(self, _job_id: str, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get_job_stats(self, _job_id: str) -> dict[str, int]:
        return {"total": len(self.blocks)}


@dataclass
class _Config:
    qe_threshold: float = 0.75


@dataclass
class _Ctx:
    ledger: _Ledger
    job_id: str = "job"
    target_lang: str = "zh"
    source_lang: str = "en"
    is_mock_run: bool = False
    config: _Config = field(default_factory=_Config)

    async def create_event(self, *_a: Any, **_kw: Any) -> Any:
        return object()


@dataclass
class _Services:
    qe_runner: BaseQERunner
    fast_pass: FastPassFilter
    qe_runner_explicit: bool = True
    concurrency_sem: Any = None

    def repair_loop_for(self, *_a: Any, **_kw: Any) -> Any:  # pragma: no cover
        return None


@dataclass
class _Scoring:
    qe_runner: BaseQERunner | None = None
    repair_loop: Any = None


@dataclass
class _Terminology:
    glossary_dicts: list[dict[str, Any]] = field(default_factory=list)
    abbreviation_entries: list[dict[str, Any]] = field(default_factory=list)


def _counting_tiered() -> tuple[TieredQERunner, list[str]]:
    """A tiered runner whose judge answers every call and records each one."""
    calls: list[str] = []

    async def judge_fn(*, user_prompt: str = "", **_kwargs: Any) -> str:
        calls.append(user_prompt)
        return '{"score": 0.40, "reason": "dropped a word"}'

    runner = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
        pass_sample=1.0,
    )
    return runner, calls


def _run(runner: BaseQERunner) -> tuple[list[list[dict[str, Any]]], _Ledger]:
    ledger = _Ledger([_block("b1")])
    ctx = _Ctx(ledger)
    services = _Services(
        qe_runner=runner,
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
    )

    async def drive() -> None:
        gen: AsyncIterator[Any] = run_quality_gate_stage(
            ctx,  # type: ignore[arg-type]
            services,  # type: ignore[arg-type]
            _Terminology(),  # type: ignore[arg-type]
            _Scoring(),  # type: ignore[arg-type]
        )
        async for _event in gen:
            pass

    asyncio.run(drive())
    return ledger.batches, ledger


def test_a_demoted_block_is_judged_once() -> None:
    """The regression: the audit's demotion was re-scored by the gate."""
    runner, calls = _counting_tiered()
    _batches, _ledger = _run(runner)

    assert len(calls) == 1, f"the judge was asked {len(calls)} times for one block"
    assert runner.judge_calls == 1


def test_a_demoted_block_still_reaches_repair_with_its_score() -> None:
    """Billing once must not mean losing the verdict."""
    runner, _calls = _counting_tiered()
    batches, _ledger = _run(runner)

    flattened = [update for batch in batches for update in batch]
    demoted = [u for u in flattened if u.get("block_id") == "b1"]
    assert demoted, "no checkpoint was written for the demoted block"
    final = demoted[-1]
    assert final["status"] is BlockStatus.REPAIR_PENDING
    assert final["mtqe_score"] is not None and final["mtqe_score"] < 0.75


def test_a_passing_audit_still_records_the_measured_score() -> None:
    """The other half of the audit: a good sample keeps its measurement."""

    async def high(**_kwargs: Any) -> str:
        return '{"score": 0.95, "reason": "faithful"}'

    runner = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=high),
        pass_sample=1.0,
    )
    batches, _ledger = _run(runner)
    flattened = [update for batch in batches for update in batch]
    passed = [u for u in flattened if u.get("status") is BlockStatus.MTQE_PASSED]
    assert passed, "the audited pass was not recorded"
    assert passed[-1]["mtqe_score"] == pytest.approx(0.95, abs=0.06)
