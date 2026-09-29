"""Tests for the terminology-consistency enforcement planner and stage."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from tests.block_builder import make_test_block
from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.stages.consistency import run_consistency_stage
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.qe.consistency_enforce import (
    CONSTRAINT_PREFIX,
    ConsistencyTask,
    plan_consistency_tasks,
)
from ubt.core.qe.term_metrics import TermHit, TermMetrics
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

_GLOSSARY = [{"source": "Working Memory", "translation": "工作记忆"}]


# -- planner ----------------------------------------------------------------


def _metrics(*hits: TermHit) -> TermMetrics:
    return TermMetrics(
        terms_expected=0,
        terms_rendered=0,
        term_precision=0.0,
        fuzzy_term_precision=0.0,
        term_recall=0.0,
        per_hit=hits,
    )


def test_planner_skips_exact_and_dedups() -> None:
    metrics = _metrics(
        TermHit("Working Memory", "工作记忆", "b2", exact=True, fuzzy=True),
        TermHit("Working Memory", "工作记忆", "b1", exact=False, fuzzy=True),
        TermHit("Working Memory", "工作记忆", "b1", exact=False, fuzzy=False),
    )
    tasks = plan_consistency_tasks(metrics, max_blocks=10)
    assert tasks == (ConsistencyTask(block_id="b1", source="Working Memory", expected="工作记忆"),)


def test_planner_is_stable_and_capped() -> None:
    metrics = _metrics(
        *[TermHit(f"Term{i}", "术语", f"b{i}", exact=False, fuzzy=False) for i in range(5)]
    )
    tasks = plan_consistency_tasks(metrics, max_blocks=3)
    assert [t.block_id for t in tasks] == ["b0", "b1", "b2"]
    assert plan_consistency_tasks(metrics, max_blocks=0) == ()


def test_task_flag_is_greppable_and_instructive() -> None:
    task = ConsistencyTask(block_id="b1", source="Working Memory", expected="工作记忆")
    assert task.flag.startswith(CONSTRAINT_PREFIX)
    assert "render 'Working Memory' as '工作记忆'" in task.flag


# -- stage ------------------------------------------------------------------


class _FakeLedger:
    def __init__(self, blocks: list[IRBlock]) -> None:
        self._blocks = blocks
        self.saved: list[dict[str, Any]] = []

    def get_all_blocks(self, job_id: str) -> list[IRBlock]:
        return list(self._blocks)

    def save_checkpoints_batch(self, updates: list[dict[str, Any]]) -> None:
        self.saved.extend(updates)


def _block(bid: str, source: str, target: str) -> IRBlock:
    return make_test_block(
        id=bid,
        source_text=source,
        draft_text=target,
        target_text=target,
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=0.8,
    )


def _repair_loop() -> RepairLoop:
    router = ModelRouter(provider=MockModelProvider(default_response="工作记忆"), max_retries=0)
    return RepairLoop(router=router, qe_runner=MockQERunner(default_score=0.95))


async def _event(*args: Any, **kwargs: Any) -> object:
    return object()


async def _run_stage(
    tmp_path: Path,
    fake: _FakeLedger,
    mode: Literal["report", "repair"],
    repair_loop: RepairLoop | None = None,
) -> list[object]:
    """One consistency pass under the given enforcement mode.

    The stage reads ``consistency_enforce`` / ``consistency_max_repairs`` off the
    job's config now, which is also what the pipeline does — so the mode is
    exercised through config rather than a shadow parameter.
    """
    events: list[object] = []
    async for event in run_consistency_stage(
        build_stage_ctx(
            tmp_path,
            ledger=cast(SQLiteJobLedger, fake),
            job_id="job_c",
            repair_loop=repair_loop or _repair_loop(),
            glossary_dicts=_GLOSSARY,
            target_lang="zh",
            source_lang="en",
            fast_pass=None,
            concurrency_sem=asyncio.Semaphore(2),
            create_event=_event,
            config=UBTConfig(consistency_enforce=mode, consistency_max_repairs=10),
        )
    ):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_report_mode_plans_without_touching_blocks(tmp_path: Path) -> None:
    fake = _FakeLedger([_block("b1", "Working Memory is central.", "工作内存是核心。")])
    events = await _run_stage(tmp_path, fake, mode="report")
    assert events == []
    assert fake.saved == []


@pytest.mark.asyncio
async def test_repair_mode_retranslates_only_the_drifted_block(
    tmp_path: Path,
) -> None:
    fake = _FakeLedger(
        [
            _block("b1", "Working Memory is central.", "工作内存是核心。"),
            _block("b2", "Working Memory holds data.", "工作记忆保存数据。"),
        ]
    )
    events = await _run_stage(tmp_path, fake, mode="repair")
    assert len(events) == 1
    saved = {update["block_id"]: update for update in fake.saved}
    assert saved["b1"]["target_text"] == "工作记忆"
    assert saved["b1"]["status"] is BlockStatus.REPAIRED
    assert "b2" not in saved  # already consistent: never re-translated


@pytest.mark.asyncio
async def test_repair_mode_skips_human_and_failed_blocks(tmp_path: Path) -> None:
    """A resumed ledger may already hold BLOCKED_HUMAN / NEEDS_HUMAN /
    FAILED blocks whose target is a quarantine placeholder (or a rejected
    draft). Re-translating them and stamping REPAIRED would drop them out of
    the human queue (triage later re-reads only REPAIR_PENDING/FAILED), letting
    a Critical ship as clean. They must be excluded from consistency."""

    def _blocked(bid: str, status: BlockStatus) -> IRBlock:
        return IRBlock(
            id=bid,
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Working Memory is central.",  # drifts vs '工作记忆' if evaluated
            draft_text="反应堆占位",
            target_text="【待人工审校】Working Memory is central.",
            status=status,
            mtqe_score=0.3,
        )

    fake = _FakeLedger(
        [
            _blocked("b_blocked", BlockStatus.BLOCKED_HUMAN),
            _blocked("b_needs", BlockStatus.NEEDS_HUMAN),
            _blocked("b_failed", BlockStatus.FAILED),
        ]
    )
    await _run_stage(tmp_path, fake, mode="repair")
    # None of the human/failed blocks were pulled back for repair.
    assert fake.saved == []


def test_planner_caps_by_block_and_keeps_all_its_terms() -> None:
    """The cap counts blocks, not tasks.

    Regression: ``consistency_max_repairs`` (blocks) was applied per
    ``(block, term)`` task, so one block with several drifted terms consumed the
    whole budget — later blocks were never planned, and a block whose tasks were
    cut off was repaired with an incomplete constraint set.
    """
    metrics = _metrics(
        TermHit("A", "甲", "b0", exact=False, fuzzy=False),
        TermHit("B", "乙", "b0", exact=False, fuzzy=False),
        TermHit("A", "甲", "b1", exact=False, fuzzy=False),
    )
    tasks = plan_consistency_tasks(metrics, max_blocks=1)
    assert [(t.block_id, t.source) for t in tasks] == [("b0", "A"), ("b0", "B")]


@pytest.mark.asyncio
async def test_repair_that_leaves_the_term_drifted_stays_pending(tmp_path: Path) -> None:
    """The terminology postcondition is checked independently of the engine.

    Regression: the constraint flag is not ``GLOSSARY_VIOLATION_MARKER``, so on a
    term-blind QE runner the repair loop laundered it and promoted a still-drifted
    block to REPAIRED — which triage never re-reads, so it shipped.
    """
    router = ModelRouter(provider=MockModelProvider(default_response="工作内存"), max_retries=0)
    drifted_loop = RepairLoop(router=router, qe_runner=MockQERunner(default_score=0.95))
    fake = _FakeLedger([_block("b1", "Working Memory is central.", "工作内存是核心。")])
    events = await _run_stage(tmp_path, fake, mode="repair", repair_loop=drifted_loop)
    assert len(events) == 1
    saved = {update["block_id"]: update for update in fake.saved}
    assert saved["b1"]["status"] is BlockStatus.REPAIR_PENDING
    assert any("Glossary term violation" in flag for flag in saved["b1"]["error_flags"])
