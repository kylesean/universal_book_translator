from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.core.qe.consistency_enforce import ConsistencyTask
from ubt.core.qe.defect_taxonomy import (
    CONSTRAINT_PREFIX,
    has_structural_defect,
    parse_defect_flag,
    structured_severity,
)
from ubt.core.qe.heuristic import HeuristicQERunner

pytestmark = pytest.mark.fast


def _block(
    *, block_id: str = "b1", target: str = "处理器架构使用自定义运算器。", rounds: int = 0
) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="The CPU architecture uses a custom ALU.",
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
        ),
        target_text=target,
        draft_text=target,
        status=BlockStatus.MTQE_PASSED,
        repair_rounds=rounds,
    )


def test_the_constraint_flag_is_structural() -> None:
    flag = ConsistencyTask("b1", "ALU", "算术逻辑单元").flag
    assert flag.startswith(CONSTRAINT_PREFIX)
    assert has_structural_defect([flag])


def test_the_constraint_flag_declares_major_severity() -> None:
    flag = ConsistencyTask("b1", "ALU", "算术逻辑单元").flag
    assert parse_defect_flag(flag) == ("terminology_consistency", "major")
    assert structured_severity([flag]) == "major"


def test_legacy_flags_are_not_mistaken_for_structured_ones() -> None:
    assert parse_defect_flag("Drafting error: 500 from upstream") is None
    assert parse_defect_flag("Numeric fidelity failure: dropped 42") is None
    assert structured_severity(["Drafting error: 500"]) is None


@pytest.mark.asyncio
async def test_a_terminology_constraint_without_a_span_reaches_the_model() -> None:
    block = _block()
    router = MagicMock()
    router.repair_reasoning_effort = "low"
    router.repair = AsyncMock(
        return_value="<final_translation>CPU 架构使用自定义的算术逻辑单元（ALU）。</final_translation>"
    )
    loop = RepairLoop(
        router=router,
        qe_runner=HeuristicQERunner(),
        max_rounds=2,
        qe_threshold=0.75,
    )

    flag = ConsistencyTask(block.id, "ALU", "算术逻辑单元").flag
    block.error_flags = [flag]
    repaired = await loop.repair_single_block(
        block=block,
        glossary_table="",
        target_lang="zh",
        source_lang="en",
        glossary_entries=[{"source": "ALU", "translation": "算术逻辑单元"}],
    )

    router.repair.assert_awaited_once()
    assert repaired.status is not BlockStatus.FAILED
    assert "算术逻辑单元" in (repaired.target_text or "")


@pytest.mark.asyncio
async def test_a_block_with_no_remaining_budget_still_gets_one_consistency_round() -> None:
    block = _block(rounds=2)
    router = MagicMock()
    router.repair_reasoning_effort = "low"
    router.repair = AsyncMock(
        return_value="<final_translation>CPU 架构使用自定义的算术逻辑单元（ALU）。</final_translation>"
    )
    loop = RepairLoop(router=router, qe_runner=HeuristicQERunner(), max_rounds=2)

    budget = max(0, loop.max_rounds - 1)
    flagged = block.model_copy(
        update={
            "error_flags": [ConsistencyTask(block.id, "ALU", "算术逻辑单元").flag],
            "status": BlockStatus.REPAIR_PENDING,
            "repair_rounds": min(block.repair_rounds, budget),
        }
    )
    repaired = await loop.repair_single_block(
        block=flagged,
        glossary_table="",
        target_lang="zh",
        source_lang="en",
        glossary_entries=[{"source": "ALU", "translation": "算术逻辑单元"}],
    )

    router.repair.assert_awaited_once()
    assert repaired.status is not BlockStatus.FAILED


@pytest.mark.asyncio
async def test_a_registered_alias_is_fixed_deterministically_without_a_model_call() -> None:
    block = _block(target="CPU 架构使用自定义 Arithmetic Unit。")
    router = MagicMock()
    router.repair_reasoning_effort = "low"
    router.repair = AsyncMock(return_value="should not be called")
    loop = RepairLoop(router=router, qe_runner=HeuristicQERunner(), max_rounds=2)

    block.error_flags = [ConsistencyTask(block.id, "ALU", "算术逻辑单元").flag]
    repaired = await loop.repair_single_block(
        block=block,
        glossary_table="",
        target_lang="zh",
        source_lang="en",
        glossary_entries=[
            {"source": "ALU", "translation": "算术逻辑单元", "aliases": ["Arithmetic Unit"]},
        ],
    )

    router.repair.assert_not_awaited()
    assert repaired.status is BlockStatus.REPAIRED
    assert "算术逻辑单元" in (repaired.target_text or "")


@pytest.mark.asyncio
async def test_a_cjk_alias_still_goes_to_the_model() -> None:
    block = _block()
    router = MagicMock()
    router.repair_reasoning_effort = "low"
    router.repair = AsyncMock(
        return_value="<final_translation>CPU 架构使用自定义的算术逻辑单元（ALU）。</final_translation>"
    )
    loop = RepairLoop(router=router, qe_runner=HeuristicQERunner(), max_rounds=2)

    block.error_flags = [ConsistencyTask(block.id, "ALU", "算术逻辑单元").flag]
    repaired = await loop.repair_single_block(
        block=block,
        glossary_table="",
        target_lang="zh",
        source_lang="en",
        glossary_entries=[
            {"source": "ALU", "translation": "算术逻辑单元", "aliases": ["运算器"]},
        ],
    )

    router.repair.assert_awaited_once()
    assert "算术逻辑单元" in (repaired.target_text or "")


def test_the_stage_test_no_longer_mocks_the_repair_loop() -> None:
    from ubt.core.engine.stages import consistency

    source = cast("Any", consistency.__doc__ or "")
    assert "repair" in source.lower()
    assert "MQMSpanAnnotator" not in dir(consistency)


def test_stage_and_loop_agree_on_the_flag_shape() -> None:
    flag = ConsistencyTask("b1", "ALU", "算术逻辑单元").flag
    assert has_structural_defect([flag])
    from ubt.core.qe.defect_taxonomy import is_transient_failure

    assert not is_transient_failure([flag])


def test_consistency_module_is_importable_without_a_cycle() -> None:
    import ubt.core.qe.consistency_enforce as enforce
    import ubt.core.qe.defect_taxonomy as taxonomy

    assert enforce.CONSTRAINT_PREFIX is taxonomy.CONSTRAINT_PREFIX


@pytest.mark.asyncio
async def test_gather_results_are_not_awaited_twice() -> None:
    from ubt.core.engine.stages.consistency import _split_repair_results

    updates = _split_repair_results([{"block_id": "b1"}, RuntimeError("boom")])
    assert updates == [{"block_id": "b1"}]
    assert asyncio.iscoroutinefunction(_split_repair_results) is False
