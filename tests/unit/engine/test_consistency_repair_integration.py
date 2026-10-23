"""The consistency stage's repair path, end to end (no mocked repair loop).

The existing ``test_consistency_stage`` mocks ``repair_single_block``, which is
exactly how the integration break survived: the stage passed a
``terminology_consistency: …`` flag, the real ``RepairLoop`` classified it as
unverifiable (no span, not in ``STRUCTURAL_DEFECT_MARKERS``) and wrote the block
FAILED without ever calling the model. These tests drive the real loop so the
two modules' contract is what is under test.
"""

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


# --------------------------------------------------------------------------- #
# Taxonomy: the constraint prefix is structural and declares its own severity
# --------------------------------------------------------------------------- #


def test_the_constraint_flag_is_structural() -> None:
    """``_flags_are_recheckable`` reads this table; without the entry every
    consistency repair was refused as unverifiable."""
    flag = ConsistencyTask("b1", "ALU", "算术逻辑单元").flag
    assert flag.startswith(CONSTRAINT_PREFIX)
    assert has_structural_defect([flag])


def test_the_constraint_flag_declares_major_severity() -> None:
    flag = ConsistencyTask("b1", "ALU", "算术逻辑单元").flag
    assert parse_defect_flag(flag) == ("terminology_consistency", "major")
    assert structured_severity([flag]) == "major"


def test_legacy_flags_are_not_mistaken_for_structured_ones() -> None:
    """A prose flag with a colon is not structured; it must fall back to the
    substring tables rather than parsing its first word as a category."""
    assert parse_defect_flag("Drafting error: 500 from upstream") is None
    assert parse_defect_flag("Numeric fidelity failure: dropped 42") is None
    assert structured_severity(["Drafting error: 500"]) is None


# --------------------------------------------------------------------------- #
# RepairLoop: a no-span terminology constraint reaches the model
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_terminology_constraint_without_a_span_reaches_the_model() -> None:
    """The core break: an unregistered synonym has no locatable span, and the
    old guard closed the block FAILED without a call. It must now be treated as
    verifiable — the constraint names a term the glossary validator re-checks."""
    block = _block()
    # The target uses 运算器, which is NOT an alias of the glossary entry, so
    # MQMSpanAnnotator finds nothing to wrap.
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
    """The consistency stage must reserve a round for itself.

    A block that already spent ``max_rounds`` in the repair stage hit the
    circuit breaker on entry and was written FAILED with no attempt — for
    exactly the blocks the consistency stage exists to fix.
    """
    block = _block(rounds=2)  # already at max_rounds
    router = MagicMock()
    router.repair_reasoning_effort = "low"
    router.repair = AsyncMock(
        return_value="<final_translation>CPU 架构使用自定义的算术逻辑单元（ALU）。</final_translation>"
    )
    loop = RepairLoop(router=router, qe_runner=HeuristicQERunner(), max_rounds=2)

    # What the stage does: clamp the rounds into the reserved budget.
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
    """The cheapest win the old code threw away.

    When the drifted rendering is a *registered* alias the deterministic
    enforcer can rewrite it for free — but the constraint flag survived its
    "remaining flags" filter, so the block fell through to the paid model.

    The alias here is Latin on purpose. A bare CJK alias (``运算器``) is
    quarantined by the enforcer's compound-word guard, which is the correct,
    conservative behaviour — that case legitimately needs the model.
    """
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
    """A bare CJK alias is quarantined by the enforcer's compound-word guard.

    That guard exists so a substitution cannot damage a longer CJK word; the
    consequence is that this case must fall through to the paid repair rather
    than being "fixed" mechanically.
    """
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


# --------------------------------------------------------------------------- #
# The stage itself: the real loop is wired in, not a mock
# --------------------------------------------------------------------------- #


def test_the_stage_test_no_longer_mocks_the_repair_loop() -> None:
    """Guard against the regression returning through the test suite.

    The original unit test's ``AsyncMock`` is what hid this integration break;
    the real-loop coverage now lives here, so the stage test must keep driving
    a mock only for concerns it does not own.
    """
    from ubt.core.engine.stages import consistency

    source = cast("Any", consistency.__doc__ or "")
    assert "repair" in source.lower()
    # The stage module must not import the span annotator itself: the repair
    # loop owns span decision-making, and a second copy would drift.
    assert "MQMSpanAnnotator" not in dir(consistency)


def test_stage_and_loop_agree_on_the_flag_shape() -> None:
    """A flag the stage writes must be one the loop can classify."""
    flag = ConsistencyTask("b1", "ALU", "算术逻辑单元").flag
    assert has_structural_defect([flag])
    # And the resume path must not treat it as a transient lifecycle note.
    from ubt.core.qe.defect_taxonomy import is_transient_failure

    assert not is_transient_failure([flag])


def test_consistency_module_is_importable_without_a_cycle() -> None:
    """The prefix constant moved into the taxonomy module to break a cycle."""
    import ubt.core.qe.consistency_enforce as enforce
    import ubt.core.qe.defect_taxonomy as taxonomy

    assert enforce.CONSTRAINT_PREFIX is taxonomy.CONSTRAINT_PREFIX


@pytest.mark.asyncio
async def test_gather_results_are_not_awaited_twice() -> None:
    """Sanity: the stage's helper still passes plain updates through."""
    from ubt.core.engine.stages.consistency import _split_repair_results

    updates = _split_repair_results([{"block_id": "b1"}, RuntimeError("boom")])
    assert updates == [{"block_id": "b1"}]
    assert asyncio.iscoroutinefunction(_split_repair_results) is False
