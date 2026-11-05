from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, patch

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.facts import Scoring, Terminology
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.consistency import run_consistency_stage
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_run_consistency_stage_passes_abbreviations_to_table(tmp_path: Path) -> None:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="The CPU architecture uses a custom ALU.",
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
        ),
        target_text="处理器架构使用自定义运算器。",
        status=BlockStatus.MTQE_PASSED,
        error_flags=[],
    )

    class FakeLedger:
        def get_all_blocks(self, job_id: str) -> list[IRBlock]:
            return [block]

        def save_checkpoints_batch(
            self, updates: list[dict[str, Any]], allow_terminal_override: bool = False
        ) -> None:
            pass

    config = SimpleNamespace(
        consistency_enforce="repair",
        consistency_max_repairs=5,
    )

    async def fake_create_event(*args: Any, **kwargs: Any) -> TranslationProgressEvent:
        return SimpleNamespace()  # type: ignore[return-value]

    ctx = StageContext(
        config=cast("UBTConfig", config),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", FakeLedger()),
        manifest=cast("Any", SimpleNamespace(title="Test", source_path="test.pdf")),
        job_id="test-job",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=fake_create_event,
    )

    mock_repair_loop = SimpleNamespace()
    mock_repair_loop.repair_single_block = AsyncMock(
        return_value=block.model_copy(
            update={
                "status": BlockStatus.REPAIRED,
                "target_text": "CPU 架构使用自定义算术逻辑单元 (ALU)。",
                "error_flags": [],
            }
        )
    )

    services = cast(
        "RunServices",
        SimpleNamespace(
            repair_loop=mock_repair_loop,
            fast_pass=None,
            concurrency_sem=asyncio.Semaphore(1),
        ),
    )

    terminology = Terminology(
        glossary_dicts=[
            {"source": "ALU", "target": "算术逻辑单元", "kind": "term"},
        ],
        abbreviation_entries=[
            {"source": "CPU", "target": "CPU", "kind": "abbrev"},
        ],
    )

    scoring = Scoring(repair_loop=cast("Any", mock_repair_loop))

    from ubt.core.qe.consistency_enforce import ConsistencyTask

    with (
        patch("ubt.core.engine.stages.consistency.evaluate_terms", return_value=[]),
        patch(
            "ubt.core.engine.stages.consistency.plan_consistency_tasks",
            return_value=[ConsistencyTask(block_id="b1", source="ALU", expected="算术逻辑单元")],
        ),
        patch("ubt.core.engine.stages.consistency.build_chunk_glossary_table") as mock_build_table,
    ):
        mock_build_table.return_value = "term table with CPU"
        events = [e async for e in run_consistency_stage(ctx, services, terminology, scoring)]

        assert len(events) == 1
        mock_build_table.assert_called_once_with(
            terminology.glossary_dicts,
            terminology.abbreviation_entries,
            block.source_text,
            context=ANY,
        )


def _drift_stage(
    tmp_path: Path, *, repair_rounds: int, max_repair_rounds: int
) -> tuple[Any, Terminology, list[list[dict[str, Any]]]]:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="The CPU architecture uses a custom ALU.",
        ),
        target_text="处理器架构使用自定义运算器。",
        status=BlockStatus.MTQE_PASSED,
        repair_rounds=repair_rounds,
        error_flags=[],
    )
    batches: list[list[dict[str, Any]]] = []

    class FakeLedger:
        def get_all_blocks(self, job_id: str) -> list[IRBlock]:
            return [block]

        def save_checkpoints_batch(self, updates: list[dict[str, Any]], **_kw: Any) -> int:
            batches.append(list(updates))
            return len(updates)

    config = SimpleNamespace(
        consistency_enforce="repair",
        consistency_max_repairs=5,
        max_repair_rounds=max_repair_rounds,
    )

    async def fake_create_event(*_args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(detail=kwargs.get("message", ""))

    ctx = StageContext(
        config=cast("UBTConfig", config),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", FakeLedger()),
        manifest=cast("Any", SimpleNamespace(title="Test", source_path="test.pdf")),
        job_id="test-job",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=fake_create_event,
    )
    terminology = Terminology(
        glossary_dicts=[{"source": "ALU", "target": "算术逻辑单元", "kind": "term"}],
    )
    return ctx, terminology, batches


@pytest.mark.asyncio
async def test_host_driven_consistency_returns_drift_for_repair(tmp_path: Path) -> None:
    from ubt.core.qe.consistency_enforce import ConsistencyTask

    ctx, terminology, batches = _drift_stage(tmp_path, repair_rounds=0, max_repair_rounds=2)
    repair_loop = SimpleNamespace()
    repair_loop.repair_single_block = AsyncMock(
        side_effect=AssertionError("host-driven consistency must not call the model")
    )
    services = cast(
        "RunServices",
        SimpleNamespace(
            repair_loop=repair_loop, fast_pass=None, concurrency_sem=asyncio.Semaphore(1)
        ),
    )
    task = ConsistencyTask(block_id="b1", source="ALU", expected="算术逻辑单元")

    with (
        patch("ubt.core.engine.stages.consistency.evaluate_terms", return_value=[]),
        patch("ubt.core.engine.stages.consistency.plan_consistency_tasks", return_value=[task]),
    ):
        events = [
            e
            async for e in run_consistency_stage(
                ctx,
                services,
                terminology,
                Scoring(repair_loop=cast("Any", repair_loop)),
                repair=False,
            )
        ]

    assert len(events) == 1
    assert len(batches) == 1
    update = batches[0][0]
    assert update["block_id"] == "b1"
    assert update["status"] is BlockStatus.REPAIR_PENDING
    assert "target_text" not in update
    assert task.flag in update["error_flags"]


@pytest.mark.asyncio
async def test_host_driven_consistency_leaves_a_block_at_the_repair_ceiling(
    tmp_path: Path,
) -> None:
    from ubt.core.qe.consistency_enforce import ConsistencyTask

    ctx, terminology, batches = _drift_stage(tmp_path, repair_rounds=2, max_repair_rounds=2)
    repair_loop = SimpleNamespace()
    repair_loop.repair_single_block = AsyncMock(
        side_effect=AssertionError("host-driven consistency must not call the model")
    )
    services = cast(
        "RunServices",
        SimpleNamespace(
            repair_loop=repair_loop, fast_pass=None, concurrency_sem=asyncio.Semaphore(1)
        ),
    )
    task = ConsistencyTask(block_id="b1", source="ALU", expected="算术逻辑单元")

    with (
        patch("ubt.core.engine.stages.consistency.evaluate_terms", return_value=[]),
        patch("ubt.core.engine.stages.consistency.plan_consistency_tasks", return_value=[task]),
    ):
        events = [
            e
            async for e in run_consistency_stage(
                ctx,
                services,
                terminology,
                Scoring(repair_loop=cast("Any", repair_loop)),
                repair=False,
            )
        ]

    assert len(events) == 1
    assert batches == []
