from __future__ import annotations

from typing import Any, cast

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.core.qe.base import BaseQERunner
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


class _UncalibratedRunner(BaseQERunner):
    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        return [0.9 for _ in pairs]

    def is_calibrated(self) -> bool:
        return False


class _FakeRouter:
    repair_reasoning_effort = "low"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def repair(self, **_kw: Any) -> str:
        return self._reply


def _loop(reply: str) -> RepairLoop:
    return RepairLoop(router=cast(ModelRouter, _FakeRouter(reply)), qe_runner=_UncalibratedRunner())


def _block(*, source: str, draft: str, flags: list[str]) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=source,
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
        ),
        target_text=draft,
        draft_text=draft,
        status=BlockStatus.REPAIR_PENDING,
        error_flags=flags,
    )


_SOURCE = "The trial ran for 42 days."
_DRAFT = "该试验持续了43天。"
_FLAGS = ["Numeric fidelity: mismatched number '43', expected '42'"]


@pytest.mark.asyncio
async def test_in_place_splice_to_a_wrong_number_is_not_promoted() -> None:
    block = _block(source=_SOURCE, draft=_DRAFT, flags=_FLAGS)
    loop = _loop('<correction id="1">41</correction>')

    result = await loop.repair_single_block(block)

    assert result.status is not BlockStatus.REPAIRED, (
        "a candidate that still does not state the source number must not ship as REPAIRED"
    )
    assert any("repair_structural_failure" in f for f in result.error_flags)


@pytest.mark.asyncio
async def test_in_place_splice_restoring_the_number_is_promoted() -> None:
    block = _block(source=_SOURCE, draft=_DRAFT, flags=_FLAGS)
    loop = _loop('<correction id="1">42</correction>')

    result = await loop.repair_single_block(block)

    assert result.status is BlockStatus.REPAIRED
    assert result.error_flags == []


@pytest.mark.asyncio
async def test_residual_check_adds_no_flag_on_a_clean_candidate() -> None:
    block = _block(
        source="Chapter 3 lists 12 species.",
        draft="第3章列出了11个物种。",
        flags=["Numeric fidelity: mismatched number '11', expected '12'"],
    )
    loop = _loop('<correction id="1">12</correction>')

    result = await loop.repair_single_block(block)

    assert result.error_flags == []
    assert result.status is BlockStatus.REPAIRED


_GLOSSARY: list[dict[str, Any]] = [
    {"source": "framekernel", "translation": "帧内核", "aliases": [], "inflected_variants": []}
]
_DRIFT = "Glossary term violation: Glossary terms drifted: ['framekernel -> 帧内核']"


def _pending(draft: str, flags: list[str]) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="We propose framekernel.",
        )
    )
    block.target_text = draft
    block.draft_text = draft
    block.error_flags = list(flags)
    block.status = BlockStatus.REPAIR_PENDING
    return block


def _bare_loop() -> RepairLoop:
    from ubt.core.qe.heuristic import HeuristicQERunner

    loop = RepairLoop.__new__(RepairLoop)
    loop.qe_runner = HeuristicQERunner()
    return loop


def _try(block: IRBlock) -> IRBlock | None:
    import asyncio

    return asyncio.run(
        _bare_loop()._try_deterministic_glossary(block, block.draft_text or "", _GLOSSARY, "zh")
    )


def test_glossary_drift_is_fixed_without_calling_the_model() -> None:
    block = _pending("我们提出了 framekernel。", [_DRIFT])
    result = _try(block)
    assert result is not None
    assert "framekernel" not in (result.target_text or "")
    assert "帧内核" in (result.target_text or "")
    assert result.status is BlockStatus.REPAIRED
    assert result.error_flags == []


def test_a_deterministic_repair_is_re_scored_not_inherited() -> None:
    block = _pending("我们提出了 framekernel。", [_DRIFT])
    block.mtqe_score = 0.25
    result = _try(block)
    assert result is not None
    assert result.mtqe_score is not None
    assert result.mtqe_score > 0.25, "the repaired text kept the defect's score"


def test_a_drift_that_leaves_another_defect_still_reaches_the_model() -> None:
    block = _pending("我们提出了 framekernel。", [_DRIFT, "Omission suspected: verbatim residue"])
    assert _try(block) is None


def test_a_triage_verdict_is_never_cleared_by_the_enforcer() -> None:
    block = _pending("我们提出了 framekernel。", [_DRIFT, "needs_human_review"])
    assert _try(block) is None


def test_an_already_canonical_draft_is_not_a_repair() -> None:
    block = _pending("我们提出了帧内核。", [_DRIFT])
    assert _try(block) is None


def test_the_export_stage_writes_the_canonical_terminology_marker() -> None:
    from unittest.mock import MagicMock

    from ubt.core.qe.heuristic import GLOSSARY_VIOLATION_MARKER, glossary_violation_flag
    from ubt.core.validators.base import ValidationResult

    validator = MagicMock()
    validator.validate.return_value = ValidationResult.failure(
        error_code="GLOSSARY_DRIFT",
        message="Glossary terms drifted: [Transformer -> 变换器]",
        suggested_action="RETRY",
    )

    flag = glossary_violation_flag("The Transformer model", "这个模型", validator)

    assert flag is not None
    assert flag.startswith(GLOSSARY_VIOLATION_MARKER), flag
    assert any(marker in flag for marker in ("Glossary term violation",))
    assert "glossary_inconsistency" not in flag


def test_the_export_stage_does_not_mint_its_own_glossary_string() -> None:
    import inspect

    from ubt.core.engine.stages import export as export_stage

    source = inspect.getsource(export_stage)
    code_lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    assert "glossary_inconsistency" not in code, (
        "export must emit the canonical marker via glossary_violation_flag"
    )
    assert "glossary_violation_flag" in code
