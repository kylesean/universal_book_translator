"""``finalize_draft`` parses the model's bold markers into target-side runs.

The draft prompt asks the model to bracket emphasized target text with
``⟦B⟧ … ⟦/B⟧``; the finalized target text must be clean (no marker ever leaks)
and the parsed runs must be checkpointed so the render can re-apply the weight.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ubt.core.engine.stages.draft import _DraftInputs, _DraftProcessor
from ubt.core.ir.models import BlockType, InlineRun, IRBlock, StyleMeta, make_element

pytestmark = pytest.mark.fast


class _FakeEngine:
    def __init__(self, resolved: str) -> None:
        self.resolved = resolved

    def resolve(self, raw: str, masked: Any) -> SimpleNamespace:
        return SimpleNamespace(text=self.resolved, dirty=[], clean=True)


class _FakeMemory:
    def record_drafted_block(self, block: IRBlock) -> None:
        self.recorded = block


class _FakeFlusher:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    async def enqueue(self, update: dict[str, Any]) -> None:
        self.updates.append(update)


def _runtime(resolved: str) -> SimpleNamespace:
    return SimpleNamespace(
        fail_fast_consecutive=0,
        engine=_FakeEngine(resolved),
        memory_mgr=_FakeMemory(),
        flusher=_FakeFlusher(),
        counters={},
    )


def _inputs() -> _DraftInputs:
    return _DraftInputs(
        masked_source="source",
        code_map=None,
        cite_map=None,
        math_map=None,
        glossary_table="",
        neighbor_ctx="",
        macro_ctx="",
        few_shot_reference="",
    )


def _block(style: StyleMeta | None = None) -> IRBlock:
    element = make_element(
        id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="source"
    )
    return IRBlock(element=element, style=style)


async def _finalize(resolved: str, block: IRBlock | None = None) -> SimpleNamespace:
    runtime = _runtime(resolved)
    processor = _DraftProcessor(runtime=runtime, policy=None)  # type: ignore[arg-type]
    await processor.finalize_draft(block or _block(), "raw", _inputs())
    return runtime


@pytest.mark.asyncio
async def test_finalize_parses_markers_into_target_runs() -> None:
    runtime = await _finalize("成本降低了 ⟦B⟧50.0%⟦/B⟧；随后")
    update = runtime.flusher.updates[0]
    assert update["target_text"] == "成本降低了 50.0%；随后"
    assert "⟦" not in update["target_text"]
    assert [(r.text, r.bold) for r in update["style"].target_runs] == [("50.0%", True)]


@pytest.mark.asyncio
async def test_finalize_strips_markers_even_when_unbalanced() -> None:
    runtime = await _finalize("前⟦B⟧重点⟦/B⟧尾⟦/B⟧")
    update = runtime.flusher.updates[0]
    assert update["target_text"] == "前重点尾"
    assert [r.text for r in update["style"].target_runs] == ["重点"]


@pytest.mark.asyncio
async def test_finalize_overwrites_a_stale_run_with_an_empty_result() -> None:
    stale = StyleMeta(target_runs=(InlineRun(text="旧", bold=True),))
    runtime = await _finalize("没有标记的译文", _block(stale))
    update = runtime.flusher.updates[0]
    assert update["target_text"] == "没有标记的译文"
    assert update["style"].target_runs == ()


@pytest.mark.asyncio
async def test_finalize_leaves_a_styleless_block_styleless() -> None:
    runtime = await _finalize("没有标记的译文", _block(None))
    assert runtime.flusher.updates[0]["style"] is None


@pytest.mark.asyncio
async def test_finalize_preserves_existing_source_inline_runs() -> None:
    style = StyleMeta(inline_runs=(InlineRun(text="2020", bold=False, color_hex="#0000ff"),))
    runtime = await _finalize("译文 ⟦B⟧重点⟦/B⟧", _block(style))
    update = runtime.flusher.updates[0]
    assert update["style"].inline_runs == style.inline_runs
    assert [r.text for r in update["style"].target_runs] == ["重点"]
