"""A macro-chunk failure marks only the block that failed, never its group.

The draft stage finalizes each block in a macro-chunk group one at a time, and
``finalize_draft`` checkpoints a block the moment it succeeds. If a later
block's finalize raised and the exception escaped to ``run_draft_batch``'s
``asyncio.gather``, the handler there marked EVERY block in the group FAILED —
including blocks already drafted, paid for and checkpointed. On resume the
``Drafting error:`` flag routed those paid blocks to ``reset_blocks_to_pending``,
which cleared their target text and re-billed them. This pins the fault
isolation: the failure stays on the block that produced it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ubt.core.engine.stages.draft import _DraftInputs, _DraftProcessor
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


class _FakeEngine:
    """Stands in for the translation engine: no cache, identity resolve."""

    cache = None

    def resolve(self, raw: str, masked: Any) -> SimpleNamespace:
        return SimpleNamespace(text=raw, dirty=[], clean=True)


class _FakeRouter:
    """Returns one extracted draft per requested block, keyed by block id."""

    def __init__(self, drafts: dict[str, str]) -> None:
        self.drafts = drafts

    async def draft_macro_chunk(self, *, blocks: list[IRBlock], **_kwargs: Any) -> dict[str, str]:
        return {b.id: self.drafts.get(b.id, "") for b in blocks}


class _FakeMemory:
    def record_drafted_block(self, block: IRBlock) -> None:
        pass


class _FakeFlusher:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    async def enqueue(self, update: dict[str, Any]) -> None:
        self.updates.append(update)


def _runtime(drafts: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        fail_fast_consecutive=0,
        engine=_FakeEngine(),
        router=_FakeRouter(drafts),
        memory_mgr=_FakeMemory(),
        flusher=_FakeFlusher(),
        counters={},
        concurrency_sem=asyncio.Semaphore(4),
    )


def _block(block_id: str) -> IRBlock:
    element = make_element(
        id=block_id, spine_index=1, block_type=BlockType.NARRATIVE, source_text="source"
    )
    return IRBlock(element=element)


def _inputs(block_id: str) -> _DraftInputs:
    return _DraftInputs(
        masked_source=f"masked-{block_id}",
        code_map=None,
        cite_map=None,
        math_map=None,
        glossary_table="",
        neighbor_ctx="",
        macro_ctx="",
        few_shot_reference="",
    )


class _Policy:
    """The handful of policy knobs ``draft_macro_chunk_group`` reads."""

    glossary_dicts: list[Any] = []
    draft_max_retries = 0
    draft_retry_base_delay = 0.0
    target_lang = "zh"
    source_lang = "en"
    profile_name = "default"
    domain = ""
    global_glossary_table = ""


@pytest.mark.asyncio
async def test_a_finalize_failure_does_not_fail_already_drafted_siblings() -> None:
    runtime = _runtime({"b1": "译文一", "b2": "译文二"})
    processor = _DraftProcessor(runtime=runtime, policy=_Policy())  # type: ignore[arg-type]
    drafted: list[str] = []
    original = processor.finalize_draft

    async def _finalize(block: IRBlock, raw_text: str, inputs: _DraftInputs) -> bool:
        if block.id == "b2":
            raise RuntimeError("finalize blew up on b2")
        drafted.append(block.id)
        return await original(block, raw_text, inputs)

    processor.finalize_draft = _finalize  # type: ignore[method-assign]

    b1, b2 = _block("b1"), _block("b2")
    await processor.draft_macro_chunk_group([(b1, _inputs("b1")), (b2, _inputs("b2"))], [b1, b2])

    # b1 was drafted and checkpointed; only b2 carries a FAILED update.
    assert drafted == ["b1"]
    updates = runtime.flusher.updates
    failed_ids = {u["block_id"] for u in updates if str(u.get("status")) == "failed"}
    assert failed_ids == {"b2"}, updates
    # b1's successful checkpoint is present and intact — it was never re-marked.
    b1_updates = [u for u in updates if u["block_id"] == "b1"]
    assert b1_updates and all(str(u.get("status")) != "failed" for u in b1_updates)
    assert any(u.get("target_text") == "译文一" for u in b1_updates)
