from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ubt.core.engine.stages.draft import _DraftInputs, _DraftProcessor
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


class _FakeEngine:
    cache = None

    async def draft(self, masked_source: str, generate: Any, *, context: str = "") -> object:
        return await generate(masked_source)

    def resolve(self, raw: str, masked: Any) -> SimpleNamespace:
        return SimpleNamespace(text=raw, dirty=[], clean=True)


class _FakeRouter:
    def __init__(self, drafts: dict[str, str]) -> None:
        self.drafts = drafts
        self.calls: list[list[IRBlock]] = []

    async def draft_macro_chunk(self, *, blocks: list[IRBlock], **_kwargs: Any) -> dict[str, str]:
        self.calls.append(list(blocks))
        return {b.id: self.drafts.get(b.id, "") for b in blocks}

    async def draft(self, *, block: IRBlock, **_kwargs: Any) -> str:
        self.calls.append([block])
        return f"单块-{block.id}"


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

    assert drafted == ["b1"]
    updates = runtime.flusher.updates
    failed_ids = {u["block_id"] for u in updates if str(u.get("status")) == "failed"}
    assert failed_ids == {"b2"}, updates
    b1_updates = [u for u in updates if u["block_id"] == "b1"]
    assert b1_updates and all(str(u.get("status")) != "failed" for u in b1_updates)
    assert any(u.get("target_text") == "译文一" for u in b1_updates)


@pytest.mark.asyncio
async def test_a_failed_macro_chunk_redrafts_blocks_individually() -> None:
    runtime = _runtime({})

    class _ChunkFailsRouter(_FakeRouter):
        async def draft_macro_chunk(
            self, *, blocks: list[IRBlock], **kwargs: Any
        ) -> dict[str, str]:
            self.calls.append(list(blocks))
            if len(blocks) > 1:
                raise RuntimeError("macro-chunk exhausted its output budget")
            block = blocks[0]
            return {block.id: f"单块-{block.id}"}

    router = _ChunkFailsRouter({})
    runtime.router = router
    processor = _DraftProcessor(runtime=runtime, policy=_Policy())  # type: ignore[arg-type]
    b1, b2 = _block("b1"), _block("b2")

    await processor.draft_macro_chunk_group([(b1, _inputs("b1")), (b2, _inputs("b2"))], [b1, b2])

    assert [len(call) for call in router.calls] == [2, 1, 1]
    statuses = {u["block_id"]: str(u.get("status")) for u in runtime.flusher.updates}
    assert statuses["b1"] == "drafted"
    assert statuses["b2"] == "drafted"


def test_a_group_raise_leaves_already_paid_blocks_alone() -> None:
    from ubt.core.engine.stages.draft import _paid_block_ids
    from ubt.core.ir.models import BlockStatus

    drafted = _block("b1")
    drafted.target_text = "译文一"
    passed = _block("b2")
    passed.status = BlockStatus.MTQE_PASSED
    repaired = _block("b3")
    repaired.status = BlockStatus.REPAIR_PENDING
    untouched = _block("b4")

    assert _paid_block_ids([drafted, passed, repaired, untouched]) == {"b1", "b2", "b3"}
