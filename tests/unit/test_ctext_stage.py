"""C-track formula-span translation must be chapter-scoped under streaming.

The chapter-streaming QE worker calls ``run_c_text_stage`` once per chapter.
The stage used to fetch every FORMULA block job-wide, so chapter 1 translated
the whole book's formula spans (defeating memory bounding) and a formula that
failed closed (target left source-verbatim, so ``_already_translated`` stays
false) was re-sent to the model on every later chapter — N-times billing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.stages.ctext import run_c_text_stage
from ubt.core.ir.models import BlockType


class _FakeLedger:
    def __init__(self) -> None:
        self.fetch_calls: list[tuple[str, BlockType, str | None]] = []
        self.saved: list[Any] = []

    def fetch_blocks_by_type(
        self, job_id: str, block_type: BlockType, chapter_id: str | None = None
    ) -> list[Any]:
        self.fetch_calls.append((job_id, block_type, chapter_id))
        return []

    def save_checkpoints_batch(self, checkpoints: list[dict[str, Any]]) -> None:
        self.saved.extend(checkpoints)


async def _create_event(
    event_type: EventType, job_id: str, ledger: Any, message: str = "", **kwargs: Any
) -> TranslationProgressEvent:
    return TranslationProgressEvent(
        event_type=event_type, job_id=job_id, total_blocks=0, completed_blocks=0, message=message
    )


def _ctx(ledger: _FakeLedger) -> Any:
    return SimpleNamespace(
        ledger=ledger,
        job_id="j1",
        router=SimpleNamespace(),
        target_lang="zh",
        source_lang="en",
        create_event=_create_event,
        config=SimpleNamespace(max_concurrency=2),
    )


def test_c_text_stage_scopes_the_fetch_to_the_chapter() -> None:
    ledger = _FakeLedger()

    async def _run() -> None:
        async for _ in run_c_text_stage(_ctx(ledger), chapter_id="ch_001_a_b"):
            pass

    asyncio.run(_run())
    assert ledger.fetch_calls == [("j1", BlockType.FORMULA, "ch_001_a_b")]


def test_c_text_stage_without_chapter_fetches_job_wide() -> None:
    ledger = _FakeLedger()

    async def _run() -> None:
        async for _ in run_c_text_stage(_ctx(ledger)):
            pass

    asyncio.run(_run())
    assert ledger.fetch_calls == [("j1", BlockType.FORMULA, None)]
