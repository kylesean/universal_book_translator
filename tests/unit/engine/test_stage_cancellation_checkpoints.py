"""Tests for cancellation and lease loss checkpoints in pipeline stages."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.facts import Scoring, Terminology
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.bible import run_bible_stage
from ubt.core.engine.stages.consistency import run_consistency_stage
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.exceptions import JobInterruptedError, LedgerWriterLeaseLostError
from ubt.core.ir.models import IRBlock
from ubt.core.memory.abbreviation_backfill import backfill_abbreviation_translations

pytestmark = pytest.mark.fast


def _make_dummy_ctx(
    tmp_path: Path,
    cancel_token: asyncio.Event | None = None,
    lease_lost: asyncio.Event | None = None,
    blocks: list[IRBlock] | None = None,
) -> StageContext:
    class FakeLedger:
        def __init__(self) -> None:
            self.blocks = blocks or []

        def get_all_blocks(self, job_id: str) -> list[IRBlock]:
            return self.blocks

        def fetch_blocks_by_status(
            self, job_id: str, status: Any, chapter_id: Any = None
        ) -> list[IRBlock]:
            return [b for b in self.blocks if b.status == status]

        def get_job_metadata_value(self, job_id: str, key: str) -> Any:
            return None

        def set_job_metadata_value(self, job_id: str, key: str, value: Any) -> None:
            pass

        def save_checkpoints_batch(
            self, updates: Any, allow_terminal_override: bool = False
        ) -> None:
            pass

    config = SimpleNamespace(
        fresh=True,
        glossary_path=None,
        qe_threshold=0.85,
        consistency_enforce="report",
        consistency_max_repairs=5,
    )

    async def fake_create_event(*args: Any, **kwargs: Any) -> TranslationProgressEvent:
        return SimpleNamespace()  # type: ignore[return-value]

    return StageContext(
        config=cast("UBTConfig", config),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", FakeLedger()),
        manifest=cast("Any", SimpleNamespace(doc_id="doc1", chapters=[], metadata={})),
        job_id="test-job",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=fake_create_event,
        cancel_token=cancel_token,
        lease_lost=lease_lost,
    )


@pytest.mark.asyncio
async def test_bible_stage_cancellation_and_lease_lost(tmp_path: Path) -> None:
    # 1. Cancel token raises JobInterruptedError
    cancel_token = asyncio.Event()
    cancel_token.set()
    ctx_cancel = _make_dummy_ctx(tmp_path, cancel_token=cancel_token)
    with pytest.raises(JobInterruptedError):
        async for _ in run_bible_stage(ctx_cancel, Terminology()):
            pass

    # 2. Lease lost raises LedgerWriterLeaseLostError
    lease_lost = asyncio.Event()
    lease_lost.set()
    ctx_lease = _make_dummy_ctx(tmp_path, lease_lost=lease_lost)
    with pytest.raises(LedgerWriterLeaseLostError):
        async for _ in run_bible_stage(ctx_lease, Terminology()):
            pass


@pytest.mark.asyncio
async def test_quality_gate_stage_cancellation_and_lease_lost(tmp_path: Path) -> None:
    services = cast(
        "RunServices", SimpleNamespace(fast_pass=None, qe_runner=None, qe_runner_explicit=False)
    )
    # 1. Cancel token raises JobInterruptedError
    cancel_token = asyncio.Event()
    cancel_token.set()
    ctx_cancel = _make_dummy_ctx(tmp_path, cancel_token=cancel_token)
    with pytest.raises(JobInterruptedError):
        async for _ in run_quality_gate_stage(ctx_cancel, services, Terminology(), Scoring()):
            pass

    # 2. Lease lost raises LedgerWriterLeaseLostError
    lease_lost = asyncio.Event()
    lease_lost.set()
    ctx_lease = _make_dummy_ctx(tmp_path, lease_lost=lease_lost)
    with pytest.raises(LedgerWriterLeaseLostError):
        async for _ in run_quality_gate_stage(ctx_lease, services, Terminology(), Scoring()):
            pass


@pytest.mark.asyncio
async def test_consistency_stage_cancellation_and_lease_lost(tmp_path: Path) -> None:
    services = cast(
        "RunServices",
        SimpleNamespace(repair_loop=None, fast_pass=None, concurrency_sem=asyncio.Semaphore(1)),
    )
    # 1. Cancel token raises JobInterruptedError
    cancel_token = asyncio.Event()
    cancel_token.set()
    ctx_cancel = _make_dummy_ctx(tmp_path, cancel_token=cancel_token)
    with pytest.raises(JobInterruptedError):
        async for _ in run_consistency_stage(ctx_cancel, services, Terminology(), Scoring()):
            pass

    # 2. Lease lost raises LedgerWriterLeaseLostError
    lease_lost = asyncio.Event()
    lease_lost.set()
    ctx_lease = _make_dummy_ctx(tmp_path, lease_lost=lease_lost)
    with pytest.raises(LedgerWriterLeaseLostError):
        async for _ in run_consistency_stage(ctx_lease, services, Terminology(), Scoring()):
            pass


@pytest.mark.asyncio
async def test_backfill_abbreviation_translations_cancels_cooperatively() -> None:
    entries = [{"source": "BERT", "aliases": ["BERT"], "translation": ""}]
    complete_mock = AsyncMock(return_value="BERT = 变换器双向编码器表征")

    # Cancel check raises
    def cancel_check() -> None:
        raise JobInterruptedError("cancelled")

    with pytest.raises(JobInterruptedError):
        await backfill_abbreviation_translations(
            entries,
            complete_mock,
            target_lang="zh",
            cancel_check=cancel_check,
        )

    # complete_mock must NOT have been called
    complete_mock.assert_not_called()
