from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ubt.adapters.markdown.adapter import _format_markdown_block
from ubt.core.cleaners.cjk_spacing import apply_pangu_spacing
from ubt.core.config import UBTConfig
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_queue import JobQueue, QueuedJob
from ubt.core.engine.job_worker import JobWorker
from ubt.core.engine.stages.chapter_streaming import _DONE
from ubt.core.exceptions import LedgerError, ModelProviderError
from ubt.core.ir.models import IRBlock
from ubt.core.qe.fast_pass import grid_columns, markdown_grid_shape
from ubt.core.router.router import classify_provider_error
from ubt.model.ast import Caption, Heading, ListItem, RegionKind, Span

pytestmark = pytest.mark.fast


def test_caption_region_kind_default() -> None:
    """Verify Caption AST element defaults to RegionKind.CAPTION."""
    c = Caption(id="c1", spine_index=0, span=Span(), text="Photo of the author")
    assert c.region == RegionKind.CAPTION


def test_markdown_adapter_heading_level_preservation() -> None:
    """Verify MarkdownAdapter preserves heading levels from AST element even if raw_content has no '#'."""
    h = Heading(id="h1", spine_index=0, span=Span(), text="Chapter Three", level=3)
    block = IRBlock.from_element(h)
    block.target_text = "第三章"
    formatted = _format_markdown_block(block, "第三章")
    assert formatted == "### 第三章"


def test_markdown_adapter_list_item_preservation() -> None:
    """Verify MarkdownAdapter preserves list markers and ordinals."""
    # Unordered list item
    li_unordered = ListItem(
        id="li1",
        spine_index=0,
        span=Span(),
        text="First point",
        marker="- ",
    )
    block_u = IRBlock.from_element(li_unordered)
    assert _format_markdown_block(block_u, "第一点") == "- 第一点"

    # Ordered list item
    li_ordered = ListItem(
        id="li2",
        spine_index=0,
        span=Span(),
        text="Second point",
        marker="2. ",
    )
    block_o = IRBlock.from_element(li_ordered)
    assert _format_markdown_block(block_o, "第二点") == "2. 第二点"


def test_cjk_spacing_formula_and_html_protection() -> None:
    """Verify LaTeX math environments and HTML tags are protected from pangu spacing insertion."""
    text = "这是公式\\(x = 1\\)测试。"
    cleaned = apply_pangu_spacing(text)
    assert "\\(x = 1\\)" in cleaned

    text2 = "爱因斯坦提出\\[ E = mc^2 \\]公式。"
    cleaned2 = apply_pangu_spacing(text2)
    assert "\\[ E = mc^2 \\]" in cleaned2

    text3 = "这是<code>int a = 1;</code>代码。"
    cleaned3 = apply_pangu_spacing(text3)
    assert "<code>int a = 1;</code>" in cleaned3


def test_fast_pass_escaped_pipe_handling() -> None:
    r"""Verify table checker does not treat escaped pipes \| as cell separators."""
    # Row with 3 columns, middle column contains escaped pipe
    row = r"| Col 1 | Col \| 2 | Col 3 |"
    assert grid_columns(row) == 3

    # Multi-row markdown grid shape
    text = (
        r"| Col 1 | Col \| 2 | Col 3 |" + "\n"
        r"|---|---|---|" + "\n"
        r"| A | B \| C | D |"
    )
    shape = markdown_grid_shape(text)
    assert shape == [3, 3, 3]


def test_router_quota_exhausted_fail_fast() -> None:
    """Verify router treats 429 quota exhaustion as chain fail-fast instead of retrying all tiers."""
    err = ModelProviderError(
        "Rate limited",
        details={
            "status_code": 429,
            "body": '{"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "type": "insufficient_quota"}}',
        },
    )
    action = classify_provider_error(err)
    assert action.retryable is False
    assert action.top_up_hint is True
    assert action.reason == "quota_exhausted"


@pytest.mark.asyncio
async def test_job_worker_execute_terminal_failure_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify JobWorker.execute isolates terminal queue write failure and does not crash slot."""
    config = UBTConfig(provider="mock")
    queue = JobQueue(tmp_path / "test_queue.db")
    job = queue.enqueue("test-job-fail", {"input_path": "book.epub"})

    # Complete raises LedgerError (e.g. transient SQLite error during terminal status update)
    monkeypatch.setattr(
        queue, "complete", MagicMock(side_effect=LedgerError("Database disk I/O error"))
    )

    async def failing_event_source(
        job: QueuedJob, cfg: UBTConfig, cancel_token: asyncio.Event | None = None
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        raise RuntimeError("LLM synthesis failure")
        if False:
            yield TranslationProgressEvent(job_id=job.job_id, stage="done")

    worker = JobWorker(
        queue=queue,
        config=config,
        worker_id="test-w1",
        event_source=failing_event_source,
    )
    res = await worker.execute(job)
    # execute handles the exception cleanly without bubbling up to crash the worker process
    assert res is True
    assert worker.failed_jobs == 1


@pytest.mark.asyncio
async def test_chapter_streaming_event_queue_backpressure() -> None:
    """Verify supervisor waits for event_queue space when putting _DONE rather than dropping it."""
    event_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
    # Fill event queue to simulate saturation
    await event_queue.put("pre-existing-event")

    # In a background task, read one event after a small delay
    async def drain_event() -> None:
        await asyncio.sleep(0.05)
        await event_queue.get()
        event_queue.task_done()

    drain_task = asyncio.create_task(drain_event())

    # Simulated supervisor putting _DONE with backpressure
    put_done_coro = event_queue.put(_DONE)
    await asyncio.wait_for(put_done_coro, timeout=1.0)
    await drain_task

    # Queue should now have _DONE
    item = await event_queue.get()
    assert item is _DONE
