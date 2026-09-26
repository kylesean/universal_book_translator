"""Text adapters must not run their synchronous render on the event loop.

Rendering a DOM/zip (EPUB/HTML) or building the bilingual Markdown is
synchronous CPU + file IO. Run inline in an ``async def`` it stalls every other
task on the loop for the whole document; each adapter now offloads that work to
a worker thread. This file pins the contract with a heartbeat: while the sync
work is sleeping in a worker thread the loop must keep ticking. A regression
back to inline work makes the heartbeat tick ~0 times.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.epub.adapter import EPUBAdapter
from ubt.adapters.html.adapter import HTMLAdapter
from ubt.adapters.markdown.adapter import MarkdownAdapter


async def _ticks_during(run: Callable[[], Awaitable[Any]]) -> int:
    ticks = 0
    stop = False

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    task = asyncio.create_task(heartbeat())
    try:
        await run()
    finally:
        stop = True
        await task
    return ticks


@pytest.mark.parametrize(
    "adapter_cls",
    [EPUBAdapter, HTMLAdapter, MarkdownAdapter],
    ids=["epub", "html", "markdown"],
)
@pytest.mark.asyncio
async def test_render_blocks_runs_off_the_event_loop(
    adapter_cls: type[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = adapter_cls()
    calls: list[int] = []

    def slow_sync(*_args: Any, **_kwargs: Any) -> Path:
        calls.append(1)
        time.sleep(0.2)
        return Path("/tmp/ubt-offload-test.out")

    monkeypatch.setattr(adapter, "_render_blocks_sync", slow_sync)

    async def run() -> None:
        await adapter.render_blocks(
            None,  # type: ignore[arg-type]
            [],
            "zh",
            Path("/tmp/ubt-offload-test.out"),
        )

    ticks = await _ticks_during(run)
    assert calls, "render_blocks did not route through the offloaded sync helper"
    # ~40 ticks fit in a 0.2s sleep; a blocked loop ticks ~0.
    assert ticks >= 10, f"event loop stalled during render_blocks (ticks={ticks})"
