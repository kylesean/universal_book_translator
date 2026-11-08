"""The extraction hop must carry the run's usage sink across the thread boundary.

``run_in_executor`` does not copy contextvars, so the remote table reader
inside ``_extract_blocks_sync`` read a ``None`` sink and every table-crop
token left the job's bill and the ``UBT_BUDGET_USD`` floor unrecorded.
``to_thread`` copies the context; this test pins that invariant at the seam
so the hop cannot regress to the executor form.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast


def test_extraction_thread_inherits_the_usage_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.core.router.transports.base import _usage_sink, attach_usage_sink

    adapter = DoclingPDFAdapter()
    seen: dict[str, Any] = {}

    def _fake_extract(path: Path, page_range: tuple[int, int] | None = None) -> list[Any]:
        seen["sink"] = _usage_sink.get()
        return []

    monkeypatch.setattr(adapter, "_extract_blocks_sync", _fake_extract)

    async def scenario() -> list[Any]:
        attach_usage_sink()
        return await adapter._extract_blocks_off_loop(Path("book.pdf"), None)

    assert asyncio.run(scenario()) == []
    assert seen["sink"] is not None, "usage sink was not visible inside the worker thread"


def test_without_a_sink_the_hop_still_returns_the_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs that never attach a sink (dry-run, scripts) must be unaffected."""
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.core.router.transports.base import _usage_sink

    adapter = DoclingPDFAdapter()
    marker = object()

    def _fake_extract(path: Path, page_range: tuple[int, int] | None = None) -> list[Any]:
        return [marker]

    monkeypatch.setattr(adapter, "_extract_blocks_sync", _fake_extract)

    async def scenario() -> list[Any]:
        assert _usage_sink.get() is None
        return await adapter._extract_blocks_off_loop(Path("book.pdf"), None)

    assert asyncio.run(scenario()) == [marker]
