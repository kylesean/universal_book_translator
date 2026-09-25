"""Page-range ingest plumbing: --pages must cut parse cost, not just output.

Before this, ``--pages`` only filtered blocks after a full document parse:
Docling still converted and formula-VLM'd every page of a 26-page chapter for
a two-page sample. A contiguous selection now travels into the engine, so the
expensive pass only sees the requested range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter, _contiguous_page_range
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.core.ir.models import BoundingBox, IRBlock


class TestContiguousPageRange:
    @pytest.mark.parametrize(
        ("pages", "expected"),
        [
            (None, None),
            (set(), None),
            ({2}, (2, 2)),
            ({1, 2, 3}, (1, 3)),
            ({5, 6, 7, 8}, (5, 8)),
            ({1, 3}, None),
            ({1, 2, 4, 5}, None),
        ],
    )
    def test_collapse(self, pages: set[int] | None, expected: tuple[int, int] | None) -> None:
        assert _contiguous_page_range(pages) == expected


@pytest.mark.asyncio
async def test_docling_parse_stream_pushes_contiguous_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = DoclingPDFAdapter()
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    captured: dict[str, Any] = {}

    def fake_extract(_path: Path, page_range: tuple[int, int] | None = None) -> list[IRBlock]:
        captured["page_range"] = page_range
        return []

    monkeypatch.setattr(adapter, "_extract_blocks_sync", fake_extract)
    monkeypatch.setattr(adapter, "_vlm_fallback_missing_pages", lambda *args, **kwargs: args[1])

    chapters = [chapter async for chapter in adapter.parse_stream(path, {2, 3, 4})]
    assert len(chapters) == 1
    assert captured["page_range"] == (2, 4)


@pytest.mark.asyncio
async def test_docling_parse_stream_leaves_non_contiguous_selection_to_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = DoclingPDFAdapter()
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    captured: dict[str, Any] = {}

    def fake_extract(_path: Path, page_range: tuple[int, int] | None = None) -> list[IRBlock]:
        captured["page_range"] = page_range
        return []

    monkeypatch.setattr(adapter, "_extract_blocks_sync", fake_extract)
    monkeypatch.setattr(adapter, "_vlm_fallback_missing_pages", lambda *args, **kwargs: args[1])

    _chapters = [chapter async for chapter in adapter.parse_stream(path, {1, 3})]
    assert captured["page_range"] is None


def test_pdfium_extraction_filters_to_page_range(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = PDFiumAdapter()
    blocks = [
        IRBlock(
            id=f"p{page}",
            spine_index=page,
            source_text="text",
            bbox=BoundingBox(page=page, x0=0, y0=0, x1=10, y1=10),
        )
        for page in (1, 2, 3, 4)
    ]
    monkeypatch.setattr(adapter, "_extract_with_pdfium", lambda _path: blocks)

    filtered = adapter._extract_blocks_sync(Path("sample.pdf"), (2, 3))
    assert [block.bbox.page for block in filtered if block.bbox is not None] == [2, 3]

    unfiltered = adapter._extract_blocks_sync(Path("sample.pdf"))
    assert len(unfiltered) == 4
