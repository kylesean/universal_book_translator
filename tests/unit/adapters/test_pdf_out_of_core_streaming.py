"""Unit tests for out-of-core PDF page streaming and memory eviction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.page_chunking import (
    compute_pdf_page_chunks,
    get_pdf_page_chunk_size,
)
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.core.ir.models import BlockType, BookManifest, ChapterMeta, IRBlock, make_element

pytestmark = pytest.mark.fast


def test_compute_pdf_page_chunks_unconstrained() -> None:
    chunks = compute_pdf_page_chunks(total_pages=120, chunk_size=50)
    assert len(chunks) == 3
    assert chunks[0] == list(range(1, 51))
    assert chunks[1] == list(range(51, 101))
    assert chunks[2] == list(range(101, 121))


def test_compute_pdf_page_chunks_with_explicit_pages() -> None:
    requested_pages = {1, 2, 5, 55, 56, 120}
    chunks = compute_pdf_page_chunks(total_pages=200, pages=requested_pages, chunk_size=3)
    assert len(chunks) == 2
    assert chunks[0] == [1, 2, 5]
    assert chunks[1] == [55, 56, 120]


def test_compute_pdf_page_chunks_fallback_on_zero() -> None:
    chunks = compute_pdf_page_chunks(total_pages=0, chunk_size=50)
    assert chunks == [[]]


def test_get_pdf_page_chunk_size_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_PDF_STREAM_CHUNK_SIZE", "25")
    assert get_pdf_page_chunk_size() == 25

    monkeypatch.delenv("UBT_PDF_STREAM_CHUNK_SIZE", raising=False)
    assert get_pdf_page_chunk_size() == 50


@pytest.mark.asyncio
async def test_pdfium_adapter_out_of_core_streaming(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "large_book.pdf"
    dummy_pdf.write_bytes(b"%PDF-dummy")

    adapter = PDFiumAdapter()

    manifest = BookManifest(
        doc_id="test_doc_123",
        title="large_book",
        source_path=str(dummy_pdf),
        chapters=[
            ChapterMeta(
                chapter_id="c0001", title="Pages 1-50", spine_index=1, source_file=dummy_pdf.name
            ),
            ChapterMeta(
                chapter_id="c0002", title="Pages 51-100", spine_index=2, source_file=dummy_pdf.name
            ),
        ],
        metadata={"pdf_parser_engine": "pdfium", "page_count": 100},
    )

    def fake_extract_blocks(path: Path, page_range: tuple[int, int] | None) -> list[IRBlock]:
        assert page_range is not None
        p_start, p_end = page_range
        b1 = IRBlock(
            element=make_element(
                id="raw_1",
                spine_index=1,
                block_type=BlockType.NARRATIVE,
                source_text=f"Text on page {p_start}",
            )
        )
        b2 = IRBlock(
            element=make_element(
                id="raw_2",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text=f"Text on page {p_end}",
            )
        )
        return [b1, b2]

    with (
        patch.object(adapter, "extract_manifest", return_value=manifest),
        patch.object(adapter, "_extract_blocks_sync", side_effect=fake_extract_blocks),
        patch("ubt.adapters.pdf.page_chunking.get_pdf_page_chunk_size", return_value=50),
        patch("gc.collect") as mock_gc,
    ):
        yielded_chapters = []
        async for chapter in adapter.parse_stream(dummy_pdf):
            yielded_chapters.append(chapter)

        assert len(yielded_chapters) == 2
        assert yielded_chapters[0].chapter_id == "c0001"
        assert yielded_chapters[1].chapter_id == "c0002"

        # Check block ID uniqueness and monotonic spine index across chunks
        ch1_blocks = yielded_chapters[0].blocks
        ch2_blocks = yielded_chapters[1].blocks

        assert len(ch1_blocks) == 2
        assert len(ch2_blocks) == 2

        assert ch1_blocks[0].id == "pdf_main#b0001"
        assert ch1_blocks[0].spine_index == 1
        assert ch1_blocks[1].id == "pdf_main#b0002"
        assert ch1_blocks[1].spine_index == 2

        assert ch2_blocks[0].id == "pdf_main#b0003"
        assert ch2_blocks[0].spine_index == 3
        assert ch2_blocks[1].id == "pdf_main#b0004"
        assert ch2_blocks[1].spine_index == 4

        # Eviction called after each chunk
        assert mock_gc.call_count == 2


@pytest.mark.asyncio
async def test_docling_adapter_out_of_core_streaming(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "long_manual.pdf"
    dummy_pdf.write_bytes(b"%PDF-dummy")

    adapter = DoclingPDFAdapter()

    manifest = BookManifest(
        doc_id="docling_doc_789",
        title="long_manual",
        source_path=str(dummy_pdf),
        chapters=[
            ChapterMeta(
                chapter_id="c0001", title="Pages 1-50", spine_index=1, source_file=dummy_pdf.name
            ),
            ChapterMeta(
                chapter_id="c0002", title="Pages 51-75", spine_index=2, source_file=dummy_pdf.name
            ),
        ],
        metadata={"pdf_parser_engine": "docling", "page_count": 75},
    )

    def fake_extract_blocks(path: Path, page_range: tuple[int, int] | None) -> list[IRBlock]:
        p_start = page_range[0] if page_range else 1
        return [
            IRBlock(
                element=make_element(
                    id=f"doc_b_{p_start}",
                    spine_index=1,
                    block_type=BlockType.NARRATIVE,
                    source_text=f"Content starting page {p_start}",
                )
            )
        ]

    with (
        patch.object(adapter, "extract_manifest", return_value=manifest),
        patch.object(adapter, "_extract_blocks_sync", side_effect=fake_extract_blocks),
        patch.object(
            adapter, "_vlm_fallback_missing_pages", side_effect=lambda *args, **kw: args[1]
        ),
        patch.object(adapter, "_annotate_page_kinds", return_value={1: "body"}),
        patch("ubt.adapters.pdf.page_chunking.get_pdf_page_chunk_size", return_value=50),
        patch("gc.collect") as mock_gc,
    ):
        yielded_chapters = []
        async for chapter in adapter.parse_stream(dummy_pdf):
            yielded_chapters.append(chapter)

        assert len(yielded_chapters) == 2
        assert yielded_chapters[0].chapter_id == "c0001"
        assert yielded_chapters[1].chapter_id == "c0002"
        assert yielded_chapters[0].blocks[0].id == "pdf_main#b0001"
        assert yielded_chapters[1].blocks[0].id == "pdf_main#b0002"
        assert mock_gc.call_count == 2
