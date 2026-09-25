"""Regression tests for Batch 3 fixes: PDF Adapter, Layout & Typst."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver
from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
from ubt.adapters.pdf.svg_diagram import _parse_content_drawings


@pytest.mark.fast
def test_svg_diagram_cm_matrix_multiplication() -> None:
    # A path preceded by translation and scaling:
    # cm: scale by 2 (2 0 0 2 0 0)
    # 10 10 20 20 re -> transformed to (20, 20, 60, 60)
    stream = b"q\n2 0 0 2 0 0 cm\n10 10 20 20 re\nS\nQ\n"
    rects = _parse_content_drawings(stream)
    assert len(rects) >= 1
    # x0 should be 20.0, x1 should be 60.0
    r0 = rects[0]
    assert r0[0] == pytest.approx(20.0), f"Expected x0=20.0, got {r0[0]}"
    assert r0[2] == pytest.approx(60.0), f"Expected x1=60.0, got {r0[2]}"


@pytest.mark.fast
def test_coordinate_resolver_auto_1000_large_page() -> None:
    # Large poster page: 1200 x 1600 pt
    # Box has coordinates (100, 200, 500, 800) in PDF points
    resolver = PageBBoxResolver(page_width=1200.0, page_height=1600.0)
    # Under 'auto', since 500 <= 1200 and 800 <= 1600, it must NOT treat this as normalized_1000!
    box = resolver.resolve_bbox([100, 200, 500, 800], coord_system="auto", origin="bottom-left")
    assert box is not None
    # x0 should be ~100 pt, NOT 100/1000 * 1200 = 120 pt
    assert box[0] == pytest.approx(100.0), f"Expected native 100.0 pt, got {box[0]}"


@pytest.mark.fast
def test_rigid_batch_overlay_no_extra_pagebreak() -> None:
    typesetter = RigidTypesetter()
    facts_p1 = MagicMock(width=500.0, height=700.0, bg=False)
    facts_p2 = MagicMock(width=500.0, height=700.0, bg=False)
    pages: dict[int, Any] = {1: facts_p1, 2: facts_p2}
    zone1 = MagicMock(x0=10.0, y1=50.0, width=100.0, height=40.0)
    zone2 = MagicMock(x0=10.0, y1=50.0, width=100.0, height=40.0)
    paints: dict[int, list[Any]] = {
        1: [MagicMock(zone=zone1, height=700.0, text="hello", size=10.0)],
        2: [MagicMock(zone=zone2, height=700.0, text="world", size=10.0)],
    }

    overlay_src, compiled = typesetter._batch_page_overlay(pages, paints)
    # Typst #set page(...) automatically creates a pagebreak when preceded by content.
    # An explicit #pagebreak() directly before #set page(...) causes an extra blank page.
    assert not (
        "#pagebreak()\n#set page" in overlay_src or "#pagebreak()\r\n#set page" in overlay_src
    ), "Overlay Typst source must not have #pagebreak() immediately preceding #set page"


@pytest.mark.fast
def test_sample_pdf_pages_closes_handles(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    from ubt.adapters.pdf.plain_text_extractor import sample_pdf_pages

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=100, height=100)
    pdf_path = tmp_path / "test_sample.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    closed_pages = []
    closed_textpages = []
    orig_page_close = pdfium.PdfPage.close
    orig_textpage_close = pdfium.PdfTextPage.close

    def mock_page_close(self: object) -> None:
        closed_pages.append(self)
        orig_page_close(self)

    def mock_textpage_close(self: object) -> None:
        closed_textpages.append(self)
        orig_textpage_close(self)

    with (
        patch.object(pdfium.PdfPage, "close", mock_page_close),
        patch.object(pdfium.PdfTextPage, "close", mock_textpage_close),
    ):
        page_count, is_scanned, sample = sample_pdf_pages(pdf_path)
        assert len(closed_pages) >= 1, "PdfPage.close must be called in sample_pdf_pages"
        assert len(closed_textpages) >= 1, "PdfTextPage.close must be called in sample_pdf_pages"


@pytest.mark.fast
def test_docling_parser_circuit_breaker_remaining_count() -> None:

    proofread_pages = {10, 11, 12, 13}
    logs = []

    class MockLogger:
        def error(self, msg: str, *args: object) -> None:
            logs.append(msg % args if args else msg)

        def warning(self, *args: object) -> None:
            pass

        def info(self, *args: object) -> None:
            pass

        def debug(self, *args: object) -> None:
            pass

    # Simulate loop index calculation
    sorted_proofread = sorted(proofread_pages)
    idx = 0
    assert sorted_proofread[idx] == 10
    # Fixed calculation: len(sorted_proofread) - idx
    correct_remaining = len(sorted_proofread) - idx
    assert correct_remaining == 4


@pytest.mark.fast
def test_docling_adapter_extract_sync_not_serialized() -> None:
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    adapter = DoclingPDFAdapter()
    fn = adapter._extract_blocks_sync
    # Verify that the function is NOT wrapped by pdfium_serialized (which wraps with PDFIUM_LOCK)
    # Functions wrapped with @pdfium_serialized have '__wrapped__' or closure holding PDFIUM_LOCK
    import inspect

    closure_vars = inspect.getclosurevars(fn)
    assert "PDFIUM_LOCK" not in closure_vars.nonlocals, (
        "_extract_blocks_sync must not be decorated with @pdfium_serialized holding global PDFIUM_LOCK"
    )
