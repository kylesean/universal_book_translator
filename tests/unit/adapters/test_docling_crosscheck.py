"""Unit tests for PDFium vs Docling layout IoU cross-check (Defense 1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_crosscheck import (
    DEFAULT_MIN_IOU,
    cross_check_blocks_with_pdfium,
)
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.model.ast import Confidence

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    spine: int = 1,
    block_type: BlockType = BlockType.NARRATIVE,
    x0: float = 50.0,
    y0: float = 600.0,
    x1: float = 300.0,
    y1: float = 700.0,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            block_type=block_type,
            id=block_id,
            source_text=text,
            spine_index=spine,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page),
        )
    )


def test_high_iou_blocks_are_preserved() -> None:
    # A block whose bounding box closely matches the physical PDFium text lines
    # passes verification and retains its original confidence and skip_translate state.
    block = _block("b1", "Clean body paragraph text.", x0=50.0, y0=600.0, x1=300.0, y1=700.0)
    mock_lines = [
        LineBox(text="Clean body paragraph", rect=(52.0, 602.0, 298.0, 698.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"))

    assert len(result) == 1
    assert result[0].id == "b1"
    assert result[0].skip_translate is False
    assert result[0].element.confidence is not Confidence.UNKNOWN


def test_phantom_block_with_no_physical_lines_is_sunk_to_opaque() -> None:
    # A block claiming substantial text where PDFium finds 0 intersecting lines
    # is recognized as a layout hallucination and sunk to PRESERVED_OPAQUE.
    block = _block(
        "b1",
        "A hallucinated paragraph over an empty white area.",
        x0=50.0,
        y0=600.0,
        x1=300.0,
        y1=700.0,
    )
    # The page has text elsewhere, but none under this block.
    mock_lines = [
        LineBox(text="Header text at the very top", rect=(50.0, 750.0, 200.0, 780.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"))

    assert len(result) == 1
    demoted = result[0]
    assert demoted.skip_translate is True
    assert demoted.element.confidence is Confidence.UNKNOWN
    assert "no_physical_lines" in (demoted.provenance.iou_crosscheck or "")


def test_low_iou_displaced_box_is_sunk_to_opaque() -> None:
    # A block whose bounding box severely disagrees with the physical text lines (IoU < 0.60)
    # is demoted to PRESERVED_OPAQUE.
    # Block box area = 250 * 500 = 125000; mock line area = 50 * 50 = 2500; IoU << 0.60
    block = _block("b1", "Sprawling hallucinated box", x0=50.0, y0=100.0, x1=300.0, y1=600.0)
    mock_lines = [
        LineBox(text="Tiny line in corner", rect=(50.0, 100.0, 100.0, 150.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"), min_iou=DEFAULT_MIN_IOU)

    assert len(result) == 1
    demoted = result[0]
    assert demoted.skip_translate is True
    assert demoted.element.confidence is Confidence.UNKNOWN
    assert "iou=" in (demoted.provenance.iou_crosscheck or "")


def test_scanned_textless_pages_skip_crosscheck() -> None:
    # On a pure image/scanned page where PDFium returns 0 lines, cross-check skips
    # without demoting blocks (leaving transcription to OCR / VLM).
    block = _block("b1", "Transcribed OCR text", x0=50.0, y0=600.0, x1=300.0, y1=700.0)

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines", return_value=([], (600.0, 800.0))
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"))

    assert len(result) == 1
    assert result[0].skip_translate is False
    assert result[0].element.confidence is not Confidence.UNKNOWN


def test_non_prose_blocks_are_not_checked() -> None:
    # Tables and images are already canvas assets handled by Layer 0; they are untouched.
    table = _block("t1", "a | b", block_type=BlockType.TABLE)
    image = _block("i1", "", block_type=BlockType.IMAGE)

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines", return_value=([], (600.0, 800.0))
    ):
        result = cross_check_blocks_with_pdfium([table, image], Path("dummy.pdf"))

    assert len(result) == 2
    assert result[0].id == "t1"
    assert result[1].id == "i1"


def test_page_caches_are_evicted_as_the_page_advances() -> None:
    # Blocks arrive in reading order (page 1, page 2, then page 1 again). The
    # per-page caches must be dropped when the page advances: without eviction
    # they held every page's lines and char styles for the whole book. The third
    # block therefore re-extracts page 1 instead of reusing a book-wide cache.
    pages_seen: list[int] = []

    def fake_extract_lines(_path: object, page: int) -> tuple[list[LineBox], tuple[float, float]]:
        pages_seen.append(page)
        return [LineBox(text="body", rect=(50.0, 600.0, 300.0, 700.0))], (600.0, 800.0)

    b1 = _block("b1", "page one text", page=1)
    b2 = _block("b2", "page two text", page=2)
    b3 = _block("b3", "page one again", page=1)

    with (
        patch(
            "ubt.adapters.pdf.docling_crosscheck.extract_lines",
            side_effect=fake_extract_lines,
        ),
        patch("ubt.adapters.pdf.docling_crosscheck.extract_char_styles", return_value=[]),
    ):
        cross_check_blocks_with_pdfium([b1, b2, b3], Path("dummy.pdf"))

    assert pages_seen == [1, 2, 1]
