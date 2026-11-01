from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from pdf_builders import write_text_pdf, write_two_column_pdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, FloatObject, NameObject

from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.adapters.pdf.textgeom import extract_lines

pytestmark = pytest.mark.fast

_TOL = 0.5


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _TOL


def _assert_rect_close(
    rect: tuple[float, float, float, float], expected: tuple[float, float, float, float]
) -> None:
    assert all(_close(got, want) for got, want in zip(rect, expected, strict=True)), (
        f"rect {tuple(round(v, 2) for v in rect)} != expected {expected}"
    )


_PAGE_W = 300.0
_PAGE_H = 500.0
_LINE = ("The rotated page body text for the golden geometry run.",)


def test_rotated_page_mediabox_and_text_share_the_user_space_frame(tmp_path: Path) -> None:
    source = write_text_pdf(
        tmp_path / "rot90.pdf", [_LINE], width=_PAGE_W, height=_PAGE_H, rotation=90
    )
    lines, size = extract_lines(source, 1)
    assert size == (_PAGE_W, _PAGE_H), f"size must be the unrotated mediabox, got {size}"
    assert len(lines) == 1
    _assert_rect_close(lines[0].rect, (54.17, 443.37, 350.46, 454.74))


async def test_rotated_page_adapter_blocks_stay_in_user_space(tmp_path: Path) -> None:
    source = write_text_pdf(
        tmp_path / "rot90.pdf", [_LINE], width=_PAGE_W, height=_PAGE_H, rotation=90
    )

    blocks = [b async for ch in PDFiumAdapter().parse_stream(source) for b in ch.blocks]
    assert len(blocks) == 1
    bbox = blocks[0].bbox
    assert bbox is not None and bbox.page == 1
    assert _close(bbox.x0, 54.2) and _close(bbox.x1, 350.5), (bbox.x0, bbox.x1)
    assert _close(bbox.y0, 443.4) and _close(bbox.y1, 454.7), (bbox.y0, bbox.y1)
    assert (bbox.x1 - bbox.x0) > (bbox.y1 - bbox.y0)


_LEFT = ("Left heading.", "Left body line one.", "Left body line two.")
_RIGHT = ("Right heading.", "Right body line one.", "Right body line two.")


def test_two_column_lines_are_read_in_column_order(tmp_path: Path) -> None:
    source = write_two_column_pdf(tmp_path / "two_col.pdf", [(_LEFT, _RIGHT)])
    lines, size = extract_lines(source, 1)
    assert size == (612.0, 792.0)
    texts = [line.text for line in lines]
    assert texts == [*_LEFT, *_RIGHT], texts
    assert all(line.rect[0] < 200.0 for line in lines[: len(_LEFT)])
    assert all(line.rect[0] > 300.0 for line in lines[len(_LEFT) :])


async def test_two_column_adapter_blocks_split_on_the_gutter(tmp_path: Path) -> None:
    source = write_two_column_pdf(tmp_path / "two_col.pdf", [(_LEFT, _RIGHT)])

    blocks = [b async for ch in PDFiumAdapter().parse_stream(source) for b in ch.blocks]
    assert len(blocks) == 2, [b.source_text for b in blocks]
    first, second = blocks
    assert first.source_text.startswith("Left heading.")
    assert second.source_text.startswith("Right heading.")
    left_box = first.bbox
    right_box = second.bbox
    assert left_box is not None and right_box is not None
    assert left_box.x1 < right_box.x0
    _assert_rect_close(
        (left_box.x0, left_box.y0, left_box.x1, left_box.y1), (54.9, 703.4, 151.0, 746.7)
    )
    _assert_rect_close(
        (right_box.x0, right_box.y0, right_box.x1, right_box.y1), (334.1, 703.4, 438.0, 746.6)
    )


def _page_with_boxes(
    path: Path,
    mediabox: tuple[float, float, float, float],
    cropbox: tuple[float, float, float, float],
) -> Path:
    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.pages[0][NameObject("/MediaBox")] = ArrayObject([FloatObject(v) for v in mediabox])
    writer.pages[0][NameObject("/CropBox")] = ArrayObject([FloatObject(v) for v in cropbox])
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_cropbox_page_reports_both_boxes_in_one_frame(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "crop.pdf", [_LINE], width=600.0, height=800.0)
    source = _page_with_boxes(
        source, mediabox=(50.0, 60.0, 650.0, 860.0), cropbox=(100.0, 120.0, 500.0, 700.0)
    )
    pdf = pdfium.PdfDocument(str(source))
    try:
        page = pdf[0]
        mediabox = tuple(float(v) for v in page.get_mediabox())
        cropbox = tuple(float(v) for v in page.get_cropbox())
        assert all(
            _close(g, w) for g, w in zip(mediabox, (50.0, 60.0, 650.0, 860.0), strict=True)
        ), mediabox
        assert all(
            _close(g, w) for g, w in zip(cropbox, (100.0, 120.0, 500.0, 700.0), strict=True)
        ), cropbox
        assert page.get_size() == (400.0, 580.0)
    finally:
        pdf.close()

    lines, size = extract_lines(source, 1)
    assert size == (600.0, 800.0)
    assert len(lines) == 1
    _assert_rect_close(lines[0].rect, (54.17, 743.37, 350.46, 754.74))


__all__ = ["pytest"]
