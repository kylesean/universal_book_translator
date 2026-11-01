from __future__ import annotations

import struct
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.pdf.oxide_render import extract_page_texts, render_page_png

pytestmark = pytest.mark.fast


def _png_dimensions(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width, height = struct.unpack(">II", png[16:24])
    return width, height


def test_the_four_symbols_behave_as_oxide_render_assumes(tmp_path: Path) -> None:
    from pdf_oxide import PdfDocument

    pdf = tmp_path / "contract.pdf"
    write_text_pdf(pdf, ["The four symbols the engine relies on."])
    doc = PdfDocument(str(pdf))
    assert doc.page_count == 1

    rendered_72 = bytes(doc.render_page(0, dpi=72))
    assert rendered_72[:4] == b"\x89PNG"
    rendered_144 = bytes(doc.render_page(0, dpi=144))
    w72, h72 = _png_dimensions(rendered_72)
    w144, h144 = _png_dimensions(rendered_144)
    assert (w144, h144) == (w72 * 2, h72 * 2)

    text = doc.extract_text(0)
    assert isinstance(text, str)
    assert "symbols" in "".join(text.split())


def test_oxide_render_legs_map_the_conventions(tmp_path: Path) -> None:
    pdf = tmp_path / "legs.pdf"
    write_text_pdf(pdf, ["First page body.", "Second page body."])
    png = render_page_png(pdf, page=1, dpi=72)
    assert png is not None and png[:4] == b"\x89PNG"
    assert render_page_png(pdf, page=2, dpi=72) is not None
    assert render_page_png(pdf, page=3, dpi=72) is None
    pages = extract_page_texts(pdf)
    assert len(pages) == 2
    assert "Second" in "".join(pages[1].split())
