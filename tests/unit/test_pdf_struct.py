"""Structural census helpers in ubt.adapters.pdf.pdf_struct."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf import pdf_struct

pytestmark = pytest.mark.fast


def _image_pdf(path: Path, pages: int, image_pages: set[int]) -> Path:
    with pikepdf.new() as doc:
        for i in range(pages):
            page = doc.add_blank_page(page_size=(595, 842))
            if i in image_pages:
                stream = pikepdf.Stream(doc, b"\x00")
                stream.Type = pikepdf.Name("/XObject")
                stream.Subtype = pikepdf.Name("/Image")
                stream.Width = 1
                stream.Height = 1
                stream.ColorSpace = pikepdf.Name("/DeviceGray")
                stream.BitsPerComponent = 8
                page.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=stream))
                page.Contents = doc.make_stream(b"q 100 0 0 100 72 72 cm /Im0 Do Q")
        doc.save(path)
    return path


def test_resource_image_count_sees_image_streams(tmp_path: Path) -> None:
    """An /Image XObject is a pikepdf Stream, not a Dictionary.

    The census required ``isinstance(obj, Dictionary)``, which a Stream never
    satisfies, so every PDF on earth counted zero images and the
    vector-diagram tell never fired from an illustration.
    """
    pdf = _image_pdf(tmp_path / "img.pdf", pages=2, image_pages={0})
    with pdf_struct.open_pdf(pdf) as doc:
        assert pdf_struct.resource_image_count(doc.pages[0]) == 1
        assert pdf_struct.resource_image_count(doc.pages[1]) == 0
