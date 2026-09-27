"""Rotated-page coordinate-frame guards.

pdfium's ``get_rect`` returns unrotated user-space rects while
``get_width/height`` return the rotated display size. Mixing them (as the
geometry layer used to) clips zones and misplaces the anchored overlay. The
line extractor now reports the unrotated mediabox size, and the rigid
engine skips rotated pages outright (demote to source-visible, never abort
the whole book).
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from ubt.adapters.pdf.rigid.extract import extract_pages
from ubt.adapters.pdf.textgeom import extract_lines


def _make_pdf(path: Path, rotation: int | None = None) -> Path:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=100)
    if rotation is not None:
        page.rotate(rotation)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_extract_lines_reports_unrotated_mediabox(tmp_path: Path) -> None:
    pdf_path = _make_pdf(tmp_path / "rot90.pdf", rotation=90)
    _lines, size = extract_lines(pdf_path, 1)
    # Unrotated mediabox, matching the rect frame — NOT the rotated (100, 200).
    assert size == (200.0, 100.0)


def test_extract_pages_skips_rotated_pages(tmp_path: Path) -> None:
    # A rotated page is demoted to source-visible (omitted from facts),
    # NOT a whole-book abort — the rest of the document still renders rigid.
    pdf_path = _make_pdf(tmp_path / "rot270.pdf", rotation=270)
    facts = extract_pages(pdf_path, [1])
    assert 1 not in facts


def test_extract_pages_allows_unrotated_pages(tmp_path: Path) -> None:
    pdf_path = _make_pdf(tmp_path / "plain.pdf")
    facts = extract_pages(pdf_path, [1])
    assert facts[1].width == 200.0
    assert facts[1].height == 100.0
