from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pdf_builders import write_dotted_toc_pdf, write_marker_cloud_pdf, write_text_pdf

from ubt.adapters.pdf.textgeom import LineBox, extract_lines

pytestmark = pytest.mark.fast

_REPO = Path(__file__).resolve().parents[3]
_NESTED_RECT_PAGE = _REPO / "corpus" / "documents" / "2609.20519v1.pdf"


def _chars(lines: Sequence[LineBox]) -> int:
    return sum(len(line.text) for line in lines)


def test_an_ordinary_page_is_unchanged(tmp_path: Path) -> None:
    body = [f"plain body line number {i}" for i in range(20)]
    pdf = write_text_pdf(tmp_path / "prose.pdf", [body])
    lines, _size = extract_lines(pdf, 1)
    assert [line.text for line in lines] == body


def test_a_dotted_toc_keeps_its_rows(tmp_path: Path) -> None:
    pdf = write_dotted_toc_pdf(tmp_path / "toc.pdf", rows=30, dots_per_row=70)
    lines, _size = extract_lines(pdf, 1)
    assert len(lines) >= 30
    assert max(len(line.text) for line in lines) <= 240


def test_a_marker_cloud_still_reports_its_drop(tmp_path: Path) -> None:
    pdf = write_marker_cloud_pdf(tmp_path / "cloud.pdf", 700, body=["Real body line one."])
    lines, _size = extract_lines(pdf, 1)
    assert any("Real body line one." in line.text for line in lines)


@pytest.mark.skipif(
    not _NESTED_RECT_PAGE.exists(),
    reason="corpus/documents is gitignored; fetch it to run the measured-shape test",
)
def test_a_nested_rect_page_does_not_duplicate_its_text() -> None:
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(str(_NESTED_RECT_PAGE)) as pdf:
        page = pdf[0]
        textpage = page.get_textpage()
        try:
            layer_chars = int(textpage.count_chars())
        finally:
            textpage.close()
            page.close()

    lines, _size = extract_lines(_NESTED_RECT_PAGE, 1)
    extracted = _chars(lines)
    assert extracted <= 2.5 * layer_chars, (
        f"page 1 extracted {extracted:,} chars from a {layer_chars:,} char text "
        f"layer ({extracted / layer_chars:.1f}x) — the rect sweep is duplicating"
    )
