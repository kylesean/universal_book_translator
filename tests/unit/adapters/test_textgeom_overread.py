"""A rect may not contribute more text than its own box can hold.

Regression (arXiv 2609.20519, measured 2026-10-25): page 1 emits 9,395 pdfium
rects over 104,456 characters, and ``get_text_bounded`` over each of them sums
to 3,220,105 characters — 30.8x the page's own text. The rects are nested, so
each box selects its neighbours' glyphs as well as its own. ``extract_lines``
turned that into 1,331,556 characters (12.7x the page), one block reached the
translator carrying 146,289 characters of duplicated fragments, and the whole
document parsed to 1,298,561 characters against an assessed 160,848 — the
delivery was wrong *and* the quote under-billed it 8x.

The per-rect over-read guard already existed but ran only when
``_drop_scatter_debris`` had dropped something, i.e. only on a marker cloud.
This page clears that precondition: its rects are 3.6pt tall, above the cloud's
2pt median, so 91% of its debris-shaped rects were kept and every one of them
over-read. The guard now runs unconditionally; these tests pin both halves —
the tell fires on a nested-rect page, and it stays silent on ordinary ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pdf_builders import write_dotted_toc_pdf, write_marker_cloud_pdf, write_text_pdf

from ubt.adapters.pdf.textgeom import LineBox, extract_lines

pytestmark = pytest.mark.fast

_REPO = Path(__file__).resolve().parents[3]
#: The page the regression was measured on. ``corpus/documents`` is gitignored,
#: so the exact-shape test skips without it; the synthetic tests below always run.
_NESTED_RECT_PAGE = _REPO / "corpus" / "documents" / "2609.20519v1.pdf"


def _chars(lines: Sequence[LineBox]) -> int:
    return sum(len(line.text) for line in lines)


def test_an_ordinary_page_is_unchanged(tmp_path: Path) -> None:
    """The guard must not eat a normal page: every line survives verbatim."""
    body = [f"plain body line number {i}" for i in range(20)]
    pdf = write_text_pdf(tmp_path / "prose.pdf", [body])
    lines, _size = extract_lines(pdf, 1)
    assert [line.text for line in lines] == body


def test_a_dotted_toc_keeps_its_rows(tmp_path: Path) -> None:
    """Leader dots are many rects but short ones — never over-read."""
    pdf = write_dotted_toc_pdf(tmp_path / "toc.pdf", rows=30, dots_per_row=70)
    lines, _size = extract_lines(pdf, 1)
    assert len(lines) >= 30
    assert max(len(line.text) for line in lines) <= 240


def test_a_marker_cloud_still_reports_its_drop(tmp_path: Path) -> None:
    """The cloud path is the guard's original home and must keep working."""
    pdf = write_marker_cloud_pdf(tmp_path / "cloud.pdf", 700, body=["Real body line one."])
    lines, _size = extract_lines(pdf, 1)
    assert any("Real body line one." in line.text for line in lines)


@pytest.mark.skipif(
    not _NESTED_RECT_PAGE.exists(),
    reason="corpus/documents is gitignored; fetch it to run the measured-shape test",
)
def test_a_nested_rect_page_does_not_duplicate_its_text() -> None:
    """The measured page: extracted text must stay near its text-layer size.

    Before the fix this page extracted 1,331,556 characters from a 104,456
    character text layer. The bar is deliberately loose (2.5x) because a page
    can legitimately extract slightly more than ``count_chars`` reports — the
    failure it guards against is 12.7x, not 1.05x.
    """
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
