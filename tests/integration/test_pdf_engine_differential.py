"""Differential test: the two PDF engines must agree on the book's content.

The pdfium and Docling adapters are two independent readers of the same bytes:
pdfium is geometric text harvesting, Docling is a neural layout model -- if
they disagree about what a page *says*, one of them is dropping or inventing
prose, and neither a masker nor the renderer can recover it later. The engines
are allowed to segment differently (Docling merges a heading into its
paragraph; pdfium keeps it separate -- both are valid), so the invariant is
not block-for-block equality but the one the e2e ingest test already pins for
pdfium alone (``tests/unit/adapters/test_pdf_adapter_e2e.py``):

1. **Content parity** -- over a synthetic document, each engine's extraction
   carries exactly the source's non-space characters: nothing missing,
   nothing invented. Checking each engine against the *source* (not engine
   against engine) localizes the defect: a missing char says which engine
   lost it.
2. **Page attribution** -- both engines stamp every block with the page its
   text sits on, so a resume or a page preview reads the same book no matter
   which engine analyzed it.
3. **Page-range subsets** -- the resume path extracts ``pages={n}``; both
   engines return exactly that page's content and no neighbour's.
4. **Fail-closed parity** -- on an unparseable file both engines must raise
   rather than return a silently empty book (a false "0 pages" document
   would flow through the pipeline as a completed empty translation).

The documents are built with :mod:`tests.pdf_builders` (pypdf, an independent
writer) -- single-column, two-column, and ``/Rotate``-displayed pages, the
three shapes the roadmap names (golden-geometry T4 covers the rest).

Why ``slow``: the Docling half runs the real layout model (~4 s/document
after the local HF cache warms; a cold cache downloads weights, which the
``network`` marker of the default tier forbids). CI installs only
``--extra dev``, so the module skips when the ``docling`` extra is absent --
mirroring the corpus harnesses. Run it locally with
``uv sync --extra pdf && pytest -m slow``.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf, write_two_column_pdf

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.core.ir.models import IRBlock

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not DoclingPDFAdapter().is_docling_installed(),
        reason="the docling extra is not installed; the differential needs both engines",
    ),
]

#: Single-column prose: heading + body per page (Docling merges the pair,
#: pdfium splits them -- the segmentation difference the parity invariant
#: deliberately tolerates).
_PAGES: tuple[tuple[str, ...], ...] = (
    (
        "The Differential Fixture",
        "The first paragraph states the shared text between engines.",
    ),
    (
        "The Descendants",
        "The second page carries more prose for both engines to read.",
    ),
)

_LEFT = (
    "Left column heading.",
    "The left column continues its argument here.",
    "More left prose follows.",
)
_RIGHT = (
    "Right column heading.",
    "The right column answers the left one.",
    "More right prose follows.",
)


def _chars(text: str) -> Counter[str]:
    """Whitespace-insensitive character histogram (segmentation may differ)."""
    return Counter(ch for ch in text if not ch.isspace())


def _source_text(pages: Sequence[Sequence[str]]) -> str:
    return "\n".join(line for page in pages for line in page)


async def _collect(adapter: object, path: Path, pages: set[int] | None = None) -> list[IRBlock]:
    blocks: list[IRBlock] = []
    stream = adapter.parse_stream(path, pages=pages)  # type: ignore[attr-defined]
    assert isinstance(stream, AsyncIterator)
    async for chapter in stream:
        blocks.extend(chapter.blocks)
    return blocks


def _histogram(blocks: Sequence[IRBlock]) -> Counter[str]:
    histogram: Counter[str] = Counter()
    for block in blocks:
        histogram.update(_chars(block.source_text))
    return histogram


def _assert_content_parity(label: str, source: str, blocks: Sequence[IRBlock]) -> None:
    """Every non-space source char survives extraction; nothing was invented."""
    assert blocks, f"{label}: the adapter extracted no blocks from a born-digital page"
    assert all(block.source_text.strip() for block in blocks), (
        f"{label}: an extracted block carries empty text"
    )
    want = _chars(source)
    got = _histogram(blocks)
    missing = want - got
    invented = got - want
    assert sum(missing.values()) == 0, f"{label}: extraction dropped {dict(missing)}"
    assert sum(invented.values()) == 0, f"{label}: extraction invented {dict(invented)}"


def test_both_engines_extract_the_full_document_losslessly(tmp_path: Path) -> None:
    """Content parity over a multi-page single-column document, engine vs source."""
    pdf = write_text_pdf(tmp_path / "single.pdf", _PAGES)
    source = _source_text(_PAGES)
    pdfium_blocks = asyncio.run(_collect(PDFiumAdapter(), pdf))
    docling_blocks = asyncio.run(_collect(DoclingPDFAdapter(), pdf))
    _assert_content_parity("pdfium", source, pdfium_blocks)
    _assert_content_parity("docling", source, docling_blocks)


def test_both_engines_agree_on_page_attribution(tmp_path: Path) -> None:
    """Each engine stamps its blocks with the page the text actually sits on."""
    pdf = write_text_pdf(tmp_path / "pages.pdf", _PAGES)
    pdfium_pages = {b.bbox.page for b in asyncio.run(_collect(PDFiumAdapter(), pdf)) if b.bbox}
    docling_pages = {b.bbox.page for b in asyncio.run(_collect(DoclingPDFAdapter(), pdf)) if b.bbox}
    assert pdfium_pages == {1, 2}, f"pdfium pages: {sorted(pdfium_pages)}"
    assert docling_pages == {1, 2}, f"docling pages: {sorted(docling_pages)}"


def test_both_engines_read_a_two_column_page_losslessly(tmp_path: Path) -> None:
    """The column layout must not cost either engine any content."""
    pdf = write_two_column_pdf(tmp_path / "two_col.pdf", [(_LEFT, _RIGHT)])
    source = _source_text([_LEFT, _RIGHT])
    pdfium_blocks = asyncio.run(_collect(PDFiumAdapter(), pdf))
    docling_blocks = asyncio.run(_collect(DoclingPDFAdapter(), pdf))
    _assert_content_parity("pdfium/two-column", source, pdfium_blocks)
    _assert_content_parity("docling/two-column", source, docling_blocks)


def test_both_engines_read_a_rotated_page_losslessly(tmp_path: Path) -> None:
    """A ``/Rotate 90`` display rotation must not rotate the extracted text away."""
    lines = ("Rotated heading.", "The rotated page body text for both engines.")
    pdf = write_text_pdf(tmp_path / "rotated.pdf", [lines], rotation=90)
    source = "\n".join(lines)
    pdfium_blocks = asyncio.run(_collect(PDFiumAdapter(), pdf))
    docling_blocks = asyncio.run(_collect(DoclingPDFAdapter(), pdf))
    _assert_content_parity("pdfium/rotated", source, pdfium_blocks)
    _assert_content_parity("docling/rotated", source, docling_blocks)


def test_both_engines_extract_the_same_page_range_subset(tmp_path: Path) -> None:
    """The resume path (``pages={2}``) returns exactly page 2 from both engines."""
    pdf = write_text_pdf(tmp_path / "book.pdf", _PAGES)
    page_two = _PAGES[1]
    source = "\n".join(page_two)
    for label, adapter in (("pdfium", PDFiumAdapter()), ("docling", DoclingPDFAdapter())):
        blocks = asyncio.run(_collect(adapter, pdf, pages={2}))
        pages = {b.bbox.page for b in blocks if b.bbox}
        assert pages == {2}, f"{label}: page subset returned pages {sorted(pages)}"
        _assert_content_parity(f"{label}/page-2", source, blocks)


def test_both_engines_fail_closed_on_an_unparseable_file(tmp_path: Path) -> None:
    """A corrupt PDF must raise in both engines, never yield an empty book."""
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"%PDF-1.4 not actually a pdf \x00\x01 garbage")
    for label, adapter in (("pdfium", PDFiumAdapter()), ("docling", DoclingPDFAdapter())):
        with pytest.raises(Exception, match=r"."):
            asyncio.run(_collect(adapter, bad))
        # The message contract: something parseable enough to act on.
        try:
            asyncio.run(_collect(adapter, bad))
        except Exception as exc:
            assert str(exc), f"{label}: raised {type(exc).__name__} with no message"


__all__ = ["pytest"]
