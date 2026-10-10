"""Engine selection: preset routing, the pdfium default, and scan escape.

The probe (``inspect_pdf_route_plan``) reports layout facts; the factory
(``_resolve_pdf_adapter``) turns them into an adapter choice. These tests pin
the factory half: which engine each entry point selects, and the one case
that is a capability boundary rather than a preference — a scanned PDF
yields zero blocks under pdfium, so it always escapes to docling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf, write_two_column_pdf

from ubt.adapters.pdf.engine_selector import inspect_pdf_route_plan

pytestmark = pytest.mark.fast

_PROSE = "The quick brown fox jumps over the lazy dog near the river bank today"


def _prose_pages(n: int) -> list[list[str]]:
    return [[_PROSE, _PROSE, _PROSE, _PROSE, _PROSE, _PROSE] for _ in range(n)]


def _two_column_pages() -> list[tuple[list[str], list[str]]]:
    return [
        (
            ["left column line one", "left column line two", "left column three"],
            ["right column line one", "right column line two", "right column three"],
        )
        for _ in range(10)
    ]


def test_scanned_pdf_reports_scan_signal(tmp_path: Path) -> None:
    # A text-less page is the capability boundary: pdfium extracts 0 blocks from
    # it, so the plan must carry the scan flag for the factory to escape to docling.
    import pypdfium2 as pdfium

    pdf = write_text_pdf(tmp_path / "blank.pdf", [[] for _ in range(10)])
    plan = inspect_pdf_route_plan(pdf)
    assert plan.has_scan
    pdfium.PdfDocument(str(pdf)).close()  # the fixture really is text-less


def test_pdfium_default_keeps_fast_path_on_layout_signal(tmp_path: Path) -> None:
    # The fast mode honours pdfium even when the probe saw a two-column layout.
    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter

    pages = [
        (
            ["left column line one", "left column line two", "left column three"],
            ["right column line one", "right column line two", "right column three"],
        )
        for _ in range(10)
    ]
    pdf = write_two_column_pdf(tmp_path / "twocol.pdf", pages)
    assert inspect_pdf_route_plan(pdf).has_multicolumn
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="pdfium"), PDFiumAdapter)


def test_auto_routing_sends_two_column_to_docling(tmp_path: Path) -> None:
    # ``auto`` is the quality-routing mode the publication/standard presets use.
    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    pages = [
        (
            ["left column line one", "left column line two", "left column three"],
            ["right column line one", "right column line two", "right column three"],
        )
        for _ in range(10)
    ]
    pdf = write_two_column_pdf(tmp_path / "twocol.pdf", pages)
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="auto"), DoclingPDFAdapter)


def test_pdfium_default_escapes_a_scanned_pdf_to_docling(tmp_path: Path) -> None:
    # pdfium cannot read a scanned document (0 blocks), so even the fast mode
    # must route it to docling — a capability boundary, not a preference.
    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    pdf = write_text_pdf(tmp_path / "scan.pdf", [[] for _ in range(10)])
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="pdfium"), DoclingPDFAdapter)


def test_explicit_docling_beats_everything(tmp_path: Path) -> None:
    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    pdf = write_text_pdf(tmp_path / "prose.pdf", _prose_pages(6))
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="docling"), DoclingPDFAdapter)
