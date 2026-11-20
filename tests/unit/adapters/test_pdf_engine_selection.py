# ubt-pdfium-allow: single-threaded pytest; the test owns its fixture document and finishes before exit.

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf, write_two_column_pdf

from ubt.adapters.pdf.pdf_profiler import inspect_pdf_profile

pytestmark = pytest.mark.fast

_PROSE = "The quick brown fox jumps over the lazy dog near the river bank today"


def _prose_pages(n: int) -> list[list[str]]:
    return [[_PROSE, _PROSE, _PROSE, _PROSE, _PROSE, _PROSE] for _ in range(n)]


def test_scanned_pdf_reports_scan_signal(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    pdf = write_text_pdf(tmp_path / "blank.pdf", [[] for _ in range(10)])
    plan = inspect_pdf_profile(pdf)
    assert plan.has_scan
    pdfium.PdfDocument(str(pdf)).close()


def test_auto_routing_sends_born_digital_to_native_and_scanned_to_docling(tmp_path: Path) -> None:
    import importlib.util

    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import ScanPDFToTypstCodec
    from ubt.adapters.pdf.native_adapter import NativePDFAdapter

    pages = [
        (
            ["left column line one", "left column line two", "left column three"],
            ["right column line one", "right column line two", "right column three"],
        )
        for _ in range(10)
    ]
    pdf = write_two_column_pdf(tmp_path / "twocol.pdf", pages)
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="auto"), NativePDFAdapter)

    if importlib.util.find_spec("docling") is not None:
        scan = write_text_pdf(tmp_path / "scan.pdf", [[] for _ in range(10)])
        assert isinstance(get_adapter_for_path(scan, pdf_engine="auto"), ScanPDFToTypstCodec)


def test_explicit_docling_beats_everything(tmp_path: Path) -> None:
    import importlib.util

    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import ScanPDFToTypstCodec
    from ubt.core.exceptions import OptionalDependencyError

    pdf = write_text_pdf(tmp_path / "prose.pdf", _prose_pages(6))
    if importlib.util.find_spec("docling") is None:
        with pytest.raises(OptionalDependencyError, match="docling"):
            get_adapter_for_path(pdf, pdf_engine="docling")
        return
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="docling"), ScanPDFToTypstCodec)


def test_docling_without_package_is_refused_not_degraded(tmp_path: Path) -> None:
    import importlib.util

    from ubt.adapters import get_adapter_for_path
    from ubt.core.exceptions import OptionalDependencyError

    if importlib.util.find_spec("docling") is not None:
        pytest.skip("docling is installed; the refusal path cannot be exercised")

    pdf = write_text_pdf(tmp_path / "prose.pdf", _prose_pages(6))
    with pytest.raises(OptionalDependencyError, match="uv sync --all-extras"):
        get_adapter_for_path(pdf, pdf_engine="docling")
