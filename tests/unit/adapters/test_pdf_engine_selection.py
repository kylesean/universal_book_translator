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

from ubt.adapters.pdf.pdf_profiler import inspect_pdf_profile as inspect_pdf_route_plan

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
    # ``auto`` is the quality-routing mode the high_fidelity/standard presets use.
    import importlib.util

    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    if importlib.util.find_spec("docling") is None:
        pytest.skip("docling is not installed; the refusal path is pinned separately")

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
    import importlib.util

    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    if importlib.util.find_spec("docling") is None:
        pytest.skip("docling is not installed; the refusal path is pinned separately")

    pdf = write_text_pdf(tmp_path / "scan.pdf", [[] for _ in range(10)])
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="pdfium"), DoclingPDFAdapter)


def test_explicit_docling_beats_everything(tmp_path: Path) -> None:
    import importlib.util

    from ubt.adapters import get_adapter_for_path
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.core.exceptions import OptionalDependencyError

    pdf = write_text_pdf(tmp_path / "prose.pdf", _prose_pages(6))
    if importlib.util.find_spec("docling") is None:
        # An explicit request for an engine that cannot run is refused before
        # ingest: silently degrading would deliver a book with every table,
        # figure and formula dropped, under a job id a later (fixed) run would
        # resume into.
        with pytest.raises(OptionalDependencyError, match="docling"):
            get_adapter_for_path(pdf, pdf_engine="docling")
        return
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="docling"), DoclingPDFAdapter)


def test_auto_without_docling_is_refused_not_degraded(tmp_path: Path) -> None:
    """``auto`` resolves to docling for every PDF, so a missing package refuses.

    The old behavior warned in server logs and degraded to the geometric
    pdfium reader — a translated "success" whose tables, figures and formulas
    were silently omitted, invisible to the web console that never shows logs.
    PDF translation is the product's primary scenario: a broken environment
    must stop the job with the repair command, not ship a residue book under
    a job id a later (fixed) run would resume into.
    """
    import importlib.util

    from ubt.adapters import get_adapter_for_path
    from ubt.core.exceptions import OptionalDependencyError

    if importlib.util.find_spec("docling") is not None:
        pytest.skip("docling is installed; the refusal path cannot be exercised")

    pdf = write_text_pdf(tmp_path / "prose.pdf", _prose_pages(6))
    with pytest.raises(OptionalDependencyError, match="uv sync --all-extras"):
        get_adapter_for_path(pdf, pdf_engine="auto")
