"""Unit tests for Typst physical layout spill and overflow diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ubt.render.typst.layout_diagnostics import (
    A4_HEIGHT_PT,
    A4_WIDTH_PT,
    AnomalyType,
    PageLayoutSpec,
    Severity,
    diagnose_pdf_layout,
    is_page_furniture,
    render_ascii_heatmap,
    scan_page_anomalies,
)

pytestmark = pytest.mark.fast


def _make_line(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    font_size: float = 10.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        rect=(float(x0), float(y0), float(x1), float(y1)),
        font_size=float(font_size),
    )


def test_layout_spec_geometry_defaults() -> None:
    """Verify standard page layout specifications and safe boundaries."""
    book = PageLayoutSpec.book()
    assert book.name == "book"
    assert book.columns == 1
    assert pytest.approx(book.page_width, 0.1) == A4_WIDTH_PT
    assert pytest.approx(book.page_height, 0.1) == A4_HEIGHT_PT
    assert pytest.approx(book.safe_x0, 0.1) == 62.36
    assert pytest.approx(book.safe_x1, 0.1) == 532.92
    assert pytest.approx(book.safe_y0, 0.1) == 68.03
    assert pytest.approx(book.safe_y1, 0.1) == 773.86
    assert book.printable_width > 400.0

    dual = PageLayoutSpec.academic_dual()
    assert dual.name == "academic_dual"
    assert dual.columns == 2
    assert dual.gutter == 29.0
    assert pytest.approx(dual.safe_x0, 0.1) == 51.02
    assert pytest.approx(dual.safe_x1, 0.1) == 544.26
    assert dual.column_width < 250.0

    single = PageLayoutSpec.academic_single()
    assert single.name == "academic_single"
    assert single.columns == 1
    assert pytest.approx(single.safe_x0, 0.1) == 70.87

    from_name_dual = PageLayoutSpec.from_name("academic-dual")
    assert from_name_dual.columns == 2
    from_name_book = PageLayoutSpec.from_name("book")
    assert from_name_book.columns == 1


def test_page_furniture_recognition() -> None:
    """Ensure footer page numbers and top running headers are recognized as furniture."""
    ph = A4_HEIGHT_PT
    margin_b = 68.0

    # Legitimate page numbers in footer
    assert is_page_furniture("1", 40.0, 50.0, ph, margin_b)
    assert is_page_furniture("14", 42.0, 52.0, ph, margin_b)
    assert is_page_furniture("- 42 -", 38.0, 48.0, ph, margin_b)
    assert is_page_furniture("xiv", 40.0, 50.0, ph, margin_b)
    assert is_page_furniture("", 40.0, 50.0, ph, margin_b)

    # Regular body text in footer zone is NOT furniture (it is a spill!)
    assert not is_page_furniture(
        "This is an overflowing paragraph that leaked into the bottom margin.",
        40.0,
        50.0,
        ph,
        margin_b,
    )

    # Running header at top of page
    assert is_page_furniture("Chapter 3: Semiconductor Physics", ph - 30.0, ph - 20.0, ph, margin_b)


def test_scan_page_clean_layout() -> None:
    """A page with well-behaved text within safe margins emits no anomalies."""
    spec = PageLayoutSpec.book()
    lines = [
        _make_line("First body paragraph perfectly placed", 70.0, 700.0, 450.0, 712.0),
        _make_line("Second body paragraph within margins", 70.0, 680.0, 500.0, 692.0),
        _make_line("1", 290.0, 40.0, 305.0, 50.0),  # Footer page number
    ]
    report = scan_page_anomalies(lines, 1, spec)
    assert not report.has_errors
    assert not report.has_warnings
    assert len(report.anomalies) == 0


def test_scan_page_detects_right_margin_spill() -> None:
    """Text spilling horizontally beyond the right margin boundary is flagged."""
    spec = PageLayoutSpec.book(tolerance_pt=5.0)
    # safe_x1 is ~532.9pt. A line reaching 545pt is a moderate WARN; reaching 560pt is a FAIL.
    warn_line = _make_line("Slight table spill", 70.0, 600.0, 545.0, 612.0)
    fail_line = _make_line("Massive math overflow \\int_0^\\infty ...", 70.0, 550.0, 570.0, 562.0)

    report_warn = scan_page_anomalies([warn_line], 1, spec)
    assert len(report_warn.anomalies) == 1
    assert report_warn.anomalies[0].anomaly_type == AnomalyType.MARGIN_SPILL_RIGHT
    assert report_warn.anomalies[0].severity == Severity.WARN

    report_fail = scan_page_anomalies([fail_line], 2, spec)
    assert len(report_fail.anomalies) == 1
    assert report_fail.anomalies[0].anomaly_type == AnomalyType.MARGIN_SPILL_RIGHT
    assert report_fail.anomalies[0].severity == Severity.FAIL
    assert report_fail.anomalies[0].delta_pt > 30.0
    assert "Wrap long inline math" in report_fail.anomalies[0].suggested_action


def test_scan_page_detects_left_and_bottom_spills() -> None:
    """Left margin breaches and bottom footer leaks are detected."""
    spec = PageLayoutSpec.book(tolerance_pt=5.0)
    # safe_x0 is ~62.4pt; left edge at 40pt
    left_spill = _make_line("Shifted table column", 40.0, 600.0, 400.0, 612.0)
    # safe_y0 is ~68.0pt; bottom edge at 10pt (critical overflow off page bottom)
    bottom_spill = _make_line("Trailing line clipped at page bottom", 70.0, 8.0, 400.0, 18.0)

    report = scan_page_anomalies([left_spill, bottom_spill], 3, spec)
    assert len(report.anomalies) == 2
    types = {a.anomaly_type for a in report.anomalies}
    assert AnomalyType.MARGIN_SPILL_LEFT in types
    assert AnomalyType.MARGIN_SPILL_BOTTOM in types
    assert any(a.severity == Severity.FAIL for a in report.anomalies)


def test_scan_page_dual_column_gutter_overrun() -> None:
    """In multi-column mode, text overflowing from Column 1 into the central gutter is caught."""
    spec = PageLayoutSpec.academic_dual(tolerance_pt=4.0)
    # safe_x0 = 51.02, col1_width = 232.12 -> col1_x1 = 283.14. Gutter is [283.14, 312.14]
    # Line starting at 55pt and ending at 305pt overruns into gutter by ~21pt without spanning Col 2
    gutter_line = _make_line("Column 1 line overrunning into gutter", 55.0, 600.0, 305.0, 610.0)

    report = scan_page_anomalies([gutter_line], 4, spec)
    assert any(a.anomaly_type == AnomalyType.GUTTER_OVERRUN for a in report.anomalies)
    gutter_anom = next(a for a in report.anomalies if a.anomaly_type == AnomalyType.GUTTER_OVERRUN)
    assert gutter_anom.delta_pt > 15.0
    assert gutter_anom.severity == Severity.FAIL


def test_scan_page_detects_type_collapse() -> None:
    """Text shrunk below readable font threshold is flagged as type collapse."""
    spec = PageLayoutSpec.book()
    collapsed_line = _make_line(
        "Microscopic unreadable text", 70.0, 500.0, 300.0, 505.0, font_size=3.2
    )
    warn_line = _make_line("Small footnote text", 70.0, 480.0, 300.0, 485.0, font_size=4.4)

    report = scan_page_anomalies([collapsed_line, warn_line], 5, spec)
    type_anoms = [a for a in report.anomalies if a.anomaly_type == AnomalyType.TYPE_COLLAPSE]
    assert len(type_anoms) == 2
    assert any(a.severity == Severity.FAIL for a in type_anoms)
    assert any(a.severity == Severity.WARN for a in type_anoms)


def test_ascii_heatmap_generation() -> None:
    """ASCII heatmap rendering produces valid 2D grid containing boundary glyphs and anomaly highlights."""
    spec = PageLayoutSpec.book()
    spill_line = _make_line("Spill line", 70.0, 550.0, 580.0, 562.0)
    report = scan_page_anomalies([spill_line], 1, spec)

    heatmap = render_ascii_heatmap(report, spec, width_chars=30, height_chars=12)
    assert "┌" in heatmap
    assert "┘" in heatmap
    assert "┆" in heatmap  # Margin outline
    assert "█" in heatmap  # Critical anomaly highlight


def test_diagnose_pdf_layout_real_document() -> None:
    """End-to-end diagnosis over real corpus PDF file."""
    doc_path = Path("corpus/documents/chapter-3.pdf")
    if not doc_path.is_file():
        pytest.skip("chapter-3.pdf not available")

    report = diagnose_pdf_layout(doc_path, spec=PageLayoutSpec.book(), pages=[1, 2])
    assert report.total_pages == 26
    assert report.checked_pages == [1, 2]
    assert 1 in report.page_reports
    assert 2 in report.page_reports

    data = report.to_dict()
    assert "pdf_path" in data
    assert "total_pages" in data
    assert "anomalies" in data
    assert isinstance(report.to_json(), str)


def test_diagnose_pdf_layout_non_existent_file() -> None:
    """Missing PDF raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        diagnose_pdf_layout(Path("non_existent_12345.pdf"))
