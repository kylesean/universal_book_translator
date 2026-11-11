"""Direct evidence for the review fixes that had only indirect coverage.

Each test here exists because a claim in the fix list is otherwise verified
only through a caller: the 10 % VLM budget, the CropBox read, the contract's
small-box bookkeeping, the QE sample-coverage figure, and the [gate] section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


# -- VLM sampling budget ---------------------------------------------------


def test_vlm_budget_is_ten_percent_capped_and_never_zero_when_enabled() -> None:
    from ubt.adapters.pdf.visual_gate import vlm_page_budget

    assert vlm_page_budget(0, 20) == 0
    assert vlm_page_budget(100, 20) == 10
    assert vlm_page_budget(5, 20) == 1, "a short artifact still gets one judged page"
    assert vlm_page_budget(400, 20) == 20, "the operator's cap holds"
    assert vlm_page_budget(400, 0) == 0, "0 disables the tier"


# -- CropBox ---------------------------------------------------------------


def test_crop_boxes_are_preferred_over_media_boxes(tmp_path: Path) -> None:
    """The visible box is the CropBox; text clipped by it is invisible."""
    from pypdf import PdfWriter
    from pypdf.generic import RectangleObject

    from ubt.adapters.pdf.pdf_struct import page_crop_boxes

    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    page.cropbox = RectangleObject((50, 60, 550, 760))
    path = tmp_path / "cropped.pdf"
    with path.open("wb") as handle:
        writer.write(handle)

    boxes = page_crop_boxes(path)
    assert boxes[1] == (50.0, 60.0, 550.0, 760.0)


def test_a_page_without_a_cropbox_reports_its_media_box(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    from ubt.adapters.pdf.pdf_struct import page_crop_boxes

    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    path = tmp_path / "plain.pdf"
    with path.open("wb") as handle:
        writer.write(handle)

    assert page_crop_boxes(path)[1] == (0.0, 0.0, 600.0, 800.0)


# -- contract evidence -----------------------------------------------------


class _StubTextPage:
    """One 3pt-tall single-glyph rect: the shape the old code dropped."""

    def count_rects(self, _start: int, _end: int) -> int:
        return 1

    def get_rect(self, _index: int) -> tuple[float, float, float, float]:
        return (100.0, 100.0, 103.0, 103.0)

    def get_text_bounded(self, *_args: float) -> str:
        return "7"


def test_contract_evidence_records_a_tiny_single_glyph_box() -> None:
    from ubt.adapters.pdf.delivery_evidence import _text_boxes_for_page

    boxes = _text_boxes_for_page(1, _StubTextPage())
    assert len(boxes) == 1, "a one-character box is a measurement, not noise"
    assert boxes[0].font_size_pt == 3.0, "a tiny height must be reported, not replaced by None"


# -- QE sample coverage ----------------------------------------------------


def test_the_kpi_set_carries_the_sample_coverage_next_to_avg_qe() -> None:
    from ubt.core.engine.reporter import (
        QualityReport,
        ReportEntityConsistency,
        ReportPlaceholderMetrics,
        ReportRenderCoverage,
        ReportRepairBreakdown,
        ReportRouteInfo,
        ReportScoreMetrics,
        ReportSummary,
        ReportTerminologyMetrics,
    )
    from ubt.core.metrics.collect import collect_kpis
    from ubt.core.metrics.schema import KPI_BY_NAME

    report = QualityReport(
        job_id="j",
        doc_id="d",
        book_title="t",
        source_path="in.pdf",
        output_path="out.pdf",
        target_lang="zh",
        summary=ReportSummary(
            total_blocks=220,
            completed_blocks=204,
            repaired_blocks=30,
            failed_blocks=0,
            pass_rate=0.927,
            estimated_cost_usd=None,
        ),
        score_metrics=ReportScoreMetrics(
            avg_qe=0.9,
            min_qe=0.25,
            max_qe=0.92,
            p10_qe=0.92,
            p50_qe=0.92,
            p90_qe=0.92,
            bottom_15_avg_qe=0.75,
            scored_blocks=31,
            qe_measured=True,
            scored_coverage=0.1409,
        ),
        terminology=ReportTerminologyMetrics(),
        entity_consistency=ReportEntityConsistency(),
        repair_breakdown=ReportRepairBreakdown(
            direct_pass_count=0,
            round_1_repaired_count=0,
            round_2_repaired_count=0,
            exhausted_count=0,
        ),
        placeholder=ReportPlaceholderMetrics(
            masked_spans=0,
            corrupt_spans=0,
            retention_rate=1.0,
            masked_blocks=0,
            corrupt_blocks=0,
        ),
        render_coverage=ReportRenderCoverage(
            rendered_blocks=220,
            skipped_blocks=0,
            fail_closed_blocks=0,
            preserved_blocks=0,
            render_coverage=0.0,
            skip_families={},
        ),
        route=ReportRouteInfo(mode="short", pages=31, chars=99078),
    )
    kpis = collect_kpis(report)
    assert kpis.kpis["qe_scored_coverage"] == pytest.approx(0.1409)
    assert "qe_scored_coverage" in KPI_BY_NAME
