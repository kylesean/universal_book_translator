"""Tests for engine_selector PageIngestPlan and PDFRoutePlan."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.corpus_markers import requires_synthetic_mono
from ubt.adapters.pdf.engine_selector import (
    PageIngestPlan,
    PDFRoutePlan,
    _scan_detected_from_probe,
    build_page_ingest_plans,
    inspect_pdf_route_plan,
)


@requires_synthetic_mono
def test_build_page_ingest_plans_real_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # NOTE: tests/fixtures/*.pdf were removed (HEAD "update" slimming);
    # docs/synthetic-mono.pdf is the same 13-page sample, kept as the live fixture.
    pdf = Path("docs/synthetic-mono.pdf").resolve()
    monkeypatch.chdir(tmp_path)
    plans = build_page_ingest_plans(pdf, cache_dir=tmp_path / ".ubt_cache")
    assert len(plans) == 13
    assert plans[0].page_number == 1


def test_build_page_ingest_plans_missing_file_returns_empty() -> None:
    plans = build_page_ingest_plans(Path("/nonexistent/dummy.pdf"))
    assert plans == []


def test_scan_detected_from_probe_uses_all_sampled_pages() -> None:
    """Scan detection must aggregate sampled pages, not just page index 0."""
    assert _scan_detected_from_probe([]) is False
    assert _scan_detected_from_probe([500, 500, 500, 500, 500]) is False
    assert _scan_detected_from_probe([0, 0, 0, 0, 0]) is True
    # Text cover, image-only body: 4/5 low-text must still route to Docling.
    assert _scan_detected_from_probe([500, 0, 0, 0, 0]) is True
    # One image page among text pages stays born-digital.
    assert _scan_detected_from_probe([500, 0, 500, 500, 0]) is False


@requires_synthetic_mono
def test_inspect_pdf_route_plan_populates_page_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = Path("docs/synthetic-mono.pdf").resolve()
    monkeypatch.chdir(tmp_path)
    route = inspect_pdf_route_plan(pdf, cache_dir=tmp_path / ".ubt_cache", include_page_plans=True)
    assert isinstance(route, PDFRoutePlan)
    assert len(route.page_plans) == 13
    assert route.page_plans[0].page_number == 1


@requires_synthetic_mono
def test_route_plan_probe_does_not_profile_every_page_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-page plans used to be built on every engine selection.

    The probe samples a handful of pages; building ``page_plans`` profiles all
    of them, and ``select_pdf_engine`` (called twice per auto job by the adapter
    factory) threw the result away — no production reader consumes the field.
    """
    import ubt.adapters.pdf.engine_selector as selector

    pdf = Path("docs/synthetic-mono.pdf").resolve()
    monkeypatch.chdir(tmp_path)
    calls: list[Path] = []

    def _spy(path: Path, cache_dir: Path | None = None) -> list[PageIngestPlan]:
        calls.append(path)
        return []

    monkeypatch.setattr(selector, "build_page_ingest_plans", _spy)

    assert selector.select_pdf_engine(pdf) in ("docling", "pdfium")
    assert inspect_pdf_route_plan(pdf).page_plans == ()
    assert calls == []

    # And the opt-in still works for callers that do want the plans.
    inspect_pdf_route_plan(pdf, cache_dir=tmp_path / ".ubt_cache", include_page_plans=True)
    assert calls == [pdf]


@pytest.mark.fast
def test_engine_selector_pikepdf_does_not_shadow_pdfium(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    from ubt.adapters.pdf import engine_selector

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=200, height=200)
    pdf_path = tmp_path / "probe_test.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    closed_docs: list[Any] = []
    real_pdfium_doc = pdfium.PdfDocument

    def monitored_pdf_doc(*args: Any, **kwargs: Any) -> Any:
        doc = real_pdfium_doc(*args, **kwargs)
        real_close = doc.close

        def _close() -> None:
            closed_docs.append(doc)
            real_close()

        doc.close = _close
        return doc

    with patch("pypdfium2.PdfDocument", side_effect=monitored_pdf_doc):
        plan = engine_selector.inspect_pdf_route_plan(pdf_path)
        assert plan is not None
        assert len(closed_docs) == 1, "Outer pdfium document must be closed in finally block"
