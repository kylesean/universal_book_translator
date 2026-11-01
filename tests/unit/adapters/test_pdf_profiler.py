from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.pdf_profiler import (
    PROFILE_CACHE_DIR,
    PDFDocumentProfile,
    PDFRoutePlan,
    inspect_pdf_profile,
)
from ubt.core.policy.layout_policy import PROBE_MIN_SIGNAL_PAGES, PROBE_SAMPLE_FRACTIONS

pytestmark = pytest.mark.fast

from pdf_builders import write_text_pdf, write_two_column_pdf  # noqa: E402

_PROSE = "The quick brown fox jumps over the lazy dog near the river bank today"
_REFERENCE = "P. Zijlstra, J. Fernandes, and V. Pillai. Core scheduling, 2021."


def _prose_pages(n: int) -> list[list[str]]:
    return [[_PROSE, _PROSE, _PROSE, _PROSE, _PROSE, _PROSE] for _ in range(n)]


def test_probe_sample_fractions_skip_the_last_page() -> None:
    assert max(PROBE_SAMPLE_FRACTIONS) < 1.0
    assert PROBE_SAMPLE_FRACTIONS[0] == 0.0


def test_signal_bar_needs_more_than_one_page() -> None:
    assert PROBE_MIN_SIGNAL_PAGES == 2


def test_probe_routes_every_document_to_docling(tmp_path: Path) -> None:
    pdf = write_text_pdf(tmp_path / "prose.pdf", _prose_pages(10))
    plan = inspect_pdf_profile(pdf)
    assert not plan.has_formulas
    assert not plan.has_multicolumn
    assert plan.has_scan is False
    assert plan.primary_engine == "docling"


def test_probe_routes_two_column_to_docling(tmp_path: Path) -> None:
    pages = [
        (
            ["left column line one", "left column line two", "left column three"],
            ["right column line one", "right column line two", "right column three"],
        )
        for _ in range(10)
    ]
    pdf = write_two_column_pdf(tmp_path / "twocol.pdf", pages)
    plan = inspect_pdf_profile(pdf)
    assert plan.has_multicolumn
    assert plan.primary_engine == "docling"


def test_single_reference_page_does_not_veto_the_document(tmp_path: Path) -> None:
    pages = _prose_pages(9) + [[_REFERENCE, _REFERENCE, _REFERENCE, _REFERENCE]]
    pdf = write_text_pdf(tmp_path / "refs.pdf", pages)
    plan = inspect_pdf_profile(pdf)
    assert not plan.has_formulas, "one reference page must not read as formula-dense"


def test_single_formula_page_does_not_veto_the_document(tmp_path: Path) -> None:
    debris = "K = x plus V t h"
    pages = _prose_pages(9) + [[debris, debris, debris, debris]]
    pdf = write_text_pdf(tmp_path / "one_debris.pdf", pages)
    plan = inspect_pdf_profile(pdf)
    assert not plan.has_formulas


def test_two_formula_pages_route_to_docling(tmp_path: Path) -> None:
    debris = "K = x plus V t h"
    pages = _prose_pages(10)
    pages[2] = [debris, debris, debris, debris]
    pages[7] = [debris, debris, debris, debris]
    pdf = write_text_pdf(tmp_path / "two_debris.pdf", pages)
    plan = inspect_pdf_profile(pdf)
    assert plan.has_formulas
    assert plan.primary_engine == "docling"


def test_majority_formula_debris_routes_to_docling(tmp_path: Path) -> None:
    debris = "K = x plus V t h"
    pages = [[debris, debris, debris, debris] for _ in range(10)]
    pdf = write_text_pdf(tmp_path / "all_debris.pdf", pages)
    plan = inspect_pdf_profile(pdf)
    assert plan.has_formulas
    assert plan.primary_engine == "docling"


def test_all_reference_pages_do_not_report_a_formula_signal(tmp_path: Path) -> None:
    pages = [[_REFERENCE, _REFERENCE, _REFERENCE, _REFERENCE] for _ in range(10)]
    pdf = write_text_pdf(tmp_path / "all_refs.pdf", pages)
    plan = inspect_pdf_profile(pdf)
    assert not plan.has_formulas
    assert isinstance(plan, PDFDocumentProfile)


def test_alias_and_cache_parity(tmp_path: Path) -> None:
    assert PDFRoutePlan is PDFDocumentProfile
    assert PROFILE_CACHE_DIR.name == "profile_cache"

    pdf = write_text_pdf(tmp_path / "alias_test.pdf", _prose_pages(5))
    profile = inspect_pdf_profile(pdf)
    assert isinstance(profile, PDFRoutePlan)
