"""Contract tests for the PDF engine probe (``engine_selector``).

The probe samples a handful of pages and routes a document to ``docling`` or
``pdfium``. Its two failure shapes are asymmetric: routing a clean document to
docling costs time and GPU memory, while routing a two-column or scanned
document to pdfium silently interleaves columns (the delivery contract still
reports PASS — it checks that text was not *lost*, not that it is *correct*).

These tests pin the 2026-10-10 fixes:

* a single reference page no longer vetoes the whole document (a bibliography is
  dense with name initials that scored as formula debris);
* a genuinely two-column document still routes to docling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.engine_selector import inspect_pdf_route_plan
from ubt.core.policy.layout_policy import PROBE_MIN_SIGNAL_PAGES, PROBE_SAMPLE_FRACTIONS

pytestmark = pytest.mark.fast

from pdf_builders import write_text_pdf, write_two_column_pdf  # noqa: E402

_PROSE = "The quick brown fox jumps over the lazy dog near the river bank today"
_REFERENCE = "P. Zijlstra, J. Fernandes, and V. Pillai. Core scheduling, 2021."


def _prose_pages(n: int) -> list[list[str]]:
    return [[_PROSE, _PROSE, _PROSE, _PROSE, _PROSE, _PROSE] for _ in range(n)]


def test_probe_sample_fractions_skip_the_last_page() -> None:
    # 1.0 is the last page of a paper — its reference list. Sampling it made a
    # bibliography out-score the body and route the book to docling.
    assert max(PROBE_SAMPLE_FRACTIONS) < 1.0
    assert PROBE_SAMPLE_FRACTIONS[0] == 0.0


def test_signal_bar_needs_more_than_one_page() -> None:
    # One page is noise; two is a signal. A strict majority is too high for a
    # five-page sample: a 296-page maths textbook with 41% formula pages draws
    # only 2 of 5 samples, and a >50% bar would send it to pdfium.
    assert PROBE_MIN_SIGNAL_PAGES == 2


def test_probe_routes_every_document_to_docling(tmp_path: Path) -> None:
    """`auto` must be lossless, so it is always docling.

    pdfium's reader emits only headings, paragraphs, list items and code — it
    cannot represent a table, a figure or a formula. That used to be survivable
    because the overlay renderer kept the source page as a canvas, so an
    unextracted table was still visible in the original language. The reflow
    renderer rebuilds the page, so the same omission is a hole in the book.

    The probe cannot rule structure out: measured on the reference corpus every
    document carried structure, and one had 19 figures and 16 tables with every
    cheap signal reading zero (vector-drawn figures are invisible to
    ``resource_image_count``). So the probe reports the signals it saw, and
    ``auto`` takes the lossless engine regardless.
    """
    pdf = write_text_pdf(tmp_path / "prose.pdf", _prose_pages(10))
    plan = inspect_pdf_route_plan(pdf)
    assert not plan.has_formulas
    assert not plan.has_multicolumn
    # The signals are reported honestly (assess shows them to the operator)…
    assert plan.has_scan is False
    # …and the engine is the fixed lossless policy.
    assert plan.primary_engine == "docling"


def test_probe_routes_two_column_to_docling(tmp_path: Path) -> None:
    # A genuine two-column body: pdfium would interleave the columns, so the
    # probe must keep routing these to docling.
    pages = [
        (
            ["left column line one", "left column line two", "left column three"],
            ["right column line one", "right column line two", "right column three"],
        )
        for _ in range(10)
    ]
    pdf = write_two_column_pdf(tmp_path / "twocol.pdf", pages)
    plan = inspect_pdf_route_plan(pdf)
    assert plan.has_multicolumn
    assert plan.primary_engine == "docling"


def test_single_reference_page_does_not_veto_the_document(tmp_path: Path) -> None:
    # Nine clean prose pages plus one reference page: the reference page scores
    # as formula debris on its own, but one page must not flip the whole book.
    pages = _prose_pages(9) + [[_REFERENCE, _REFERENCE, _REFERENCE, _REFERENCE]]
    pdf = write_text_pdf(tmp_path / "refs.pdf", pages)
    plan = inspect_pdf_route_plan(pdf)
    assert not plan.has_formulas, "one reference page must not read as formula-dense"


def test_single_formula_page_does_not_veto_the_document(tmp_path: Path) -> None:
    # One page of formula debris among clean prose must not flip the route: a
    # single figure page reads as debris (2026-10-10).
    debris = "K = x plus V t h"
    pages = _prose_pages(9) + [[debris, debris, debris, debris]]
    pdf = write_text_pdf(tmp_path / "one_debris.pdf", pages)
    plan = inspect_pdf_route_plan(pdf)
    assert not plan.has_formulas


def test_two_formula_pages_route_to_docling(tmp_path: Path) -> None:
    # Two sampled pages is a signal, even though it is below a strict majority
    # of a five-page sample (the maths-textbook case). With 10 pages the probe
    # samples 1, 3, 6, 8, 10, so the debris goes on pages 3 and 8.
    debris = "K = x plus V t h"
    pages = _prose_pages(10)
    pages[2] = [debris, debris, debris, debris]
    pages[7] = [debris, debris, debris, debris]
    pdf = write_text_pdf(tmp_path / "two_debris.pdf", pages)
    plan = inspect_pdf_route_plan(pdf)
    assert plan.has_formulas
    assert plan.primary_engine == "docling"


def test_majority_formula_debris_routes_to_docling(tmp_path: Path) -> None:
    # When most pages carry real shattered-equation debris, the signal is real
    # and the majority rule must fire. (Reference entries are deliberately NOT
    # used here: after the name-initial fix they no longer read as debris, which
    # is the point of the previous test.)
    debris = "K = x plus V t h"
    pages = [[debris, debris, debris, debris] for _ in range(10)]
    pdf = write_text_pdf(tmp_path / "all_debris.pdf", pages)
    plan = inspect_pdf_route_plan(pdf)
    assert plan.has_formulas
    assert plan.primary_engine == "docling"


def test_all_reference_pages_do_not_report_a_formula_signal(tmp_path: Path) -> None:
    # A book of nothing but references is still not a formula book. Pinned so a
    # future tweak cannot quietly reintroduce the 2026-10-10 false positive.
    pages = [[_REFERENCE, _REFERENCE, _REFERENCE, _REFERENCE] for _ in range(10)]
    pdf = write_text_pdf(tmp_path / "all_refs.pdf", pages)
    plan = inspect_pdf_route_plan(pdf)
    assert not plan.has_formulas
