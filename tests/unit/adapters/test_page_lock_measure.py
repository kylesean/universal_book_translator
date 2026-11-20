"""The plan-time page-lock probe: what it measures, and what it refuses to guess.

The router promises a 1:1 page lock when ``page_lock_feasible`` is true. That
signal used to be a constant, so these tests pin the two properties that make
it a measurement: a page that cannot hold its own text is vetoed, and a
document the probe could not read is reported as *unmeasured* rather than
feasible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.page_lock_measure import (
    _body_lines,
    _capacity_at_floor,
    _column_groups,
    _line_records,
    _sample_page_numbers,
    measure_page_lock,
)
from ubt.delivery.contracts import ExpansionRisk, PageKind
from ubt.delivery.planning import build_delivery_signals, plan_for_document
from ubt.render.layout_solver import check_source_overlap, solve_page_2d

pytestmark = pytest.mark.fast

_BODY_LINES = [
    f"Line {i} of body text on this page, reasonably long as prose goes." for i in range(30)
]


def _pdf(tmp_path: Path, pages: list[list[str]]) -> Path:
    from pdf_builders import write_text_pdf

    return write_text_pdf(tmp_path / "doc.pdf", pages)


def test_sample_page_numbers_are_even_and_bounded() -> None:
    assert _sample_page_numbers(0, 8) == []
    assert _sample_page_numbers(3, 8) == [1, 2, 3]
    sampled = _sample_page_numbers(100, 8)
    assert len(sampled) == 8
    assert sampled[0] == 1 and sampled[-1] == 100
    assert sampled == sorted(set(sampled))


def test_line_records_measure_glyph_aspect_from_the_line() -> None:
    class _Line:
        def __init__(self, text: str, rect: tuple[float, float, float, float], size: float):
            self.text = text
            self.rect = rect
            self.font_size = size

    lines = [
        _Line("abcdefghij", (50.0, 700.0, 110.0, 712.0), 12.0),
        _Line("klmnopqrst", (50.0, 684.0, 110.0, 696.0), 12.0),
    ]
    records, size, leading = _line_records(lines)
    assert size == pytest.approx(12.0)
    assert len(records) == 2
    # 60pt advance over 10 glyphs at 12pt = 0.5em per glyph.
    assert records[0]["glyph_aspect"] == pytest.approx(0.5)
    assert leading == pytest.approx(16.0)


class _Line:
    def __init__(self, text: str, rect: tuple[float, float, float, float], size: float):
        self.text = text
        self.rect = rect
        self.font_size = size


def _body_line(top: float, text: str = "A reasonably long body line of prose.") -> _Line:
    return _Line(text, (60.0, top - 11.0, 552.0, top), 10.0)


def test_a_punctuation_fragment_is_not_a_paragraph() -> None:
    """pdfium reports the comma after a body line as its own 1.3pt "line" that
    sits inside the body box. Counting it vetoed every academic paper."""
    body = _body_line(700.0)
    fragment = _Line(",", (413.5, 709.5, 414.8, 711.5), 10.0)

    kept = _body_lines([body, fragment])

    assert kept == [body]
    records, _, _ = _line_records(kept)
    assert check_source_overlap([records]) is False


def test_a_small_table_header_is_not_body_text() -> None:
    lines = [_body_line(700.0 - i * 12.0) for i in range(8)]
    header = _Line("Mean Δ", (60.0, 380.0, 90.0, 386.5), 6.5)

    kept = _body_lines([*lines, header])

    assert header not in kept


def test_a_real_body_overlap_is_still_detected() -> None:
    first = _Line("First paragraph line", (60.0, 700.0, 552.0, 712.0), 10.0)
    second = _Line("Second paragraph line", (60.0, 705.0, 552.0, 717.0), 10.0)

    records, _, _ = _line_records(_body_lines([first, second]))

    assert check_source_overlap([records]) is True


def test_two_columns_are_compared_per_column() -> None:
    left = [
        _Line(f"left line {i}", (60.0, 700.0 - i * 12.0, 300.0, 711.0 - i * 12.0), 10.0)
        for i in range(6)
    ]
    right = [
        _Line(f"right line {i}", (324.0, 700.0 - i * 12.0, 552.0, 711.0 - i * 12.0), 10.0)
        for i in range(6)
    ]

    records, _, _ = _line_records(_body_lines([*left, *right]))

    # Split at the page midpoint, each column has its full 12pt pitch.
    groups = _column_groups(records, 612.0)
    assert len(groups) == 2
    assert check_source_overlap(groups) is False
    for group in groups:
        solution = solve_page_2d([group], source_size_pt=10.0, source_leading_pt=12.0, floor_pt=7.5)
        assert solution.fits is True


def test_capacity_shrinks_as_the_page_fills() -> None:
    roomy = [
        {
            "top": 750.0 - i * 12,
            "bot": 740.0 - i * 12,
            "w": 480.0,
            "glyphs": 80,
            "glyph_aspect": 0.5,
        }
        for i in range(10)
    ]
    packed = [
        {
            "top": 750.0 - i * 7.2,
            "bot": 742.0 - i * 7.2,
            "w": 480.0,
            "glyphs": 80,
            "glyph_aspect": 0.5,
        }
        for i in range(40)
    ]
    assert _capacity_at_floor(roomy, 1.33, 7.5) > _capacity_at_floor(packed, 1.33, 7.5)


def test_a_normal_page_measures_feasible_with_headroom(tmp_path: Path) -> None:
    measurement = measure_page_lock(_pdf(tmp_path, [_BODY_LINES, _BODY_LINES]))
    assert measurement.measured is True
    assert measurement.feasible is True
    assert measurement.source_overlap is False
    assert measurement.expansion_capacity > 0.0
    assert measurement.expansion_risk in {"low", "medium", "high"}


def test_an_unreadable_document_is_unmeasured_not_feasible(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    measurement = measure_page_lock(broken)
    assert measurement.measured is False
    assert measurement.reasons


def test_a_text_free_document_is_unmeasured(tmp_path: Path) -> None:
    measurement = measure_page_lock(_pdf(tmp_path, [[], []]))
    assert measurement.measured is False


def test_measured_signals_replace_the_census_defaults() -> None:
    signals = build_delivery_signals(
        document_mode="paper",
        is_pdf=True,
        page_lock_feasible=False,
        source_overlap=True,
        expansion_risk="high",
    )
    assert signals.page_lock_measured is True
    assert signals.page_lock_feasible is False
    assert signals.source_overlap is True
    assert signals.expansion_risk is ExpansionRisk.HIGH
    assert signals.page_kind is PageKind.NATIVE


def test_a_scan_is_vetoed_even_when_the_probe_says_feasible() -> None:
    """The probe cannot make a scan's text layer trustworthy: the census veto
    outranks it."""
    signals = build_delivery_signals(
        document_mode="paper",
        is_pdf=True,
        has_scan=True,
        page_lock_feasible=True,
        source_overlap=False,
        expansion_risk="low",
    )
    assert signals.page_lock_feasible is False


def test_a_measured_faithful_plan_is_not_labelled_provisional() -> None:
    decision = plan_for_document(
        "faithful",
        document_mode="paper",
        is_pdf=True,
        page_lock_feasible=True,
        source_overlap=False,
        expansion_risk="low",
    )
    assert decision.plan.contract is not None
    assert decision.plan.eligibility.provisional is False
    assert decision.plan.reasons


def test_an_unmeasured_faithful_plan_stays_provisional() -> None:
    decision = plan_for_document("faithful", document_mode="paper", is_pdf=True)
    assert decision.plan.eligibility.provisional is True
    assert any("provisional" in reason for reason in decision.plan.reasons)
