"""Tests for pdfium geometric text extraction helpers (``ubt/adapters/pdf/textgeom.py``)."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.textgeom import LineBox, column_order

pytestmark = pytest.mark.fast


def _order(lines: list[LineBox], page_width: float = 600.0) -> list[str]:
    return [ln.text for ln in column_order(lines, page_width)]


def test_centered_banner_does_not_bridge_two_columns() -> None:
    """A centered line spanning the gutter must not collapse the page to top-down.

    Regression: connected-component clustering merged every overlapping interval,
    so one centered banner (width in (0.15, 0.7]·span) chained the left and right
    columns into a single cluster and the page read ``TITLE,L0,R0,L1,R1`` —
    interleaved, wrong reading order.
    """
    lines = [
        LineBox(text="TITLE", rect=(150, 700, 450, 720)),
        LineBox(text="L0", rect=(50, 650, 290, 665)),
        LineBox(text="R0", rect=(310, 650, 550, 665)),
        LineBox(text="L1", rect=(50, 630, 290, 645)),
        LineBox(text="R1", rect=(310, 630, 550, 645)),
    ]
    order = _order(lines)
    assert [t for t in order if t in {"L0", "L1"}] == ["L0", "L1"], order
    assert [t for t in order if t in {"R0", "R1"}] == ["R0", "R1"], order
    # No interleaving: every left-column line precedes every right-column line.
    assert order.index("L1") < order.index("R0"), order
    # Banner placed above both columns must be read before the columns.
    assert order.index("TITLE") < order.index("L0"), order


def test_two_columns_read_left_before_right() -> None:
    """Without any spanning line the two columns must still read left-first."""
    lines = [
        LineBox(text="L0", rect=(50, 700, 290, 715)),
        LineBox(text="R0", rect=(310, 700, 550, 715)),
        LineBox(text="L1", rect=(50, 680, 290, 695)),
        LineBox(text="R1", rect=(310, 680, 550, 695)),
    ]
    assert _order(lines) == ["L0", "L1", "R0", "R1"]


def test_single_column_ragged_lines_stay_top_down() -> None:
    """A ragged single column must not be split into fake columns."""
    lines = [
        LineBox(text="a long line spanning most of the column", rect=(50, 700, 550, 715)),
        LineBox(text="short", rect=(50, 680, 150, 695)),
        LineBox(text="a medium-length line here", rect=(50, 660, 300, 675)),
        LineBox(text="another long line spanning the column", rect=(50, 640, 545, 655)),
    ]
    assert _order(lines) == [
        "a long line spanning most of the column",
        "short",
        "a medium-length line here",
        "another long line spanning the column",
    ]


def test_full_width_lines_do_not_form_columns() -> None:
    """Only full-width lines: there is no gutter, so order is strict top-down."""
    lines = [
        LineBox(text="a", rect=(0, 700, 600, 715)),
        LineBox(text="b", rect=(0, 680, 600, 695)),
    ]
    assert _order(lines) == ["a", "b"]
