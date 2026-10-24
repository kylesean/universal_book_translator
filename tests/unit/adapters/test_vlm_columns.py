"""VLM line ordering: a two-column scan must read column by column."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.vlm.transcribe import (
    group_lines_to_paragraphs,
    order_lines_by_column,
)

pytestmark = pytest.mark.fast

# Two columns; the driver returned them interleaved top-down across the page.
_LEFT_TOP = (70.0, 700.0, 280.0, 712.0)
_RIGHT_TOP = (320.0, 700.0, 530.0, 712.0)
_LEFT_BOT = (70.0, 680.0, 280.0, 692.0)
_RIGHT_BOT = (320.0, 680.0, 530.0, 692.0)


def test_two_column_lines_are_ordered_by_column() -> None:
    lines = [
        ("L top", _LEFT_TOP),
        ("R top", _RIGHT_TOP),
        ("L bottom", _LEFT_BOT),
        ("R bottom", _RIGHT_BOT),
    ]
    order = order_lines_by_column([box for _t, box in lines], page_width=595.0)
    ordered = [lines[i][0] for i in order]
    # Left column first, top-down, then the right column.
    assert ordered == ["L top", "L bottom", "R top", "R bottom"]


def test_single_column_page_keeps_its_order() -> None:
    lines = [("a", (70.0, 700.0, 530.0, 712.0)), ("b", (70.0, 680.0, 530.0, 692.0))]
    order = order_lines_by_column([box for _t, box in lines], page_width=595.0)
    assert [lines[i][0] for i in order] == ["a", "b"]


def test_unequal_columns_do_not_sink_the_longer_column_to_the_footer() -> None:
    # The left column runs lower than the right. Its lower lines belong to the
    # column, not to a footer bucket: the inverted min/max threshold used to
    # misread them as footers and emit them after both columns.
    lines = [
        ("L top", (70.0, 700.0, 280.0, 712.0)),
        ("R top", (320.0, 700.0, 530.0, 712.0)),
        ("L mid", (70.0, 400.0, 280.0, 412.0)),
        ("R bottom", (320.0, 680.0, 530.0, 692.0)),
        ("L bottom", (70.0, 120.0, 280.0, 132.0)),
    ]
    order = order_lines_by_column([box for _t, box in lines], page_width=595.0)
    ordered = [lines[i][0] for i in order]
    assert ordered == ["L top", "L mid", "L bottom", "R top", "R bottom"]


def test_grouping_preserves_the_given_order() -> None:
    lines = [("a", (70.0, 700.0, 280.0, 712.0)), ("b", (320.0, 700.0, 530.0, 712.0))]
    # Different columns -> separate groups, in the order given.
    assert group_lines_to_paragraphs(lines) == [[0], [1]]


# --------------------------------------------------------------------------- #
# Three-column pages: the recursive gutter cut.
# --------------------------------------------------------------------------- #

_C1 = (54.0, 219.0)
_C2 = (250.0, 415.0)
_C3 = (446.0, 611.0)


def _column_lines(
    col: tuple[float, float], rows: int, *, label: str, top: float = 700.0
) -> list[tuple[str, tuple[float, float, float, float]]]:
    return [
        (f"{label}{i}", (col[0], top - i * 20.0, col[1], top - i * 20.0 + 12.0))
        for i in range(rows)
    ]


def test_three_column_lines_read_column_major() -> None:
    """A single gutter cut interleaved columns 2 and 3 line by line; the
    recursive cut reads each column whole, left to right."""
    lines = (
        _column_lines(_C1, 6, label="A")
        + _column_lines(_C2, 6, label="B")
        + _column_lines(_C3, 6, label="C")
    )
    order = order_lines_by_column([box for _t, box in lines], page_width=660.0)
    ordered = [lines[i][0] for i in order]
    assert ordered == [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(6)] + [
        f"C{i}" for i in range(6)
    ]


def test_three_columns_with_banner_and_footer() -> None:
    lines = (
        [("banner", (54.0, 720.0, 611.0, 732.0))]
        + _column_lines(_C1, 5, label="A")
        + _column_lines(_C2, 5, label="B")
        + _column_lines(_C3, 5, label="C")
        + [("footer", (54.0, 80.0, 611.0, 92.0))]
    )
    order = order_lines_by_column([box for _t, box in lines], page_width=660.0)
    ordered = [lines[i][0] for i in order]
    assert ordered[0] == "banner"
    assert ordered[-1] == "footer"
    assert ordered[1:-1] == [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)] + [
        f"C{i}" for i in range(5)
    ]


def test_stacked_layout_single_column_abstract_over_three_columns() -> None:
    """A page-wide abstract above a three-column body: the abstract is page
    furniture for the sub-split (excluded from the non-furniture crossing
    count), so the columns below it still split."""
    lines = (
        [
            ("abstract 1", (54.0, 720.0, 611.0, 732.0)),
            ("abstract 2", (54.0, 700.0, 611.0, 712.0)),
        ]
        + _column_lines(_C1, 5, label="A", top=660.0)
        + _column_lines(_C2, 5, label="B", top=660.0)
        + _column_lines(_C3, 5, label="C", top=660.0)
    )
    order = order_lines_by_column([box for _t, box in lines], page_width=660.0)
    ordered = [lines[i][0] for i in order]
    assert ordered[:2] == ["abstract 1", "abstract 2"]
    assert ordered[2:] == [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)] + [
        f"C{i}" for i in range(5)
    ]


def test_two_column_page_with_short_lines_inside_one_column() -> None:
    """A centered short title and a short line INSIDE one column of a real
    two-column page produce a zero-crossing pseudo-cut at the sub-split level;
    the guards must refuse it so the column reads plain top-down — exactly what
    the single-cut implementation did (verified element-for-element against it).
    """
    left = [
        ("L1", (70.0, 700.0, 280.0, 712.0)),
        ("L2", (70.0, 680.0, 280.0, 692.0)),
        ("L3", (70.0, 660.0, 280.0, 672.0)),
        ("L short centre", (140.0, 630.0, 210.0, 642.0)),
        ("L4", (70.0, 600.0, 280.0, 612.0)),
        ("L5", (70.0, 580.0, 280.0, 592.0)),
        ("L6", (70.0, 560.0, 280.0, 572.0)),
        ("L short aside", (70.0, 530.0, 130.0, 542.0)),
    ]
    right = [
        ("R1", (320.0, 700.0, 530.0, 712.0)),
        ("R2", (320.0, 680.0, 530.0, 692.0)),
        ("R3", (320.0, 660.0, 530.0, 672.0)),
        ("R4", (320.0, 600.0, 530.0, 612.0)),
    ]
    lines = left + right
    order = order_lines_by_column([box for _t, box in lines], page_width=595.0)
    ordered = [lines[i][0] for i in order]
    # Left column whole, top-down (short lines in their y position), then right.
    assert ordered == [t for t, _ in left] + [t for t, _ in right]


def test_column_order_returns_the_input_objects() -> None:
    """The transcription path maps results back through ``id()``; the recursive
    implementation must keep returning the input objects themselves."""
    from ubt.adapters.pdf.textgeom import LineBox, column_order

    boxes = [
        LineBox(f"t{i}", (c[0], 700.0 - i * 20.0, c[1], 712.0 - i * 20.0))
        for c in (_C1, _C2, _C3)
        for i in range(4)
    ]
    ordered = column_order(boxes, 660.0)
    assert {id(lb) for lb in ordered} == {id(lb) for lb in boxes}
    assert len(ordered) == len(boxes)
