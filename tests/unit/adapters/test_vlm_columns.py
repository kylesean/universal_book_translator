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
