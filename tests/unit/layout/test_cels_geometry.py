"""CELS Stage 1 geometry: gulter refusal, track assignment, overrun reporting.

The two guards under test are the ones measured on ``chapter-3.pdf`` p12 — a
margin fragment that looks like a gutter by width alone, and a lone right-side
box with no vertical overlap. Synthetic pages pin the behaviour so a future
threshold change has to face the same cases the corpus did.
"""

from __future__ import annotations

import pytest

from ubt.core.layout import (
    Box,
    ColumnTrack,
    assign_lines,
    detect_columns,
    pin_from,
    track_overruns,
)

pytestmark = pytest.mark.fast


def _line(x0: float, y0: float, width: float, height: float = 9.0) -> Box:
    return Box(x0=x0, y0=y0, x1=x0 + width, y1=y0 + height)


def _two_column_page() -> list[Box]:
    boxes: list[Box] = []
    for row in range(12):
        y = 600.0 - row * 12.0
        boxes.append(_line(84.0, y, 224.0))
        boxes.append(_line(332.0, y, 224.0))
    return boxes


def test_single_column_page_yields_one_track() -> None:
    boxes = [_line(84.0, 600.0 - row * 12.0, 372.0) for row in range(10)]

    detection = detect_columns(boxes, page_width=540.0, page_height=666.0)

    assert len(detection.tracks) == 1
    track = detection.tracks[0]
    assert track.x0 == pytest.approx(84.0)
    assert track.x1 == pytest.approx(456.0)
    assert detection.ink_extent == pytest.approx((84.0, 456.0))
    assert not detection.is_multicolumn


def test_genuine_two_column_page_is_split_at_the_gutter() -> None:
    boxes = _two_column_page()

    detection = detect_columns(boxes, page_width=540.0, page_height=666.0)

    assert len(detection.tracks) == 2
    left, right = detection.tracks
    assert left.x0 == pytest.approx(84.0)
    assert left.x1 == pytest.approx(308.0)
    assert right.x0 == pytest.approx(332.0)
    assert right.x1 == pytest.approx(556.0)
    assert len(detection.accepted_gutters) == 1
    assert detection.accepted_gutters[0].width == pytest.approx(24.0)


def test_margin_fragment_is_refused_because_one_side_has_no_ink() -> None:
    """p12's trap: a wide band in the right margin has nothing on its right.

    Accepting it would split one column into a body column plus a 4% strip,
    and every translated line would then be forced into that strip, so the
    guard has to refuse it and say why.
    """
    boxes = [_line(84.0, 600.0 - row * 12.0, 333.4) for row in range(20)]
    boxes.append(_line(434.8, 556.4, 21.1))

    detection = detect_columns(boxes, page_width=540.0, page_height=666.0)

    assert len(detection.tracks) == 1, "a margin note must not become a column"
    assert detection.rejected_gutters, "the refusal has to be reported, not dropped"
    rejection = max(detection.rejected_gutters, key=lambda gap: gap.width)
    assert rejection.width == pytest.approx(38.5, abs=0.6)
    assert "margin" in rejection.reason
    assert rejection.right_ink_share == pytest.approx(0.0)


def test_narrow_sidebar_is_refused_by_ink_share() -> None:
    """A narrow column of numbers is a binder, not a second text column."""
    body = [_line(84.0, 600.0 - row * 12.0, 200.0) for row in range(20)]
    sidebar = [_line(320.0, 600.0 - row * 12.0, 30.0) for row in range(6)]

    detection = detect_columns(body + sidebar, page_width=612.0, page_height=792.0)

    assert len(detection.tracks) == 1
    rejection = max(detection.rejected_gutters, key=lambda gap: gap.width)
    assert "ink share" in rejection.reason
    assert rejection.right_ink_share < 0.15


def test_gutter_without_vertical_overlap_is_refused() -> None:
    """Two bands stacked vertically are two blocks, not two columns."""
    left = [_line(84.0, 600.0 - row * 12.0, 200.0) for row in range(20)]
    right = [_line(320.0, 260.0 - row * 12.0, 200.0) for row in range(10)]

    detection = detect_columns(left + right, page_width=612.0, page_height=792.0)

    assert len(detection.tracks) == 1
    assert any("vertical overlap" in gap.reason for gap in detection.rejected_gutters)


def test_full_width_title_does_not_erase_the_gutter() -> None:
    """A title spanning both columns would fill the gutter's empty bins."""
    boxes = _two_column_page()
    boxes.append(_line(84.0, 640.0, 448.0, height=16.0))

    detection = detect_columns(boxes, page_width=540.0, page_height=666.0)

    assert len(detection.tracks) == 2
    assignments, leftover = assign_lines(boxes, detection.tracks)
    assert leftover == (boxes[-1],), "the title spans both tracks and is pinned"
    assert all(len(items) == 12 for items in assignments.values())


def test_wide_title_alone_is_one_track() -> None:
    boxes = [_line(84.0, 640.0, 448.0, height=16.0)]

    detection = detect_columns(boxes, page_width=540.0, page_height=666.0)

    assert len(detection.tracks) == 1


def test_three_column_page_recurses() -> None:
    boxes: list[Box] = []
    for row in range(10):
        y = 600.0 - row * 12.0
        boxes.extend(
            [
                _line(84.0, y, 140.0),
                _line(248.0, y, 140.0),
                _line(412.0, y, 140.0),
            ]
        )

    detection = detect_columns(boxes, page_width=612.0, page_height=792.0, max_tracks=4)

    assert len(detection.tracks) == 3
    assert [round(track.x0, 1) for track in detection.tracks] == [84.0, 248.0, 412.0]


def test_empty_page_yields_the_page_box() -> None:
    detection = detect_columns([], page_width=612.0, page_height=792.0)

    assert len(detection.tracks) == 1
    assert detection.tracks[0].width == pytest.approx(612.0)


def test_assign_lines_reports_margin_objects_as_leftovers() -> None:
    tracks = (
        ColumnTrack(index=0, x0=84.0, x1=308.0, y0=500.0, y1=600.0),
        ColumnTrack(index=1, x0=332.0, x1=556.0, y0=500.0, y1=600.0),
    )
    inside = _line(90.0, 580.0, 200.0)
    margin = _line(560.0, 585.0, 20.0)
    spanning = Box(x0=84.0, y0=620.0, x1=556.0, y1=636.0)

    assignments, leftover = assign_lines([inside, margin, spanning], tracks)

    assert assignments[0] == (inside,)
    assert assignments[1] == ()
    assert leftover == (margin, spanning)


def test_track_overruns_reports_the_right_limit_breach() -> None:
    track = ColumnTrack(index=0, x0=84.0, x1=308.0, y0=500.0, y1=600.0)
    inside = _line(90.0, 580.0, 200.0)
    breach = _line(90.0, 560.0, 230.0)

    overruns = track_overruns([track], [inside, breach])

    assert len(overruns) == 1
    assert overruns[0].track_index == 0
    assert overruns[0].box == breach
    assert overruns[0].overrun_pt == pytest.approx(12.0)


def test_pin_from_keeps_the_kind_and_flow_behaviour() -> None:
    box = Box(x0=84.0, y0=620.0, x1=556.0, y1=636.0)

    pin = pin_from(box)
    beside = pin_from(box, "figure", flow_beside=True)

    assert pin.kind == "full_width"
    assert not pin.flow_beside
    assert beside.kind == "figure"
    assert beside.flow_beside
