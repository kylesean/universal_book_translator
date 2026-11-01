from __future__ import annotations

import pytest

from ubt.core.layout.cels import Box, ColumnTrack
from ubt.core.layout.distortion import compute_page_distortion

pytestmark = pytest.mark.fast


def test_clean_page_produces_zero_distortion() -> None:
    printable = Box(x0=50.0, y0=50.0, x1=500.0, y1=700.0)
    track = ColumnTrack(index=0, x0=50.0, x1=500.0, y0=50.0, y1=700.0)
    boxes = [
        Box(x0=50.0, y0=60.0, x1=500.0, y1=80.0),
        Box(x0=50.0, y0=90.0, x1=500.0, y1=110.0),
    ]

    res = compute_page_distortion(
        page_no=1,
        boxes=boxes,
        tracks=[track],
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=10.0,
    )

    assert res.score == 0.0
    assert not res.is_critical
    assert res.verdict == "faithful_deliverable"
    assert res.subscores.overlap_ratio == 0.0
    assert res.subscores.out_of_bounds_ratio == 0.0
    assert res.subscores.style_drift == 0.0


def test_overlap_is_critical_fail() -> None:
    printable = Box(x0=50.0, y0=50.0, x1=500.0, y1=700.0)
    track = ColumnTrack(index=0, x0=50.0, x1=500.0, y0=50.0, y1=700.0)
    boxes = [
        Box(x0=50.0, y0=60.0, x1=500.0, y1=100.0),
        Box(x0=50.0, y0=80.0, x1=500.0, y1=120.0),
    ]

    res = compute_page_distortion(
        page_no=1,
        boxes=boxes,
        tracks=[track],
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=10.0,
    )

    assert res.is_critical
    assert res.verdict == "critical_fail"
    assert res.subscores.overlap_ratio > 0.0


def test_out_of_bounds_is_critical_fail() -> None:
    printable = Box(x0=50.0, y0=50.0, x1=500.0, y1=700.0)
    track = ColumnTrack(index=0, x0=50.0, x1=500.0, y0=50.0, y1=700.0)
    boxes = [
        Box(x0=50.0, y0=60.0, x1=550.0, y1=80.0),
    ]

    res = compute_page_distortion(
        page_no=1,
        boxes=boxes,
        tracks=[track],
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=10.0,
    )

    assert res.is_critical
    assert res.verdict == "critical_fail"
    assert res.subscores.out_of_bounds_ratio == 1.0


def test_style_drift_and_gutter_violation_within_thresholds() -> None:
    printable = Box(x0=50.0, y0=50.0, x1=550.0, y1=700.0)
    track1 = ColumnTrack(index=0, x0=50.0, x1=280.0, y0=50.0, y1=700.0)
    track2 = ColumnTrack(index=1, x0=320.0, x1=550.0, y0=50.0, y1=700.0)

    boxes = [
        Box(x0=50.0, y0=60.0, x1=280.0, y1=80.0),
        Box(x0=50.0, y0=90.0, x1=280.0, y1=110.0),
    ]

    res = compute_page_distortion(
        page_no=1,
        boxes=boxes,
        tracks=[track1, track2],
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=9.5,
    )

    assert not res.is_critical
    assert res.subscores.style_drift == pytest.approx(0.05)
    assert res.score < 0.08
    assert res.verdict == "faithful_deliverable"
