"""M2: rigid margin-reclaim geometry (pure, no PDF/Typst)."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.rigid.typesetter import _reclaim_zone_down
from ubt.adapters.pdf.rigid.zones import Zone


def _zone(x0: float, y0: float, x1: float, y1: float, block_id: str = "b1") -> Zone:
    return Zone(block_id=block_id, page=1, x0=x0, y0=y0, x1=x1, y1=y1, base_size=11.0)


def test_reclaim_disabled_returns_same_zone() -> None:
    z = _zone(50, 200, 400, 300)
    assert _reclaim_zone_down(z, [], min_y=30.0, max_reclaim_pt=0.0) is z


def test_reclaim_grows_down_into_blank_space() -> None:
    z = _zone(50, 200, 400, 300)
    out = _reclaim_zone_down(z, [], min_y=30.0, max_reclaim_pt=60.0)
    assert out is not z
    assert out.y0 == pytest.approx(140.0)  # 200 - 60
    assert out.y1 == 300.0  # top unchanged
    assert (out.x0, out.x1) == (50.0, 400.0)  # never grows sideways


def test_reclaim_stops_at_neighbour_below() -> None:
    z = _zone(50, 200, 400, 300)
    # A neighbour occupying y=[120,180] that overlaps the horizontal span.
    occupancy = [("other", (60.0, 120.0, 380.0, 180.0))]
    out = _reclaim_zone_down(z, occupancy, min_y=30.0, max_reclaim_pt=100.0, gap_pt=5.0)
    assert out.y0 == pytest.approx(185.0)  # neighbour top 180 + gap 5


def test_reclaim_ignores_lateral_columns() -> None:
    z = _zone(50, 200, 400, 300)
    # Neighbour below but in a different column (no horizontal overlap).
    occupancy = [("other", (450.0, 120.0, 700.0, 180.0))]
    out = _reclaim_zone_down(z, occupancy, min_y=30.0, max_reclaim_pt=60.0)
    assert out.y0 == pytest.approx(140.0)  # grows full budget, ignoring the column


def test_reclaim_clamps_to_footer_band() -> None:
    z = _zone(50, 40, 400, 300)
    out = _reclaim_zone_down(z, [], min_y=30.0, max_reclaim_pt=1000.0)
    assert out.y0 == pytest.approx(30.0)  # never below the footer clamp


def test_noop_when_no_room_below_neighbour() -> None:
    z = _zone(50, 200, 400, 300)
    # Neighbour top already above the zone bottom → nothing to reclaim.
    occupancy = [("other", (60.0, 150.0, 380.0, 210.0))]
    out = _reclaim_zone_down(z, occupancy, min_y=30.0, max_reclaim_pt=60.0, gap_pt=5.0)
    assert out is z  # floor(210+5=215) >= zone.y0(200) → unchanged
