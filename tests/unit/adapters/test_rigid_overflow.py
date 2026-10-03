"""Bounded overflow prefers clean space, then the page bottom — never silently.

When a translation cannot fit its zone even at the degraded floor, the rigid
planner extends the last zone downward ("bounded overflow", Axiom B: a
translated paragraph beats a blank box). The preferred extension stops at the
nearest occupied rect below — running to the page's content bottom *ignoring*
neighbours (the old behaviour) let a Docling fragment with text far larger
than its box (arXiv 2609.32391 p22) paint over the adjacent fragments,
destroying both. When the neighbour-bounded floor still cannot fit, the
agreed priority order applies — completeness above pixel layout — so the
extension falls back to the page's content bottom before the caller's
``spill`` fail-closed path (source visible, loss recorded).
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.rigid.typesetter import _overflow_zone_down
from ubt.adapters.pdf.rigid.zones import Zone

pytestmark = pytest.mark.fast


def _zone(y0: float, y1: float, x0: float = 75.0, x1: float = 185.0) -> Zone:
    return Zone(block_id="b1", page=1, x0=x0, y0=y0, x1=x1, y1=y1, base_size=8.0)


def test_extends_into_blank_space_below() -> None:
    z = _zone(168.0, 174.0)
    out = _overflow_zone_down(z, 60.0, [])
    assert out.y0 == 60.0
    assert (out.x0, out.y1) == (z.x0, z.y1)


def test_stops_at_nearest_occupied_rect_below() -> None:
    z = _zone(168.0, 174.0)
    occupied = [("b2", (75.0, 150.0, 185.0, 160.0))]  # directly below, same column
    out = _overflow_zone_down(z, 60.0, occupied)
    assert out.y0 == pytest.approx(160.0 + 5.0)  # neighbour top + RECLAIM_GAP_PT


def test_ignores_neighbours_in_other_columns() -> None:
    z = _zone(168.0, 174.0, x0=75.0, x1=185.0)
    occupied = [("b2", (300.0, 100.0, 540.0, 160.0))]  # right column only
    out = _overflow_zone_down(z, 60.0, occupied)
    assert out.y0 == 60.0


def test_ignores_own_rects() -> None:
    z = _zone(168.0, 174.0)
    occupied = [("b1", (75.0, 100.0, 185.0, 160.0))]  # the block's own continuation
    out = _overflow_zone_down(z, 60.0, occupied)
    assert out.y0 == 60.0


def test_no_op_when_neighbour_blocks_the_whole_gap() -> None:
    z = _zone(168.0, 174.0)
    occupied = [("b2", (75.0, 160.0, 185.0, 167.0))]  # directly beneath the zone
    out = _overflow_zone_down(z, 60.0, occupied)
    assert out is z  # unchanged: caller falls through to spill


def test_no_op_when_already_at_content_bottom() -> None:
    z = _zone(50.0, 58.0)
    assert _overflow_zone_down(z, 60.0, []) is z
