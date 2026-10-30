"""Unit tests for InlineRigidBox and horizontal fitting with rigid obstacles.

Specification: ``docs/design/adaptive_delivery_prd.md`` §8.2 & §1194. Resolves GAP-14.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.cels_inject import FontMetrics, advance_width, solve_horizontal
from ubt.core.layout import InlineRigidBox

pytestmark = pytest.mark.fast


def test_inline_rigid_box_properties_and_immutability() -> None:
    box = InlineRigidBox(
        token_id="math_1",
        width_pt=45.5,
        height_pt=18.0,
        baseline_offset_pt=-2.0,
        xobject_ref="/Fm1",
    )
    assert box.token_id == "math_1"
    assert box.width_pt == 45.5
    assert box.height_pt == 18.0
    assert box.baseline_offset_pt == -2.0
    assert box.xobject_ref == "/Fm1"

    with pytest.raises(AttributeError):
        box.width_pt = 50.0  # type: ignore[misc]


def test_advance_width_includes_rigid_boxes_without_scale() -> None:
    # Font metrics where each char has 0.5 em advance
    metrics = FontMetrics(first_char=0x20, widths=tuple([500.0] * 96))
    size_pt = 10.0
    text = "Hello"  # 5 chars * 5.0pt = 25.0pt

    rigid = InlineRigidBox(
        token_id="m1",
        width_pt=30.0,
        height_pt=10.0,
        xobject_ref="/Fm1",
    )

    # At horizontal_scale=1.0: 25.0 + 30.0 = 55.0pt
    w_normal = advance_width(text, metrics, size_pt, horizontal_scale=1.0, rigid_boxes=[rigid])
    assert w_normal == pytest.approx(55.0)

    # At horizontal_scale=0.9: 25.0 * 0.9 + 30.0 = 22.5 + 30.0 = 52.5pt
    # (Rigid box DOES NOT scale with horizontal_scale)
    w_scaled = advance_width(text, metrics, size_pt, horizontal_scale=0.9, rigid_boxes=[rigid])
    assert w_scaled == pytest.approx(52.5)


def test_solve_horizontal_reserves_rigid_box_budget() -> None:
    metrics = FontMetrics(first_char=0x20, widths=tuple([500.0] * 96))
    size_pt = 10.0
    text = "ABCD"  # 4 chars * 5.0pt = 20.0pt natural

    rigid = InlineRigidBox(
        token_id="m1",
        width_pt=25.0,
        height_pt=10.0,
        xobject_ref="/Fm1",
    )

    # Available = 45.0pt. Natural = 20.0 + 25.0 = 45.0pt -> Fits at 1.0 scale
    fit = solve_horizontal(text, metrics, size_pt, available_pt=45.0, rigid_boxes=[rigid])
    assert fit.fits
    assert fit.horizontal_scale == 1.0
    assert fit.width_pt == pytest.approx(45.0)

    # Available = 43.0pt. Text must shrink to 18.0pt (18.0 / 20.0 = 0.90 scale)
    fit_shrink = solve_horizontal(text, metrics, size_pt, available_pt=43.0, rigid_boxes=[rigid])
    assert fit_shrink.fits
    assert fit_shrink.width_pt <= 43.001
    assert fit_shrink.width_pt >= 42.0

    # Available = 20.0pt < rigid width 25.0pt -> Deficit!
    fit_fail = solve_horizontal(text, metrics, size_pt, available_pt=20.0, rigid_boxes=[rigid])
    assert not fit_fail.fits
    assert fit_fail.deficit_pt > 0.0
