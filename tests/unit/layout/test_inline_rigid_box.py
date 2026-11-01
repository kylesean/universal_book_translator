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
    metrics = FontMetrics(first_char=0x20, widths=tuple([500.0] * 96))
    size_pt = 10.0
    text = "Hello"

    rigid = InlineRigidBox(
        token_id="m1",
        width_pt=30.0,
        height_pt=10.0,
        xobject_ref="/Fm1",
    )

    w_normal = advance_width(text, metrics, size_pt, horizontal_scale=1.0, rigid_boxes=[rigid])
    assert w_normal == pytest.approx(55.0)

    w_scaled = advance_width(text, metrics, size_pt, horizontal_scale=0.9, rigid_boxes=[rigid])
    assert w_scaled == pytest.approx(52.5)


def test_solve_horizontal_reserves_rigid_box_budget() -> None:
    metrics = FontMetrics(first_char=0x20, widths=tuple([500.0] * 96))
    size_pt = 10.0
    text = "ABCD"

    rigid = InlineRigidBox(
        token_id="m1",
        width_pt=25.0,
        height_pt=10.0,
        xobject_ref="/Fm1",
    )

    fit = solve_horizontal(text, metrics, size_pt, available_pt=45.0, rigid_boxes=[rigid])
    assert fit.fits
    assert fit.horizontal_scale == 1.0
    assert fit.width_pt == pytest.approx(45.0)

    fit_shrink = solve_horizontal(text, metrics, size_pt, available_pt=43.0, rigid_boxes=[rigid])
    assert fit_shrink.fits
    assert fit_shrink.width_pt <= 43.001
    assert fit_shrink.width_pt >= 42.0

    fit_fail = solve_horizontal(text, metrics, size_pt, available_pt=20.0, rigid_boxes=[rigid])
    assert not fit_fail.fits
    assert fit_fail.deficit_pt > 0.0
