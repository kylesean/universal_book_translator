"""Page-level coordinate-system inference and bbox normalization.

A per-box ``auto`` guess cannot tell a normalized-1000 box that fits inside an
A4 page from a native PDF-point box, so ``infer_coord_system`` decides once per
page from the boxes' extremes. These tests pin that decision and the resolved
geometry it feeds, because a wrong choice scales only some boxes and silently
misplaces the translation overlay.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver

pytestmark = pytest.mark.fast

_A4 = (595.0, 842.0)


def _resolver() -> PageBBoxResolver:
    return PageBBoxResolver(page_width=_A4[0], page_height=_A4[1])


def test_normalized_1000_page_is_inferred_from_its_extremes() -> None:
    boxes = [(500.0, 200.0, 900.0, 300.0), (100.0, 50.0, 300.0, 150.0)]
    assert PageBBoxResolver.infer_coord_system(boxes, *_A4) == "normalized_1000"


def test_a_top_left_box_is_placed_correctly_once_the_page_is_known() -> None:
    # The top-left box (100, 50, 300, 150) fits inside A4 and, seen alone, looks
    # like native points. Page-level inference sees the other box reach x=900 and
    # decides normalized_1000, so the small box is scaled instead of misplaced.
    boxes = [(500.0, 200.0, 900.0, 300.0), (100.0, 50.0, 300.0, 150.0)]
    system = PageBBoxResolver.infer_coord_system(boxes, *_A4)
    resolved = _resolver().resolve_bbox(boxes[1], coord_system=system)
    assert resolved is not None
    x0, y0, x1, y1 = resolved
    assert (x0, y0, x1, y1) == pytest.approx((59.5, 715.7, 178.5, 799.9))


def test_native_pdf_points_page_is_inferred_when_all_boxes_fit() -> None:
    boxes = [(100.0, 50.0, 300.0, 150.0), (400.0, 600.0, 580.0, 800.0)]
    assert PageBBoxResolver.infer_coord_system(boxes, *_A4) == "pdf_points"


def test_normalized_1_page_is_inferred() -> None:
    boxes = [(0.1, 0.05, 0.3, 0.15)]
    assert PageBBoxResolver.infer_coord_system(boxes, *_A4) == "normalized_1"


def test_image_pixel_page_is_inferred_from_image_dimensions() -> None:
    boxes = [(100.0, 50.0, 1100.0, 1600.0)]
    assert (
        PageBBoxResolver.infer_coord_system(
            boxes, _A4[0], _A4[1], image_width=1190.0, image_height=1684.0
        )
        == "image_pixel"
    )


def test_infer_falls_back_to_auto_without_usable_boxes() -> None:
    assert PageBBoxResolver.infer_coord_system([], *_A4) == "auto"
    assert PageBBoxResolver.infer_coord_system([None, (1.0, 2.0)], *_A4) == "auto"
