"""Page-level coordinate-system inference and bbox normalization.

A per-box ``auto`` guess cannot tell a normalized-1000 box that fits inside an
A4 page from a native PDF-point box, so ``infer_coord_system`` decides once per
page from the boxes' extremes. These tests pin that decision and the resolved
geometry it feeds, because a wrong choice scales only some boxes and silently
misplaces the extracted block on the page.
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


def test_cropbox_origin_offset_normalized_1() -> None:
    # Page with CropBox starting at (50, 60) with size 200x300
    resolver = PageBBoxResolver(page_width=200.0, page_height=300.0, origin_x=50.0, origin_y=60.0)
    box = (0.1, 0.2, 0.6, 0.7)
    resolved = resolver.resolve_bbox(box, coord_system="normalized_1")
    assert resolved is not None
    # x: 50 + [0.1*200, 0.6*200] = [70, 170]
    # y: 60 + [(1-0.7)*300, (1-0.2)*300] = [150, 300]
    assert resolved == pytest.approx((70.0, 150.0, 170.0, 300.0))


def test_cropbox_origin_offset_normalized_1000() -> None:
    resolver = PageBBoxResolver(page_width=200.0, page_height=300.0, origin_x=50.0, origin_y=60.0)
    box = (100.0, 200.0, 600.0, 700.0)
    resolved = resolver.resolve_bbox(box, coord_system="normalized_1000")
    assert resolved is not None
    assert resolved == pytest.approx((70.0, 150.0, 170.0, 300.0))


def test_cropbox_origin_offset_image_pixel() -> None:
    resolver = PageBBoxResolver(
        page_width=200.0,
        page_height=300.0,
        image_width=400.0,
        image_height=600.0,
        origin_x=50.0,
        origin_y=60.0,
    )
    # At scale 2 (400x600 px), box (40, 120, 240, 420) is identical to the 100x200pt box above
    box = (40.0, 120.0, 240.0, 420.0)
    resolved = resolver.resolve_bbox(box, coord_system="image_pixel")
    assert resolved is not None
    assert resolved == pytest.approx((70.0, 150.0, 170.0, 300.0))


def test_cropbox_origin_offset_top_left_points() -> None:
    resolver = PageBBoxResolver(page_width=200.0, page_height=300.0, origin_x=50.0, origin_y=60.0)
    box = (20.0, 60.0, 120.0, 210.0)
    resolved = resolver.resolve_bbox(box, origin="top-left")
    assert resolved is not None
    assert resolved == pytest.approx((70.0, 150.0, 170.0, 300.0))


def test_cropbox_origin_offset_with_page_rotation() -> None:
    # A rotated 90 page displayed as 500x300 with unrotated CropBox origin (50, 60)
    # rotation=270 un-rotates the displayed box into unrotated user-space orientation
    resolver = PageBBoxResolver(
        page_width=500.0,
        page_height=300.0,
        rotation=270,
        origin_x=50.0,
        origin_y=60.0,
    )
    # Box at pixels (445, 54)-(456, 102) on 500x300 image
    # Without origin offset, it lands at unrotated points (54, 445, 102, 456)
    # With origin offset (50, 60), it lands at (104, 505, 152, 516)
    box = (445.0, 54.0, 456.0, 102.0)
    resolved = resolver.resolve_bbox(box, coord_system="image_pixel")
    assert resolved is not None
    assert resolved == pytest.approx((104.0, 505.0, 152.0, 516.0), abs=0.5)
