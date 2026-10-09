"""A crop must subtract the MediaBox origin, not assume it is (0, 0).

The raster covers only the page's MediaBox, so a user-space point maps to pixel
``(point - mediabox_origin) * scale``. ``_compute_crop_coords`` treated the box
as if the MediaBox started at the origin, so a page cropped or imposed to a
non-zero lower-left (common in print PDFs) had every crop shifted by that
offset — the wrong region, and the further the origin, the further off. On a
zero-origin page (the overwhelmingly common case) the fix is a no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, FloatObject, NameObject

from ubt.adapters.pdf.visual_scalpel import _compute_crop_coords

pytestmark = pytest.mark.fast


def _page_with_mediabox(path: Path, box: tuple[float, float, float, float]) -> Path:
    """Rewrite a one-page PDF's /MediaBox to *box*."""
    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.pages[0][NameObject("/MediaBox")] = ArrayObject([FloatObject(v) for v in box])
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_a_zero_origin_page_crops_at_the_box_itself() -> None:
    # 100x200pt box at (10, 20) on a 200x300pt page, scale 1 (dpi 72).
    x0, y0, x1, y1 = _compute_crop_coords(
        (10.0, 20.0, 110.0, 220.0), 200.0, 300.0, 1.0, bleed_pt=0.0
    )
    assert (x0, x1) == (10, 110)
    # Raster y is measured down from the top: top of box = 300 - 220 = 80.
    assert (y0, y1) == (80, 280)


def test_a_non_zero_media_origin_is_subtracted() -> None:
    # Same box on a page whose MediaBox starts at (50, 60): the raster begins
    # at that corner, so the box's top-left pixel is (10-50, ...) — negative,
    # clamped to 0 — and its right edge is (110-50)=60.
    x0, y0, x1, y1 = _compute_crop_coords(
        (10.0, 20.0, 110.0, 220.0),
        200.0,
        300.0,
        1.0,
        bleed_pt=0.0,
        origin_x=50.0,
        origin_y=60.0,
    )
    assert (x0, x1) == (0, 60)
    # height: 300 - (220-60) = 140 down from the raster top; bottom 300-(20-60)=340.
    assert (y0, y1) == (140, 300)


def test_a_box_entirely_inside_the_shifted_page_crops_in_bounds() -> None:
    # MediaBox (50, 60, 250, 260): a box at (60, 70, 160, 170) sits 10pt in from
    # the corner and must crop to (10, ...) in raster pixels, not (60, ...).
    x0, y0, x1, y1 = _compute_crop_coords(
        (60.0, 70.0, 160.0, 170.0),
        200.0,
        200.0,
        1.0,
        bleed_pt=0.0,
        origin_x=50.0,
        origin_y=60.0,
    )
    assert (x0, x1) == (10, 110)
    # 200 - (170-60) = 90 from the top; 200 - (70-60) = 190.
    assert (y0, y1) == (90, 190)


def test_crop_block_pil_uses_the_real_mediabox_origin(tmp_path: Path) -> None:
    """End-to-end: a box in the cropped-away margin fails closed.

    The fixture's MediaBox is (50, 60, 250, 260); a box at (40, 50, 45, 55)
    lies entirely below-left of the visible page. Subtracting the origin, its
    raster extent is negative and clamps to empty, so the crop refuses — the
    honest outcome. A crop that ignored the origin would treat (40,50)-(45,55)
    as in-page pixels and hand back a 5x5 sliver of the wrong region.
    """
    from ubt.adapters.pdf import visual_scalpel

    pdf_path = tmp_path / "shifted.pdf"
    write_text_pdf(pdf_path, [("ABCDEF",)], width=200.0, height=200.0)
    _page_with_mediabox(pdf_path, (50.0, 60.0, 250.0, 260.0))

    visual_scalpel._PAGE_CACHE = None
    try:
        with pytest.raises(ValueError, match="empty intersection"):
            visual_scalpel.crop_block_pil(
                pdf_path, 1, (40.0, 50.0, 45.0, 55.0), dpi=72, bleed_pt=0.0
            )
    finally:
        visual_scalpel._PAGE_CACHE = None
