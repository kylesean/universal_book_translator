"""The occlusion probe: text the layer reports but the raster does not show.

A near-white paint over a paragraph leaves the text line boxes intact, so only
the rendered pixels can tell. These pin the three behaviours the probe must
keep: white-on-white is reported, real ink is not, and a handful of stray dark
pixels (a neighbour's antialiased edge bleeding into the box) does not turn a
light crop into an occlusion finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ubt.adapters.pdf.visual_gate import (
    _ArtifactBBox,
    _ArtifactBox,
    text_occlusion_findings,
)

pytestmark = pytest.mark.fast

#: 72 dpi maps PDF points to PNG pixels 1:1, so the box below can be read
#: directly against the 200x100 pt media box.
_DPI = 72
_MEDIA = (0.0, 0.0, 200.0, 100.0)
#: One text line inside the media box, well above OCCLUSION_MIN_BOX_PX.
_BOX = _ArtifactBox(id="l1", bbox=_ArtifactBBox(page=1, x0=10.0, y0=10.0, x1=110.0, y1=30.0))


def _page(
    tmp_path: Path,
    *,
    ink: Sequence[tuple[int, int, int, int]] = (),
    name: str = "page.png",
) -> Path:
    """A white 200x100 page with each ``(x0, y0, x1, y1)`` rect filled black."""
    img = Image.new("L", (200, 100), 255)
    draw = ImageDraw.Draw(img)
    for rect in ink:
        draw.rectangle(rect, fill=0)
    path = tmp_path / name
    img.save(path)
    return path


def _stray_pixels(count: int) -> tuple[tuple[int, int, int, int], ...]:
    """``count`` isolated black pixels on one row inside the box's crop."""
    return tuple((20 + i, 80, 20 + i, 80) for i in range(count))


def test_white_on_white_is_reported_as_occluded(tmp_path: Path) -> None:
    findings = text_occlusion_findings({1: _page(tmp_path)}, [_BOX], {1: _MEDIA}, dpi=_DPI)
    assert [f.code for f in findings] == ["text_occluded"]
    assert findings[0].severity == "major"
    assert findings[0].page == 1
    assert "1 text line(s)" in findings[0].message


def test_a_line_carrying_ink_is_not_reported(tmp_path: Path) -> None:
    # 960 of the crop's 2000 pixels are black: ink far above the 2% floor.
    page = _page(tmp_path, ink=((20, 74, 100, 86),))
    assert text_occlusion_findings({1: page}, [_BOX], {1: _MEDIA}, dpi=_DPI) == []


def test_stray_dark_pixels_suppress_a_near_white_reading(tmp_path: Path) -> None:
    # 12 isolated dark pixels are 0.6% of the crop -- below the ink floor --
    # but enough truly dark pixels that something *is* rendered in the box.
    page = _page(tmp_path, ink=_stray_pixels(12))
    assert text_occlusion_findings({1: page}, [_BOX], {1: _MEDIA}, dpi=_DPI) == []
    # Under ten dark pixels the same near-white crop is still an occlusion.
    sparse = _page(tmp_path, ink=_stray_pixels(6), name="sparse.png")
    findings = text_occlusion_findings({1: sparse}, [_BOX], {1: _MEDIA}, dpi=_DPI)
    assert [f.code for f in findings] == ["text_occluded"]


def test_visual_gate_tmpdir_tracking_and_cleanup(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import (
        _OWNED_TMP_DIRS,
        cleanup_visual_gate_tmpdirs,
        render_pages_to_png,
    )

    # Empty pages should return empty dict and not leak
    res = render_pages_to_png(tmp_path / "nonexistent.pdf", [])
    assert res == {}

    # Cleanup function should clear any tracked tmpdirs safely
    cleanup_visual_gate_tmpdirs()
    assert len(_OWNED_TMP_DIRS) == 0


def test_rotated_page_occlusion_findings(tmp_path: Path) -> None:
    # 200x100 PDF page rotated 90 deg rasterizes to a 100x200 image.
    # _BOX in unrotated PDF is (10, 10, 110, 30).
    # Under 90 deg clockwise rotation, it maps to x=[10, 30], y=[10, 110].
    img_90 = Image.new("L", (100, 200), 255)
    draw_90 = ImageDraw.Draw(img_90)
    # Fill ink in the rotated bounding box (placed in y=[15, 55], which is
    # inside the rotated crop y=[10, 110] but misses the unrotated crop y=[70, 90])
    draw_90.rectangle((12, 15, 28, 55), fill=0)
    p90 = tmp_path / "p90.png"
    img_90.save(p90)

    # With rotation 90 declared, the crop correctly lands on the ink:
    findings_with_rot = text_occlusion_findings(
        {1: p90}, [_BOX], {1: _MEDIA}, dpi=_DPI, rotations={1: 90}
    )
    assert findings_with_rot == []

    # Without rotation (0 deg assumption), crop misses the ink and flags occlusion:
    findings_without_rot = text_occlusion_findings(
        {1: p90}, [_BOX], {1: _MEDIA}, dpi=_DPI, rotations={1: 0}
    )
    assert [f.code for f in findings_without_rot] == ["text_occluded"]


def test_pdf_struct_page_rotation(tmp_path: Path) -> None:
    import pikepdf

    from ubt.adapters.pdf.pdf_struct import page_rotation, page_rotations

    pdf = pikepdf.new()
    _ = pdf.add_blank_page(page_size=(200, 100))
    p2 = pdf.add_blank_page(page_size=(200, 100))
    p2.Rotate = 90
    out = tmp_path / "test_rot.pdf"
    pdf.save(out)

    with pikepdf.open(out) as opened:
        assert page_rotation(opened.pages[0]) == 0
        assert page_rotation(opened.pages[1]) == 90

    rots = page_rotations(out)
    assert rots == {1: 0, 2: 90}
