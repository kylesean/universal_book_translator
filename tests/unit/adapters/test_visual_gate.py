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
    VisualFinding,
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


# --------------------------------------------------------------------------- #
# The gate's decision logic (pure; no PDF, no rendering).
#
# These were the uncovered 2/3 of the module: the pixel helpers had tests, but
# the functions that decide whether an artifact may SHIP did not. Each branch
# below is a delivery decision (2026-10-14 review, W5-3).
# --------------------------------------------------------------------------- #


def _finding(severity: str, code: str = "some_defect") -> VisualFinding:
    return VisualFinding(severity=severity, code=code, message="m")


def test_the_blocking_gate_returns_critical_only_when_enabled_and_short() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    findings = [_finding("critical"), _finding("major"), _finding("info")]
    tripped = blocking_gate_tripped(findings, total_pages=10, enabled=True)
    assert [f.severity for f in tripped] == ["critical"]
    # Disabled: the gate never blocks, even with criticals.
    assert blocking_gate_tripped(findings, total_pages=10, enabled=False) == []


def test_a_long_document_never_blocks() -> None:
    """Long books fail OPEN (warn-only): the page window is the opt-in bound."""
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped
    from ubt.core.policy.layout_policy import QA_FULL_GATE_MAX_PAGES

    findings = [_finding("critical")]
    assert (
        blocking_gate_tripped(findings, total_pages=QA_FULL_GATE_MAX_PAGES + 1, enabled=True) == []
    )


def test_zero_pages_means_not_measured_and_does_not_block() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    assert blocking_gate_tripped([_finding("critical")], total_pages=0, enabled=True) == []


def test_an_unreadable_artifact_fails_closed_even_when_disabled() -> None:
    """``total_pages < 0`` is the pdf_page_count sentinel: nobody can read it.

    The gate must not report passed=True on an artifact it could not measure,
    regardless of the opt-in switch or the page window.
    """
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    findings = [_finding("critical"), _finding("major")]
    tripped = blocking_gate_tripped(findings, total_pages=-1, enabled=False)
    assert [f.severity for f in tripped] == ["critical"]


def test_a_missing_target_language_fails_closed_even_when_disabled() -> None:
    """An artifact carrying none of the commissioned language is unshippable."""
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    absent = _finding("critical", code="target_language_absent")
    tripped = blocking_gate_tripped([absent], total_pages=999, enabled=False)
    assert tripped == [absent]
    # A crashed gate is the same class: "unknown" must not read as a pass.
    crashed = _finding("critical", code="visual_gate_crashed")
    assert blocking_gate_tripped([crashed], total_pages=999, enabled=False) == [crashed]


def test_a_sparse_target_language_major_fails_closed() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    sparse = _finding("major", code="target_language_sparse")
    assert blocking_gate_tripped([sparse], total_pages=999, enabled=False) == [sparse]
    # A different major stays warn-only.
    other = _finding("major", code="something_else")
    assert blocking_gate_tripped([other], total_pages=999, enabled=False) == []


def test_the_adaptive_budget_floors_short_books_and_caps_long_ones() -> None:
    from ubt.adapters.pdf.visual_gate import adaptive_sample_budget

    # Disabled / degenerate.
    assert adaptive_sample_budget(0, 6) == 0
    assert adaptive_sample_budget(10, 0) == 0
    # Short books get full inspection (configured is a floor, not a ceiling).
    assert adaptive_sample_budget(8, 6) == 8
    assert adaptive_sample_budget(8, 100) == 8
    # A long book gets a fixed anchor set, at least the configured floor.
    long_budget = adaptive_sample_budget(900, 6)
    assert long_budget >= 6
    assert long_budget == adaptive_sample_budget(901, 6)  # constant past the band


def test_the_sample_puts_flagged_pages_first_and_covers_the_anchors() -> None:
    from ubt.adapters.pdf.visual_gate import select_sample_pages

    # Degenerate inputs.
    assert select_sample_pages(0, [1], 5) == []
    assert select_sample_pages(10, [1], 0) == []
    # Flagged pages win, out-of-range flags are dropped.
    assert select_sample_pages(10, [3, 7, 99, 0], 2) == [3, 7]
    # More flags than budget: the earliest flags fill it.
    assert select_sample_pages(10, [5, 1, 9], 2) == [1, 5]
    # A long book with no flags covers cover/TOC/tail.
    sample = select_sample_pages(500, [], 5)
    assert {1, 2, 500}.issubset(set(sample))
    assert len(sample) <= 5
    # Every returned page is inside the document.
    assert all(1 <= page <= 500 for page in sample)
