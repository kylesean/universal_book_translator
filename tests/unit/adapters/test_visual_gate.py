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
    Severity,
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


def _finding(severity: Severity, code: str = "some_defect") -> VisualFinding:
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


# --------------------------------------------------------------------------- #
# A skipped tier must not read as "verified clean"
#
# The VLM judge is opt-in, so a default run always ends with vlm_checked == 0 and
# the old report said only `passed: false, skipped_reason: "vlm judge disabled"`.
# That invites the reader to conclude the visual check failed, when the
# geometry/pixel tiers ran and the semantic tier was never asked (measured: the
# ATC'25 report was read exactly that way).
# --------------------------------------------------------------------------- #


def test_the_report_names_the_tiers_that_ran_and_the_one_that_did_not() -> None:
    from ubt.adapters.pdf.visual_gate import _tier_coverage

    default_run = _tier_coverage((), ["vlm judge disabled"], rendered_pages=5, sampled_pages=5)
    assert "geometry" in default_run
    assert "type" in default_run
    assert "pixel" in default_run
    assert "not run: vlm" in default_run


def test_a_run_with_the_vlm_judge_reports_three_tiers() -> None:
    from ubt.adapters.pdf.visual_gate import _tier_coverage

    full = _tier_coverage((1, 2), [], rendered_pages=5, sampled_pages=5)
    assert "vlm" in full
    assert "not run" not in full


def test_a_report_never_claims_a_tier_that_did_not_run() -> None:
    """The coverage line contradicted the skipped-reason in the same string.

    ``ran`` was the hardcoded constant ``["geometry", "type", "pixel"]`` and
    ``skip_notes`` was never read, so when every sampled page failed to render
    the report still said "tiers run: geometry, type, pixel" beside "page render
    failed (pdf_oxide); pixel checks skipped". With no findings, ``passed`` was
    also True: a gate that inspected nothing reported success. Each tier is now
    derived from what actually ran.
    """
    from ubt.adapters.pdf.visual_gate import _tier_coverage

    nothing_rendered = _tier_coverage(
        (),
        ["page render failed (pdf_oxide); pixel checks skipped"],
        rendered_pages=0,
        sampled_pages=5,
    )
    assert "type" not in nothing_rendered.split("not run")[0], (
        "the type tier needs a rendered page and must not be claimed"
    )
    assert "pixel" not in nothing_rendered.split("not run")[0]
    assert "not run: type, pixel, vlm" in nothing_rendered
    assert "0 of 5 sampled page(s) rendered" in nothing_rendered
    # The geometry tier reads the PDF's own text rects, so it does still run.
    assert "tiers run: geometry" in nothing_rendered


# --------------------------------------------------------------------------- #
# Type-size audit
#
# The tier that catches a whole-document collapse. It was added after an 83-page
# paper shipped at 22 pages with 10 pages below 4pt and the last at 0.49pt: every
# structural check passed (no text lost, no overflow, no blank page), so nothing
# else in the gate could see it.
#
# The audit's two measurement traps were found by running it over the corpus's 13
# delivered files, and these tests pin both:
#   * FPDFText_GetFontSize is a state size a text matrix can scale (chapter-3
#     reports 1.00pt while rendering ~5.8pt), so height is measured instead;
#   * a rect's height is ink height, and punctuation ink is short (a TOC leader
#     measures 1.29pt on a healthy page), so only letters/digits count.
# --------------------------------------------------------------------------- #


def _typst_pdf(tmp_path: Path, body: str, *, size: str = "10pt") -> Path:
    """Compile a one-page Typst document at ``size`` into ``tmp_path``."""
    import shutil
    import subprocess

    typst = shutil.which("typst")
    if typst is None:
        pytest.skip("typst is not installed")
    source = tmp_path / "main.typ"
    pdf = tmp_path / "main.pdf"
    source.write_text(
        f"#set page(width: 420pt, height: 600pt, margin: 20pt)\n#set text(size: {size})\n{body}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [typst, "compile", str(source), str(pdf)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:  # pragma: no cover - environment failure
        pytest.skip(f"typst failed: {result.stderr[:200]}")
    return pdf


def test_a_page_set_too_small_is_reported_as_illegible(tmp_path: Path) -> None:
    """The delivered-defect shape: body text rendered at a fraction of a point."""
    from ubt.adapters.pdf.visual_gate import type_size_findings

    pdf = _typst_pdf(tmp_path, "The quick brown fox jumps over the lazy dog. " * 40, size="0.5pt")
    findings = type_size_findings(pdf, [1])
    assert [f.code for f in findings] == ["type_illegible"]
    assert findings[0].severity == "critical"
    assert findings[0].page == 1


def test_a_normally_set_page_passes(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import type_size_findings

    pdf = _typst_pdf(tmp_path, "The quick brown fox jumps over the lazy dog. " * 40)
    assert type_size_findings(pdf, [1]) == []


def test_a_table_of_contents_leader_dots_do_not_fail_the_page(tmp_path: Path) -> None:
    """book_mono p3 was a false positive: leader dots measure 1.29pt of ink.

    The audit must judge the page by its letters, not by its punctuation, or
    every book with a table of contents refuses export.
    """
    from ubt.adapters.pdf.visual_gate import type_size_findings

    body = "\n".join(f"#h(1fr) Chapter {i} #h(1fr) {i}" for i in range(30))
    pdf = _typst_pdf(tmp_path, body)
    assert type_size_findings(pdf, [1]) == []


def test_six_point_type_is_small_but_legible(tmp_path: Path) -> None:
    """The corpus's densest healthy pages render at ~5.4-5.8pt and must pass.

    ``book_mono`` and ``chapter-3`` sit there (footnote-heavy technical books).
    4pt is chosen as the floor precisely so those stay legal while the 0.49pt
    collapse does not.
    """
    from ubt.adapters.pdf.visual_gate import type_size_findings

    pdf = _typst_pdf(tmp_path, "The quick brown fox jumps over the lazy dog. " * 40, size="6pt")
    assert type_size_findings(pdf, [1]) == []


def test_a_matrix_scaled_font_is_measured_by_rendered_height(tmp_path: Path) -> None:
    """chapter-3_mono.pdf reports 1.00pt via GetFontSize while rendering at 5.8pt.

    A ``#scale``-transformed run keeps a nominal font size that is unrelated to
    its rendered size. Measuring heights rather than the reported size is what
    keeps this healthy page out of the fail-closed path.
    """
    from ubt.adapters.pdf.visual_gate import type_size_findings

    body = "#scale(x: 90%, y: 90%)[The quick brown fox jumps over the lazy dog. " * 40 + "]"
    pdf = _typst_pdf(tmp_path, body)
    assert type_size_findings(pdf, [1]) == []


def test_the_illegible_code_fails_closed_with_the_blocking_gate_off() -> None:
    """A page nobody can read must not ship because an opt-in flag is off.

    ``blocking_gate_tripped`` normally returns [] when ``enabled`` is False and
    the document is long. Type illegibility joins the small set of codes that
    ignore both conditions, alongside an unreadable PDF.
    """
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    finding = VisualFinding(
        severity="critical",
        code="type_illegible",
        message="Page 3 is typeset at a median glyph height of 0.49pt",
        page=3,
    )
    for enabled in (False, True):
        tripped = blocking_gate_tripped([finding], total_pages=400, enabled=enabled)
        assert [f.code for f in tripped] == ["type_illegible"]


def test_a_collapsed_region_warns_but_does_not_fail_closed() -> None:
    """A small table can legitimately be tiny; only a majority collapse warns.

    ``type_collapsed`` is a MAJOR, so it blocks under the opt-in gate and stays a
    warning otherwise -- unlike ``type_illegible``, which is about the page as a
    whole.
    """
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    finding = VisualFinding(
        severity="major",
        code="type_collapsed",
        message="Page 7 has 80% of its glyphs below 4pt",
        page=7,
    )
    assert blocking_gate_tripped([finding], total_pages=400, enabled=False) == []
    # A LONG document only blocks on CRITICAL findings even with the opt-in gate
    # on (the window is QA_FULL_GATE_MAX_PAGES); this is the existing contract,
    # not a property of the type codes.
    assert blocking_gate_tripped([finding], total_pages=400, enabled=True) == []
