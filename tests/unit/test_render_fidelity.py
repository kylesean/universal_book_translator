"""Unit tests for the rigid-render fidelity ruler (pure Pillow core + advisory)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ubt.adapters.pdf.render_fidelity import (
    _COVERAGE_WARN,
    _IN_BOX_INK_WARN,
    _RESIDUAL_WARN,
    diff_outside_masks,
    fidelity_findings,
    in_box_ink_retention,
)


def test_identical_pages_no_masks_is_zero_residual() -> None:
    a = Image.new("RGB", (40, 40), (200, 200, 200))
    b = Image.new("RGB", (40, 40), (200, 200, 200))
    residual, coverage = diff_outside_masks(a, b, [])
    assert residual == 0.0
    assert coverage == 0.0


def test_change_outside_mask_shows_up_in_residual() -> None:
    a = Image.new("RGB", (40, 40), (255, 255, 255))
    b = Image.new("RGB", (40, 40), (255, 255, 255))
    b.putpixel((30, 30), (0, 0, 0))  # a stray ink change in the kept region
    residual, _ = diff_outside_masks(a, b, [(0, 0, 10, 10)])  # mask elsewhere
    assert residual > 0.0


def test_change_inside_mask_only_is_excluded() -> None:
    a = Image.new("RGB", (40, 40), (255, 255, 255))
    b = Image.new("RGB", (40, 40), (255, 255, 255))
    b.putpixel((5, 5), (0, 0, 0))  # the only difference sits under the mask
    residual, coverage = diff_outside_masks(a, b, [(0, 0, 12, 12)])
    assert residual == 0.0
    assert coverage > 0.0


def test_size_mismatch_is_maximally_suspect() -> None:
    a = Image.new("RGB", (40, 40))
    b = Image.new("RGB", (50, 50))
    residual, coverage = diff_outside_masks(a, b, [])
    assert residual == 1.0
    assert coverage == 0.0


def test_findings_report_bad_residual_and_low_coverage_as_advisory() -> None:
    findings = fidelity_findings(
        {
            "pages_measured": 3,
            "non_text_diff_ratio": _RESIDUAL_WARN * 5,
            "masked_coverage_ratio": _COVERAGE_WARN * 0.1,
        }
    )
    codes = {f.code for f in findings}
    assert {"fidelity_non_text_residual", "fidelity_low_coverage"} <= codes
    # Advisory: never escalate above info, so delivery is never blocked.
    assert all(f.severity == "info" for f in findings)


def test_in_box_ink_loss_is_measured() -> None:
    """In-box ink retention drops when the artifact lost glyphs inside a box."""
    src = Image.new("RGB", (40, 40), (255, 255, 255))
    art = Image.new("RGB", (40, 40), (255, 255, 255))
    for y in range(10, 30):
        for x in range(5, 35):
            src.putpixel((x, y), (0, 0, 0))
    for y in range(10, 15):  # only a quarter of the ink survives (clipped block)
        for x in range(5, 35):
            art.putpixel((x, y), (0, 0, 0))
    assert in_box_ink_retention(src, art, [(0, 0, 40, 40)]) < 0.5
    # Identical ink is a perfect score; a box with no source ink is neutral.
    assert in_box_ink_retention(src, src, [(0, 0, 40, 40)]) == 1.0
    assert in_box_ink_retention(src, art, []) == 1.0


def test_in_box_ink_loss_finding_is_advisory() -> None:
    """A severe in-box ink drop is reported, but never escalates above info."""
    findings = fidelity_findings(
        {
            "pages_measured": 1,
            "non_text_diff_ratio": 0.0,
            "masked_coverage_ratio": 0.9,
            "in_box_ink_retention": _IN_BOX_INK_WARN * 0.5,
        }
    )
    assert "fidelity_in_box_ink_loss" in {f.code for f in findings}
    assert all(f.severity == "info" for f in findings)


def test_good_fidelity_produces_no_findings() -> None:
    findings = fidelity_findings(
        {
            "pages_measured": 5,
            "non_text_diff_ratio": 0.0,
            "masked_coverage_ratio": 0.9,
        }
    )
    assert findings == []


def test_unmeasured_stats_produce_no_findings() -> None:
    assert fidelity_findings({"pages_measured": 0}) == []


def test_select_probe_pages_whole_page_when_no_blocks() -> None:
    """Empty blocks must sample the common pages (whole-page compare), not zero.

    The offline fidelity harness passes no IR blocks; the old page selection
    returned an empty list, so it measured nothing and reported a perfect 0.0.
    """
    from ubt.adapters.pdf.render_fidelity import _select_probe_pages

    assert _select_probe_pages({}, 3, 8) == [1, 2, 3]
    assert _select_probe_pages({}, 10, 4) == [1, 2, 3, 4]
    # With blocks, only the pages they name are sampled.
    assert _select_probe_pages({2: [], 5: []}, 9, 8) == [2, 5]


def test_compute_render_fidelity_measures_whole_page_without_blocks() -> None:
    """The harness's ``blocks=[]`` path must actually measure (E2E)."""
    pytest.importorskip("pypdfium2")
    from ubt.adapters.pdf.render_fidelity import compute_render_fidelity

    pdf = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "synthetic-mono.pdf"
    if not pdf.exists():
        pytest.skip("synthetic corpus unavailable")
    stats = compute_render_fidelity(pdf, pdf, [], dpi=72, max_pages=2)
    assert stats["pages_measured"] > 0
    assert stats["non_text_diff_ratio"] == 0.0
