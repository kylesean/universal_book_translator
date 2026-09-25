"""Unit tests for the rigid-render fidelity ruler (pure Pillow core + advisory)."""

from __future__ import annotations

from PIL import Image

from ubt.adapters.pdf.render_fidelity import (
    _COVERAGE_WARN,
    _RESIDUAL_WARN,
    diff_outside_masks,
    fidelity_findings,
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
