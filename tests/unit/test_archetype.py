"""Archetype detection lives in ``ubt.core.archetype``; ``ubt.core.advisor`` re-exports it."""

from pathlib import Path

import pytest

from ubt.core.archetype import (
    DocCategory,
    MathDensity,
    analyze_archetype,
    classify_category,
    detect_domain,
    detect_math_density,
)


def test_advisor_reexports_are_the_same_objects() -> None:
    # The recommendation layer must not fork the enums it classifies with.
    from ubt.core.advisor import DocCategory as AdvisorDocCategory
    from ubt.core.advisor import MathDensity as AdvisorMathDensity

    assert AdvisorDocCategory is DocCategory
    assert AdvisorMathDensity is MathDensity


def test_math_density_thresholds() -> None:
    assert detect_math_density("") is MathDensity.NONE
    assert detect_math_density("Once upon a time in a quiet village.") is MathDensity.NONE
    dense = "\\frac{\\partial u}{\\partial t} + \\int \\sqrt{1-x^2} dx " * 4
    assert detect_math_density(dense) is MathDensity.HIGH


def test_domain_detection_and_floor() -> None:
    domain, conf = detect_domain("general prose about cooking")
    assert domain == "general"
    semi = "The finfet gate drainer shows subthreshold doping and bandgap capacitance."
    domain, conf = detect_domain(semi)
    assert domain == "semiconductor"
    assert conf >= 0.4


def test_category_rules() -> None:
    assert (
        classify_category("pdf", MathDensity.HIGH, "semiconductor", 10)
        is DocCategory.ACADEMIC_PAPER
    )
    assert (
        classify_category("pdf", MathDensity.HIGH, "semiconductor", 200)
        is DocCategory.TECHNICAL_BOOK
    )
    assert classify_category("epub", MathDensity.NONE, "general", 40) is DocCategory.LITERATURE
    assert classify_category("md", MathDensity.LOW, "general", 3) is DocCategory.GENERAL


def test_analyze_archetype_on_markdown(tmp_path: Path) -> None:
    f = tmp_path / "book.md"
    f.write_text("# One\n\nprose " * 20, encoding="utf-8")
    arch = analyze_archetype(f)
    assert arch.format_ext == "md"
    assert arch.category is DocCategory.LITERATURE
    assert arch.is_scanned is False
    assert arch.sample_chars > 0


@pytest.mark.parametrize("ext", ["md", "txt"])
def test_sample_never_raises_on_garbage(tmp_path: Path, ext: str) -> None:
    f = tmp_path / f"broken.{ext}"
    f.write_bytes(b"\xff\x00binary junk")
    arch = analyze_archetype(f)  # errors="ignore" path; must still return facts
    assert arch.page_or_ch_count >= 1
