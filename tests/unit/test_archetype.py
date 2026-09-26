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


@pytest.mark.fast
def test_detect_math_density_recognizes_unicode_type_theory_and_greek_math() -> None:
    """detect_math_density must flag Unicode math (Greek letters, turnstile ⊢,
    tensor ⊗, arrows →, Theorem/Definition)."""
    from ubt.core.archetype import MathDensity, detect_math_density

    sample = (
        "1. Introduction\nWe study spatiotemporal composability.\n"
        "Definition 2.1. A context transformation Γ ⊢ M : A ⊗ B → C ⊸ D satisfies "
        "f ∘ g = id_Γ for all α, β ∈ Φ(Γ) and ∀x ∈ Δ, ρ(x) ≤ σ(x)."
    )
    assert detect_math_density(sample) == MathDensity.HIGH


def test_analyze_archetype_on_html_and_markdown(tmp_path: Path) -> None:
    html_f = tmp_path / "article.html"
    html_f.write_text(
        "<html><body><h1>Chapter 1</h1><p>Prose content here.</p></body></html>", encoding="utf-8"
    )
    arch = analyze_archetype(html_f)
    assert arch.format_ext == "html"
    assert arch.sample_chars > 0
    assert arch.category is DocCategory.LITERATURE

    md_f = tmp_path / "book.markdown"
    md_f.write_text("# Chapter 1\n\nSome great literature.", encoding="utf-8")
    arch_md = analyze_archetype(md_f)
    assert arch_md.format_ext == "markdown"
    assert arch_md.sample_chars > 0
    assert arch_md.category is DocCategory.LITERATURE
