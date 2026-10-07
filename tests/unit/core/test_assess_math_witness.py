from __future__ import annotations

from ubt.adapters.pdf.plain_text_extractor import sample_page_indices
from ubt.core.archetype import Archetype, DocCategory, MathDensity
from ubt.core.assess import RouteRecommendation, _synthesize_warnings
from ubt.core.config import UBTConfig


def test_sample_page_indices_edge_cases() -> None:
    assert sample_page_indices(0) == []
    assert sample_page_indices(1) == [0]
    assert sample_page_indices(5) == [0, 1, 2, 3, 4]
    assert sample_page_indices(10) == list(range(10))

    # Multi-page book: must start with page 0 (title/cover) and distribute across the book
    indices_300 = sample_page_indices(300, max_samples=10)
    assert len(indices_300) == 10
    assert indices_300[0] == 0
    # Must NOT have clustered front-matter indices [0, 1, 2, 3, 4, 6, 8, 9]
    assert indices_300[-1] >= 260
    assert indices_300[len(indices_300) // 2] >= 140


def test_witness_warnings_no_false_positive_when_unconfirmed() -> None:
    arch = Archetype(
        format_ext="pdf",
        page_or_ch_count=100,
        is_scanned=False,
        math_density=MathDensity.NONE,
        detected_domain="general",
        domain_confidence=1.0,
        category=DocCategory.TECHNICAL_BOOK,
        sample_chars=5000,
    )
    route = RouteRecommendation(
        mode="long",
        reason="test",
        recommended_preset="standard",
        recommended_render_engine="reflow",
        recommended_dual_mode="inline",
        recommended_profile="general",
        confidence=0.9,
        confidence_basis="",
    )
    config = UBTConfig()
    config.draft_model = "test-model"

    # Only unconfirmed at-risk pages (e.g. Type1 fonts without /ToUnicode, but 0 residue chars)
    pdf = {
        "witness": {"confirmed_pages": 0, "residue_chars": 0, "at_risk_pages": 50},
        "scan_page_share": 0.0,
    }
    signals: list[str] = []
    warnings = _synthesize_warnings(arch, pdf, config, route, signals)

    # Must NOT emit FONT_RESIDUE_RISK warning
    codes = [w.code for w in warnings]
    assert "FONT_RESIDUE_RISK" not in codes
    # Must record in quality_signals
    assert any("缺少 /ToUnicode" in s for s in signals)


def test_witness_warnings_contextual_remedy_on_confirmed_damage() -> None:
    arch_digital = Archetype(
        format_ext="pdf",
        page_or_ch_count=100,
        is_scanned=False,
        math_density=MathDensity.NONE,
        detected_domain="general",
        domain_confidence=1.0,
        category=DocCategory.TECHNICAL_BOOK,
        sample_chars=5000,
    )
    route = RouteRecommendation(
        mode="long",
        reason="test",
        recommended_preset="publication",
        recommended_render_engine="reflow",
        recommended_dual_mode="monolingual",
        recommended_profile="textbook",
        confidence=0.9,
        confidence_basis="",
    )
    config = UBTConfig()
    config.draft_model = "test-model"

    pdf_damaged = {
        "witness": {"confirmed_pages": 4, "residue_chars": 15, "at_risk_pages": 10},
        "scan_page_share": 0.0,
    }
    signals: list[str] = []
    warnings = _synthesize_warnings(arch_digital, pdf_damaged, config, route, signals)

    font_warns = [w for w in warnings if w.code == "FONT_RESIDUE_RISK"]
    assert len(font_warns) == 1
    # For digital PDF, should not ask user to re-scan
    assert "重新扫描" not in font_warns[0].detail_zh
    assert "OCR" in font_warns[0].detail_zh or "rigid" in font_warns[0].detail_zh

    # For scanned PDF, re-scan advice is appropriate
    arch_scanned = Archetype(
        format_ext="pdf",
        page_or_ch_count=100,
        is_scanned=True,
        math_density=MathDensity.NONE,
        detected_domain="general",
        domain_confidence=1.0,
        category=DocCategory.TECHNICAL_BOOK,
        sample_chars=5000,
    )
    signals_scanned: list[str] = []
    warnings_scanned = _synthesize_warnings(
        arch_scanned, pdf_damaged, config, route, signals_scanned
    )
    scanned_warns = [w for w in warnings_scanned if w.code == "FONT_RESIDUE_RISK"]
    assert len(scanned_warns) == 1
    assert "重新扫描" in scanned_warns[0].detail_zh
