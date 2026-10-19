import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from ubt.adapters.pdf.plain_text_extractor import sample_page_indices
from ubt.api.models import AssessDocumentFacts, JobAssessRequest
from ubt.core.archetype import Archetype, DocCategory, MathDensity
from ubt.core.assess import RouteRecommendation, _synthesize_warnings, assess_document_async
from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast


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
        recommended_dual_mode="dual",
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
        recommended_dual_mode="mono",
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
    assert "OCR" in font_warns[0].detail_zh

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


def test_job_assess_request_pages_validation() -> None:
    req = JobAssessRequest(input_path="sample.pdf", pages="1-5")
    assert req.pages == "1-5"

    with pytest.raises(ValidationError):
        JobAssessRequest(input_path="sample.pdf", pages="invalid-range-format")

    facts = AssessDocumentFacts(
        file_name="test.pdf",
        file_size_bytes=100,
        format_ext="pdf",
        pages=100,
        chapters=5,
        source_chars=20000,
        estimated_tokens=5000,
        category="technical_book",
        detected_domain="textbook",
        domain_confidence=0.9,
        math_density="high",
        is_scanned=False,
        selected_pages=5,
    )
    assert facts.selected_pages == 5


@pytest.mark.asyncio
async def test_assess_document_async_page_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_path = tmp_path / "test.pdf"
    doc_path.write_bytes(b"%PDF-1.4 mock")

    monkeypatch.setattr(
        "ubt.core.assess._pdf_facts",
        lambda *args, **kwargs: {
            "page_count": 100,
            "probed_pages": 100,
            "probed_chars": 50000,
            "scan_page_share": 0.0,
            "has_formulas": False,
        },
    )

    config = UBTConfig()
    rep = await assess_document_async(doc_path, config, pages="1-10")
    assert rep.document.pages == 100
    assert rep.document.selected_pages == 10
    assert rep.document.source_chars == 5000
    assert any(w.code == "PAGE_RANGE_FILTERED" for w in rep.warnings)


def test_build_cost_disables_rolling_summary_for_page_slice() -> None:
    from ubt.core.assess import _build_cost

    config = UBTConfig(enable_rolling_summary=True)
    cost_normal = _build_cost(
        config,
        billable_blocks=50,
        source_chars=5000,
        prefix=100,
        is_exact=True,
        pages=10,
        scan_pages=0,
        chapters=5,
        route_mode="long",
        profile_name="general",
        warnings=[],
        is_page_slice=False,
    )
    assert cost_normal.rollup_calls == 5

    cost_slice = _build_cost(
        config,
        billable_blocks=50,
        source_chars=5000,
        prefix=100,
        is_exact=True,
        pages=10,
        scan_pages=0,
        chapters=5,
        route_mode="long",
        profile_name="general",
        warnings=[],
        is_page_slice=True,
    )
    assert cost_slice.rollup_calls == 0


# --- the warning must not overstate or understate the damage -----------------


def test_the_warning_says_the_residue_count_is_a_lower_bound() -> None:
    """``residue_chars`` counts six non-ASCII codepoints, so it is a floor.

    The same misdecoding rewrites ASCII glyphs into other ASCII glyphs — on the
    reference document ``$`` stands for a minus sign and ``%`` for a
    multiplication sign (``cm$3`` for ``cm⁻³``, ``%1018`` for ``×10¹⁸``) — and
    those are invisible to the confirm set by design, because they occur
    legitimately in ordinary technical prose.

    A reader who takes 275 as the total damage will under-scope the proofing
    work, so the warning has to say it is a lower bound.
    """
    from ubt.adapters.pdf.extraction_witness import CONFIRM_CHARS

    # The blind spot is structural: every confirm char is non-ASCII.
    assert all(ord(ch) > 127 for ch in CONFIRM_CHARS), (
        "an ASCII char entered CONFIRM_CHARS; it would fire on ordinary prose"
    )
    # And the characters that actually substitute for operators here are ASCII.
    assert ord("$") < 128 and ord("%") < 128

    source = Path("ubt/core/assess.py").read_text(encoding="utf-8")
    block = re.search(r'"FONT_RESIDUE_RISK",\s*\n\s*"warn",\s*\n(.*?)\n\s*\{', source, re.S)
    assert block, "could not locate the FONT_RESIDUE_RISK message"
    message = block.group(1)
    assert "下界" in message, "the warning no longer says the count is a lower bound"


def test_the_warning_distinguishes_recoverable_from_lost_glyphs() -> None:
    """``部分文字可能不可恢复`` was true but useless.

    Brackets and equals signs survive a MacRoman remap and are recoverable by
    rule; superscripts, subscripts and operators that landed on ASCII
    codepoints are gone. The two cases need different handling, so the copy
    names both instead of averaging them into "some text may be unrecoverable".
    """
    source = Path("ubt/core/assess.py").read_text(encoding="utf-8")
    block = re.search(r'"FONT_RESIDUE_RISK",\s*\n\s*"warn",\s*\n(.*?)\n\s*\{', source, re.S)
    assert block, "could not locate the FONT_RESIDUE_RISK message"
    message = block.group(1)

    assert "还原" in message, "the warning no longer mentions what is recoverable"
    assert "永久丢失" in message, "the warning no longer mentions what is lost"
    assert "部分文字可能不可恢复" not in message, (
        "the old blanket phrasing is back; it collapses two distinct cases"
    )


def test_both_locales_carry_the_same_two_qualifiers() -> None:
    """The console renders the i18n copy, not the engine's ``detail_zh``.

    Fixing only the engine message would leave the console overstating it,
    which is the surface the operator actually reads.
    """
    repo = Path(__file__).resolve().parents[3]
    for name, needles in (
        ("zh.ts", ("下界", "永久丢失")),
        ("en.ts", ("lower bound", "lost for good")),
    ):
        path = repo / "web" / "src" / "i18n" / "translations" / name
        if not path.exists():
            pytest.skip("web/ frontend is not present in this checkout")
        text = path.read_text(encoding="utf-8")
        block = re.search(r"FONT_RESIDUE_RISK:\s*\n?\s*'([^']*)'", text)
        assert block, f"{name} has no FONT_RESIDUE_RISK copy"
        copy = block.group(1)
        for needle in needles:
            assert needle in copy, f"{name} lost {needle!r}"
