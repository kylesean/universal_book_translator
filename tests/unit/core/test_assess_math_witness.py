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

    indices_300 = sample_page_indices(300, max_samples=10)
    assert len(indices_300) == 10
    assert indices_300[0] == 0
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

    pdf = {
        "witness": {"confirmed_pages": 0, "residue_chars": 0, "at_risk_pages": 50},
        "scan_page_share": 0.0,
    }
    signals: list[str] = []
    warnings = _synthesize_warnings(arch, pdf, config, route, signals)

    codes = [w.code for w in warnings]
    assert "FONT_RESIDUE_RISK" not in codes
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
        recommended_preset="high_fidelity",
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
    assert "重新扫描" not in font_warns[0].detail_zh
    assert "OCR" in font_warns[0].detail_zh

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


def test_the_warning_says_the_residue_count_is_a_lower_bound() -> None:
    from ubt.adapters.pdf.extraction_witness import CONFIRM_CHARS

    assert all(ord(ch) > 127 for ch in CONFIRM_CHARS), (
        "an ASCII char entered CONFIRM_CHARS; it would fire on ordinary prose"
    )
    assert ord("$") < 128 and ord("%") < 128

    source = Path("ubt/core/assess.py").read_text(encoding="utf-8")
    block = re.search(r'"FONT_RESIDUE_RISK",\s*\n\s*"warn",\s*\n(.*?)\n\s*\{', source, re.S)
    assert block, "could not locate the FONT_RESIDUE_RISK message"
    message = block.group(1)
    assert "下界" in message, "the warning no longer says the count is a lower bound"


def test_the_warning_distinguishes_recoverable_from_lost_glyphs() -> None:
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


def test_pathological_gate_reads_extraction_cost() -> None:
    arch = Archetype(
        format_ext="pdf",
        page_or_ch_count=92,
        is_scanned=False,
        math_density=MathDensity.HIGH,
        detected_domain="general",
        domain_confidence=1.0,
        category=DocCategory.TECHNICAL_BOOK,
        sample_chars=5000,
    )
    route = RouteRecommendation(
        mode="long",
        reason="test",
        recommended_preset="high_fidelity",
        recommended_dual_mode="mono",
        recommended_profile="textbook",
        confidence=0.9,
        confidence_basis="",
    )
    config = UBTConfig()
    config.draft_model = "test-model"

    toc_shaped = {
        "max_extract_cost": 2289 * 5363,
        "max_extract_page": 2,
        "max_extract_rects": 2289,
        "max_extract_chars": 5363,
        "scan_page_share": 0.0,
    }
    codes = [w.code for w in _synthesize_warnings(arch, toc_shaped, config, route, [])]
    assert "PATHOLOGICAL_PAGE_RISK" not in codes

    for stale in (
        {"max_rect_rows": 2289, "scan_page_share": 0.0},
        {"max_text_rows": 2500, "scan_page_share": 0.0},
    ):
        codes = [w.code for w in _synthesize_warnings(arch, stale, config, route, [])]
        assert "PATHOLOGICAL_PAGE_RISK" not in codes, stale

    pathological = {
        "max_extract_cost": 9395 * 104456,
        "max_extract_page": 1,
        "max_extract_rects": 9395,
        "max_extract_chars": 104456,
        "scan_page_share": 0.0,
    }
    (warning,) = [
        w
        for w in _synthesize_warnings(arch, pathological, config, route, [])
        if w.code == "PATHOLOGICAL_PAGE_RISK"
    ]
    assert warning.params["rects"] == 9395
    assert warning.params["chars"] == 104456
    assert warning.params["page"] == 1
    assert "9395" in warning.detail_zh


def test_pathological_copy_reports_the_cost_in_both_locales() -> None:
    repo = Path(__file__).resolve().parents[3]
    for name, needles in (
        ("zh.ts", ("{page}", "{rects}", "{chars}")),
        ("en.ts", ("{page}", "{rects}", "{chars}")),
    ):
        path = repo / "web" / "src" / "i18n" / "translations" / name
        if not path.exists():
            pytest.skip("web/ frontend is not present in this checkout")
        text = path.read_text(encoding="utf-8")
        block = re.search(r"PATHOLOGICAL_PAGE_RISK:\s*\n?\s*'([^']*)'", text)
        assert block, f"{name} has no PATHOLOGICAL_PAGE_RISK copy"
        copy = block.group(1)
        for needle in needles:
            assert needle in copy, f"{name} lost {needle!r}"


def test_recommend_route_penalizes_confidence_on_font_damage() -> None:
    from ubt.core.assess import _recommend_route
    from ubt.core.router_mode import RouteDecision

    arch = Archetype(
        format_ext="pdf",
        page_or_ch_count=26,
        is_scanned=False,
        math_density=MathDensity.HIGH,
        detected_domain="semiconductor",
        domain_confidence=1.0,
        category=DocCategory.ACADEMIC_PAPER,
        sample_chars=10000,
    )
    route = RouteDecision(
        mode="long",
        pages=26,
        chars=40000,
        has_scan=False,
        formula_heavy=False,
        reason="test",
    )
    config = UBTConfig()

    pdf_damaged = {
        "page_count": 26,
        "scan_page_share": 0.0,
        "has_formulas": False,
        "witness": {"confirmed_pages": 20, "residue_chars": 275, "at_risk_pages": 5},
    }
    rec = _recommend_route(arch, route, pdf_damaged, config)

    assert rec.confidence < 0.7
    assert "字体编码损伤" in rec.confidence_basis
    assert "20 页" in rec.confidence_basis


def test_runtime_estimate_incorporates_docling_overhead() -> None:
    from ubt.core.assess import _runtime_estimate

    config = UBTConfig()
    runtime_fast = _runtime_estimate(
        0, 0, config, pages=20, format_ext="pdf", pdf_engine="pdfium", preset="standard"
    )
    runtime_docling = _runtime_estimate(
        0, 0, config, pages=20, format_ext="pdf", pdf_engine="docling", preset="high_fidelity"
    )

    assert runtime_docling.est_seconds_low > runtime_fast.est_seconds_low
    assert runtime_docling.est_seconds_low >= 50.0
