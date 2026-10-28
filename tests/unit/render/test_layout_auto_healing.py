"""Unit tests for Typst adaptive layout auto-healing pass and multiline formula folding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.typst.auto_healing import (
    AutoHealingAction,
    AutoHealingReport,
    LayoutAutoHealer,
    fold_multiline_math,
)
from ubt.render.typst.document_renderer import TypstDocumentRenderer, render_table
from ubt.render.typst.layout_diagnostics import (
    AnomalyType,
    DocumentLayoutReport,
    LayoutAnomaly,
    PageLayoutReport,
    PageLayoutSpec,
    Severity,
)

pytestmark = pytest.mark.fast


def _make_dummy_pdf(path: Path) -> None:
    """Create minimal valid PDF binary."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] /Contents 4 0 R >> endobj\n"
        b"4 0 obj << /Length 50 >> stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello AutoHeal) Tj ET\n"
        b"endstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000214 00000 n \n"
        b"trailer << /Size 5 /Root 1 0 R >>\nstartxref\n314\n%%EOF\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_content)


def test_fold_multiline_math_short_formula() -> None:
    """Short formulas within max_chars remain unchanged."""
    short = "y = a + b"
    assert fold_multiline_math(short, max_chars=30) == short


def test_fold_multiline_math_nested_groups_protected() -> None:
    """Operators inside nested parentheses or brackets are never broken."""
    formula = "f(x) = frac(a + b + c + d, e + f + g) + h"
    # Even with small max_chars, it should NOT break inside frac(..., ...)
    folded = fold_multiline_math(formula, max_chars=10)
    assert "frac(a + b + c + d" in folded
    # But breaks at the top-level ' + h'
    assert "\\\n + h" in folded or " \\\n + h" in folded


def test_fold_multiline_math_long_relation_and_binary() -> None:
    """Long formulas fold at top-level relation and binary operators."""
    formula = "L(theta) = E[log p(x | z)] - D_KL(q(z | x) || p(z)) + lambda * R(theta)"
    folded = fold_multiline_math(formula, max_chars=25)
    assert "\\\n" in folded
    lines = folded.split("\\\n")
    assert len(lines) >= 2


def test_auto_healing_models_serialization() -> None:
    """Verify AutoHealingAction and AutoHealingReport serialization."""
    action = AutoHealingAction(
        action_type="adaptive_layout_scaling",
        description="Pass 1: scale tables",
        pass_num=1,
        delta_errors_resolved=2,
    )
    d_act = action.to_dict()
    assert d_act["pass_num"] == 1
    assert d_act["delta_errors_resolved"] == 2

    report = AutoHealingReport(
        initial_passed=False,
        final_passed=True,
        initial_error_count=2,
        final_error_count=0,
        initial_max_overflow_pt=12.5,
        final_max_overflow_pt=0.0,
        passes_attempted=1,
        healed=True,
        actions=[action],
    )
    d_rep = report.to_dict()
    assert d_rep["healed"] is True
    assert d_rep["final_error_count"] == 0
    assert len(d_rep["actions"]) == 1


def test_layout_auto_healer_no_spills_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When initial PDF passes layout diagnostics, auto-healer returns immediately."""
    pdf_path = tmp_path / "doc.pdf"
    _make_dummy_pdf(pdf_path)

    clean_report = DocumentLayoutReport(
        pdf_path=str(pdf_path),
        total_pages=1,
        checked_pages=[1],
        layout_template="book",
        tolerance_pt=5.0,
        page_reports={
            1: PageLayoutReport(page=1, page_width=595.28, page_height=841.89, line_count=10)
        },
    )

    import ubt.render.typst.auto_healing as ah_mod

    monkeypatch.setattr(ah_mod, "diagnose_pdf_layout", lambda *a, **kw: clean_report)

    healer = LayoutAutoHealer(spec=PageLayoutSpec.book(), max_passes=2)
    passed, report = healer.run_healing_loop(
        output_path=pdf_path,
        build_dir=tmp_path / "build",
        blocks=[],
        target_lang="zh",
        renderer_builder_fn=lambda **kw: None,
        compile_fn=lambda *a: (True, ""),
    )

    assert passed is True
    assert report.healed is False
    assert report.passes_attempted == 0


def test_layout_auto_healer_heals_spill_successfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When initial PDF has spills, healer adjusts options, recompiles, and clears spills."""
    pdf_path = tmp_path / "output.pdf"
    _make_dummy_pdf(pdf_path)
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    # Diagnosis 1 (spill): 1 error
    anomaly = LayoutAnomaly(
        anomaly_type=AnomalyType.MARGIN_SPILL_RIGHT,
        severity=Severity.FAIL,
        page=1,
        bbox=(50.0, 100.0, 610.0, 115.0),
        delta_pt=14.72,
        text_snippet="Very wide text table line",
    )
    spill_page_rep = PageLayoutReport(
        page=1,
        page_width=595.28,
        page_height=841.89,
        line_count=10,
        anomalies=[anomaly],
    )
    spill_report = DocumentLayoutReport(
        pdf_path=str(pdf_path),
        total_pages=1,
        checked_pages=[1],
        layout_template="book",
        tolerance_pt=5.0,
        page_reports={1: spill_page_rep},
    )

    # Diagnosis 2 (healed): 0 errors
    clean_report = DocumentLayoutReport(
        pdf_path=str(pdf_path),
        total_pages=1,
        checked_pages=[1],
        layout_template="book",
        tolerance_pt=5.0,
        page_reports={
            1: PageLayoutReport(page=1, page_width=595.28, page_height=841.89, line_count=10)
        },
    )

    call_count = 0

    def mock_diagnose(path: Path | str, *a: Any, **kw: Any) -> DocumentLayoutReport:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return spill_report
        return clean_report

    import ubt.render.typst.auto_healing as ah_mod

    monkeypatch.setattr(ah_mod, "diagnose_pdf_layout", mock_diagnose)

    class MockRenderer:
        def __init__(self, **opts: Any) -> None:
            self.opts = opts

        def build_source(self, blocks: Any, build_dir: Path, target_lang: str) -> tuple[str, Any]:
            return "healed source", None

    def mock_compile(typ_p: str, out_p: str) -> tuple[bool, str]:
        _make_dummy_pdf(Path(out_p))
        return True, ""

    healer = LayoutAutoHealer(spec=PageLayoutSpec.book(), max_passes=2)
    passed, report = healer.run_healing_loop(
        output_path=pdf_path,
        build_dir=build_dir,
        blocks=[],
        target_lang="zh",
        renderer_builder_fn=lambda **kw: MockRenderer(**kw),
        compile_fn=mock_compile,
    )

    assert passed is True
    assert report.healed is True
    assert report.initial_passed is False
    assert report.final_passed is True
    assert report.initial_error_count == 1
    assert report.final_error_count == 0
    assert len(report.actions) == 1
    assert report.actions[0].delta_errors_resolved == 1


def test_document_renderer_with_auto_healing_options(tmp_path: Path) -> None:
    """Verify TypstDocumentRenderer applies auto-healing table/formula scaling and paragraph spacing."""
    table_elem = make_element(
        id="b_table",
        spine_index=1,
        block_type=BlockType.TABLE,
        source_text="| Col A | Col B |\n|---|---|\n| Val 1 | Val 2 |",
    )
    table_block = IRBlock(
        element=table_elem,
        target_text="| 列甲 | 列乙 |\n|---|---|\n| 内容1 | 内容2 |",
    )

    formula_elem = make_element(
        id="b_formula",
        spine_index=2,
        block_type=BlockType.FORMULA,
        source_text="$$ L(theta) = A + B + C + D + E + F + G + H + I + J + K + L + M + N + O + P + Q + R $$",
    )
    formula_block = IRBlock(element=formula_elem)

    renderer = TypstDocumentRenderer(
        table_scale_factor=0.85,
        formula_scale=0.90,
        break_long_formulas=True,
        par_spacing_em=0.82,
    )

    build_dir = tmp_path / "typst_build"
    source, _ = renderer.build_source(
        [table_block, formula_block], build_dir=build_dir, target_lang="zh"
    )

    # Check paragraph spacing in preamble
    assert "spacing: 0.82em" in source
    # Check formula scaled in #text block
    assert "#text(size: 9.45pt)" in source
    # Check long formula folded across lines
    assert "\\\n" in source


def test_render_table_scale_factor() -> None:
    """Verify render_table accepts table_scale_factor and scales text accordingly."""
    markdown_table = (
        "| Identifier | Description | Status |\n"
        "| :--- | :--- | :--- |\n"
        "| `metadata.openclaw.requires` | Core pipeline dependencies | ACTIVE |\n"
    )
    scale_normal: list[float] = []
    _ = render_table(markdown_table, scale_out=scale_normal, table_scale_factor=1.0)
    assert scale_normal

    scale_shrunk: list[float] = []
    out_shrunk = render_table(markdown_table, scale_out=scale_shrunk, table_scale_factor=0.85)
    assert scale_shrunk
    assert scale_shrunk[0] < scale_normal[0]
    assert "#table(" in out_shrunk
