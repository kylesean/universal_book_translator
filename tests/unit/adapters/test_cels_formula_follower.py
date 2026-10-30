"""Unit tests for CELS Stage 5 Formula Follower (GAP-14).

Tests Form XObject extraction, mixed CJK + rigid formula stream injection,
and text extraction roundtrip via PDFium.
"""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
import pytest

from ubt.adapters.pdf.cels_font import embed_subset_font, install_font
from ubt.adapters.pdf.cels_formula import (
    create_formula_form_xobject,
    emit_line_with_rigids,
    inject_mixed_cid_line,
)
from ubt.adapters.pdf.cels_inject import LineSite, ShowSite, scan_lines
from ubt.core.layout import Box, InlineRigidBox

pytestmark = pytest.mark.fast

_CJK_FONT = Path("/usr/share/fonts/MapleMono-NF-CN-unhinted/MapleMono-NF-CN-Regular.ttf")


@pytest.fixture
def cjk_font() -> Path:
    if not _CJK_FONT.exists():
        pytest.skip("MapleMono font not found")
    return _CJK_FONT


def test_create_formula_form_xobject_builds_valid_clipping_matrix() -> None:
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(b"0 0 1 rg 10 10 80 40 re f")

    formula_box = Box(x0=20.0, y0=15.0, x1=70.0, y1=35.0)  # width 50, height 20
    rigid = create_formula_form_xobject(
        pdf,
        page,
        formula_box,
        token_id="formula_test_1",
        resource_name="/FmTest",
        baseline_offset_pt=-2.0,
    )

    assert rigid.token_id == "formula_test_1"
    assert rigid.width_pt == 50.0
    assert rigid.height_pt == 20.0
    assert rigid.baseline_offset_pt == -2.0
    assert rigid.xobject_ref == "/FmTest"

    # Verify Form XObject in page resources
    assert "/XObject" in page.Resources
    form = page.Resources["/XObject"]["/FmTest"]
    assert str(form.Subtype) == "/Form"
    assert list(form.BBox) == [20.0, 15.0, 70.0, 35.0]
    assert list(form.Matrix) == [1.0, 0.0, 0.0, 1.0, -20.0, -15.0]


def test_emit_line_with_rigids_generates_isolated_graphics_state(cjk_font: Path) -> None:
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(400, 200))
    page.Resources = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({})})

    text_content = "漏极电流为正"
    font = embed_subset_font(pdf, font_path=cjk_font, text=text_content)
    install_font(page, font, pdf)

    dummy_site = ShowSite(
        index=0,
        op="Tj",
        text="orig",
        x=50.0,
        baseline=100.0,
        font_size=10.0,
        text_scale=1.0,
        font_key="/F1",
        horizontal_scale=1.0,
        char_spacing=0.0,
        word_spacing=0.0,
    )
    line = LineSite(
        sites=(dummy_site,),
        text="orig",
        x0=50.0,
        x1=150.0,
        baseline=100.0,
        font_size=10.0,
        text_scale=1.0,
        font_key="/F1",
        horizontal_scale=1.0,
        char_spacing=0.0,
        word_spacing=0.0,
    )

    rigid = InlineRigidBox(
        token_id="math_curr",
        width_pt=30.0,
        height_pt=12.0,
        baseline_offset_pt=-1.0,
        xobject_ref="/FmMath",
    )

    segments = ["漏极电流", rigid, "为正"]
    ops = emit_line_with_rigids(
        line,
        segments,
        font=font,
        size_pt=10.0,
        horizontal_scale=1.0,
        char_spacing=0.0,
    )

    op_names = [str(op) for args, op in ops]
    # Assert ET precedes Do and BT follows Do to ensure graphics state validity
    assert "ET" in op_names
    assert "Do" in op_names
    assert "BT" in op_names
    do_idx = op_names.index("Do")
    et_indices = [i for i, name in enumerate(op_names) if name == "ET"]
    bt_indices = [i for i, name in enumerate(op_names) if name == "BT"]

    assert any(et_i < do_idx for et_i in et_indices)
    assert any(bt_i > do_idx for bt_i in bt_indices)


def test_formula_follower_roundtrip_on_chapter3_page(cjk_font: Path, tmp_path: Path) -> None:
    """Full roundtrip: extract formula on chapter-3 p9, rewrite line with Chinese + formula XObject."""
    source_pdf = Path("corpus/documents/chapter-3.pdf")
    if not source_pdf.exists():
        pytest.skip("chapter-3.pdf not found in corpus")

    pdf = pikepdf.open(str(source_pdf))
    page = pdf.pages[8]  # Page 9

    lines = scan_lines(page)
    assert len(lines) > 0

    # Line 0 is 'Here,' (x0=83.7, x1=105.5, baseline=579.7)
    target_line = lines[0]

    # Create a Form XObject for the Q_bulk symbol at x: 111.2..131.7, y: 575..588
    q_bulk_box = Box(x0=111.2, y0=575.0, x1=131.7, y1=588.0)
    rigid_formula = create_formula_form_xobject(
        pdf,
        page,
        q_bulk_box,
        token_id="formula_q_bulk",
        resource_name="/FmQBulk",
        baseline_offset_pt=0.0,
    )

    chinese_prose = "此处固定电荷由给出"
    font = embed_subset_font(pdf, font_path=cjk_font, text=chinese_prose)
    install_font(page, font, pdf)

    segments = ["此处固定电荷由", rigid_formula, "给出"]

    inject_mixed_cid_line(
        page,
        target_line,
        segments,
        font=font,
        size_pt=target_line.font_size,
        horizontal_scale=1.0,
        char_spacing=0.0,
        pdf=pdf,
    )

    out_file = tmp_path / "cels_formula_follower.pdf"
    pdf.save(str(out_file))

    # Verify with PDFium: text extracted properly, no crash
    pdoc = pdfium.PdfDocument(str(out_file))
    tpage = pdoc[8].get_textpage()
    extracted_text = tpage.get_text_range()

    assert "此处固定电荷由" in extracted_text
    assert "给出" in extracted_text

    # Render page to verify graphics stream does not fail
    render_image = pdoc[8].render(scale=1.0).to_pil()
    assert render_image.width > 0
    assert render_image.height > 0
