"""Unit tests for the 1:1 facsimile delivery channel (Track A).

Tests:
1. strip_pdf_text: BT...ET stripping, Form XObject recursion, selective rect keeping.
2. flow: solve_flow, punctuation preference, CJK 避头尾 rules.
3. layout_solver: solve_page_2d binary search, dynamic leading stretch, overlap resolution.
4. facsimile_renderer: coordinate transformation, markup generation, and full compilation.
"""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf.strip_pdf_text import (
    BoxRect,
    strip_form_xobjects_recursive,
    strip_stream_bt_et,
    strip_text_streams,
)
from ubt.model.span import BBox, PhysicalBox
from ubt.render.flow import (
    approximate_measure,
    solve_flow,
)
from ubt.render.layout_solver import (
    check_source_overlap,
    resolve_local_overlaps,
    solve_page_2d,
)
from ubt.render.typst.facsimile_renderer import (
    FacsimileBlockPlacement,
    FacsimilePageSpec,
    FacsimileRenderer,
)

pytestmark = pytest.mark.fast


def test_strip_stream_bt_et_removes_text_and_keeps_vectors() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    raw_content = b"10 20 m BT /F1 12 Tf (Hello World) Tj ET 30 40 l S"
    page.Contents = pdf.make_stream(raw_content)

    stripped = strip_stream_bt_et(page, pdf)
    assert b"Hello World" not in stripped
    assert b"BT" not in stripped
    assert b"ET" not in stripped
    assert b"10 20 m" in stripped
    assert b"30 40 l" in stripped
    assert b"S" in stripped


def test_strip_form_xobject_recursive() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    form = pdf.make_stream(b"5 5 m BT /F1 10 Tf (Form Text) Tj ET 25 25 l S")
    form[pikepdf.Name("/Type")] = pikepdf.Name("/XObject")
    form[pikepdf.Name("/Subtype")] = pikepdf.Name("/Form")
    form[pikepdf.Name("/BBox")] = pikepdf.Array([0, 0, 100, 100])

    page[pikepdf.Name("/Resources")] = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form))

    strip_form_xobjects_recursive(page.Resources, pdf)
    unparsed_form = form.read_bytes()
    assert b"Form Text" not in unparsed_form
    assert b"5 5 m" in unparsed_form
    assert b"25 25 l" in unparsed_form


def test_strip_text_streams_selective_keep() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    # Two text blocks: one at (50, 50), one at (200, 200)
    raw = (
        b"10 10 m "
        b"BT 1 0 0 1 50 50 Tm (Keep This Math Formula) Tj ET "
        b"BT 1 0 0 1 200 200 Tm (Strip This Prose) Tj ET "
        b"S"
    )
    page.Contents = pdf.make_stream(raw)

    keep_rect = BoxRect(page_index=1, x0=40, y0=40, x1=100, y1=100)
    stripped = strip_stream_bt_et(page, pdf, keep_rects=[keep_rect])
    assert b"Keep This Math Formula" in stripped
    assert b"Strip This Prose" not in stripped


def test_strip_text_streams_file_roundtrip(tmp_path: Path) -> None:
    src_pdf = tmp_path / "source.pdf"
    dst_pdf = tmp_path / "stripped.pdf"

    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 500))
    p1.Contents = pdf.make_stream(b"10 20 m BT /F1 10 Tf (Page 1 Text) Tj ET S")
    p2 = pdf.add_blank_page(page_size=(400, 500))
    p2.Contents = pdf.make_stream(b"30 40 m BT /F1 10 Tf (Page 2 Text) Tj ET S")
    pdf.save(src_pdf)

    res = strip_text_streams(src_pdf, dst_pdf)
    assert res.exists()

    with pikepdf.open(dst_pdf) as out:
        assert len(out.pages) == 2
        p1_bytes = out.pages[0].Contents.read_bytes()
        p2_bytes = out.pages[1].Contents.read_bytes()
        assert b"Page 1 Text" not in p1_bytes
        assert b"Page 2 Text" not in p2_bytes
        assert b"10 20 m" in p1_bytes
        assert b"30 40 m" in p2_bytes


def test_flow_solver_punctuation_preference() -> None:
    text = "这是第一句话。这是第二句话！这是第三句话？"
    bbox1: BBox = (0.0, 0.0, 100.0, 30.0)
    bbox2: BBox = (0.0, 0.0, 100.0, 30.0)
    box1 = PhysicalBox.of(1, bbox1)
    box2 = PhysicalBox.of(1, bbox2)

    measure = approximate_measure(font_size_pt=10.0, leading_pt=12.0)
    placements = solve_flow(text, [box1, box2], measure)
    assert len(placements) == 2
    assert placements[0].text.endswith(("。", "！", "？"))
    assert placements[0].text + placements[1].text == text


def test_flow_solver_cjk_kinsoku() -> None:
    text = "一个没有任何标点符号的长长中文段落用来测试避头尾规则是否正确生效"
    bbox1: BBox = (0.0, 0.0, 50.0, 20.0)
    bbox2: BBox = (0.0, 0.0, 100.0, 50.0)
    box1 = PhysicalBox.of(1, bbox1)
    box2 = PhysicalBox.of(1, bbox2)

    measure = approximate_measure(font_size_pt=10.0, leading_pt=12.0)
    placements = solve_flow(text, [box1, box2], measure)
    assert len(placements) == 2
    assert len(placements[0].text) > 0
    assert len(placements[1].text) > 0


def test_layout_solver_binary_search_convergence() -> None:
    # 2 paragraphs in one column
    records = [
        {"id": "p1", "top": 500.0, "bot": 420.0, "w": 300.0, "glyphs": 50.0, "n": 3},
        {"id": "p2", "top": 400.0, "bot": 320.0, "w": 300.0, "glyphs": 60.0, "n": 4},
    ]
    sol = solve_page_2d([records], source_size_pt=10.0, source_leading_pt=13.0)
    assert sol.fits is True
    assert 7.5 <= sol.uniform_size_pt <= 11.0  # 1.10 * 10 = 11.0
    assert sol.leading_em >= 0.15
    assert sol.delivery_tier == "L1"


def test_layout_solver_dynamic_leading_stretch() -> None:
    # Very short paragraphs with plenty of vertical room (EN->ZH contraction)
    records = [
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 400.0, "glyphs": 20.0, "n": 2},
        {"id": "p2", "top": 450.0, "bot": 350.0, "w": 400.0, "glyphs": 20.0, "n": 2},
    ]
    sol = solve_page_2d([records], source_size_pt=10.0, source_leading_pt=13.0)
    assert sol.fits is True
    assert sol.uniform_size_pt == 11.0  # Capped at ceiling
    assert sol.leading_em >= 0.30  # Leading stretched


def test_layout_solver_overlap_detection_and_resolution() -> None:
    # Overlapping records: p1.bot = 400, p2.top = 401.5 (overlap of 1.5pt)
    records = [
        {"id": "p1", "top": 500.0, "bot": 400.0, "w": 300.0, "glyphs": 30.0, "n": 2},
        {"id": "p2", "top": 401.5, "bot": 300.0, "w": 300.0, "glyphs": 30.0, "n": 2},
    ]
    # Without resolution, overlaps trigger source_overlap
    assert check_source_overlap([records]) is True

    # With local resolution, micro-overlap <= 2.0pt is resolved
    resolved = resolve_local_overlaps(records, max_merge_overlap_pt=2.0)
    assert check_source_overlap([resolved]) is False


def test_facsimile_renderer_markup_and_coordinates() -> None:
    placement1 = FacsimileBlockPlacement(
        block_id="b1",
        page=1,
        x0=50.0,
        y0=600.0,
        x1=350.0,
        y1=650.0,
        text="Chapter 1: Facsimile Architecture",
        is_heading=True,
        text_color_hex="#0078ae",
    )
    placement2 = FacsimileBlockPlacement(
        block_id="b2",
        page=1,
        x0=50.0,
        y0=500.0,
        x1=350.0,
        y1=580.0,
        text="This is 1:1 visual isomorphism prose.",
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.5,
        leading_em=0.38,
        placements=[placement1, placement2],
    )

    renderer = FacsimileRenderer()
    markup = renderer.build_markup([page_spec], "bg_stripped.pdf", target_lang="zh")

    assert "#page(width: 500.0pt, height: 700.0pt" in markup
    assert 'image("bg_stripped.pdf", page: 1' in markup
    # dy = page_h - y1 = 700 - 650 = 50.0pt for placement1
    assert "dx: 50.0pt, dy: 50.0pt" in markup
    # dy = 700 - 580 = 120.0pt for placement2
    assert "dx: 50.0pt, dy: 120.0pt" in markup
    assert '#text(weight: "bold", fill: rgb("#0078ae"))' in markup


def test_facsimile_renderer_full_e2e_compile(tmp_path: Path) -> None:
    # 1. Create a 2-page source PDF with vector background lines
    src_pdf = tmp_path / "book_source.pdf"
    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 600))
    p1.Contents = pdf.make_stream(b"20 20 m 200 200 l S BT /F1 12 Tf (Source text p1) Tj ET")
    p2 = pdf.add_blank_page(page_size=(400, 600))
    p2.Contents = pdf.make_stream(b"40 40 m 300 300 l S BT /F1 12 Tf (Source text p2) Tj ET")
    pdf.save(src_pdf)

    # 2. Strip background
    bg_stripped = tmp_path / "bg_stripped.pdf"
    strip_text_streams(src_pdf, bg_stripped)

    # 3. Create page specifications
    p1_spec = FacsimilePageSpec(
        page_no=1,
        page_w=400.0,
        page_h=600.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[
            FacsimileBlockPlacement(
                block_id="p1_h",
                page=1,
                x0=40.0,
                y0=520.0,
                x1=360.0,
                y1=560.0,
                text="第一章：原位同构系统",
                is_heading=True,
            ),
            FacsimileBlockPlacement(
                block_id="p1_b",
                page=1,
                x0=40.0,
                y0=440.0,
                x1=360.0,
                y1=500.0,
                text="第一页翻译正文，原位填充在矩形内。",
            ),
        ],
    )
    p2_spec = FacsimilePageSpec(
        page_no=2,
        page_w=400.0,
        page_h=600.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[
            FacsimileBlockPlacement(
                block_id="p2_b",
                page=2,
                x0=40.0,
                y0=480.0,
                x1=360.0,
                y1=540.0,
                text="第二页翻译正文，矢量背景完全保留。",
            )
        ],
    )

    out_pdf = tmp_path / "delivered_facsimile.pdf"
    build_dir = tmp_path / ".build_facsimile"

    renderer = FacsimileRenderer()
    renderer.render_document(
        [p1_spec, p2_spec],
        bg_stripped,
        out_pdf,
        build_dir,
        target_lang="zh",
    )

    assert out_pdf.exists()
    with pikepdf.open(out_pdf) as res:
        assert len(res.pages) == 2
        # Verify background vector XObject exists in output pages
        assert "/XObject" in res.pages[0].Resources
        assert "/XObject" in res.pages[1].Resources
