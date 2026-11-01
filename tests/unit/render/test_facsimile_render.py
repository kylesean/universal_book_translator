from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pikepdf
import pytest

from ubt.adapters.pdf.docling_render import _block_font_size, _facsimile_strip_plan
from ubt.adapters.pdf.strip_pdf_text import (
    BoxRect,
    strip_form_xobjects_recursive,
    strip_stream_text,
    strip_text_streams,
)
from ubt.core.ir.models import BlockType, BookManifest, BoundingBox, IRBlock, make_element
from ubt.model.span import BBox, PhysicalBox
from ubt.render.flow import (
    approximate_measure,
    solve_flow,
)
from ubt.render.layout_solver import (
    FLOOR_PT,
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


def _instructions_in(page: pikepdf.Page) -> list[tuple[Any, Any]]:
    pairs: list[tuple[Any, Any]] = []
    for item in pikepdf.parse_content_stream(page):
        operands = getattr(item, "operands", None)
        operator = getattr(item, "operator", None)
        if operands is None or operator is None:
            continue
        if not hasattr(operands, "__iter__"):
            continue
        pairs.append((operands, operator))
    return pairs


def _texts_in(stream_bytes: bytes) -> list[str]:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(stream_bytes)
    out: list[str] = []
    for operands, operator in _instructions_in(page):
        if str(operator) not in ("Tj", "TJ", "'", '"'):
            continue
        for operand in operands:
            if isinstance(operand, pikepdf.String):
                out.append(str(operand))
            elif isinstance(operand, pikepdf.Array):
                out.extend(str(item) for item in operand if isinstance(item, pikepdf.String))
    return [text for text in out if text]


def _operators_in(stream_bytes: bytes) -> list[str]:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(stream_bytes)
    return [str(operator) for _operands, operator in _instructions_in(page)]


def _source_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "source_666.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(540, 666))
    pdf.save(path)
    return path


def _fake_block(
    block_id: str,
    *,
    page: int,
    bbox: tuple[float, float, float, float],
    block_type: BlockType = BlockType.NARRATIVE,
    skip_translate: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=block_id,
        block_type=block_type,
        bbox=BoundingBox(page=page, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
        skip_translate=skip_translate,
        element=None,
        style=None,
        provenance=None,
        source_text="",
    )


def test_strip_stream_text_no_rects_keeps_everything() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"10 20 m BT /F1 12 Tf 1 0 0 1 50 200 Tm (Hello World) Tj ET 30 40 l S"
    )

    stripped = strip_stream_text(page)
    assert _texts_in(stripped) == ["Hello World"]
    assert b"10 20 m" in stripped
    assert b"30 40 l" in stripped


def test_strip_stream_text_erases_only_lines_in_the_box() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"10 10 m "
        b"BT 1 0 0 1 50 50 Tm (Keep This Math Formula) Tj ET "
        b"BT 1 0 0 1 200 200 Tm (Strip This Prose) Tj ET "
        b"S"
    )

    erase = [BoxRect(page_index=1, x0=190.0, y0=190.0, x1=290.0, y1=210.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == ["Keep This Math Formula"]
    assert b"10 10 m" in stripped


def test_pen_left_of_erase_box_still_erases() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(b"BT /F1 16 Tf 1 0 0 1 60 100 Tm (Leaked Title) Tj ET")

    erase = [BoxRect(page_index=1, x0=60.6, y0=95.0, x1=250.0, y1=115.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == []


def test_pen_pad_does_not_erase_text_left_of_the_box() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(b"BT /F1 9 Tf 1 0 0 1 40 100 Tm (Neighbour) Tj ET")

    erase = [BoxRect(page_index=1, x0=60.6, y0=95.0, x1=250.0, y1=115.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == ["Neighbour"]


def test_pen_pad_covers_the_measured_toc_entry_bearing() -> None:
    from ubt.adapters.pdf.strip_pdf_text import _keep_point

    erase = [BoxRect(page_index=1, x0=87.89, y0=669.21, x1=524.10, y1=680.44)]
    assert _keep_point(87.23, 675.0, erase, None, font_size=11.0) is False


def test_pen_pad_scales_with_the_font_size() -> None:
    from ubt.adapters.pdf.strip_pdf_text import PEN_PAD_EM, _keep_point

    erase = [BoxRect(page_index=1, x0=100.0, y0=0.0, x1=200.0, y1=20.0)]

    assert _keep_point(98.1, 10.0, erase, None, font_size=40.0) is False
    assert 9.0 * PEN_PAD_EM < 1.9
    assert _keep_point(98.1, 10.0, erase, None, font_size=9.0) is True


def test_keep_rect_wins_over_erase_rect() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(b"BT 1 0 0 1 60 60 Tm (Protected Formula) Tj ET")

    erase = [BoxRect(page_index=1, x0=40.0, y0=40.0, x1=200.0, y1=80.0)]
    keep = [BoxRect(page_index=1, x0=50.0, y0=50.0, x1=150.0, y1=70.0)]
    stripped = strip_stream_text(page, erase_rects=erase, keep_rects=keep)
    assert _texts_in(stripped) == ["Protected Formula"]


def test_strip_blanking_preserves_the_operator_sequence() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"BT /F1 12 Tf 14 TL 1 0 0 1 50 200 Tm (erase me) Tj "
        b"T* (keep me) Tj "
        b"0 -20 Td (erase me too) Tj "
        b"[(a) -120 (b) 50 (c)] TJ "
        b"ET"
    )

    erase = [BoxRect(page_index=1, x0=0.0, y0=180.0, x1=300.0, y1=205.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _operators_in(stripped) == [
        "BT",
        "Tf",
        "TL",
        "Tm",
        "Tj",
        "T*",
        "Tj",
        "Td",
        "Tj",
        "TJ",
        "ET",
    ]
    assert _texts_in(stripped) == ["erase me too", "a", "b", "c"]


def test_strip_splits_a_bt_block_by_line() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"BT /F1 10 Tf 12 TL 1 0 0 1 40 250 Tm "
        b"(top line) Tj T* "
        b"(middle line) Tj T* "
        b"(bottom line) Tj "
        b"ET"
    )

    erase = [BoxRect(page_index=1, x0=0.0, y0=232.0, x1=300.0, y1=244.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == ["top line", "bottom line"]


def test_strip_respects_the_ctm_when_placing_a_line() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(b"q 10 0 0 10 0 0 cm BT 1 0 0 1 5 5 Tm (scaled) Tj ET Q")

    erase = [BoxRect(page_index=1, x0=40.0, y0=40.0, x1=60.0, y1=60.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == []
    assert b"q" in stripped
    assert b"Q" in stripped


def test_strip_leaves_an_unparseable_stream_intact() -> None:
    pdf = pikepdf.Pdf.new()

    broken = pdf.make_stream(b"garbage")
    broken.write(b"\x00\xff not a content stream \x00")
    stripped = strip_stream_text(broken)
    assert stripped != b""


def test_strip_form_xobjects_keeps_figure_labels_without_erase_rects() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    form = pdf.make_stream(b"5 5 m BT /F1 10 Tf 1 0 0 1 5 5 Tm (Axis 68) Tj ET 25 25 l S")
    form[pikepdf.Name("/Type")] = pikepdf.Name("/XObject")
    form[pikepdf.Name("/Subtype")] = pikepdf.Name("/Form")
    form[pikepdf.Name("/BBox")] = pikepdf.Array([0, 0, 100, 100])
    page[pikepdf.Name("/Resources")] = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form))

    resources = page.Resources
    assert isinstance(resources, pikepdf.Dictionary)
    visited = strip_form_xobjects_recursive(resources)
    assert visited == 1
    assert b"Axis 68" in form.read_bytes()
    assert b"25 25 l" in form.read_bytes()


def test_strip_text_streams_absent_page_keeps_its_text(tmp_path: Path) -> None:
    src_pdf = tmp_path / "source.pdf"
    dst_pdf = tmp_path / "stripped.pdf"

    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 500))
    p1.Contents = pdf.make_stream(b"10 20 m BT /F1 10 Tf 1 0 0 1 50 300 Tm (Page 1 Text) Tj ET S")
    p2 = pdf.add_blank_page(page_size=(400, 500))
    p2.Contents = pdf.make_stream(b"30 40 m BT /F1 10 Tf 1 0 0 1 50 300 Tm (Page 2 Text) Tj ET S")
    pdf.save(src_pdf)

    erase = {1: [BoxRect(page_index=1, x0=40.0, y0=290.0, x1=200.0, y1=310.0)]}
    strip_text_streams(src_pdf, dst_pdf, per_page_erase_rects=erase)

    with pikepdf.open(dst_pdf) as out:
        assert len(out.pages) == 2
        p1_bytes = out.pages[0].Contents.read_bytes()
        p2_bytes = out.pages[1].Contents.read_bytes()
        assert _texts_in(p1_bytes) == []
        assert _texts_in(p2_bytes) == ["Page 2 Text"]
        assert b"10 20 m" in p1_bytes
        assert b"30 40 m" in p2_bytes


def test_strip_text_streams_rects_are_page_scoped(tmp_path: Path) -> None:
    src_pdf = tmp_path / "source.pdf"
    dst_pdf = tmp_path / "stripped.pdf"

    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(400, 500))
    page.Contents = pdf.make_stream(b"BT /F1 10 Tf 1 0 0 1 50 300 Tm (Body Text) Tj ET")
    pdf.save(src_pdf)

    erase = {1: [BoxRect(page_index=2, x0=40.0, y0=290.0, x1=200.0, y1=310.0)]}
    strip_text_streams(src_pdf, dst_pdf, per_page_erase_rects=erase)

    with pikepdf.open(dst_pdf) as out:
        assert _texts_in(out.pages[0].Contents.read_bytes()) == ["Body Text"]


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
    records = [
        {"id": "p1", "top": 500.0, "bot": 420.0, "w": 300.0, "glyphs": 50.0, "n": 3},
        {"id": "p2", "top": 400.0, "bot": 320.0, "w": 300.0, "glyphs": 60.0, "n": 4},
    ]
    sol = solve_page_2d([records], source_size_pt=10.0, source_leading_pt=13.0)
    assert sol.fits is True
    assert 7.5 <= sol.uniform_size_pt <= 11.0
    assert sol.leading_em >= 0.15
    assert sol.delivery_tier == "L1"


def test_layout_solver_dynamic_leading_stretch() -> None:
    records = [
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 400.0, "glyphs": 20.0, "n": 2},
        {"id": "p2", "top": 450.0, "bot": 350.0, "w": 400.0, "glyphs": 20.0, "n": 2},
    ]
    sol = solve_page_2d([records], source_size_pt=10.0, source_leading_pt=13.0)
    assert sol.fits is True
    assert sol.uniform_size_pt == 11.0
    assert sol.leading_em >= 0.30


def test_layout_solver_uses_the_page_source_size() -> None:
    records = [
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 400.0, "glyphs": 20.0, "n": 2},
    ]
    sol = solve_page_2d([records], source_size_pt=11.3, source_leading_pt=15.0)
    assert sol.uniform_size_pt == pytest.approx(12.0)


def test_layout_solver_never_returns_a_size_below_the_floor() -> None:
    records = [
        {"id": "p1", "top": 600.0, "bot": 300.0, "w": 400.0, "glyphs": 400.0, "n": 20},
    ]
    sol = solve_page_2d([records], source_size_pt=6.2, source_leading_pt=16.2)
    assert sol.uniform_size_pt >= FLOOR_PT
    assert sol.delivery_tier == "L2"
    assert sol.fits is False


def test_layout_solver_floor_is_not_a_target() -> None:
    records = [
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 400.0, "glyphs": 20.0, "n": 2},
    ]
    sol = solve_page_2d([records], source_size_pt=9.0, source_leading_pt=12.0)
    assert sol.uniform_size_pt == pytest.approx(9.9)
    assert sol.uniform_size_pt > FLOOR_PT


def test_layout_solver_overlap_detection_and_resolution() -> None:
    records = [
        {"id": "p1", "top": 500.0, "bot": 400.0, "w": 300.0, "glyphs": 30.0, "n": 2},
        {"id": "p2", "top": 401.5, "bot": 300.0, "w": 300.0, "glyphs": 30.0, "n": 2},
    ]
    assert check_source_overlap([records]) is True

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
        font_weight="bold",
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
    assert "dx: 50.0pt, dy: 50.0pt" in markup
    assert "dx: 50.0pt, dy: 120.0pt" in markup
    assert '#text(weight: "bold", fill: rgb("#0078ae"))' in markup


def test_facsimile_renderer_clones_the_heading_size() -> None:
    placement = FacsimileBlockPlacement(
        block_id="h1",
        page=1,
        x0=50.0,
        y0=600.0,
        x1=350.0,
        y1=650.0,
        text="Chapter One",
        is_heading=True,
        font_size_pt=16.2,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "size: 16.2pt" in markup


def test_block_font_size_clones_a_small_caption_below_the_body_floor() -> None:
    block = SimpleNamespace(
        style=SimpleNamespace(font_size=6.45),
        provenance=SimpleNamespace(font_size=6.45),
    )
    size = _block_font_size(block, is_hdr=True, body_size=8.0)
    assert size is not None, "a heading with a plausible measurement must not inherit"
    assert size == pytest.approx(6.45)
    assert size < FLOOR_PT, "the caption is below the body floor by design"


def test_block_font_size_ignores_an_implausible_measurement() -> None:
    block = SimpleNamespace(
        style=SimpleNamespace(font_size=1.0),
        provenance=SimpleNamespace(font_size=1.0),
    )
    size = _block_font_size(block, is_hdr=True, body_size=10.0)
    assert size == pytest.approx(16.0)


def test_block_font_size_lets_body_prose_inherit_the_page_size() -> None:
    block = SimpleNamespace(
        style=SimpleNamespace(font_size=11.0),
        provenance=SimpleNamespace(font_size=11.0),
    )
    assert _block_font_size(block, is_hdr=False, body_size=9.0) is None


def test_facsimile_renderer_heading_weight_is_not_inverted() -> None:
    placement = FacsimileBlockPlacement(
        block_id="h1",
        page=1,
        x0=50.0,
        y0=600.0,
        x1=350.0,
        y1=650.0,
        text="A Regular Heading",
        is_heading=True,
        font_weight="regular",
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert 'weight: "bold"' not in markup


def test_facsimile_renderer_tables_are_tables() -> None:
    placement = FacsimileBlockPlacement(
        block_id="t1",
        page=1,
        x0=50.0,
        y0=400.0,
        x1=400.0,
        y1=520.0,
        text="| Metric | Value |\n|---|---|\n| Acc | 91.2 |",
        is_table=True,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "#table(" in markup
    assert "|" not in markup


def test_facsimile_strip_plan_does_not_band_the_furniture_margins(tmp_path: Path) -> None:
    body = _fake_block(
        "b1",
        page=1,
        bbox=(80.0, 31.3, 456.0, 95.2),
        block_type=BlockType.NARRATIVE,
        skip_translate=False,
    )
    running_head = _fake_block(
        "b2",
        page=1,
        bbox=(83.7, 624.0, 294.6, 633.9),
        block_type=BlockType.HEADING,
        skip_translate=True,
    )

    erase, keep = _facsimile_strip_plan({1: [body, running_head]}, 1, _source_pdf(tmp_path))

    assert len(erase[1]) == 1, "the translated body block must be erased"
    assert len(keep[1]) == 1, (
        "only the untranslated block may be protected; a furniture band would "
        "add a second full-width rect"
    )
    assert (keep[1][0].x0, keep[1][0].x1) == (83.7, 294.6), (
        "the keep rect must be the running head's own box, not the full page width"
    )


def test_facsimile_strip_plan_keeps_body_prose_that_reaches_into_the_margin(tmp_path: Path) -> None:
    page_h = 666.0
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(540, page_h))
    page.Contents = pdf.make_stream(
        b"BT 1 0 0 1 121 626 Tm (70 CHAPTER 3 Core model) Tj ET "
        b"BT 1 0 0 1 121 37 Tm (last line of the paragraph) Tj ET"
    )

    body = _fake_block(
        "b1",
        page=1,
        bbox=(121.0, 31.3, 456.0, 95.2),
        block_type=BlockType.NARRATIVE,
        skip_translate=False,
    )
    erase, keep = _facsimile_strip_plan({1: [body]}, 1, _source_pdf(tmp_path))
    stripped = strip_stream_text(page, erase_rects=erase[1], keep_rects=keep.get(1, []))

    texts = _texts_in(stripped)
    assert "70 CHAPTER 3 Core model" in texts, "the running head must survive"
    assert "last line of the paragraph" not in texts, (
        "the body line at y=37 is inside a translated block and must be erased, "
        "even though it sits below 0.08·H"
    )


def test_facsimile_renderer_full_e2e_compile(tmp_path: Path) -> None:
    src_pdf = tmp_path / "book_source.pdf"
    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 600))
    p1.Contents = pdf.make_stream(b"20 20 m 200 200 l S BT /F1 12 Tf (Source text p1) Tj ET")
    p2 = pdf.add_blank_page(page_size=(400, 600))
    p2.Contents = pdf.make_stream(b"40 40 m 300 300 l S BT /F1 12 Tf (Source text p2) Tj ET")
    pdf.save(src_pdf)

    bg_stripped = tmp_path / "bg_stripped.pdf"
    erase = {
        page_no: [BoxRect(page_index=page_no, x0=0.0, y0=0.0, x1=400.0, y1=600.0)]
        for page_no in (1, 2)
    }
    strip_text_streams(src_pdf, bg_stripped, per_page_erase_rects=erase)

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
        assert "/XObject" in res.pages[0].Resources
        assert "/XObject" in res.pages[1].Resources


def test_solve_flow_protects_atomic_inline_math_and_markers() -> None:
    from ubt.model.span import PhysicalBox
    from ubt.render.flow import approximate_measure, solve_flow

    box1 = PhysicalBox(
        page=1,
        bbox=(50.0, 400.0, 140.0, 415.0),
        available_width=90.0,
        available_height=15.0,
    )
    box2 = PhysicalBox(
        page=1,
        bbox=(50.0, 300.0, 250.0, 360.0),
        available_width=200.0,
        available_height=60.0,
    )
    measure = approximate_measure(font_size_pt=10.0, leading_pt=13.0)

    text = "前置说明文字 $f(x, y) = a + b + c$ 后续说明文字以及 ⟦FIG_REF_01⟧ 结尾。"
    placements = solve_flow(text, [box1, box2], measure)
    assert len(placements) == 2
    for pl in placements:
        assert pl.text.count("$") % 2 == 0, f"Unbalanced '$' in box slice: {pl.text!r}"
        assert pl.text.count("⟦") == pl.text.count("⟧"), (
            f"Split sentinel marker in box slice: {pl.text!r}"
        )
    combined = "".join(pl.text for pl in placements)
    assert "$f(x, y) = a + b + c$" in combined
    assert "⟦FIG_REF_01⟧" in combined


def test_solve_page_2d_clamps_vertical_glue_at_obstacle_top() -> None:
    from ubt.render.layout_solver import solve_page_2d

    col_free = [
        {"id": "p1", "top": 500.0, "bot": 460.0, "w": 200.0, "n": 3, "glyphs": 120},
        {"id": "p2", "top": 200.0, "bot": 150.0, "w": 200.0, "n": 4, "glyphs": 60},
    ]
    sol_free = solve_page_2d([col_free], source_size_pt=10.0, source_leading_pt=12.0)

    col_blocked = [
        {
            "id": "p1",
            "top": 500.0,
            "bot": 460.0,
            "w": 200.0,
            "n": 3,
            "glyphs": 120,
            "obstacle_top": 458.0,
        },
        {"id": "p2", "top": 200.0, "bot": 150.0, "w": 200.0, "n": 4, "glyphs": 60},
    ]
    sol_blocked = solve_page_2d([col_blocked], source_size_pt=10.0, source_leading_pt=12.0)

    assert sol_blocked.uniform_size_pt < sol_free.uniform_size_pt, (
        "Clamping vertical glue at obstacle_top must prevent p1 from borrowing "
        "the preserved figure's vertical space"
    )


def test_strip_stream_text_uses_text_matrix_font_scale_for_pen_pad() -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(400, 600))
    page.Contents = pdf.make_stream(b"BT /F1 1 Tf 16 0 0 16 49.0 300.0 Tm (ScaledFontRun) Tj ET")
    erase = [BoxRect(page_index=1, x0=50.0, y0=290.0, x1=100.0, y1=320.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert "ScaledFontRun" not in _texts_in(stripped)


def test_strip_text_streams_descends_into_form_xobjects_with_copy_on_write(
    tmp_path: Path,
) -> None:
    src_pdf = tmp_path / "shared_form.pdf"
    out_pdf = tmp_path / "shared_form_stripped.pdf"

    pdf = pikepdf.Pdf.new()
    form_stream = pdf.make_stream(
        b"BT /F1 10 Tf 1 0 0 1 50 500 Tm (FormBodyText) Tj ET "
        b"BT /F1 10 Tf 1 0 0 1 50 100 Tm (FormAxisLabel) Tj ET"
    )
    form_stream.Type = pikepdf.Name("/XObject")
    form_stream.Subtype = pikepdf.Name("/Form")
    form_stream.BBox = [0, 0, 400, 600]

    p1 = pdf.add_blank_page(page_size=(400, 600))
    p1.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form_stream))
    p1.Contents = pdf.make_stream(b"q /Fm0 Do Q")

    p2 = pdf.add_blank_page(page_size=(400, 600))
    p2.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form_stream))
    p2.Contents = pdf.make_stream(b"q /Fm0 Do Q")
    pdf.save(src_pdf)

    erase = {1: [BoxRect(page_index=1, x0=40.0, y0=490.0, x1=250.0, y1=520.0)]}
    keep = {1: [BoxRect(page_index=1, x0=40.0, y0=90.0, x1=250.0, y1=120.0)]}
    strip_text_streams(
        src_pdf,
        out_pdf,
        per_page_erase_rects=erase,
        per_page_keep_rects=keep,
        descend_forms=True,
    )

    with pikepdf.open(out_pdf) as res:
        fm_p1 = res.pages[0].Resources.XObject["/Fm0"]
        fm_p2 = res.pages[1].Resources.XObject["/Fm0"]
        p1_texts = _texts_in(bytes(fm_p1.read_bytes()))
        p2_texts = _texts_in(bytes(fm_p2.read_bytes()))
        assert "FormBodyText" not in p1_texts
        assert "FormAxisLabel" in p1_texts
        assert "FormBodyText" in p2_texts


def test_facsimile_strip_plan_erases_all_physical_boxes_of_continuation_block(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    b = _fake_block(
        "cont_1",
        page=1,
        bbox=(50.0, 80.0, 260.0, 180.0),
        block_type=BlockType.NARRATIVE,
        skip_translate=False,
    )
    b.provenance = SimpleNamespace(
        physical_boxes=[
            {"page": 1, "bbox": [50.0, 80.0, 260.0, 180.0]},
            {"page": 1, "bbox": [300.0, 480.0, 510.0, 580.0]},
            {"page": 2, "bbox": [50.0, 500.0, 260.0, 580.0]},
        ]
    )
    erase, _keep = _facsimile_strip_plan({1: [b], 2: [b]}, 2, _source_pdf(tmp_path))
    assert len(erase[1]) == 2
    assert len(erase[2]) == 1
    assert (erase[1][1].x0, erase[1][1].x1) == (300.0, 510.0)
    assert (erase[2][0].y0, erase[2][0].y1) == (500.0, 580.0)


def test_l2_and_l3_uncompilable_formula_falls_back_to_300dpi_source_crop(
    tmp_path: Path,
) -> None:
    from ubt.core.ir.models import BoundingBox, IRBlock, make_element
    from ubt.render.typst.document_renderer import TypstDocumentRenderer
    from ubt.render.typst.page_locked_renderer import (
        PageGeometry,
        PageLockedPageSpec,
        PageLockedReflowRenderer,
        make_formula_cropper,
    )

    src_pdf = tmp_path / "eq_source.pdf"
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(400, 600))
    page.Contents = pdf.make_stream(b"0 0 0 rg 70 295 220 20 re f")
    pdf.save(src_pdf)

    eq_block = IRBlock(
        element=make_element(
            id="eq1",
            spine_index=1,
            block_type=BlockType.FORMULA,
            source_text=r"$$\frac{a}{}$$",
            bbox=BoundingBox(page=1, x0=60.0, y0=280.0, x1=320.0, y1=340.0),
            skip_translate=True,
        ),
        target_text=r"$$\frac{a}{}$$",
    )

    l2_build = tmp_path / "l2_build"
    cropper = make_formula_cropper(l2_build, src_pdf)
    l2_renderer = PageLockedReflowRenderer(math_gate=lambda _body: False)
    l2_spec = PageLockedPageSpec(
        page_no=1,
        geometry=PageGeometry(
            page_w=400.0,
            page_h=600.0,
            margin_top=36.0,
            margin_bottom=36.0,
            margin_left=36.0,
            margin_right=36.0,
        ),
        uniform_size_pt=10.0,
        leading_em=0.35,
        par_spacing_em=0.65,
        blocks=[eq_block],
    )
    l2_markup = l2_renderer.build_page_markup(l2_spec, formula_cropper=cropper)
    assert 'image("formula-crop-eq1-p1.png"' in l2_markup
    assert (l2_build / "formula-crop-eq1-p1.png").exists()

    l3_build = tmp_path / "l3_build"
    l3_renderer = TypstDocumentRenderer(
        math_gate=lambda _body: False,
        source_pdf=src_pdf,
    )
    l3_markup, stats = l3_renderer.build_source([eq_block], l3_build, "zh")
    assert 'image("formula-crop-eq1-p1.png"' in l3_markup
    assert (l3_build / "formula-crop-eq1-p1.png").exists()
    assert stats.formulas == 1
    assert ("eq1", "formula_raster_fallback") in l3_renderer.last_render_flags


@pytest.mark.asyncio
async def test_facsimile_auto_reroutes_colliding_l1_page_to_l2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.adapters.pdf import visual_gate
    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
    from ubt.core.ir.models import BookManifest, BoundingBox, IRBlock, make_element

    src_pdf = tmp_path / "two_page_src.pdf"
    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 600))
    p1.Contents = pdf.make_stream(b"BT /F1 10 Tf 1 0 0 1 50 500 Tm (Hello p1) Tj ET")
    p2 = pdf.add_blank_page(page_size=(400, 600))
    p2.Contents = pdf.make_stream(b"BT /F1 10 Tf 1 0 0 1 50 500 Tm (Hello p2) Tj ET")
    pdf.save(src_pdf)

    blocks = [
        IRBlock(
            element=make_element(
                id="b_p1",
                spine_index=1,
                block_type=BlockType.NARRATIVE,
                source_text="Hello p1",
                bbox=BoundingBox(page=1, x0=50.0, y0=480.0, x1=350.0, y1=530.0),
            ),
            target_text="第一页译文，模拟与底图残留线条发生碰撞。",
        ),
        IRBlock(
            element=make_element(
                id="b_p2",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Hello p2",
                bbox=BoundingBox(page=2, x0=50.0, y0=480.0, x1=350.0, y1=530.0),
            ),
            target_text="第二页译文，底图干净无碰撞。",
        ),
    ]

    rendered_tiers: list[list[str]] = []

    def fake_render_doc(
        self: FacsimileRenderer,
        pages: list[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        **kwargs: object,
    ) -> Path:
        rendered_tiers.append([p.tier for p in pages])
        output_path.write_bytes(src_pdf.read_bytes())
        return output_path

    monkeypatch.setattr(FacsimileRenderer, "render_document", fake_render_doc)
    monkeypatch.setattr(
        visual_gate,
        "detect_colliding_l1_pages",
        lambda bg_pdf, final_pdf, l1_pages, **_kw: {1: 0.085} if 1 in l1_pages else {},
    )

    strategy = DoclingRenderStrategy(delivery_mode="facsimile")
    manifest = BookManifest(
        doc_id="doc_col",
        title="Collision Test",
        source_path=str(src_pdf),
        metadata={"source_path": str(src_pdf)},
    )
    out_pdf = tmp_path / "out_facsimile.pdf"
    await strategy.render_blocks(manifest, blocks, "zh", out_pdf)

    assert rendered_tiers == [["L1", "L1"], ["L2", "L1"]]
    assert manifest.metadata.get("l1_ink_collision_rerouted_pages") == [1]
    assert strategy.last_page_geometry[0]["tier"] == "L2"
    assert strategy.last_page_geometry[1]["tier"] == "L1"


def test_dual_column_wide_figure_spanning(tmp_path: Path) -> None:
    from ubt.render.typst.document_renderer import TypstDocumentRenderer, _Figure

    renderer = TypstDocumentRenderer(columns=2)
    setup = renderer.page_setup_obj()
    page_w_pt = setup.width_mm / 0.352778
    wide_w = page_w_pt * 0.75
    narrow_w = page_w_pt * 0.35

    assert renderer.spans_dual_columns(wide_w) is True
    assert renderer.spans_dual_columns(narrow_w) is False

    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    asset_file = tmp_path / "fig_wide.png"
    asset_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfake")

    fig_wide = _Figure(
        asset=str(asset_file),
        width_pt=wide_w,
        block_id="fig1",
        caption="Wide architecture schematic across dual columns",
    )
    markup = renderer._figure_markup(fig_wide, build_dir)
    assert 'scope: "parent"' in markup
    assert "placement: auto" in markup

    fig_narrow = _Figure(
        asset=str(asset_file),
        width_pt=narrow_w,
        block_id="fig2",
        caption="Narrow single-column diagram",
    )
    markup_narrow = renderer._figure_markup(fig_narrow, build_dir)
    assert 'scope: "parent"' not in markup_narrow


def test_dual_column_wide_table_spanning() -> None:
    from ubt.render.typst.document_renderer import render_table

    table_md = (
        "| Col1 | Col2 | Col3 | Col4 | Col5 | Col6 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| A | B | C | D | E | F |\n"
        "| 1 | 2 | 3 | 4 | 5 | 6 |\n"
    )

    scale_out: list[float] = []
    markup = render_table(
        table_md,
        page_columns=2,
        span_full_width=True,
        full_page_width_pt=493.2,
        scale_out=scale_out,
    )
    assert '#place(top + center, scope: "parent", float: true, block(width: 100%)[' in markup
    assert "rect(width: 98%)" not in markup
    assert scale_out and scale_out[0] <= 8.5 / 10.5 + 0.01


@pytest.mark.asyncio
async def test_render_facsimile_rescues_bottleneck_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pikepdf

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
    from ubt.core.ir.models import BoundingBox
    from ubt.render.typst.facsimile_renderer import FacsimileRenderer

    src_pdf = tmp_path / "src_rescue.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(500, 700))
        pdf.save(src_pdf)

    verbose_text = (
        "我们可以看到，由于这个原因，可以通过使用分布式共识算法进行系统优化与高吞吐处理。"
    )
    blocks = [
        IRBlock(
            element=make_element(
                id="b_hdr",
                spine_index=1,
                block_type=BlockType.HEADING,
                source_text="Header",
                bbox=BoundingBox(page=1, x0=50.0, y0=620.0, x1=350.0, y1=650.0),
            ),
            target_text="第一章 架构",
        ),
        IRBlock(
            element=make_element(
                id="b_tight",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Tight container",
                bbox=BoundingBox(page=1, x0=50.0, y0=580.0, x1=170.0, y1=600.0),
            ),
            target_text=verbose_text,
        ),
        IRBlock(
            element=make_element(
                id="b_next",
                spine_index=3,
                block_type=BlockType.NARRATIVE,
                source_text="Next block",
                bbox=BoundingBox(page=1, x0=50.0, y0=555.0, x1=350.0, y1=575.0),
            ),
            target_text="紧随其后的后续段落。",
        ),
    ]

    rendered_pages: list[FacsimilePageSpec] = []

    def fake_render_doc(
        self: FacsimileRenderer,
        pages: list[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        **kwargs: object,
    ) -> Path:
        rendered_pages.extend(pages)
        output_path.write_bytes(src_pdf.read_bytes())
        return output_path

    monkeypatch.setattr(FacsimileRenderer, "render_document", fake_render_doc)

    strategy = DoclingRenderStrategy(delivery_mode="facsimile")
    manifest = BookManifest(
        doc_id="doc_rescue",
        title="Rescue Test",
        source_path=str(src_pdf),
        metadata={"source_path": str(src_pdf)},
    )
    out_pdf = tmp_path / "out_rescue.pdf"
    await strategy.render_blocks(manifest, blocks, "zh", out_pdf)

    # Repaired container should have allowed page to render as L1!
    assert len(rendered_pages) == 1
    assert rendered_pages[0].tier == "L1"
    assert ("b_tight", "container_fit_repaired") in strategy.last_render_flags
    assert "b_tight" in manifest.metadata.get("container_fit_repaired_segments", [])


def test_facsimile_table_carries_continuation_notices() -> None:
    grid = "\n".join(["| Op | Cost |", "|---|---|"] + [f"| op{i} | {i}.0 |" for i in range(6)])
    placement = FacsimileBlockPlacement(
        block_id="b0100",
        page=1,
        x0=50.0,
        y0=100.0,
        x1=350.0,
        y1=300.0,
        text=grid,
        is_table=True,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "ubt-tbl-start-src-b0100" in markup
    assert "（续下页）" in markup


def test_facsimile_table_respects_its_source_box_height() -> None:
    grid = "\n".join(["| Op | Cost |", "|---|---|"] + [f"| op{i} | {i}.0 |" for i in range(6)])
    placement = FacsimileBlockPlacement(
        block_id="b0100",
        page=1,
        x0=50.0,
        y0=100.0,
        x1=350.0,
        y1=140.0,
        text=grid,
        is_table=True,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    scales = [
        float(line.strip().split(":")[1].strip().removesuffix("em)"))
        for line in markup.splitlines()
        if line.strip().startswith("#set text(size:") and line.strip().endswith("em)")
    ]
    assert scales, "expected a shrunk table text size in the markup"
    assert min(scales) < 0.84


def test_facsimile_page_level_stretch_emitted() -> None:
    placement = FacsimileBlockPlacement(
        block_id="b001",
        page=1,
        x0=50.0,
        y0=100.0,
        x1=300.0,
        y1=150.0,
        text="Normal paragraph text.",
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        stretch_pct=90.0,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "#set text(size: 10.0pt, stretch: 90%)" in markup


def test_facsimile_block_level_stretch_emitted() -> None:
    placement = FacsimileBlockPlacement(
        block_id="b002",
        page=1,
        x0=50.0,
        y0=100.0,
        x1=300.0,
        y1=150.0,
        text="Dense squeezed heading.",
        is_heading=True,
        stretch_pct=95.0,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "stretch: 95%" in markup


def test_facsimile_caption_rigid_anchoring_and_no_flow_grouping() -> None:
    cap_placement = FacsimileBlockPlacement(
        block_id="b_cap",
        page=1,
        x0=50.0,
        y0=176.0,
        x1=250.0,
        y1=196.0,
        text="图 1：系统架构与执行流程示意图",
        is_caption=True,
        parent_figure_id="img_001",
        src_y0=176.0,
        src_y1=196.0,
    )
    prose_placement = FacsimileBlockPlacement(
        block_id="b_prose",
        page=1,
        x0=50.0,
        y0=140.0,
        x1=250.0,
        y1=170.0,
        text="本节详细介绍系统整体架构及其核心设计原则。",
        src_y0=140.0,
        src_y1=170.0,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[cap_placement, prose_placement],
    )
    renderer = FacsimileRenderer(enable_column_flow=True)
    markup = renderer.build_markup([page_spec], "bg.pdf", target_lang="zh")

    assert "flow-group: 2 blocks" not in markup
    assert "clip: false" in markup
    assert "图 1：系统架构与执行流程示意图" in markup


def test_facsimile_table_booktabs_rendering() -> None:
    grid = "\n".join(
        [
            "| 参数 | 基准 | 优化后 |",
            "|---|---|---|",
            "| 吞吐量 (QPS) | 1250 | 3820 |",
            "| P99 延迟 (ms) | 42.5 | 11.2 |",
        ]
    )
    placement = FacsimileBlockPlacement(
        block_id="b_tbl",
        page=1,
        x0=60.0,
        y0=200.0,
        x1=400.0,
        y1=320.0,
        text=grid,
        is_table=True,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "stroke: none" in markup
    assert "table.hline(stroke: 1.2pt)" in markup or "table.hline(stroke: 1.1pt)" in markup
    assert 'fill: rgb("#ffffff")' in markup


def test_docling_parser_reanchors_figure_captions() -> None:
    from ubt.adapters.pdf.docling_parser import reanchor_figure_captions
    from ubt.core.ir.models import (
        BlockProvenance,
        BlockType,
        BoundingBox,
        FlowID,
        IRBlock,
        make_element,
    )

    img_block = IRBlock(
        element=make_element(
            id="pdf_main#img_001",
            spine_index=1,
            block_type=BlockType.IMAGE,
            source_text="/tmp/img.png",
            bbox=BoundingBox(page=1, x0=50.0, y0=300.0, x1=300.0, y1=450.0),
            skip_translate=True,
        )
    )
    cap_block = IRBlock(
        element=make_element(
            id="pdf_main#b0002",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.CAPTION,
            source_text="Figure 1: High-fidelity layout preservation model.",
            bbox=BoundingBox(page=1, x0=50.0, y0=265.0, x1=300.0, y1=290.0),
        ),
        provenance=BlockProvenance(),
    )

    reanchored = reanchor_figure_captions([img_block, cap_block])
    assert len(reanchored) == 2
    re_cap = reanchored[1]
    assert re_cap.bbox is not None
    assert re_cap.bbox.y1 == 296.0
    assert getattr(re_cap.provenance, "caption_parent_id", None) == "pdf_main#img_001"
    assert getattr(re_cap.provenance, "is_anchored_caption", False) is True
    assert re_cap.provenance.physical_boxes[0]["bbox"][3] == 296.0


def test_docling_parser_reanchors_table_captions_above_tables() -> None:
    from ubt.adapters.pdf.docling_parser import reanchor_figure_captions
    from ubt.core.ir.models import (
        BlockProvenance,
        BlockType,
        BoundingBox,
        IRBlock,
        make_element,
    )

    tbl_block = IRBlock(
        element=make_element(
            id="pdf_main#tbl_001",
            spine_index=2,
            block_type=BlockType.TABLE,
            source_text="| H1 | H2 |\n|---|---|\n| V1 | V2 |",
            bbox=BoundingBox(page=1, x0=50.0, y0=200.0, x1=300.0, y1=320.0),
        )
    )
    cap_block = IRBlock(
        element=make_element(
            id="pdf_main#b0003",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Table 1: Comprehensive benchmark results across model families.",
            bbox=BoundingBox(page=1, x0=50.0, y0=330.0, x1=300.0, y1=350.0),
        ),
        provenance=BlockProvenance(),
    )

    reanchored = reanchor_figure_captions([tbl_block, cap_block])
    assert len(reanchored) == 2
    re_cap = reanchored[1]
    assert re_cap.bbox is not None
    assert re_cap.bbox.y0 == 324.0
    assert getattr(re_cap.provenance, "caption_parent_id", None) == "pdf_main#tbl_001"
    assert getattr(re_cap.provenance, "is_anchored_caption", False) is True
    assert re_cap.provenance.physical_boxes[0]["bbox"][1] == 324.0


def test_docling_parser_disambiguates_side_by_side_subfigures() -> None:
    from ubt.adapters.pdf.docling_parser import reanchor_figure_captions
    from ubt.core.ir.models import (
        BlockProvenance,
        BlockType,
        BoundingBox,
        FlowID,
        IRBlock,
        make_element,
    )

    img_left = IRBlock(
        element=make_element(
            id="pdf_main#img_left",
            spine_index=1,
            block_type=BlockType.IMAGE,
            source_text="/tmp/left.png",
            bbox=BoundingBox(page=1, x0=50.0, y0=300.0, x1=200.0, y1=450.0),
            skip_translate=True,
        )
    )
    img_right = IRBlock(
        element=make_element(
            id="pdf_main#img_right",
            spine_index=2,
            block_type=BlockType.IMAGE,
            source_text="/tmp/right.png",
            bbox=BoundingBox(page=1, x0=250.0, y0=300.0, x1=400.0, y1=450.0),
            skip_translate=True,
        )
    )
    cap_left = IRBlock(
        element=make_element(
            id="pdf_main#cap_left",
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.CAPTION,
            source_text="Figure 1(a): Left subfigure analysis.",
            bbox=BoundingBox(page=1, x0=50.0, y0=270.0, x1=200.0, y1=292.0),
        ),
        provenance=BlockProvenance(),
    )
    cap_right = IRBlock(
        element=make_element(
            id="pdf_main#cap_right",
            spine_index=4,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.CAPTION,
            source_text="Figure 1(b): Right subfigure analysis.",
            bbox=BoundingBox(page=1, x0=250.0, y0=270.0, x1=400.0, y1=292.0),
        ),
        provenance=BlockProvenance(),
    )

    reanchored = reanchor_figure_captions([img_left, img_right, cap_left, cap_right])
    assert getattr(reanchored[2].provenance, "caption_parent_id", None) == "pdf_main#img_left"
    assert getattr(reanchored[3].provenance, "caption_parent_id", None) == "pdf_main#img_right"


def test_facsimile_single_column_flow_grouping() -> None:
    renderer = FacsimileRenderer(enable_column_flow=True)
    p = FacsimilePageSpec(
        page_no=1,
        page_w=595.0,
        page_h=842.0,
        uniform_size_pt=10.0,
        leading_em=0.55,
        tracking_em=0.0,
        stretch_pct=100.0,
        placements=[
            FacsimileBlockPlacement(
                block_id="b1",
                page=1,
                x0=70.0,
                y0=600.0,
                x1=520.0,
                y1=700.0,
                src_y0=600.0,
                src_y1=700.0,
                text="First single column paragraph.",
            ),
            FacsimileBlockPlacement(
                block_id="b2",
                page=1,
                x0=70.0,
                y0=500.0,
                x1=520.0,
                y1=590.0,
                src_y0=500.0,
                src_y1=590.0,
                text="Second single column paragraph.",
            ),
        ],
        tier="L1",
    )
    markup = renderer.build_markup(pages=[p], bg_pdf_filename="bg.pdf")
    assert "flow-group: 2 blocks" in markup
