"""Unit and integration tests for the UBT v2 NativePDFCodec stack."""

from __future__ import annotations

import re
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
import pytest

from ubt.codecs.pdf import (
    FontEngine,
    FontSlot,
    HeuristicLayoutDetector,
    NativePDFCodec,
    ParagraphGeometry,
    analyze_page_stream,
    build_formula_xobject,
    typeset_paragraph,
)
from ubt.codecs.pdf.font_engine import resolve_slot_paths
from ubt.codecs.protocol import DeliveryMode, UnitRole


def _make_synthetic_pdf(
    path: Path,
    *,
    num_pages: int = 3,
    rotate: int = 0,
    crop_offset: tuple[float, float] = (0.0, 0.0),
    with_inline_math: bool = False,
) -> Path:
    """Create a minimal multi-page PDF with real Helvetica & Symbol font streams."""
    pdf = pikepdf.new()
    ox, oy = crop_offset
    w, h = 612.0, 792.0

    f_body = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
            }
        )
    )
    f_math = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/CMMI10"),
            }
        )
    )

    for p in range(1, num_pages + 1):
        page = pdf.add_blank_page(page_size=(w + ox, h + oy))
        page["/MediaBox"] = pikepdf.Array([ox, oy, ox + w, oy + h])
        page["/CropBox"] = pikepdf.Array([ox, oy, ox + w, oy + h])
        if rotate:
            page["/Rotate"] = rotate
        page["/Resources"] = pikepdf.Dictionary(
            {"/Font": pikepdf.Dictionary({"/F1": f_body, "/FM": f_math})}
        )
        lines = [
            f"BT /F1 16 Tf 1 0 0 1 {ox + 72} {oy + 720} Tm (Section {p}: Overview) Tj ET",
            f"BT /F1 11 Tf 1 0 0 1 {ox + 72} {oy + 680} Tm "
            f"(This is paragraph one on page {p} discussing layout fidelity.) Tj ET",
            f"BT /F1 11 Tf 1 0 0 1 {ox + 72} {oy + 664} Tm "
            f"(It spans two lines cleanly inside a standard text column.) Tj ET",
        ]
        if with_inline_math:
            lines.append(
                f"BT /F1 11 Tf 1 0 0 1 {ox + 72} {oy + 620} Tm (Where ) Tj "
                f"/FM 11 Tf 1 0 0 1 {ox + 108} {oy + 620} Tm (x) Tj "
                f"/F1 11 Tf 1 0 0 1 {ox + 120} {oy + 620} Tm ( converges fast.) Tj ET"
            )
        page["/Contents"] = pikepdf.Stream(pdf, "\n".join(lines).encode("ascii"))

    pdf.save(path)
    pdf.close()
    return path


def _extract_pdf_text(pdf_path: Path, page_index: int = 0) -> str:
    from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK

    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            page = doc[page_index]
            try:
                tp = page.get_textpage()
                try:
                    return str(tp.get_text_range())
                finally:
                    tp.close()
            finally:
                page.close()
        finally:
            doc.close()


@pytest.mark.fast
def test_font_engine_embeds_cff_and_ttf_with_searchable_tounicode(tmp_path: Path) -> None:
    engine = FontEngine(target_lang="zh")
    sample = "原生内容流排版测试 Hello 2026！"
    engine.record_usage(sample, FontSlot.SERIF_REGULAR)
    width = engine.measure_text_pt(sample, FontSlot.SERIF_REGULAR, 12.0)
    assert width > 60.0

    out_pdf = tmp_path / "font_test.pdf"
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(400, 200))
    embedded = engine.embed_all_used(pdf)
    ef = embedded[FontSlot.SERIF_REGULAR]
    page["/Resources"] = pikepdf.Dictionary(
        {"/Font": pikepdf.Dictionary({ef.resource_name: ef.pdf_font_obj})}
    )
    payload = ef.encode_text(sample).hex().upper().encode("ascii")
    page["/Contents"] = pikepdf.Stream(
        pdf,
        b"BT " + ef.resource_name.encode() + b" 12 Tf 1 0 0 1 20 100 Tm <" + payload + b"> Tj ET",
    )
    pdf.save(out_pdf)
    pdf.close()

    extracted = _extract_pdf_text(out_pdf, 0)
    assert "原生内容流排版测试" in extracted
    assert "Hello 2026" in extracted


@pytest.mark.fast
def test_stream_analyzer_normalizes_cropbox_and_rotation(tmp_path: Path) -> None:
    src = _make_synthetic_pdf(tmp_path / "shifted.pdf", num_pages=1, crop_offset=(50.0, 40.0))
    with pikepdf.open(src) as pdf:
        ana = analyze_page_stream(pdf.pages[0], page_no=1)
    assert ana.page_width == pytest.approx(612.0)
    assert ana.page_height == pytest.approx(792.0)
    assert ana.text_ops
    # Normalized x0 should be ~72pt, NOT 122pt.
    assert 68.0 <= ana.text_ops[0].bbox[0] <= 76.0
    assert 710.0 <= ana.text_ops[0].baseline_y <= 730.0

    rot = _make_synthetic_pdf(tmp_path / "rot90.pdf", num_pages=1, rotate=90)
    with pikepdf.open(rot) as pdf:
        ana_r = analyze_page_stream(pdf.pages[0], page_no=1)
    # 90-degree rotation swaps visual width/height.
    assert ana_r.page_width == pytest.approx(792.0)
    assert ana_r.page_height == pytest.approx(612.0)


@pytest.mark.fast
def test_formula_xobject_slices_only_formula_ops(tmp_path: Path) -> None:
    src = _make_synthetic_pdf(tmp_path / "math.pdf", num_pages=1, with_inline_math=True)
    with pikepdf.open(src) as pdf:
        ana = analyze_page_stream(pdf.pages[0], page_no=1)
        math_ops = [op for op in ana.text_ops if op.is_math_font]
        assert len(math_ops) == 1
        fbox = build_formula_xobject(
            pdf,
            pdf.pages[0],
            math_ops,
            tag_id="f1",
            resource_name="/UBT_FX_0",
            prose_baseline_y=math_ops[0].baseline_y,
            stream_to_logical=ana.stream_to_logical,
            logical_to_stream=ana.logical_to_stream,
        )
        assert fbox.xobject is not None
        raw = bytes(fbox.xobject.read_bytes())
        # Must NOT contain surrounding prose from the page stream.
        assert b"Overview" not in raw
        assert b"/FM" in raw
        assert len(raw) < 512


@pytest.mark.fast
def test_typesetter_reports_exact_overflow_when_exceeding_floor() -> None:
    engine = FontEngine(target_lang="zh")
    geom = ParagraphGeometry(
        bbox=(0.0, 0.0, 80.0, 22.0),
        dominant_font_size=10.0,
        line_height=12.0,
        alignment="left",
        first_line_indent=0.0,
        is_serif=True,
        is_bold=False,
        color_rgb=(0.0, 0.0, 0.0),
        max_borrow_down_pt=0.0,
    )
    long_text = "这是一段故意超长用于触发精确溢出检测的学术中文译文内容" * 4
    para = typeset_paragraph(
        unit_id="u1",
        page_no=1,
        role=UnitRole.PARAGRAPH,
        marked_text=long_text,
        tokens=(),
        geom=geom,
        formulas={},
        font_engine=engine,
        target_lang="zh",
    )
    assert para.overflow is not None
    assert para.overflow.clipped is True
    assert para.overflow.overflow_ratio > 1.0
    assert 0 < para.overflow.target_char_budget < len(long_text)


@pytest.mark.fast
def test_native_pdf_codec_mono_interleaved_sbs_and_page_slice(tmp_path: Path) -> None:
    src = _make_synthetic_pdf(tmp_path / "book.pdf", num_pages=4, with_inline_math=True)
    codec = NativePDFCodec(layout_detector=HeuristicLayoutDetector())

    prof = codec.inspect(src)
    assert prof.page_count == 4
    assert not prof.is_scanned
    assert prof.recommended_mode == DeliveryMode.MONO

    # Slice non-1-based pages {2, 3} to verify zero page-index drift!
    _, units = codec.extract(src, pages={2, 3})
    assert units
    assert all(u.page_no in {2, 3} for u in units)
    assert any(any(t.kind == "rigid" for t in u.tokens) for u in units)

    tag_re = re.compile(r"(<[^>]+>)")

    def _translate(marked: str) -> str:
        tags = "".join(tag_re.findall(marked))
        return f"这是中文译文段落测试{tags}内容。"

    translations = {u.unit_id: _translate(u.source_marked) for u in units if u.translate}

    out_mono = tmp_path / "slice_mono.pdf"
    rep_mono = codec.apply(
        src,
        units,
        translations,
        target_lang="zh",
        mode=DeliveryMode.MONO,
        output_path=out_mono,
        pages={2, 3},
    )
    assert rep_mono.page_count_source == 2
    assert rep_mono.page_count_output == 2
    assert rep_mono.patched_units > 0
    txt_mono = _extract_pdf_text(out_mono, 0)
    assert "这是中文译文段落测试" in txt_mono
    assert "<f1/>" not in txt_mono

    out_inter = tmp_path / "slice_inter.pdf"
    rep_inter = codec.apply(
        src,
        units,
        translations,
        target_lang="zh",
        mode=DeliveryMode.INTERLEAVED,
        output_path=out_inter,
        pages={2, 3},
    )
    assert rep_inter.page_count_output == 4

    out_sbs = tmp_path / "slice_sbs.pdf"
    rep_sbs = codec.apply(
        src,
        units,
        translations,
        target_lang="zh",
        mode=DeliveryMode.SIDE_BY_SIDE,
        output_path=out_sbs,
        pages={2, 3},
    )
    assert rep_sbs.page_count_output == 2
    with pikepdf.open(out_sbs) as pdf:
        mb = [float(v) for v in pdf.pages[0]["/MediaBox"]]
        assert (mb[2] - mb[0]) == pytest.approx(1224.0, abs=2.0)

    png = codec.render_page_preview(src, 2, units, translations, target_lang="zh", dpi=96)
    assert png.startswith(b"\x89PNG")


@pytest.mark.fast
def test_pdfium_recovery_keeps_a_lines_last_glyph(tmp_path: Path) -> None:
    """A glyph whose center falls just past the estimated advance is still recovered.

    The analyzer's advance estimate for a font without /ToUnicode can be a few
    points short of the glyph's real box. Ranking pdfium chars by center alone
    dropped the trailing ``s`` of ``The Descendants``; recovery must use
    horizontal overlap so the line's last character survives.
    """
    from pdf_builders import write_text_pdf

    pdf = write_text_pdf(
        tmp_path / "glyphs.pdf",
        [("The Descendants", "A short line that also ends in a full stop.")],
    )
    with pikepdf.open(pdf) as doc:
        analysis = analyze_page_stream(doc.pages[0], page_no=1, pdfium_fallback_path=pdf)
    texts = [op.text for op in analysis.text_ops]
    assert "The Descendants" in texts
    assert "A short line that also ends in a full stop." in texts


@pytest.mark.fast
def test_mixed_script_target_resolves_a_cjk_fallback_face() -> None:
    """A Chinese name in a Spanish book must not become tofu.

    The target language's own slots are Latin faces; the CJK fallback slot is
    what keeps a mixed-script name (or a rehearsal marker) renderable.
    """
    if FontSlot.CJK_FALLBACK not in resolve_slot_paths("es"):
        pytest.skip("no CJK font is installed on this host")
    engine = FontEngine(target_lang="es")
    assert engine.resolve_char_slot("模", FontSlot.SERIF_REGULAR) == FontSlot.CJK_FALLBACK
    assert engine.resolve_char_slot("A", FontSlot.SERIF_REGULAR) == FontSlot.SERIF_REGULAR
