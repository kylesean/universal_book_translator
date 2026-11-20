"""End-to-end verification of default auto->native PDF pipeline & live review preview."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf.native_adapter import NativePDFAdapter
from ubt.codecs.pdf import HeuristicLayoutDetector, NativePDFCodec
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.ir.models import BlockStatus
from ubt.core.qe.fast_pass import FastPassFilter


def _make_born_digital_pdf(path: Path) -> Path:
    pdf = pikepdf.new()
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
    for p in (1, 2):
        page = pdf.add_blank_page(page_size=(612, 792))
        page["/Resources"] = pikepdf.Dictionary(
            {"/Font": pikepdf.Dictionary({"/F1": f_body, "/FM": f_math})}
        )
        stream = (
            f"BT /F1 14 Tf 1 0 0 1 72 720 Tm (Section {p}: Native Pipeline) Tj ET\n"
            f"BT /F1 11 Tf 1 0 0 1 72 680 Tm (In this experiment ) Tj "
            f"/FM 11 Tf 1 0 0 1 175 680 Tm (x) Tj "
            f"/F1 11 Tf 1 0 0 1 186 680 Tm ( converges rapidly.) Tj ET"
        )
        page["/Contents"] = pikepdf.Stream(pdf, stream.encode("ascii"))
    pdf.save(path)
    pdf.close()
    return path


@pytest.mark.fast
def test_fast_pass_rejects_dropped_inline_anchor_tags() -> None:
    flt = FastPassFilter()
    bad = flt.validate_structural_invariants(
        "Here <f1/> converges cleanly.",
        "这里收敛得很干净。",
    )
    assert bad.passed is False
    assert "Inline anchor tag mismatch" in bad.reason

    ok = flt.validate_structural_invariants(
        "Here <f1/> converges cleanly.",
        "这里 <f1/> 收敛得很干净。",
    )
    assert ok.passed is True


@pytest.mark.fast
@pytest.mark.asyncio
async def test_pipeline_orchestrator_runs_native_adapter_dry_run_and_preview(
    tmp_path: Path,
) -> None:
    src = _make_born_digital_pdf(tmp_path / "paper.pdf")
    out = tmp_path / "paper_zh_mono.pdf"
    cfg = UBTConfig(
        db_dir=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        dry_run=True,
        draft_model="mock-draft",
        pdf_engine="native",
        dual_mode="mono",
        allowed_dirs=str(tmp_path),
    )
    assert cfg.dual_mode == "mono"

    codec = NativePDFCodec(layout_detector=HeuristicLayoutDetector())
    adapter = NativePDFAdapter(codec=codec)
    orch = create_dry_run_orchestrator(cfg, adapter=adapter)
    events = [ev async for ev in orch.run(src, out, target_lang="zh")]
    assert events
    assert out.exists()
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 2

    # Verify live single-page preview works straight from the adapter units
    from ubt.adapters.pdf.native_adapter import _unit_to_ir_block

    assert adapter.last_units
    blocks = [_unit_to_ir_block(u, idx) for idx, u in enumerate(adapter.last_units)]
    for b in blocks:
        b.target_text = f"中文预览 {b.source_text}"
        b.status = BlockStatus.MTQE_PASSED
    png = adapter.render_page_preview_sync(src, 1, blocks, "zh", dpi=96)
    assert png.startswith(b"\x89PNG")


@pytest.mark.fast
def test_tj_kerning_restores_word_spaces_without_splitting_optical_kerns() -> None:
    from ubt.codecs.pdf.stream_analyzer import analyze_page_stream

    pdf = pikepdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
            }
        )
    )
    page = pdf.add_blank_page(page_size=(595, 842))
    page.Resources = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({"/F1": font})})
    page.Contents = pdf.make_stream(
        b"BT /F1 12 Tf 1 0 0 1 72 720 Tm [(Kum) 27 (bong) -313 (Mo) -26 (deling:) -314 (Fast)] TJ ET"
    )
    analysis = analyze_page_stream(page, page_no=1)
    assert analysis.text_ops
    assert analysis.text_ops[0].text == "Kumbong Modeling: Fast"


@pytest.mark.fast
def test_placeholders_and_fast_pass_protect_inline_anchors_and_brace_emails() -> None:
    from ubt.core.ports import placeholder_engine

    eng = placeholder_engine()
    src = "See (Section <c1>4.6</c1>) and <f1/> plus {kumboh, iskhare}@stanford.edu."
    masked = eng.mask(src)
    assert "<c1>4.6</c1>" in masked.text
    assert "<f1/>" not in masked.text
    assert any(v == "<f1/>" for v in masked.math_map.values())
    assert any(v == "{kumboh, iskhare}@stanford.edu" for v in masked.email_map.values())
    restored = eng.restore(masked.text, masked)
    assert restored.clean
    assert restored.text == src

    # Verify LaTeX \llbracket...\rrbracket wrapping around mask tokens unwraps cleanly.
    math_tok = next(iter(masked.math_map))
    wrapped = masked.text.replace(math_tok, rf"$\llbracket {math_tok} \rrbracket$")
    unwrapped = eng.restore(wrapped, masked)
    assert unwrapped.clean
    assert unwrapped.text == src

    # Plain speedup multipliers like 4–11× stay unmasked prose.
    mult = eng.mask("NCM runs 4–11× faster and 21× cheaper than baselines.")
    assert mult.soup_map == {}

    flt = FastPassFilter()
    dec = flt.evaluate(
        "Correspondence to <c1>https://github.com/org/repo</c1> and {kumboh, iskhare}@stanford.edu.",
        "通信地址：<c1>https://github.com/org/repo</c1> 与 {kumboh, iskhare}@stanford.edu。",
    )
    assert dec.passed, dec.reason
    assert flt.evaluate("Ours", "我们的方法").passed
    assert flt.evaluate(
        "Gen. PPL: 14.1 |<f1/>: 3.88", "生成困惑度 (Gen. PPL): 14.1 |<f1/>: 3.88"
    ).passed
    assert flt.evaluate(
        "See Figure <c1>5</c1> and [<c2>1</c2>, <c3>2</c3>].",
        "参见图 <c1>5</c1> 与 [<c2>1</c2>, <c3>2</c3>]。",
    ).passed


@pytest.mark.fast
def test_kinsoku_never_starts_line_with_forbidden_cjk_punctuation() -> None:
    from ubt.codecs.pdf.font_engine import FontEngine
    from ubt.codecs.pdf.layout_fusion import ParagraphGeometry
    from ubt.codecs.pdf.paragraph_typesetter import _NO_LINE_START, typeset_paragraph
    from ubt.codecs.protocol import UnitRole

    fe = FontEngine("zh")
    text = (
        "扩散语言模型为比自回归模型更快的生成提供了并行采样，这是一个日益重要的优势。"
        "然而，随着采样步数的减少会迅速退化，在每一步都处理完整的序列，并且需要多次遍历。"
    )
    # Sweep several tight box widths to stress line-break boundaries near commas/periods.
    for box_w in (92.0, 95.0, 98.0, 102.0, 106.0, 110.0):
        geom = ParagraphGeometry(
            bbox=(50.0, 100.0, 50.0 + box_w, 300.0),
            dominant_font_size=10.0,
            line_height=13.0,
            first_line_indent=0.0,
            alignment="left",
            is_serif=True,
            is_bold=False,
            color_rgb=(0.0, 0.0, 0.0),
            max_borrow_down_pt=0.0,
        )
        res = typeset_paragraph(
            unit_id="u_kinsoku",
            page_no=1,
            role=UnitRole.PARAGRAPH,
            marked_text=text,
            tokens=(),
            geom=geom,
            formulas={},
            font_engine=fe,
            target_lang="zh",
        )
        # Group runs by baseline_y to inspect each rendered line's first character.
        by_y: dict[float, list[tuple[float, str]]] = {}
        for r in res.runs:
            if r.text:
                by_y.setdefault(round(r.baseline_y, 2), []).append((r.x, r.text))
        for items in by_y.values():
            items.sort(key=lambda pair: pair[0])
            first_char = items[0][1][0]
            assert first_char not in _NO_LINE_START, (
                f"box_w={box_w}: line started with forbidden punctuation {first_char!r}"
            )


@pytest.mark.fast
def test_font_engine_measure_matches_nominal_hmtx_without_kerning() -> None:
    import uharfbuzz as hb

    from ubt.codecs.pdf.font_engine import FontEngine, FontSlot

    fe = FontEngine("zh")
    face = fe._face_for(FontSlot.SERIF_REGULAR)
    word = "OpenWebText"
    measured = fe.measure_text_pt(word, FontSlot.SERIF_REGULAR, 10.0)
    buf = hb.Buffer()
    buf.add_str(word)
    buf.guess_segment_properties()
    hb.shape(face.hb_font, buf, {"kern": False, "liga": False})
    expected = sum(p.x_advance for p in buf.glyph_positions) * (10.0 / face.upem)
    assert measured == pytest.approx(expected, rel=1e-4)


@pytest.mark.fast
def test_layout_fusion_absorbs_tex_math_accent_into_formula_span() -> None:
    from ubt.codecs.pdf.layout_fusion import _detect_line_formula_flags
    from ubt.codecs.pdf.stream_analyzer import OpGraphicsSnapshot, StreamTextOp

    snap = OpGraphicsSnapshot(
        ctm=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        text_matrix=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        font_resource="/F1",
        font_size_operand=10.0,
        tc=0.0,
        tw=0.0,
        tz=100.0,
        ts=0.0,
        color_rgb=(0.0, 0.0, 0.0),
    )

    def _op(idx: int, txt: str, x0: float, x1: float, is_math: bool) -> StreamTextOp:
        return StreamTextOp(
            op_index=idx,
            operator="Tj",
            text=txt,
            bbox=(x0, 100.0, x1, 110.0),
            baseline_y=102.0,
            font_resource="/F1",
            font_name="CMMI10" if is_math else "CMR10",
            font_size_pt=10.0,
            color_rgb=(0.0, 0.0, 0.0),
            is_bold=False,
            is_italic=is_math,
            is_serif=True,
            is_math_font=is_math,
            has_tounicode=True,
            rise_pt=0.0,
            snapshot=snap,
            raw_operands=(),
        )

    ops = [
        _op(0, "reconstruction ", 50.0, 120.0, False),
        _op(1, "ˆ", 121.0, 125.0, False),
        _op(2, "z", 121.0, 126.0, True),
        _op(3, " of", 128.0, 145.0, False),
    ]
    flags = _detect_line_formula_flags(
        ops, line_base_y=102.0, body_size=10.0, inline_formula_regions=[]
    )
    assert flags == [False, True, True, False]
