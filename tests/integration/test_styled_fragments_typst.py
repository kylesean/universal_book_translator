from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_compile import typst_compile
from ubt.core.ir.models import BlockType, FlowID, InlineRun, IRBlock, StyleMeta, make_element
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.slow


def _char_colors(pdf_path: str) -> tuple[int, tuple[float, float] | None]:
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[0]
        try:
            textpage = page.get_textpage()
            try:
                blue = 0
                dagger: tuple[float, float] | None = None
                for index in range(textpage.count_chars()):
                    code = pdfium_c.FPDFText_GetUnicode(textpage, index)
                    red = ctypes.c_uint()
                    green = ctypes.c_uint()
                    blue_c = ctypes.c_uint()
                    alpha = ctypes.c_uint()
                    pdfium_c.FPDFText_GetFillColor(
                        textpage,
                        index,
                        ctypes.byref(red),
                        ctypes.byref(green),
                        ctypes.byref(blue_c),
                        ctypes.byref(alpha),
                    )
                    if (red.value, green.value, blue_c.value) == (0, 0, 255):
                        blue += 1
                    if code == 0x2021:
                        size = float(pdfium_c.FPDFText_GetFontSize(textpage, index))
                        left = ctypes.c_double()
                        right = ctypes.c_double()
                        bottom = ctypes.c_double()
                        top = ctypes.c_double()
                        pdfium_c.FPDFText_GetCharBox(
                            textpage,
                            index,
                            ctypes.byref(left),
                            ctypes.byref(right),
                            ctypes.byref(bottom),
                            ctypes.byref(top),
                        )
                        dagger = (size, top.value)
                return blue, dagger
            finally:
                textpage.close()
        finally:
            page.close()
    finally:
        pdf.close()


def _render(text: str, runs: tuple[InlineRun, ...], tmp_path: Path) -> str:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text=text,
        ),
        target_text=text,
        style=StyleMeta(inline_runs=runs),
    )
    renderer = TypstDocumentRenderer()
    source, _stats = renderer.build_source([block], tmp_path / "build", target_lang="zh")
    typ = tmp_path / "build" / "main.typ"
    typ.parent.mkdir(parents=True, exist_ok=True)
    typ.write_text(source, encoding="utf-8")
    pdf = tmp_path / "out.pdf"
    ok, err = typst_compile(str(typ), str(pdf))
    assert ok, f"the styled paragraph must compile:\n{err}\n{source}"
    return str(pdf)


def test_a_styled_run_draws_blue_text_and_a_raised_dagger(tmp_path: Path) -> None:
    text = "参见（Guo et al., 2025; Jimenez et al., 2024）。DeepSeek-AI ‡ 清华大学。"
    runs = (
        InlineRun(text="Guo et al.", color_hex="#0000ff"),
        InlineRun(text="2025", color_hex="#0000ff"),
        InlineRun(text="Jimenez et al.", color_hex="#0000ff"),
        InlineRun(text="2024", color_hex="#0000ff"),
        InlineRun(text="‡", superscript=True),
    )
    blue, dagger = _char_colors(_render(text, runs, tmp_path))

    assert blue > 0, "the citation runs should be drawn blue"
    assert dagger is not None, "the dagger should be present"
    size, _top = dagger
    assert size < 0.8 * 11.0


def test_a_bold_run_before_an_ascii_paren_still_compiles(tmp_path: Path) -> None:
    text = "Firecracker microVMs (Agache et al., 2020) provide a stronger isolation boundary."
    runs = (
        InlineRun(text="Firecracker microVMs ", bold=True),
        InlineRun(text="Agache et al.", color_hex="#0000ff"),
        InlineRun(text="2020", color_hex="#0000ff"),
    )
    blue, _ = _char_colors(_render(text, runs, tmp_path))

    assert blue > 0, "the citation runs should still be drawn blue"
