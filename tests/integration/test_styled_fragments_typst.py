"""Real Typst: a styled fragment draws a blue run and a raised dagger."""

from __future__ import annotations

import ctypes

import pytest

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
                    if code == 0x2021:  # ‡
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


def test_a_styled_fragment_draws_blue_text_and_a_raised_dagger() -> None:
    from ubt.render.outputs import StyledRun, TypstFragmentTypesetter

    typesetter = TypstFragmentTypesetter(
        font=None, size_pt=11.0, target_lang="zh", cache_dir=":temp:"
    )
    try:
        text = "参见（Guo et al., 2025; Jimenez et al., 2024）。DeepSeek-AI ‡ 清华大学。"
        runs = (
            StyledRun("Guo et al.", color_hex="#0000ff"),
            StyledRun("2025", color_hex="#0000ff"),
            StyledRun("Jimenez et al.", color_hex="#0000ff"),
            StyledRun("2024", color_hex="#0000ff"),
            StyledRun("‡", superscript=True),
        )
        fragment = typesetter.typeset_fixed(text, 480.0, 60.0, 11.0, runs=runs)
        assert fragment is not None
        blue, dagger = _char_colors(str(fragment))
    finally:
        typesetter.close()

    assert blue > 0, "the citation runs should be drawn blue"
    assert dagger is not None, "the dagger should be present"
    size, _top = dagger
    # Superscript shrinks the glyph well below the 11pt body size.
    assert size < 0.8 * 11.0


def test_a_bold_run_before_an_ascii_paren_still_compiles() -> None:
    """Regression: ``#strong[...]`` right before a literal ``(`` must compile.

    A styled run ends in ``]``; Typst read the citation's opening ``(`` as a
    further call on the run's result, so the fragment failed to compile and the
    block descended to the source page (arXiv 2609.22978 b0039, a
    ``render:no fragment`` skip). The bold run deliberately keeps its trailing
    space so the following segment starts with the bare paren.
    """
    from ubt.render.outputs import StyledRun, TypstFragmentTypesetter

    typesetter = TypstFragmentTypesetter(
        font=None, size_pt=11.0, target_lang="zh", cache_dir=":temp:"
    )
    try:
        text = "Firecracker microVMs (Agache et al., 2020) provide a stronger isolation boundary."
        runs = (
            StyledRun("Firecracker microVMs ", bold=True),
            StyledRun("Agache et al.", color_hex="#0000ff"),
            StyledRun("2020", color_hex="#0000ff"),
        )
        fragment = typesetter.typeset_fixed(text, 480.0, 60.0, 11.0, runs=runs)
        assert fragment is not None, "the fragment must compile, not descend to source"
        blue, _ = _char_colors(str(fragment))
    finally:
        typesetter.close()

    assert blue > 0, "the citation runs should still be drawn blue"
