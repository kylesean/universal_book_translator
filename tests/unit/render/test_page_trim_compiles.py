from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_compile import resolve_typst_binary
from ubt.render.page_trim import MM_PER_PT, TrimSize, resolved_page_setup
from ubt.render.typst.document_renderer import TypstDocumentRenderer

_CUSTOM_TRIM = TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(resolve_typst_binary("typst") is None, reason="typst binary not available"),
]


def _compile(source: str, tmp_path: Path) -> Path:
    from ubt.adapters.pdf.typst_compile import typst_compile

    typ = tmp_path / "main.typ"
    typ.write_text(source, encoding="utf-8")
    out = tmp_path / "main.pdf"
    ok, stderr = typst_compile(str(typ), str(out))
    assert ok, f"the emitted markup must compile:\n{stderr}"
    return out


def _page_mm(pdf_path: Path) -> tuple[float, float]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    size = (page.get_width() * MM_PER_PT, page.get_height() * MM_PER_PT)
    doc.close()
    return size


def test_the_book_channel_delivers_the_measured_trim(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(_CUSTOM_TRIM))

    pdf = _compile(renderer.preamble("zh") + "\n#lorem(120)\n", tmp_path)
    width_mm, height_mm = _page_mm(pdf)

    assert width_mm == pytest.approx(190.5, abs=0.2)
    assert height_mm == pytest.approx(234.9, abs=0.2)


def test_the_academic_channel_delivers_the_measured_trim(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(
        layout_template="academic",
        page_setup=resolved_page_setup(_CUSTOM_TRIM),
    )

    pdf = _compile(renderer.preamble("zh") + "\n#lorem(120)\n", tmp_path)
    width_mm, height_mm = _page_mm(pdf)

    assert width_mm == pytest.approx(190.5, abs=0.2)
    assert height_mm == pytest.approx(234.9, abs=0.2)


def test_the_fallback_is_a4(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(None))

    pdf = _compile(renderer.preamble("zh") + "\n#lorem(120)\n", tmp_path)
    width_mm, height_mm = _page_mm(pdf)

    assert width_mm == pytest.approx(210.0, abs=0.2)
    assert height_mm == pytest.approx(297.0, abs=0.2)


def test_the_body_text_sets_inside_the_emitted_margins(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(_CUSTOM_TRIM))
    source = renderer.preamble("zh") + "\n#lorem(300)\n"

    pdf = _compile(source, tmp_path)

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    page = doc[0]
    textpage = page.get_textpage()
    lefts = []
    for index in range(textpage.count_chars()):
        left, _bottom, right, _top = textpage.get_charbox(index)
        if right - left > 0.05:
            lefts.append(left)
    doc.close()

    assert lefts, "the compiled page must contain text to measure"
    left_mm = min(lefts) * MM_PER_PT
    assert left_mm == pytest.approx(25.0, abs=2.0), f"text starts at {left_mm:.1f}mm"
