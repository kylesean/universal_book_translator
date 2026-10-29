"""The measured trim must compile, not merely look right in a string.

This file exists because of a defect that got through: the first version of the
trim support emitted ``paper: (width: 190.5mm, height: 234.9mm)``, and the unit
test asserted that exact string. It passed. Typst rejected it —
``expected "a4", ... found dictionary`` — because ``paper:`` takes a *name* from
a closed list. An assertion on the shape of markup proves nothing until the
markup compiles, so these tests run the compiler and read the page geometry out
of the artifact.

Marked ``slow``: each test spawns ``typst``, which is exactly what the ``fast``
tier is supposed to exclude.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_compile import resolve_typst_binary
from ubt.render.page_trim import MM_PER_PT, TrimSize, resolved_page_setup
from ubt.render.typst.document_renderer import TypstDocumentRenderer

#: A 16-mo trade trim from this repository's corpus.
_CUSTOM_TRIM = TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(resolve_typst_binary("typst") is None, reason="typst binary not available"),
]


def _compile(source: str, tmp_path: Path) -> Path:
    """Compile one Typst source; return the PDF."""
    from ubt.adapters.pdf.typst_compile import typst_compile

    typ = tmp_path / "main.typ"
    typ.write_text(source, encoding="utf-8")
    out = tmp_path / "main.pdf"
    ok, stderr = typst_compile(str(typ), str(out))
    assert ok, f"the emitted markup must compile:\n{stderr}"
    return out


def _page_mm(pdf_path: Path) -> tuple[float, float]:
    """First page size in millimetres, read from the compiled artifact."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    size = (page.get_width() * MM_PER_PT, page.get_height() * MM_PER_PT)
    doc.close()
    return size


def test_the_book_channel_delivers_the_measured_trim(tmp_path: Path) -> None:
    """The page that comes out is the page that went in.

    Asserted on the compiled PDF rather than on the source string: the string
    was what the broken version got right.
    """
    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(_CUSTOM_TRIM))

    pdf = _compile(renderer.preamble("zh") + "\n#lorem(120)\n", tmp_path)
    width_mm, height_mm = _page_mm(pdf)

    assert width_mm == pytest.approx(190.5, abs=0.2)
    assert height_mm == pytest.approx(234.9, abs=0.2)


def test_the_academic_channel_delivers_the_measured_trim(tmp_path: Path) -> None:
    """A paper printed at 190.5mm keeps that page on the academic template too."""
    renderer = TypstDocumentRenderer(
        layout_template="academic",
        page_setup=resolved_page_setup(_CUSTOM_TRIM),
    )

    pdf = _compile(renderer.preamble("zh") + "\n#lorem(120)\n", tmp_path)
    width_mm, height_mm = _page_mm(pdf)

    assert width_mm == pytest.approx(190.5, abs=0.2)
    assert height_mm == pytest.approx(234.9, abs=0.2)


def test_the_fallback_is_a4(tmp_path: Path) -> None:
    """No measured trim means A4, and the compiled artifact must say so."""
    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(None))

    pdf = _compile(renderer.preamble("zh") + "\n#lorem(120)\n", tmp_path)
    width_mm, height_mm = _page_mm(pdf)

    assert width_mm == pytest.approx(210.0, abs=0.2)
    assert height_mm == pytest.approx(297.0, abs=0.2)


def test_the_body_text_sets_inside_the_emitted_margins(tmp_path: Path) -> None:
    """The claim the margin exists for, checked against real ink on the page.

    A margin is only honoured if the text stops at it. Reading the actual glyph
    boxes is the difference between "we emitted x: 25mm" and "the page is set to
    a 25mm margin".
    """
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
    # 25mm margin, less Typst's glyph sidebearing. The point is that it is a
    # 25mm margin and not the 32mm an A4 target would have produced.
    assert left_mm == pytest.approx(25.0, abs=2.0), f"text starts at {left_mm:.1f}mm"
