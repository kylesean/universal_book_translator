from __future__ import annotations

from pathlib import Path

import pytest

from ubt.render.page_trim import MM_PER_PT, TrimSize, resolved_page_setup
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast

_CUSTOM_TRIM = TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26)


def _pdf(path: Path, size_mm: tuple[float, float], pages: int = 3) -> Path:
    import pikepdf

    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        pdf.add_blank_page(
            page_size=(size_mm[0] / MM_PER_PT, size_mm[1] / MM_PER_PT),
        )
    pdf.save(path)
    pdf.close()
    return path


def _page_line(source: str) -> str:
    lines = [line for line in source.splitlines() if line.startswith("#set page(")]
    assert len(lines) == 1, f"expected exactly one page setup, got {lines}"
    return lines[0]


def test_the_book_template_emits_the_measured_trim(tmp_path: Path) -> None:
    from ubt.render.page_trim import measure_trim

    source = _pdf(tmp_path / "book.pdf", (190.5, 234.9))
    trim = measure_trim(source)
    assert trim is not None, "the fixture must measure, or this test proves nothing"

    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(trim))
    line = _page_line(renderer.preamble("zh"))

    assert "width: 190.5mm, height: 234.9mm" in line
    assert '"a4"' not in line
    assert "x: 25mm" in line


def test_a_source_with_no_pages_keeps_a4_and_says_so(tmp_path: Path) -> None:
    del tmp_path

    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(None))
    line = _page_line(renderer.preamble("zh"))

    assert 'paper: "a4"' in line
    assert "x: 32mm" in line, "the fallback margin must still hit the target measure"


def test_the_emitted_margin_keeps_the_line_length_in_convention(tmp_path: Path) -> None:
    from ubt.render.page_trim import chars_per_line, measure_trim

    source = _pdf(tmp_path / "book.pdf", (190.5, 234.9))
    trim = measure_trim(source)
    assert trim is not None
    setup = resolved_page_setup(trim)
    line = _page_line(TypstDocumentRenderer(page_setup=setup).preamble("zh"))

    for fragment in ("x: 25mm", "y: 24mm"):
        assert fragment in line
    assert chars_per_line(setup.book_measure_mm, 10.5) == pytest.approx(37.9, abs=0.1)


def test_the_academic_template_honours_the_page_size(tmp_path: Path) -> None:
    del tmp_path

    renderer = TypstDocumentRenderer(
        layout_template="academic",
        page_setup=resolved_page_setup(_CUSTOM_TRIM),
    )
    source = renderer.preamble("zh")

    assert "width: 190.5mm, height: 234.9mm" in source
    assert "margin: (x: 2.5cm, y: 2.5cm)" in source


def test_the_academic_template_keeps_its_submission_margins_on_a4() -> None:
    renderer = TypstDocumentRenderer(layout_template="academic")

    source = renderer.preamble("zh")

    assert 'paper: "a4"' in source
    assert "margin: (x: 2.5cm, y: 2.5cm)" in source
    assert "page_setup" not in source.lower()


def test_a_dual_column_academic_page_keeps_its_narrow_margin() -> None:
    renderer = TypstDocumentRenderer(
        layout_template="academic",
        columns=2,
        page_setup=resolved_page_setup(_CUSTOM_TRIM, columns=2),
    )

    source = renderer.preamble("zh")

    assert "margin: (x: 1.8cm, y: 2.0cm)" in source
    assert "columns: 2" in source


@pytest.mark.parametrize(
    ("trim", "columns", "template"),
    [
        (None, 1, "book"),
        (_CUSTOM_TRIM, 1, "book"),
        (TrimSize(215.9, 279.4, 40, 40), 1, "book"),
        (None, 1, "academic"),
        (_CUSTOM_TRIM, 1, "academic"),
        (None, 2, "academic"),
        (_CUSTOM_TRIM, 2, "academic"),
    ],
)
def test_the_audit_spec_matches_the_page_that_is_emitted(
    trim: TrimSize | None, columns: int, template: str
) -> None:
    renderer = TypstDocumentRenderer(
        layout_template=template,
        columns=columns,
        page_setup=resolved_page_setup(trim, columns=columns),
    )

    spec = renderer.layout_spec_for_audit()

    assert spec.page_width == pytest.approx(renderer.page_setup_obj().width_mm / MM_PER_PT)
    assert spec.page_height == pytest.approx(renderer.page_setup_obj().height_mm / MM_PER_PT)
    per_column = renderer.text_width_pt()
    if columns > 1:
        assert (spec.printable_width - spec.gutter) / columns == pytest.approx(per_column)
    else:
        assert spec.printable_width == pytest.approx(per_column)


def test_the_table_fit_verdict_uses_the_measured_text_block() -> None:
    narrow = TypstDocumentRenderer(page_setup=resolved_page_setup(_CUSTOM_TRIM))
    a4 = TypstDocumentRenderer(page_setup=resolved_page_setup(None))

    assert narrow._table_width_pt() < a4._table_width_pt()
    assert narrow._table_width_pt() == pytest.approx(398.27, abs=0.1)
    assert a4._table_width_pt() == pytest.approx(413.86, abs=0.1)
