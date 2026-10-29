"""The measured trim must reach the emitted Typst source, not just this module.

``test_page_trim.py`` proves the measurement and the arithmetic. That is not
enough: the previous defect was never in the arithmetic, it was that
``academic_preamble`` had no page-size parameter at all, so a measured value had
nowhere to go. These tests drive the renderer and read the markup it produces.

The assertions are on the *emitted source* rather than on a renderer attribute
because an attribute can be set and never used — which is exactly how a
``page_paper`` field would have looked correct while every page stayed A4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.render.page_trim import MM_PER_PT, TrimSize, resolved_page_setup
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast

#: A 16-mo trade trim from this repository's corpus: the page size that A4 was
#: silently substituting for.
_CUSTOM_TRIM = TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26)


def _pdf(path: Path, size_mm: tuple[float, float], pages: int = 3) -> Path:
    """A PDF of blank pages at ``size_mm``, for ``measure_trim`` to read."""
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
    """The single ``#set page(...)`` line of an emitted document."""
    lines = [line for line in source.splitlines() if line.startswith("#set page(")]
    assert len(lines) == 1, f"expected exactly one page setup, got {lines}"
    return lines[0]


# --- the book channel -----------------------------------------------------


def test_the_book_template_emits_the_measured_trim(tmp_path: Path) -> None:
    """A 190.5x234.9mm source must be delivered at 190.5x234.9mm.

    Laid out at A4 it would gain 39% page area and read like a different
    edition, which is what "isomorphic delivery" is supposed to mean.
    """
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
    """An EPUB has no pages to measure, so A4 is a choice — and it is visible."""
    del tmp_path

    renderer = TypstDocumentRenderer(page_setup=resolved_page_setup(None))
    line = _page_line(renderer.preamble("zh"))

    assert 'paper: "a4"' in line
    assert "x: 32mm" in line, "the fallback margin must still hit the target measure"


def test_the_emitted_margin_keeps_the_line_length_in_convention(tmp_path: Path) -> None:
    """The point of the whole change: whatever the paper, ~39 characters.

    Asserted through the *emitted* numbers rather than the setup object, so a
    template that reformats or rounds the margin is caught here.
    """
    from ubt.render.page_trim import chars_per_line, measure_trim

    source = _pdf(tmp_path / "book.pdf", (190.5, 234.9))
    trim = measure_trim(source)
    assert trim is not None
    setup = resolved_page_setup(trim)
    line = _page_line(TypstDocumentRenderer(page_setup=setup).preamble("zh"))

    for fragment in ("x: 25mm", "y: 24mm"):
        assert fragment in line
    # 190.5 - 2*25 = 140.5mm of text at 10.5pt.
    assert chars_per_line(setup.book_measure_mm, 10.5) == pytest.approx(37.9, abs=0.1)


# --- the academic channel -------------------------------------------------


def test_the_academic_template_honours_the_page_size(tmp_path: Path) -> None:
    """A paper printed at 190.5mm keeps that page, because it is the reader's copy.

    The *margins* stay the venue's: a submission template's 2.5cm is a
    requirement, not a trade-book readability decision, and imposing a 146mm
    measure on an author's thesis would be answering a question nobody asked.
    """
    del tmp_path

    renderer = TypstDocumentRenderer(
        layout_template="academic",
        page_setup=resolved_page_setup(_CUSTOM_TRIM),
    )
    source = renderer.preamble("zh")

    assert "width: 190.5mm, height: 234.9mm" in source
    assert "margin: (x: 2.5cm, y: 2.5cm)" in source


def test_the_academic_template_keeps_its_submission_margins_on_a4() -> None:
    """No measured trim must not change what the academic template emits today."""
    renderer = TypstDocumentRenderer(layout_template="academic")

    source = renderer.preamble("zh")

    assert 'paper: "a4"' in source
    assert "margin: (x: 2.5cm, y: 2.5cm)" in source
    assert "page_setup" not in source.lower()


def test_a_dual_column_academic_page_keeps_its_narrow_margin() -> None:
    """Two columns need the room; 1.8cm is the submission convention."""
    renderer = TypstDocumentRenderer(
        layout_template="academic",
        columns=2,
        page_setup=resolved_page_setup(_CUSTOM_TRIM, columns=2),
    )

    source = renderer.preamble("zh")

    assert "margin: (x: 1.8cm, y: 2.0cm)" in source
    assert "columns: 2" in source


# --- the audit must judge the page that was emitted ------------------------


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
    """The audit's safe area must be the page's, or a real spill reads as clean.

    This is the coupling that made the first draft of this change dangerous:
    ``PageLayoutSpec.book()`` declares A4 at 2.2cm, so healing a 190.5mm page
    against it would either miss a spill or shrink a correct page. The spec is
    therefore built from the emitted geometry, and this asserts they agree.
    """
    renderer = TypstDocumentRenderer(
        layout_template=template,
        columns=columns,
        page_setup=resolved_page_setup(trim, columns=columns),
    )

    spec = renderer.layout_spec_for_audit()

    assert spec.page_width == pytest.approx(renderer.page_setup_obj().width_mm / MM_PER_PT)
    assert spec.page_height == pytest.approx(renderer.page_setup_obj().height_mm / MM_PER_PT)
    # The spec's printable width is the text block the template actually sets.
    per_column = renderer.text_width_pt()
    if columns > 1:
        assert (spec.printable_width - spec.gutter) / columns == pytest.approx(per_column)
    else:
        assert spec.printable_width == pytest.approx(per_column)


def test_the_table_fit_verdict_uses_the_measured_text_block() -> None:
    """Tables are sized against the block they land in, not A4's.

    ``_PAGE_TEXT_WIDTH_PT`` describes A4 at the academic margins; a 190.5mm book
    has a narrower block, so reading the constant there would size a table for a
    page it is not on and put the overflow in the artifact.
    """
    narrow = TypstDocumentRenderer(page_setup=resolved_page_setup(_CUSTOM_TRIM))
    a4 = TypstDocumentRenderer(page_setup=resolved_page_setup(None))

    assert narrow._table_width_pt() < a4._table_width_pt()
    assert narrow._table_width_pt() == pytest.approx(398.27, abs=0.1)
    assert a4._table_width_pt() == pytest.approx(413.86, abs=0.1)
