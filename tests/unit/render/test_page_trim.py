"""Unit tests for the measured trim and the measure-driven margin.

The reflow channel used to emit ``paper: "a4"`` and a fixed ``x: 2.2cm``
margin regardless of the source. Both were wrong for most of this repository's
own corpus (19 of 25 documents are US Letter, 5 A4, 1 a 190.5mm trim), and the
margin was additionally too small for the text: 2.2cm on A4 sets Chinese body
text at 44.8 characters per line, past the 25-40 convention.

The property these tests protect is the *measure*, not the paper and not the
margin: any two of (paper, margin, measure) fix the third, so the readability
promise belongs on the one the eye actually reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.render.page_trim import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    COLUMN_GUTTER_SHARE,
    DEFAULT_PAGE_MARGIN_Y_MM,
    MIN_PAGE_MARGIN_MM,
    MM_PER_PT,
    TARGET_MEASURE_MM,
    TrimSize,
    body_measure_mm,
    chars_per_line,
    margin_for_measure,
    measure_trim,
    resolved_page_setup,
)

pytestmark = pytest.mark.fast

#: The three trims this repository's corpus actually contains, plus the reader's
#: expectation for each: a page whose source is this size must deliver the
#: source's own physical page, not A4.
_CORPUS_TRIMS = {
    "A4": (210.0, 297.0),
    "US Letter": (215.9, 279.4),
    "16-mo custom": (190.5, 234.9),
}

#: Chinese body size the reflow channel sets, and the convention it must stay in.
_BODY_PT = 10.5
_MIN_CHARS, _MAX_CHARS = 25.0, 40.0


def _pdf(path: Path, sizes: list[tuple[float, float]]) -> Path:
    """A PDF with one blank page per ``(width_pt, height_pt)``."""
    import pikepdf

    pdf = pikepdf.Pdf.new()
    for width, height in sizes:
        pdf.add_blank_page(page_size=(width, height))
    pdf.save(path)
    pdf.close()
    return path


def _pdf_mm(path: Path, size_mm: tuple[float, float], pages: int = 3) -> Path:
    """A PDF of ``pages`` blank pages, all at a size given in millimetres."""
    pt = (size_mm[0] / MM_PER_PT, size_mm[1] / MM_PER_PT)
    return _pdf(path, [pt] * pages)


# --- measurement ----------------------------------------------------------


@pytest.mark.parametrize(("name", "size_mm"), sorted(_CORPUS_TRIMS.items()))
def test_measure_trim_reads_the_page_the_source_declares(
    tmp_path: Path, name: str, size_mm: tuple[float, float]
) -> None:
    """The measured trim is the source's page, to printer resolution.

    A 0.5mm tolerance is not leniency: a PDF states its page in points, and
    210mm is 595.276pt, so the round trip through points and back accumulates
    below a tenth of a millimetre. Anything larger would mean the wrong box was
    read.
    """
    path = _pdf_mm(tmp_path / f"{name}.pdf", size_mm)

    trim = measure_trim(path)

    assert trim is not None
    assert trim.width_mm == pytest.approx(size_mm[0], abs=0.5)
    assert trim.height_mm == pytest.approx(size_mm[1], abs=0.5)
    assert trim.pages_measured == 3
    assert trim.agreement == pytest.approx(1.0)


@pytest.mark.parametrize("suffix", [".epub", ".md", ".docx"])
def test_a_source_with_no_pages_reports_no_trim(tmp_path: Path, suffix: str) -> None:
    """``None`` is the honest answer, not "assume A4 here".

    A source that carries no physical pages cannot declare a trim, and this
    function must not invent one: the A4 fallback is a product decision and it
    belongs at the call site, where a reader can see it was a fallback.
    """
    path = tmp_path / f"book{suffix}"
    path.write_bytes(b"not really a book")

    assert measure_trim(path) is None


def test_a_missing_file_reports_no_trim(tmp_path: Path) -> None:
    """A deleted source must not fail the render this late in the pipeline."""
    assert measure_trim(tmp_path / "gone.pdf") is None
    assert measure_trim(None) is None
    assert measure_trim("") is None


def test_rotation_is_applied_so_the_reader_gets_an_upright_page(tmp_path: Path) -> None:
    """A landscape scan stored as a rotated portrait box is portrait to the eye.

    Honouring the unrotated MediaBox would deliver a book lying on its side,
    which is the exact failure a facsimile channel exists to prevent.
    """
    import pikepdf

    path = tmp_path / "rotated.pdf"
    pdf = pikepdf.Pdf.new()
    # A4 MediaBox, rotated 90: the reader sees a 297x210 landscape page.
    page = pdf.add_blank_page(page_size=(A4_WIDTH_MM / MM_PER_PT, A4_HEIGHT_MM / MM_PER_PT))
    page.Rotate = 90
    pdf.save(path)
    pdf.close()

    trim = measure_trim(path)

    assert trim is not None
    assert trim.width_mm == pytest.approx(A4_HEIGHT_MM, abs=0.5)
    assert trim.height_mm == pytest.approx(A4_WIDTH_MM, abs=0.5)


def test_an_implausible_page_is_not_a_book_trim(tmp_path: Path) -> None:
    """A 40mm page is a mis-read box or a fold-out, not something to lay out to.

    Propagating it would produce a page no printer accepts and a measure of a
    few characters, so it is rejected at measurement rather than downstream.
    """
    path = _pdf(tmp_path / "stamp.pdf", [(40.0 / MM_PER_PT, 40.0 / MM_PER_PT)])

    assert measure_trim(path) is None


def test_a_document_with_no_majority_page_size_has_no_trim(tmp_path: Path) -> None:
    """A pasted-together collection has no single trim to honour.

    Laying it out at the *plurality* page's size would report a measurement
    that was not a consensus. Four distinct sizes over six pages: no size
    reaches the 60% needed, so the answer is ``None`` and the caller falls back.
    """
    path = _pdf(
        tmp_path / "mixed.pdf",
        [
            (595.276, 841.89),  # A4
            (595.276, 841.89),
            (612.0, 792.0),  # Letter
            (612.0, 792.0),
            (540.0, 666.0),
            (500.0, 700.0),
        ],
    )

    assert measure_trim(path) is None


def test_a_few_fold_outs_do_not_stop_a_book_from_declaring_its_trim(
    tmp_path: Path,
) -> None:
    """A book may carry a differently-sized fold-out or cover.

    The plurality still describes the book, and refusing to measure because one
    page differs would fall back to A4 for a book that plainly declared itself.
    """
    a4 = (A4_WIDTH_MM / MM_PER_PT, A4_HEIGHT_MM / MM_PER_PT)
    path = _pdf(tmp_path / "foldout.pdf", [a4] * 8 + [(1190.55, 841.89)])

    trim = measure_trim(path)

    assert trim is not None
    assert trim.width_mm == pytest.approx(A4_WIDTH_MM, abs=0.5)
    assert trim.pages_measured == 9
    assert trim.pages_agreeing == 8


# --- the readability promise ----------------------------------------------


@pytest.mark.parametrize(("name", "size_mm"), sorted(_CORPUS_TRIMS.items()))
def test_every_corpus_trim_lands_inside_the_character_convention(
    name: str, size_mm: tuple[float, float]
) -> None:
    """The promise, stated once for every page size this corpus contains.

    25-40 Chinese characters per line is the convention; past 40 the eye's
    return sweep starts skipping lines. This is the assertion that makes the
    margin derivable rather than arbitrary: whatever the paper, the line length
    is what is held fixed.
    """
    del name
    trim = TrimSize(size_mm[0], size_mm[1], pages_measured=1, pages_agreeing=1)

    setup = resolved_page_setup(trim)
    per_line = chars_per_line(setup.book_measure_mm, _BODY_PT)

    assert _MIN_CHARS <= per_line <= _MAX_CHARS, f"{per_line:.1f} chars on {size_mm}"


def test_the_previous_a4_margin_was_over_budget() -> None:
    """Why this change exists, asserted rather than described.

    The old ``x: 2.2cm`` on A4 gave a 166mm measure, which sets 44.8 characters
    at the body size — over the convention. If this assertion ever fails, the
    arithmetic behind the whole refactor has drifted.
    """
    old_measure = body_measure_mm(A4_WIDTH_MM, margin_mm=22.0)

    assert old_measure == pytest.approx(166.0)
    assert chars_per_line(old_measure, _BODY_PT) > _MAX_CHARS


def test_the_margin_is_derived_from_the_measure_not_the_reverse() -> None:
    """The same target measure needs a different margin on a different paper.

    This is the defect a fixed margin cannot express: 3.2cm on A4 gives 39.4
    characters per line but 41.0 on US Letter — inside the convention on one
    paper and outside it on the other, from one setting.
    """
    on_a4 = margin_for_measure(A4_WIDTH_MM)
    on_letter = margin_for_measure(_CORPUS_TRIMS["US Letter"][0])

    assert on_a4 == pytest.approx(32.0)
    assert on_letter == pytest.approx(34.95, abs=0.01)
    assert on_letter > on_a4, "the wider paper needs the wider margin"
    # And both land on the same measure, which is the point.
    assert body_measure_mm(A4_WIDTH_MM, margin_mm=on_a4) == pytest.approx(TARGET_MEASURE_MM)
    assert body_measure_mm(_CORPUS_TRIMS["US Letter"][0], margin_mm=on_letter) == pytest.approx(
        TARGET_MEASURE_MM
    )


def test_a_page_too_narrow_for_the_measure_keeps_a_printable_margin() -> None:
    """A 130mm pocket book cannot have 146mm of text, and must not try.

    Without the clamp the derived margin would be negative — text running off
    the trim, which is worse than a short line. The reader gets a narrower
    measure and a page that prints.
    """
    assert margin_for_measure(130.0) == MIN_PAGE_MARGIN_MM
    measure = body_measure_mm(130.0, margin_mm=margin_for_measure(130.0))
    assert measure == pytest.approx(80.0)
    assert measure < TARGET_MEASURE_MM


def test_a_multi_column_measure_accounts_for_the_gutter() -> None:
    """Two columns halve the measure, and Typst's gutter is 4% of the block.

    Measured, not assumed: ``#context layout(size => ...)`` reports 236.75pt per
    column inside a 493.24pt block, which is a 3.96% gutter. A fixed 6mm — what
    the table-fit arithmetic used — is 0.03 of the same block, so every
    two-column table was sized ~3pt too wide per column.
    """
    block = body_measure_mm(A4_WIDTH_MM, margin_mm=0.0, columns=1)
    column = body_measure_mm(A4_WIDTH_MM, margin_mm=0.0, columns=2)

    assert column == pytest.approx(block * (1 - COLUMN_GUTTER_SHARE) / 2)
    assert block - 2 * column == pytest.approx(block * COLUMN_GUTTER_SHARE)


# --- what the caller emits ------------------------------------------------


def test_no_measured_trim_falls_back_to_a4_without_pretending_to_measure() -> None:
    """The fallback is A4 at the target measure, and it says it is a fallback.

    ``paper is None`` is the tell: the caller emits ``paper: "a4"`` because it
    chose to, not because a measurement said so.
    """
    setup = resolved_page_setup(None)

    assert setup.paper is None
    assert not setup.measured
    assert setup.width_mm == A4_WIDTH_MM
    assert setup.height_mm == A4_HEIGHT_MM
    assert setup.margin_x_mm == pytest.approx(32.0)
    assert setup.margin_y_mm == DEFAULT_PAGE_MARGIN_Y_MM
    assert chars_per_line(setup.book_measure_mm, _BODY_PT) == pytest.approx(39.4, abs=0.1)


def test_a_measured_trim_is_emitted_as_a_size_not_a_paper_name() -> None:
    """A measured trim is usually not a named paper, so it is emitted in mm.

    ``paper:`` takes a name from a closed list — Typst rejects a size dictionary
    with ``expected "a4", ... found dictionary``. So the size is passed as
    ``width:``/``height:`` instead, which is the only form that compiles for a
    trim Typst has never heard of. (This test asserted the dictionary form
    first, and it was wrong: an assertion on the *shape* of markup proves
    nothing until the markup compiles.)
    """
    setup = resolved_page_setup(TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26))

    assert setup.measured
    assert setup.typst_page_args() == "width: 190.5mm, height: 234.9mm"
    assert setup.margin_x_mm == MIN_PAGE_MARGIN_MM


def test_the_a4_fallback_emits_a_paper_name() -> None:
    """A4 *is* a named paper, and naming it is shorter and clearer than sizing it."""
    assert resolved_page_setup(None).typst_page_args() == 'paper: "a4"'


def test_the_setup_carries_the_geometry_the_audit_will_judge() -> None:
    """One object, because the page and the audit of that page must agree.

    The layout audit declared A4's 2.2cm margins while the page was emitted at
    190.5mm with 25mm margins — a safe area ~28pt wider per side than the paper
    had. A real spill read as clean, and a correct page would "heal" by
    shrinking.
    """
    trim = TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26)
    setup = resolved_page_setup(trim)

    assert setup.width_mm == 190.5
    assert setup.height_mm == 234.9
    assert setup.width_mm - 2 * setup.margin_x_mm == pytest.approx(setup.book_measure_mm)
