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

_CORPUS_TRIMS = {
    "A4": (210.0, 297.0),
    "US Letter": (215.9, 279.4),
    "16-mo custom": (190.5, 234.9),
}

_BODY_PT = 10.5
_MIN_CHARS, _MAX_CHARS = 25.0, 40.0


def _pdf(path: Path, sizes: list[tuple[float, float]]) -> Path:
    import pikepdf

    pdf = pikepdf.Pdf.new()
    for width, height in sizes:
        pdf.add_blank_page(page_size=(width, height))
    pdf.save(path)
    pdf.close()
    return path


def _pdf_mm(path: Path, size_mm: tuple[float, float], pages: int = 3) -> Path:
    pt = (size_mm[0] / MM_PER_PT, size_mm[1] / MM_PER_PT)
    return _pdf(path, [pt] * pages)


@pytest.mark.parametrize(("name", "size_mm"), sorted(_CORPUS_TRIMS.items()))
def test_measure_trim_reads_the_page_the_source_declares(
    tmp_path: Path, name: str, size_mm: tuple[float, float]
) -> None:
    path = _pdf_mm(tmp_path / f"{name}.pdf", size_mm)

    trim = measure_trim(path)

    assert trim is not None
    assert trim.width_mm == pytest.approx(size_mm[0], abs=0.5)
    assert trim.height_mm == pytest.approx(size_mm[1], abs=0.5)
    assert trim.pages_measured == 3
    assert trim.agreement == pytest.approx(1.0)


@pytest.mark.parametrize("suffix", [".epub", ".md", ".docx"])
def test_a_source_with_no_pages_reports_no_trim(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"book{suffix}"
    path.write_bytes(b"not really a book")

    assert measure_trim(path) is None


def test_a_missing_file_reports_no_trim(tmp_path: Path) -> None:
    assert measure_trim(tmp_path / "gone.pdf") is None
    assert measure_trim(None) is None
    assert measure_trim("") is None


def test_rotation_is_applied_so_the_reader_gets_an_upright_page(tmp_path: Path) -> None:
    import pikepdf

    path = tmp_path / "rotated.pdf"
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(A4_WIDTH_MM / MM_PER_PT, A4_HEIGHT_MM / MM_PER_PT))
    page.Rotate = 90
    pdf.save(path)
    pdf.close()

    trim = measure_trim(path)

    assert trim is not None
    assert trim.width_mm == pytest.approx(A4_HEIGHT_MM, abs=0.5)
    assert trim.height_mm == pytest.approx(A4_WIDTH_MM, abs=0.5)


def test_an_implausible_page_is_not_a_book_trim(tmp_path: Path) -> None:
    path = _pdf(tmp_path / "stamp.pdf", [(40.0 / MM_PER_PT, 40.0 / MM_PER_PT)])

    assert measure_trim(path) is None


def test_a_document_with_no_majority_page_size_has_no_trim(tmp_path: Path) -> None:
    path = _pdf(
        tmp_path / "mixed.pdf",
        [
            (595.276, 841.89),
            (595.276, 841.89),
            (612.0, 792.0),
            (612.0, 792.0),
            (540.0, 666.0),
            (500.0, 700.0),
        ],
    )

    assert measure_trim(path) is None


def test_a_few_fold_outs_do_not_stop_a_book_from_declaring_its_trim(
    tmp_path: Path,
) -> None:
    a4 = (A4_WIDTH_MM / MM_PER_PT, A4_HEIGHT_MM / MM_PER_PT)
    path = _pdf(tmp_path / "foldout.pdf", [a4] * 8 + [(1190.55, 841.89)])

    trim = measure_trim(path)

    assert trim is not None
    assert trim.width_mm == pytest.approx(A4_WIDTH_MM, abs=0.5)
    assert trim.pages_measured == 9
    assert trim.pages_agreeing == 8


@pytest.mark.parametrize(("name", "size_mm"), sorted(_CORPUS_TRIMS.items()))
def test_every_corpus_trim_lands_inside_the_character_convention(
    name: str, size_mm: tuple[float, float]
) -> None:
    del name
    trim = TrimSize(size_mm[0], size_mm[1], pages_measured=1, pages_agreeing=1)

    setup = resolved_page_setup(trim)
    per_line = chars_per_line(setup.book_measure_mm, _BODY_PT)

    assert _MIN_CHARS <= per_line <= _MAX_CHARS, f"{per_line:.1f} chars on {size_mm}"


def test_the_previous_a4_margin_was_over_budget() -> None:
    old_measure = body_measure_mm(A4_WIDTH_MM, margin_mm=22.0)

    assert old_measure == pytest.approx(166.0)
    assert chars_per_line(old_measure, _BODY_PT) > _MAX_CHARS


def test_the_margin_is_derived_from_the_measure_not_the_reverse() -> None:
    on_a4 = margin_for_measure(A4_WIDTH_MM)
    on_letter = margin_for_measure(_CORPUS_TRIMS["US Letter"][0])

    assert on_a4 == pytest.approx(32.0)
    assert on_letter == pytest.approx(34.95, abs=0.01)
    assert on_letter > on_a4, "the wider paper needs the wider margin"
    assert body_measure_mm(A4_WIDTH_MM, margin_mm=on_a4) == pytest.approx(TARGET_MEASURE_MM)
    assert body_measure_mm(_CORPUS_TRIMS["US Letter"][0], margin_mm=on_letter) == pytest.approx(
        TARGET_MEASURE_MM
    )


def test_a_page_too_narrow_for_the_measure_keeps_a_printable_margin() -> None:
    assert margin_for_measure(130.0) == MIN_PAGE_MARGIN_MM
    measure = body_measure_mm(130.0, margin_mm=margin_for_measure(130.0))
    assert measure == pytest.approx(80.0)
    assert measure < TARGET_MEASURE_MM


def test_a_multi_column_measure_accounts_for_the_gutter() -> None:
    block = body_measure_mm(A4_WIDTH_MM, margin_mm=0.0, columns=1)
    column = body_measure_mm(A4_WIDTH_MM, margin_mm=0.0, columns=2)

    assert column == pytest.approx(block * (1 - COLUMN_GUTTER_SHARE) / 2)
    assert block - 2 * column == pytest.approx(block * COLUMN_GUTTER_SHARE)


def test_no_measured_trim_falls_back_to_a4_without_pretending_to_measure() -> None:
    setup = resolved_page_setup(None)

    assert setup.paper is None
    assert not setup.measured
    assert setup.width_mm == A4_WIDTH_MM
    assert setup.height_mm == A4_HEIGHT_MM
    assert setup.margin_x_mm == pytest.approx(32.0)
    assert setup.margin_y_mm == DEFAULT_PAGE_MARGIN_Y_MM
    assert chars_per_line(setup.book_measure_mm, _BODY_PT) == pytest.approx(39.4, abs=0.1)


def test_a_measured_trim_is_emitted_as_a_size_not_a_paper_name() -> None:
    setup = resolved_page_setup(TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26))

    assert setup.measured
    assert setup.typst_page_args() == "width: 190.5mm, height: 234.9mm"
    assert setup.margin_x_mm == MIN_PAGE_MARGIN_MM


def test_the_a4_fallback_emits_a_paper_name() -> None:
    assert resolved_page_setup(None).typst_page_args() == 'paper: "a4"'


def test_the_setup_carries_the_geometry_the_audit_will_judge() -> None:
    trim = TrimSize(190.5, 234.9, pages_measured=26, pages_agreeing=26)
    setup = resolved_page_setup(trim)

    assert setup.width_mm == 190.5
    assert setup.height_mm == 234.9
    assert setup.width_mm - 2 * setup.margin_x_mm == pytest.approx(setup.book_measure_mm)
