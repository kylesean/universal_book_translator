"""Link annotations are relocated onto the overlay's translated glyphs."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf.link_annotations import (
    _extract_candidates_from_annot,
    _rect_intersects_any,
    _standalone_number,
    relocate_page_annotations,
)

pytestmark = pytest.mark.fast


def test_rect_intersects_any() -> None:
    rect = [10.0, 10.0, 20.0, 20.0]
    # Completely outside
    assert not _rect_intersects_any(rect, [(30.0, 30.0, 40.0, 40.0)])
    # Overlapping
    assert _rect_intersects_any(rect, [(15.0, 15.0, 25.0, 25.0)])
    # Enclosed
    assert _rect_intersects_any(rect, [(5.0, 5.0, 25.0, 25.0)])


def test_extract_candidates_from_annot() -> None:
    # Bracket citation
    annot = pikepdf.Dictionary(
        {
            "/Subtype": "/Link",
            "/A": pikepdf.Dictionary({"/S": "/GoTo", "/D": "cite.author2025"}),
        }
    )
    candidates = _extract_candidates_from_annot(annot, "[12]")
    assert "[12]" in candidates
    assert "12" in candidates
    assert "author" in candidates
    assert "2025" in candidates


def test_extract_candidates_keeps_single_digit_fragment() -> None:
    # A multi-citation bracket ("[1,2,3,4]") is stored as one annotation per
    # number, so the text under each rect is a fragment ("[1,", "2,", "4]").
    # These must yield a usable token; previously the digit was dropped and only
    # the (untranslated) author/year remained, leaving single-digit links dead.
    annot = pikepdf.Dictionary(
        {
            "/Subtype": "/Link",
            "/A": pikepdf.Dictionary({"/S": "/GoTo", "/D": "cite.chen2021codex"}),
        }
    )
    for fragment, digit in (("[1,", "1"), ("2,", "2"), ("4]", "4"), ("[9,", "9")):
        candidates = _extract_candidates_from_annot(annot, fragment)
        assert fragment in candidates, fragment
        assert digit in candidates, fragment


def test_extract_candidates_rebuilds_a_translated_cross_reference() -> None:
    # A figure/table/section reference has its label translated ("Figure" ->
    # "图"), so the source fragment no longer matches. The number survives, so
    # rebuild the localized form from the target language's caption prefix.
    annot = pikepdf.Dictionary(
        {
            "/Subtype": "/Link",
            "/A": pikepdf.Dictionary({"/S": "/GoTo", "/D": "figure.caption.1"}),
        }
    )
    candidates = _extract_candidates_from_annot(
        annot, "(Figure 1(", figure_prefix="图", table_prefix="表"
    )
    assert "图 1" in candidates
    assert "图1" in candidates


def test_extract_candidates_reads_the_displayed_number_not_the_dest_counter() -> None:
    # The destination counter is not the displayed number: table.caption.5 is
    # "Table 1" in the source. The number must come from the source fragment, or
    # the link would relocate onto the wrong table.
    annot = pikepdf.Dictionary(
        {
            "/Subtype": "/Link",
            "/A": pikepdf.Dictionary({"/S": "/GoTo", "/D": "table.caption.5"}),
        }
    )
    candidates = _extract_candidates_from_annot(
        annot, "Table 1", figure_prefix="图", table_prefix="表"
    )
    assert "表 1" in candidates
    assert "表 5" not in candidates


def test_extract_candidates_section_uses_the_bare_number() -> None:
    annot = pikepdf.Dictionary(
        {
            "/Subtype": "/Link",
            "/A": pikepdf.Dictionary({"/S": "/GoTo", "/D": "subsection.3.4"}),
        }
    )
    candidates = _extract_candidates_from_annot(annot, "Sec. 3.4")
    assert "3.4" in candidates


def test_standalone_number_rejects_a_longer_number() -> None:
    assert _standalone_number("表 1", 2, 3)
    assert not _standalone_number("表 10", 2, 3)
    assert not _standalone_number("13.1", 1, 3)  # the "3.1" inside "13.1"
    assert _standalone_number("第 3.1 节", 2, 5)


def test_relocate_page_annotations_no_annots(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    pdf.save(str(pdf_path))
    pdf.close()

    with pikepdf.open(str(pdf_path)) as doc:
        page = doc.pages[0]
        count = relocate_page_annotations(
            page=page,
            page_no=1,
            source_pdf=pdf_path,
            overlay_path=str(pdf_path),
            strip_rects=[(0, 0, 100, 100)],
        )
        assert count == 0


def test_relocate_page_annotations_keeps_an_unmatched_link(tmp_path: Path) -> None:
    # A link annotation in a stripped zone that cannot be matched to translated
    # text is KEPT at its source rect: dropping it would make the citation dead,
    # and a slightly-off click zone is the lesser failure. It still counts as
    # modified (touched) so the caller knows a pass ran.
    pdf_path = tmp_path / "test.pdf"
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(200, 200))
    annot = pikepdf.Dictionary(
        {
            "/Subtype": pikepdf.Name("/Link"),
            "/Rect": [10.0, 10.0, 50.0, 50.0],
        }
    )
    page["/Annots"] = [annot]
    pdf.save(str(pdf_path))
    pdf.close()

    with pikepdf.open(str(pdf_path)) as doc:
        page = doc.pages[0]
        annots_before = page.get("/Annots")
        assert annots_before is not None and len(annots_before) == 1
        count = relocate_page_annotations(
            page=page,
            page_no=1,
            source_pdf=pdf_path,
            overlay_path=str(pdf_path),
            strip_rects=[(0, 0, 100, 100)],
        )
        assert count == 1
        annots_after = page.get("/Annots")
        assert annots_after is not None and len(annots_after) == 1
        # The rect is unchanged: no translated glyph matched.
        assert [float(v) for v in annots_after[0]["/Rect"]] == [10.0, 10.0, 50.0, 50.0]
