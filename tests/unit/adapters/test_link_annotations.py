"""Link annotations are relocated onto the overlay's translated glyphs."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf.link_annotations import (
    _extract_candidates_from_annot,
    _rect_intersects_any,
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


def test_relocate_page_annotations_prunes_dead_link(tmp_path: Path) -> None:
    # A link annotation in a stripped zone that cannot be matched to translated
    # text is pruned from /Annots, and reported in the modified count.
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
        assert annots_after is not None and len(annots_after) == 0
