"""Golden geometry files: exact extraction coordinates over deterministic PDFs.

The adapters read page geometry through several independent readers -- pdfium's
text rects (:func:`ubt.adapters.pdf.textgeom.extract_lines`), the pdfium
adapter's block boxes, pdfium's own ``get_mediabox``/``get_cropbox``/``get_size``
-- and each must agree on the *same* frame for the special page shapes:

- **CropBox ≠ MediaBox**: a print/imposed PDF renders only its CropBox, while
  ``get_mediabox`` still reports the full sheet. A consumer that measures the
  raster by the MediaBox crops the wrong region (the P0-5 regression); one
  that mixes the frames produces boxes pointing off-page.
- **/Rotate 90**: the display frame swaps the axes, while text rects stay in
  user space. A renderer that applies /Rotate and a consumer that does not
  disagree about where the text is unless every reader states its frame.
- **Two columns**: reading order must follow columns (all left lines, then all
  right), not scan order down the page.

Each fixture is written by :mod:`tests.pdf_builders` (pypdf, an independent
writer) and the expected values below were read off the extraction once and
pinned -- a change in any reader's frame arithmetic fails the module loudly
instead of silently rotating/cropping a real book. Tolerances: pdfium's rect
values are float-noisy across builds, so coordinates compare within 0.5pt.

``fast`` tier: pure extraction, no model, no compiler.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from pdf_builders import write_text_pdf, write_two_column_pdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, FloatObject, NameObject

from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.adapters.pdf.textgeom import extract_lines

pytestmark = pytest.mark.fast

#: pdfium rect float noise across builds.
_TOL = 0.5


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _TOL


def _assert_rect_close(
    rect: tuple[float, float, float, float], expected: tuple[float, float, float, float]
) -> None:
    assert all(_close(got, want) for got, want in zip(rect, expected, strict=True)), (
        f"rect {tuple(round(v, 2) for v in rect)} != expected {expected}"
    )


# --------------------------------------------------------------------------- #
# /Rotate 90
# --------------------------------------------------------------------------- #

#: A portrait page turned for display: 300x500 written, 500x300 displayed.
_PAGE_W = 300.0
_PAGE_H = 500.0
_LINE = ("The rotated page body text for the golden geometry run.",)


def test_rotated_page_mediabox_and_text_share_the_user_space_frame(tmp_path: Path) -> None:
    """``extract_lines`` reports the *unrotated* mediabox and user-space rects.

    The docstring contract of :func:`extract_lines` states the pair shares one
    frame; get_width/height would return the swapped display size (500x300),
    which here must NOT appear. The line's rect sits in the unrotated page's
    own coordinates -- the caller applies /Rotate itself when it wants display
    space (see :func:`ubt.adapters.pdf.coordinate_resolver.undo_page_rotation`).
    """
    source = write_text_pdf(
        tmp_path / "rot90.pdf", [_LINE], width=_PAGE_W, height=_PAGE_H, rotation=90
    )
    lines, size = extract_lines(source, 1)
    assert size == (_PAGE_W, _PAGE_H), f"size must be the unrotated mediabox, got {size}"
    assert len(lines) == 1
    # The line is drawn at the top-left margin of the unrotated page (margin
    # 54, one leading down from the top: y ≈ 500-54-16), and pdfium merges the
    # wrapped segments into one wide rect running to x ≈ 350 (it overflows the
    # 300pt mediabox -- the merge keeps the glyph-run extent).
    _assert_rect_close(lines[0].rect, (54.17, 443.37, 350.46, 454.74))


async def test_rotated_page_adapter_blocks_stay_in_user_space(tmp_path: Path) -> None:
    """The pdfium adapter's block bboxes sit in the same unrotated frame.

    The full ingest (reader paragraph grouping on top of extract_lines) must
    not re-project into display space: a block box for a /Rotate 90 page must
    be a thin *horizontal* strip (text runs across the 300pt width), not a
    tall sideways one.
    """
    source = write_text_pdf(
        tmp_path / "rot90.pdf", [_LINE], width=_PAGE_W, height=_PAGE_H, rotation=90
    )

    blocks = [b async for ch in PDFiumAdapter().parse_stream(source) for b in ch.blocks]
    assert len(blocks) == 1
    bbox = blocks[0].bbox
    assert bbox is not None and bbox.page == 1
    assert _close(bbox.x0, 54.2) and _close(bbox.x1, 350.5), (bbox.x0, bbox.x1)
    assert _close(bbox.y0, 443.4) and _close(bbox.y1, 454.7), (bbox.y0, bbox.y1)
    # User-space strip: wider than tall (the line runs across the page width).
    assert (bbox.x1 - bbox.x0) > (bbox.y1 - bbox.y0)


# --------------------------------------------------------------------------- #
# Two columns
# --------------------------------------------------------------------------- #

_LEFT = ("Left heading.", "Left body line one.", "Left body line two.")
_RIGHT = ("Right heading.", "Right body line one.", "Right body line two.")


def test_two_column_lines_are_read_in_column_order(tmp_path: Path) -> None:
    """All left-column lines precede all right-column lines, within each band.

    :func:`extract_lines` sorts rows top-down; ``column_order`` decides the
    within-row order. On this one-band page the contract reads: every left
    line before every right line -- scan order down the page would interleave
    them (Left heading, Right heading, Left body one, ...).
    """
    source = write_two_column_pdf(tmp_path / "two_col.pdf", [(_LEFT, _RIGHT)])
    lines, size = extract_lines(source, 1)
    assert size == (612.0, 792.0)
    texts = [line.text for line in lines]
    assert texts == [*_LEFT, *_RIGHT], texts
    # The two columns' x origins sit either side of the midline (306pt), with
    # the right column just past it (width/2 + margin/2 = 333).
    assert all(line.rect[0] < 200.0 for line in lines[: len(_LEFT)])
    assert all(line.rect[0] > 300.0 for line in lines[len(_LEFT) :])


async def test_two_column_adapter_blocks_split_on_the_gutter(tmp_path: Path) -> None:
    """The pdfium adapter groups each column into its own block."""
    source = write_two_column_pdf(tmp_path / "two_col.pdf", [(_LEFT, _RIGHT)])

    blocks = [b async for ch in PDFiumAdapter().parse_stream(source) for b in ch.blocks]
    assert len(blocks) == 2, [b.source_text for b in blocks]
    first, second = blocks
    assert first.source_text.startswith("Left heading.")
    assert second.source_text.startswith("Right heading.")
    left_box = first.bbox
    right_box = second.bbox
    assert left_box is not None and right_box is not None
    # The gutter: the two blocks' x ranges do not overlap.
    assert left_box.x1 < right_box.x0
    _assert_rect_close(
        (left_box.x0, left_box.y0, left_box.x1, left_box.y1), (54.9, 703.4, 151.0, 746.7)
    )
    _assert_rect_close(
        (right_box.x0, right_box.y0, right_box.x1, right_box.y1), (334.1, 703.4, 438.0, 746.6)
    )


# --------------------------------------------------------------------------- #
# CropBox ≠ MediaBox (non-zero MediaBox origin)
# --------------------------------------------------------------------------- #


def _page_with_boxes(
    path: Path,
    mediabox: tuple[float, float, float, float],
    cropbox: tuple[float, float, float, float],
) -> Path:
    """Rewrite a one-page PDF's /MediaBox and /CropBox."""
    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.pages[0][NameObject("/MediaBox")] = ArrayObject([FloatObject(v) for v in mediabox])
    writer.pages[0][NameObject("/CropBox")] = ArrayObject([FloatObject(v) for v in cropbox])
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_cropbox_page_reports_both_boxes_in_one_frame(tmp_path: Path) -> None:
    """pdfium's box getters agree with the PDF's declared boxes.

    A non-zero MediaBox origin (print imposition) plus a CropBox inside it is
    the shape that broke the visual scalpel (P0-5): the raster covers the
    CropBox while ``get_mediabox`` reports the full sheet. Pin both reads:
    ``get_cropbox`` returns the *absolute* box (origin included), and the
    display size is the CropBox's extent, not the MediaBox's.
    """
    source = write_text_pdf(tmp_path / "crop.pdf", [_LINE], width=600.0, height=800.0)
    source = _page_with_boxes(
        source, mediabox=(50.0, 60.0, 650.0, 860.0), cropbox=(100.0, 120.0, 500.0, 700.0)
    )
    pdf = pdfium.PdfDocument(str(source))
    try:
        page = pdf[0]
        mediabox = tuple(float(v) for v in page.get_mediabox())
        cropbox = tuple(float(v) for v in page.get_cropbox())
        assert all(
            _close(g, w) for g, w in zip(mediabox, (50.0, 60.0, 650.0, 860.0), strict=True)
        ), mediabox
        assert all(
            _close(g, w) for g, w in zip(cropbox, (100.0, 120.0, 500.0, 700.0), strict=True)
        ), cropbox
        # The display raster is the CropBox's size: 400x580, NOT 600x800.
        assert page.get_size() == (400.0, 580.0)
    finally:
        pdf.close()

    # extract_lines keeps reporting the *mediabox* size (its documented frame
    # for the line rects), so its consumers see one consistent frame.
    lines, size = extract_lines(source, 1)
    assert size == (600.0, 800.0)
    assert len(lines) == 1
    # The text sits at the page's top-left margin *in mediabox frame* -- the
    # content stream was authored against the original 600x800 page and the
    # boxes re-read in the widened mediabox frame only shift by the origin.
    _assert_rect_close(lines[0].rect, (54.17, 743.37, 350.46, 754.74))


__all__ = ["pytest"]
