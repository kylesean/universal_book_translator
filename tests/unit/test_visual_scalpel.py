"""Unit tests for Visual Scalpel (L4 multimodal precision cropping)."""

import base64
import io
from pathlib import Path

import pypdf
import pytest
from PIL import Image

from ubt.adapters.pdf.visual_scalpel import (
    _compute_crop_coords,
    crop_block_image,
    crop_block_pil,
    is_visual_scalpel_applicable,
)
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock


def _create_test_pdf(path: Path, num_pages: int = 2) -> Path:
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=600.0, height=800.0)
    with path.open("wb") as f_out:
        writer.write(f_out)
    return path


def test_crop_block_pil_uses_the_unrotated_frame(tmp_path: Path) -> None:
    """A /Rotate page must be cropped in the bbox's (unrotated) frame.

    ``IRBlock.bbox`` comes from ``textgeom`` in the unrotated frame, but the
    cropper used ``get_width/height`` (display frame) plus the default rotated
    render, so a rotated page was cropped at the wrong coordinates (or raised).
    """
    pdf_path = tmp_path / "rot90.pdf"
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=100)
    page.rotate(90)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    img = crop_block_pil(pdf_path, 1, BoundingBox(page=1, x0=0, y0=0, x1=200, y1=100), dpi=72)
    # The bbox is the whole unrotated page, so the crop is the unrotated raster.
    assert img.size == (200, 100)


def test_compute_crop_coords_invert_y() -> None:
    """PDF origin bottom-left should correctly invert to raster image origin top-left."""
    # Box at x=[100, 200], y=[700, 750] (near top of page 800pt high)
    bbox = BoundingBox(page=1, x0=100.0, y0=700.0, x1=200.0, y1=750.0)
    scale = 2.0  # e.g. 144 DPI
    bleed = 5.0
    x0, y0, x1, y1 = _compute_crop_coords(bbox, 600.0, 800.0, scale, bleed)

    # x0 = (100 - 5) * 2 = 190
    assert x0 == 190
    # x1 = (200 + 5) * 2 = 410
    assert x1 == 410
    # y0 = (800 - 750 - 5) * 2 = 45 * 2 = 90
    assert y0 == 90
    # y1 = (800 - 700 + 5) * 2 = 105 * 2 = 210
    assert y1 == 210


def test_crop_block_image_and_pil(tmp_path: Path) -> None:
    """Cropping a valid PDF bounding box returns a valid PIL Image and Base64 PNG."""
    pdf_path = _create_test_pdf(tmp_path / "sample.pdf", num_pages=2)
    bbox = BoundingBox(page=1, x0=50.0, y0=400.0, x1=250.0, y1=500.0)

    # 1. PIL Image crop
    pil_img = crop_block_pil(pdf_path, page_num=1, bbox=bbox, dpi=150, bleed_pt=4.0)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.width > 0
    assert pil_img.height > 0

    # 2. Base64 string crop
    b64_str = crop_block_image(pdf_path, page_num=1, bbox=bbox, dpi=150)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 50

    # Verify decoding back to image
    raw_bytes = base64.b64decode(b64_str)
    decoded_img = Image.open(io.BytesIO(raw_bytes))
    assert decoded_img.format == "PNG"
    assert decoded_img.size == pil_img.size


def test_crop_page_bounds_errors(tmp_path: Path) -> None:
    """Invalid page numbers or missing files raise appropriate errors."""
    pdf_path = _create_test_pdf(tmp_path / "bounds.pdf", num_pages=1)
    bbox = BoundingBox(page=1, x0=10.0, y0=10.0, x1=50.0, y1=50.0)

    with pytest.raises(FileNotFoundError):
        crop_block_image(tmp_path / "non_existent.pdf", page_num=1, bbox=bbox)

    with pytest.raises(IndexError, match="out of bounds"):
        crop_block_image(pdf_path, page_num=5, bbox=bbox)

    with pytest.raises(IndexError, match="out of bounds"):
        crop_block_image(pdf_path, page_num=0, bbox=bbox)


def test_is_visual_scalpel_applicable(tmp_path: Path) -> None:
    """Check applicability conditions for multimodal visual repair."""
    pdf_path = _create_test_pdf(tmp_path / "test.pdf")
    bbox = BoundingBox(page=1, x0=100.0, y0=200.0, x1=300.0, y1=400.0)

    # Non-PDF or missing PDF -> False
    b1 = IRBlock(id="b1", spine_index=1, source_text="text", bbox=bbox)
    assert not is_visual_scalpel_applicable(b1, None)
    assert not is_visual_scalpel_applicable(b1, tmp_path / "missing.pdf")

    # Formula block with bbox -> True (always benefits from visual check)
    b_formula = IRBlock(
        id="bf", spine_index=2, block_type=BlockType.FORMULA, source_text="x=y", bbox=bbox
    )
    assert is_visual_scalpel_applicable(b_formula, pdf_path)

    # Table block with bbox -> True
    b_table = IRBlock(
        id="bt", spine_index=3, block_type=BlockType.TABLE, source_text="| a | b |", bbox=bbox
    )
    assert is_visual_scalpel_applicable(b_table, pdf_path)

    # Regular narrative with high score -> False
    b_norm = IRBlock(
        id="bn",
        spine_index=4,
        block_type=BlockType.NARRATIVE,
        source_text="fluent prose",
        mtqe_score=0.92,
        bbox=bbox,
    )
    assert not is_visual_scalpel_applicable(b_norm, pdf_path)

    # Regular narrative with low score (severe failure / omission) -> True
    b_fail = IRBlock(
        id="b_fail",
        spine_index=5,
        block_type=BlockType.NARRATIVE,
        source_text="dropped text",
        mtqe_score=0.45,
        bbox=bbox,
    )
    assert is_visual_scalpel_applicable(b_fail, pdf_path)

    # Zero-area bbox -> False
    b_zero = IRBlock(
        id="bz",
        spine_index=6,
        block_type=BlockType.FORMULA,
        source_text="x",
        bbox=BoundingBox(page=1, x0=100.0, y0=200.0, x1=100.0, y1=200.0),
    )
    assert not is_visual_scalpel_applicable(b_zero, pdf_path)


def test_repeated_crops_reuse_one_page_raster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cropping one page several times rasterizes it once.

    A full-page raster costs ~50 ms against a 0.3 ms crop, so formula mode
    "image" — one crop per display equation — paid the whole render per equation.
    """
    import pypdfium2

    from ubt.adapters.pdf import visual_scalpel

    pdf_path = _create_test_pdf(tmp_path / "reuse.pdf", num_pages=2)
    opened: list[str] = []
    real_document = pypdfium2.PdfDocument

    def counting_document(path: object, *args: object, **kwargs: object) -> object:
        opened.append(str(path))
        return real_document(path, *args, **kwargs)

    monkeypatch.setattr(pypdfium2, "PdfDocument", counting_document)
    monkeypatch.setattr(visual_scalpel, "_PAGE_CACHE", None)

    box = BoundingBox(page=1, x0=10.0, y0=10.0, x1=50.0, y1=50.0)
    for _ in range(3):
        crop_block_pil(pdf_path, 1, box)
    assert len(opened) == 1, "the page raster was re-rendered for every crop"

    crop_block_pil(pdf_path, 2, box)
    assert len(opened) == 2, "a second page must not be served from the first one's raster"


def test_page_raster_is_refreshed_when_the_file_is_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewritten path is a new page: self-healing renders overwrite their output."""
    from ubt.adapters.pdf import visual_scalpel

    path = tmp_path / "rewrite.pdf"
    _create_test_pdf(path, num_pages=1)
    monkeypatch.setattr(visual_scalpel, "_PAGE_CACHE", None)
    before = crop_block_pil(path, 1, BoundingBox(page=1, x0=0.0, y0=0.0, x1=600.0, y1=800.0)).size

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300.0, height=400.0)
    with path.open("wb") as fh:
        writer.write(fh)

    after = crop_block_pil(path, 1, BoundingBox(page=1, x0=0.0, y0=0.0, x1=300.0, y1=400.0)).size
    assert after != before, "the cache served the raster of the previous file"
