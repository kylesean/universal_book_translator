from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, FloatObject, NameObject

from ubt.adapters.pdf.visual_scalpel import _compute_crop_coords

pytestmark = pytest.mark.fast


def _page_with_mediabox(path: Path, box: tuple[float, float, float, float]) -> Path:
    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.pages[0][NameObject("/MediaBox")] = ArrayObject([FloatObject(v) for v in box])
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_a_zero_origin_page_crops_at_the_box_itself() -> None:
    x0, y0, x1, y1 = _compute_crop_coords(
        (10.0, 20.0, 110.0, 220.0), 200.0, 300.0, 1.0, bleed_pt=0.0
    )
    assert (x0, x1) == (10, 110)
    assert (y0, y1) == (80, 280)


def test_a_non_zero_media_origin_is_subtracted() -> None:
    x0, y0, x1, y1 = _compute_crop_coords(
        (10.0, 20.0, 110.0, 220.0),
        200.0,
        300.0,
        1.0,
        bleed_pt=0.0,
        origin_x=50.0,
        origin_y=60.0,
    )
    assert (x0, x1) == (0, 60)
    assert (y0, y1) == (140, 300)


def test_a_box_entirely_inside_the_shifted_page_crops_in_bounds() -> None:
    x0, y0, x1, y1 = _compute_crop_coords(
        (60.0, 70.0, 160.0, 170.0),
        200.0,
        200.0,
        1.0,
        bleed_pt=0.0,
        origin_x=50.0,
        origin_y=60.0,
    )
    assert (x0, x1) == (10, 110)
    assert (y0, y1) == (90, 190)


def test_crop_block_pil_uses_the_real_mediabox_origin(tmp_path: Path) -> None:
    from ubt.adapters.pdf import visual_scalpel

    pdf_path = tmp_path / "shifted.pdf"
    write_text_pdf(pdf_path, [("ABCDEF",)], width=200.0, height=200.0)
    _page_with_mediabox(pdf_path, (50.0, 60.0, 250.0, 260.0))

    visual_scalpel._PAGE_CACHE = None
    try:
        with pytest.raises(ValueError, match="empty intersection"):
            visual_scalpel.crop_block_pil(
                pdf_path, 1, (40.0, 50.0, 45.0, 55.0), dpi=72, bleed_pt=0.0
            )
    finally:
        visual_scalpel._PAGE_CACHE = None


def _page_with_cropbox(path: Path, box: tuple[float, float, float, float]) -> Path:
    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.pages[0][NameObject("/CropBox")] = ArrayObject([FloatObject(v) for v in box])
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_crop_uses_the_cropbox_not_the_mediabox(tmp_path: Path) -> None:
    from ubt.adapters.pdf import visual_scalpel

    pdf_path = tmp_path / "cropped.pdf"
    write_text_pdf(pdf_path, [("ABCDEF",)], width=600.0, height=800.0, background=(0, 200, 0))
    _page_with_cropbox(pdf_path, (0.0, 0.0, 600.0, 500.0))

    visual_scalpel._PAGE_CACHE = None
    try:
        img = visual_scalpel.crop_block_pil(
            pdf_path, 1, (0.0, 0.0, 600.0, 500.0), dpi=72, bleed_pt=0.0
        )
    finally:
        visual_scalpel._PAGE_CACHE = None

    assert img.size == (600, 500)
    probe = img.convert("RGB").getpixel((5, 495))
    assert probe == (0, 255, 0), probe


def test_resolver_and_scalpel_roundtrip_with_shifted_cropbox(tmp_path: Path) -> None:
    from ubt.adapters.pdf import visual_scalpel
    from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver

    pdf_path = tmp_path / "cropbox_shift.pdf"
    write_text_pdf(pdf_path, [("TEST",)], width=400.0, height=400.0, background=(255, 0, 0))
    _page_with_cropbox(pdf_path, (50.0, 60.0, 250.0, 260.0))

    resolver = PageBBoxResolver(page_width=200.0, page_height=200.0, origin_x=50.0, origin_y=60.0)
    bbox = resolver.resolve_bbox((0.2, 0.2, 0.8, 0.8), coord_system="normalized_1")
    assert bbox is not None
    assert bbox == pytest.approx((90.0, 100.0, 210.0, 220.0))

    visual_scalpel._PAGE_CACHE = None
    try:
        img = visual_scalpel.crop_block_pil(pdf_path, 1, bbox, dpi=72, bleed_pt=0.0)
        assert img.size == (120, 120)
        probe = img.convert("RGB").getpixel((10, 10))
        assert probe == (255, 0, 0), probe
    finally:
        visual_scalpel._PAGE_CACHE = None


def test_extract_blocks_with_pdfium_normalizes_cropbox_origin(tmp_path: Path) -> None:
    from ubt.adapters.pdf.pdfium_adapter import extract_blocks_with_pdfium

    pdf_path = tmp_path / "crop_normalize.pdf"
    write_text_pdf(pdf_path, [("Hello Rust",)], width=400.0, height=400.0)
    _page_with_cropbox(pdf_path, (50.0, 60.0, 350.0, 360.0))

    blocks = extract_blocks_with_pdfium(pdf_path)
    assert len(blocks) == 1
    bbox = blocks[0].bbox
    assert bbox is not None
    assert bbox.x0 < 300.0
    assert bbox.y0 < 300.0
    assert bbox.x1 <= 300.0 + 1e-2
    assert bbox.y1 <= 300.0 + 1e-2


def test_vertical_solver_clamps_boundary_jitter_without_exception() -> None:
    from ubt.core.layout.solver import VerticalBlockInput, VerticalTrack, solve_vertical

    track = VerticalTrack(y_min_pt=50.0, y_max_pt=650.0)
    blocks = [
        VerticalBlockInput(
            index=0,
            required_height_pt=30.0,
            source_height_pt=30.0,
            source_top_pt=655.0,
        ),
        VerticalBlockInput(
            index=1,
            required_height_pt=30.0,
            source_height_pt=30.0,
            source_top_pt=580.0,
        ),
    ]
    specs = solve_vertical(blocks, track=track, min_gap_pt=2.0)
    assert len(specs) == 2
    assert specs[0].y_top_pt <= track.y_max_pt + 1e-4
    assert specs[0].y_bottom_pt >= specs[1].y_top_pt + 2.0 - 1e-4
    assert specs[1].y_bottom_pt >= track.y_min_pt - 1e-4
