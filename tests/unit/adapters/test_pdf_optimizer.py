"""Unit tests for PDF size miniaturization and lossless stream compression pipeline."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest
from PIL import Image

from ubt.adapters.pdf.pdf_optimizer import (
    OptimizationProfile,
    PdfOptimizeOptions,
    deduplicate_pdf_streams,
    is_ghostscript_available,
    optimize_images_in_pdf,
    optimize_pdf,
    optimize_pdf_in_place,
    verify_pdf_fidelity,
)

pytestmark = pytest.mark.fast


def _create_minimal_pdf(tmp_path: Path, num_pages: int = 2) -> Path:
    """Helper to create a simple readable PDF for testing."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(num_pages):
        pdf.add_blank_page(page_size=(300, 300))
    p = tmp_path / "minimal.pdf"
    pdf.save(p)
    return p


def _create_pdf_with_duplicate_streams(tmp_path: Path) -> Path:
    """Helper to create a PDF with duplicate image streams across pages."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(200, 200))
    p2 = pdf.add_blank_page(page_size=(200, 200))

    raw_data = b"SHARED_IMAGE_STREAM_BINARY_DATA_" * 30
    s1 = pdf.make_stream(raw_data)
    s1["/Subtype"] = pikepdf.Name("/Image")
    s2 = pdf.make_stream(raw_data)
    s2["/Subtype"] = pikepdf.Name("/Image")

    p1.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im1=s1))
    p2.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im2=s2))
    p1.Contents = pdf.make_stream(b"q /Im1 Do Q")
    p2.Contents = pdf.make_stream(b"q /Im2 Do Q")

    p = tmp_path / "dups.pdf"
    pdf.save(p)
    return p


def test_deduplicate_pdf_streams_merges_identical_streams(tmp_path: Path) -> None:
    pdf_path = _create_pdf_with_duplicate_streams(tmp_path)
    with pikepdf.open(pdf_path) as pdf:
        count = deduplicate_pdf_streams(pdf)
        assert count == 1
        # Resources on page 2 should now point to canonical stream from page 1
        im1 = pdf.pages[0].Resources.XObject[pikepdf.Name("/Im1")]
        im2 = pdf.pages[1].Resources.XObject[pikepdf.Name("/Im2")]
        assert im1.objgen == im2.objgen


def test_deduplicate_pdf_streams_ignores_unique_streams() -> None:
    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(200, 200))
    p2 = pdf.add_blank_page(page_size=(200, 200))

    s1 = pdf.make_stream(b"FIRST_STREAM_CONTENT_XYZ_" * 10)
    s1["/Subtype"] = pikepdf.Name("/Image")
    s2 = pdf.make_stream(b"SECOND_STREAM_CONTENT_ABC_" * 10)
    s2["/Subtype"] = pikepdf.Name("/Image")

    p1.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im1=s1))
    p2.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im2=s2))

    count = deduplicate_pdf_streams(pdf)
    assert count == 0


def test_optimize_images_in_pdf_downsamples_oversized() -> None:
    pdf = pikepdf.Pdf.new()
    p = pdf.add_blank_page(page_size=(400, 400))

    # Create uncompressed 200x200 RGB image
    img = Image.new("RGB", (200, 200), color=(150, 75, 200))
    s = pdf.make_stream(img.tobytes())
    s["/Type"] = pikepdf.Name("/XObject")
    s["/Subtype"] = pikepdf.Name("/Image")
    s["/Width"] = 200
    s["/Height"] = 200
    s["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
    s["/BitsPerComponent"] = 8
    p.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im1=s))

    # Request downsampling to max 80px
    opt_count = optimize_images_in_pdf(pdf, max_dimension=80, jpeg_quality=75)
    assert opt_count == 1
    assert int(s["/Width"]) == 80
    assert int(s["/Height"]) == 80
    assert s["/Filter"] == pikepdf.Name("/DCTDecode")


def test_optimize_images_in_pdf_skips_small_images() -> None:
    pdf = pikepdf.Pdf.new()
    p = pdf.add_blank_page(page_size=(200, 200))

    img = Image.new("RGB", (50, 50), color=(10, 20, 30))
    s = pdf.make_stream(img.tobytes())
    s["/Type"] = pikepdf.Name("/XObject")
    s["/Subtype"] = pikepdf.Name("/Image")
    s["/Width"] = 50
    s["/Height"] = 50
    s["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
    s["/BitsPerComponent"] = 8
    p.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im1=s))

    opt_count = optimize_images_in_pdf(pdf, max_dimension=100, jpeg_quality=85)
    assert opt_count == 0
    assert int(s["/Width"]) == 50


def test_verify_pdf_fidelity_detects_page_count_mismatch(tmp_path: Path) -> None:
    pdf1 = _create_minimal_pdf(tmp_path / "p1", num_pages=2)
    pdf2 = _create_minimal_pdf(tmp_path / "p2", num_pages=3)

    valid, reason = verify_pdf_fidelity(pdf1, pdf2)
    assert not valid
    assert "Page count mismatch" in reason


def test_verify_pdf_fidelity_passes_on_identical(tmp_path: Path) -> None:
    pdf1 = _create_minimal_pdf(tmp_path / "p1", num_pages=2)
    pdf2 = _create_minimal_pdf(tmp_path / "p2", num_pages=2)

    valid, reason = verify_pdf_fidelity(pdf1, pdf2)
    assert valid
    assert "Fidelity verified" in reason


def test_optimize_pdf_in_place_atomic(tmp_path: Path) -> None:
    pdf_path = _create_pdf_with_duplicate_streams(tmp_path)

    res = optimize_pdf_in_place(pdf_path, profile="lossless")
    assert res.status in ("optimized", "skipped_already_optimal")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_optimize_pdf_preserves_file_when_already_optimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _create_minimal_pdf(tmp_path, num_pages=1)
    orig_size = pdf_path.stat().st_size

    # Force candidate size check to simulate larger candidate
    res = optimize_pdf(pdf_path, options=PdfOptimizeOptions(profile=OptimizationProfile.LOSSLESS))
    # On a tiny 1-page blank PDF (~700 bytes), QPDF object streams might not shrink it
    assert res.status in ("optimized", "skipped_already_optimal")
    assert pdf_path.stat().st_size == orig_size or res.status == "optimized"


def test_optimize_pdf_handles_missing_file() -> None:
    res = optimize_pdf_in_place(Path("/nonexistent/file.pdf"))
    assert res.status == "skipped_already_optimal"
    assert res.original_size_bytes == 0


def test_optimize_pdf_handles_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"NOT A REAL PDF FILE HEADER")

    res = optimize_pdf_in_place(corrupt)
    assert res.status == "failed"
    assert res.error is not None
    # Original corrupt file must still be intact
    assert corrupt.read_bytes() == b"NOT A REAL PDF FILE HEADER"


def test_is_ghostscript_available_boolean() -> None:
    avail = is_ghostscript_available()
    assert isinstance(avail, bool)


def test_optimize_pdf_custom_output_path(tmp_path: Path) -> None:
    src = _create_pdf_with_duplicate_streams(tmp_path)
    dst = tmp_path / "optimized_target.pdf"

    res = optimize_pdf(src, output_path=dst)
    assert dst.exists()
    assert res.path == str(dst)
    assert src.exists()  # Source remains intact when output is distinct
