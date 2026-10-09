"""Visual Scalpel: precision bounding-box cropping for multimodal LLM inspection and L4 repair.

Extracts targeted, high-resolution image crops from PDF pages using zero-AGPL pypdfium2,
converting coordinates from PDF points (origin bottom-left) to raster pixels with safety
bleed margins. The output Base64 PNG strings plug directly into ``router.complete_with_images``.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized

if TYPE_CHECKING:
    from PIL import Image as PILImage

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock

logger = logging.getLogger(__name__)

# Standard points-to-DPI scale: 72 points per inch.
# 150 DPI provides ~2.08x resolution, optimal for VLM OCR and symbol recognition.
DEFAULT_SCALPEL_DPI: int = 150
DEFAULT_BLEED_PT: float = 4.0


def _compute_crop_coords(
    bbox: BoundingBox | tuple[float, float, float, float],
    page_width_pt: float,
    page_height_pt: float,
    scale: float,
    bleed_pt: float = DEFAULT_BLEED_PT,
) -> tuple[int, int, int, int]:
    """Convert PDF coordinates (origin bottom-left) to raster image coordinates (origin top-left)."""
    if isinstance(bbox, BoundingBox):
        bx0, by0, bx1, by1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    else:
        bx0, by0, bx1, by1 = bbox

    min_x = min(bx0, bx1)
    max_x = max(bx0, bx1)
    min_y = min(by0, by1)
    max_y = max(by0, by1)

    # Invert Y axis: PDF y=0 is bottom; raster y=0 is top.
    x0 = int(max(0.0, (min_x - bleed_pt) * scale))
    x1 = int(min(page_width_pt * scale, (max_x + bleed_pt) * scale))
    y0 = int(max(0.0, (page_height_pt - max_y - bleed_pt) * scale))
    y1 = int(min(page_height_pt * scale, (page_height_pt - min_y + bleed_pt) * scale))

    # Clamp into the page instead of fabricating a full-page crop: a fully
    # out-of-bounds box would otherwise degrade to x1 = page width (feeding
    # the VLM a whole page for a phantom block). Empty intersection fails
    # closed — callers (repair visual path) already catch and fall back to
    # text repair.
    img_w = int(page_width_pt * scale)
    img_h = int(page_height_pt * scale)
    x0, x1 = min(max(x0, 0), img_w), max(min(x1, img_w), 0)
    y0, y1 = min(max(y0, 0), img_h), max(min(y1, img_h), 0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"bbox {(bx0, by0, bx1, by1)} has empty intersection with "
            f"page {page_width_pt}x{page_height_pt}pt"
        )

    return x0, y0, x1, y1


# Reuse cache for the page raster. Rasterizing a page costs ~50 ms and cropping
# it ~0.3 ms, so cropping N formulas on one page would otherwise pay N full renders.
# Exactly one page is kept: an A4 raster at 300 dpi is
# ~26 MB and a wide two-column spread ~50 MB, so a longer cache would trade the
# time back for memory. Read/write happens under PDFIUM_LOCK (@pdfium_serialized).
_PAGE_CACHE: tuple[tuple[str, int, int, int, int], PILImage.Image, float, float] | None = None


# The file identity behind a raster: a self-healing Typst render rewrites the
# same path, and a stale bitmap would then certify the previous attempt.
def _page_cache_key(path: Path, page_num: int, dpi: int) -> tuple[str, int, int, int, int]:
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size, page_num, dpi)


@pdfium_serialized
def crop_block_pil(
    pdf_path: Path | str,
    page_num: int,
    bbox: BoundingBox | tuple[float, float, float, float],
    dpi: int = DEFAULT_SCALPEL_DPI,
    bleed_pt: float = DEFAULT_BLEED_PT,
) -> PILImage.Image:
    """Rasterize a page and crop the bounding box region as a PIL Image."""
    import pypdfium2 as pdfium

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if page_num < 1:
        raise IndexError(f"Page number {page_num} out of bounds (pages are 1-based)")

    global _PAGE_CACHE
    scale = dpi / 72.0
    key = _page_cache_key(path, page_num, dpi)
    if _PAGE_CACHE is not None and _PAGE_CACHE[0] == key:
        _, full_img, page_width_pt, page_height_pt = _PAGE_CACHE
    else:
        pdf = pdfium.PdfDocument(str(path))
        try:
            if page_num > len(pdf):
                raise IndexError(f"Page number {page_num} out of bounds (1..{len(pdf)})")
            page = pdf[page_num - 1]
            try:
                # ``bbox`` comes from ``textgeom`` in the UNROTATED frame
                # (``get_rect`` / ``get_mediabox``), so rasterize unrotated too.
                # ``render``'s ``rotation`` is additive to the page's /Rotate, so
                # pass its complement; using ``get_width/height`` (display frame)
                # with the default render crops the wrong region on a /Rotate
                # page.
                mediabox = page.get_mediabox()
                page_width_pt = float(mediabox[2]) - float(mediabox[0])
                page_height_pt = float(mediabox[3]) - float(mediabox[1])
                rotation = (360 - int(page.get_rotation())) % 360
                bitmap = page.render(scale=scale, rotation=rotation)
                full_img = bitmap.to_pil().convert("RGB")
            finally:
                page.close()
        finally:
            pdf.close()
        _PAGE_CACHE = (key, full_img, page_width_pt, page_height_pt)

    x0, y0, x1, y1 = _compute_crop_coords(bbox, page_width_pt, page_height_pt, scale, bleed_pt)
    return full_img.crop((x0, y0, x1, y1))


def crop_block_image(
    pdf_path: Path | str,
    page_num: int,
    bbox: BoundingBox | tuple[float, float, float, float],
    dpi: int = DEFAULT_SCALPEL_DPI,
    bleed_pt: float = DEFAULT_BLEED_PT,
    image_format: str = "PNG",
) -> str:
    """Crop a bounding box region from a PDF page into an optimized Base64 string."""
    cropped = crop_block_pil(pdf_path, page_num, bbox, dpi=dpi, bleed_pt=bleed_pt)
    buf = io.BytesIO()
    cropped.save(buf, format=image_format, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def crop_ir_block_image(
    pdf_path: Path | str,
    block: IRBlock,
    dpi: int = DEFAULT_SCALPEL_DPI,
    bleed_pt: float = DEFAULT_BLEED_PT,
    image_format: str = "PNG",
) -> str | None:
    """Crop an IRBlock's bounding box region into a Base64 string. Returns None if bbox invalid."""
    if block.bbox is None or block.bbox.page <= 0:
        return None
    return crop_block_image(
        pdf_path=pdf_path,
        page_num=block.bbox.page,
        bbox=block.bbox,
        dpi=dpi,
        bleed_pt=bleed_pt,
        image_format=image_format,
    )


def is_visual_scalpel_applicable(
    block: IRBlock,
    source_pdf_path: Path | str | None,
    qe_threshold_for_vision: float = 0.60,
) -> bool:
    """Evaluate whether an IRBlock qualifies for L4 multimodal visual repair.

    Triggers when:
    1. A valid source PDF file is available;
    2. The block has a non-zero-area bounding box and positive page number;
    3. The block is either a formula/table or has an mtqe_score below the vision threshold
       (severe structural defect, omission, or hallucination).
    """
    if source_pdf_path is None:
        return False

    pdf_p = Path(source_pdf_path)
    if not pdf_p.exists() or pdf_p.suffix.lower() != ".pdf":
        return False

    bbox = block.bbox
    if bbox is None or bbox.page <= 0:
        return False

    # Check non-empty area
    if abs(bbox.x1 - bbox.x0) <= 1.0 or abs(bbox.y1 - bbox.y0) <= 1.0:
        return False

    # Structural math/table blocks benefit immediately from visual context
    if block.block_type in (BlockType.FORMULA, BlockType.TABLE):
        return True

    # Low-scoring blocks (severe errors / omissions) trigger visual rescue
    if block.mtqe_score is not None and block.mtqe_score < qe_threshold_for_vision:
        return True

    # Check for metadata indicators (e.g. diagram text, formula markers)
    meta = getattr(block, "metadata", None) or getattr(block, "provenance", None) or {}
    return bool(meta.get("is_diagram_text") or meta.get("has_complex_symbols"))
