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

#: Resolution for *delivered* figure/formula/code crops (the PNGs the rebuilt
#: page places). The VLM path above reads a crop once to answer a prompt, but a
#: delivered asset is zoomed and printed: source figures in born-digital papers
#: are vector drawings, so the crop is the only raster that will ever exist and
#: it must meet the 300 DPI print standard. At the old 150 DPI every figure in
#: the reference corpus landed at ~150 ppi as placed — visibly soft on zoom.
#: Memory is bounded: the page raster cache holds exactly one page (~24 MB at
#: this DPI) and is replaced as the walk advances.
FIGURE_ASSET_DPI: int = 300
#: Tag folded into delivered asset filenames. The asset dir is keyed by the
#: source file's sha alone, so an old-DPI crop with the same name would be
#: reused forever by the ``not asset_path.exists()`` guards; naming the DPI in
#: the file makes a resolution change cut new files instead of replaying old
#: ones (and leaves the old files harmless).
FIGURE_ASSET_DPI_TAG = f"dpi{FIGURE_ASSET_DPI}"


def _compute_crop_coords(
    bbox: BoundingBox | tuple[float, float, float, float],
    page_width_pt: float,
    page_height_pt: float,
    scale: float,
    bleed_pt: float = DEFAULT_BLEED_PT,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> tuple[int, int, int, int]:
    """Convert PDF coordinates (origin bottom-left) to raster image coordinates (origin top-left).

    ``origin_x``/``origin_y`` are the *rendered* box's lower-left corner — the
    page's CropBox (which pdfium/pdf_oxide rasterize), falling back to the
    MediaBox when no CropBox is set. The raster covers only that box, so a
    user-space point maps to pixel ``(point - origin) * scale``; a box whose
    rendered origin is not (0, 0) (a cropped or imposed page) would otherwise be
    cropped shifted by that offset — the wrong region, and the further from the
    origin, the further off.
    """
    if isinstance(bbox, BoundingBox):
        bx0, by0, bx1, by1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    else:
        bx0, by0, bx1, by1 = bbox

    min_x = min(bx0, bx1) - origin_x
    max_x = max(bx0, bx1) - origin_x
    min_y = min(by0, by1) - origin_y
    max_y = max(by0, by1) - origin_y

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
#: ``(key, raster, width_pt, height_pt, origin_x, origin_y)`` — the last two are
#: the MediaBox lower-left, needed to map user-space rects onto the raster.
_PAGE_CACHE: (
    tuple[tuple[str, int, int, int, int], PILImage.Image, float, float, float, float] | None
) = None


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
        _, full_img, page_width_pt, page_height_pt, origin_x, origin_y = _PAGE_CACHE
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
                #
                # The raster covers the page's *CropBox*, not its MediaBox:
                # pdfium (and pdf_oxide) render the CropBox, and ``get_cropbox``
                # falls back to the MediaBox when none is set, so on the common
                # zero-crop page this is identical. Using the MediaBox on a
                # page that IS cropped (print/imposed PDFs) made every y
                # coordinate wrong — the raster was shorter than the box the
                # math assumed, so a bbox in the visible area cropped a blank
                # strip below the image (fed to the repair VLM, and by the
                # figure-asset fallback, written into the delivered PDF).
                cropbox = page.get_cropbox()
                origin_x = float(cropbox[0])
                origin_y = float(cropbox[1])
                page_width_pt = float(cropbox[2]) - origin_x
                page_height_pt = float(cropbox[3]) - origin_y
                rotation = (360 - int(page.get_rotation())) % 360
                bitmap = page.render(scale=scale, rotation=rotation)
                full_img = bitmap.to_pil().convert("RGB")
            finally:
                page.close()
        finally:
            pdf.close()
        _PAGE_CACHE = (key, full_img, page_width_pt, page_height_pt, origin_x, origin_y)

    x0, y0, x1, y1 = _compute_crop_coords(
        bbox,
        page_width_pt,
        page_height_pt,
        scale,
        bleed_pt,
        origin_x=origin_x,
        origin_y=origin_y,
    )
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
    vision_threshold: float = 0.60,
) -> bool:
    """Evaluate whether an IRBlock qualifies for L4 multimodal visual repair.

    Triggers when:
    1. A valid source PDF file is available;
    2. The block has a non-zero-area bounding box and positive page number;
    3. The block is either a formula/table or has an mtqe_score below the vision threshold
       (severe structural defect, omission, or hallucination).

    The threshold keyword is ``vision_threshold``, matching the ports bridge
    (:func:`ubt.core.ports.is_visual_scalpel_applicable`) that calls this through
    the SPI registry. The two names drifted (``qe_threshold_for_vision`` here,
    ``vision_threshold`` there), so every real call raised ``TypeError`` and the
    L4 visual-repair path never ran.
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
    if block.mtqe_score is not None and block.mtqe_score < vision_threshold:
        return True

    # Check for metadata indicators (e.g. diagram text, formula markers).
    # ``IRBlock.provenance`` is a pydantic ``BlockProvenance`` (an attribute
    # object, not a ``dict``) with ``extra="allow"``, so .get() on it raises;
    # read the two indicators with ``getattr`` so both a mapping and the typed
    # object work.
    meta = getattr(block, "metadata", None) or getattr(block, "provenance", None)
    if isinstance(meta, dict):
        return bool(meta.get("is_diagram_text") or meta.get("has_complex_symbols"))
    return bool(
        getattr(meta, "is_diagram_text", False) or getattr(meta, "has_complex_symbols", False)
    )
