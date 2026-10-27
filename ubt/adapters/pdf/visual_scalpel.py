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
#: Margin grown around a box when the crop is a *prompt* — the L4 repair path
#: reads it once to answer a question and is told which region it is looking at,
#: so a little surrounding context is harmless and often useful.
DEFAULT_BLEED_PT: float = 4.0

#: Margin for a crop that becomes a *page* (a delivered asset the Typst renderer
#: places). It must be zero-ish: the renderer rebuilds the page from the IR and
#: has no source canvas, so any source ink inside the PNG is drawn onto the
#: reader's page as part of the picture. A display equation's bbox sits as close
#: as 0.15pt to the next text line in the two-column layouts this project
#: targets, and the old blanket 4pt sliced that line in — the delivered PDF
#: showed each formula welded to a half-cut English clause, while the reflowed
#: translation of the same sentence sat directly below it, looking like a
#: mistranslated fragment.
DELIVERED_BLEED_PT: float = 1.0

#: Tag folded into delivered asset filenames *and* (through
#: :func:`ubt.adapters.pdf.analysis_cache.docling_identity`) into the block
#: cache key. Both halves are load-bearing:
#:
#: * the filename, because the asset dir is keyed by the source PDF's sha alone,
#:   so a re-run would otherwise replay the previous crop from the
#:   ``not asset_path.exists()`` guard;
#: * the cache key, because ``cached_blocks`` stores the finished IMAGE blocks
#:   (whose ``source_text`` *is* the asset path). Invalidate only the file and a
#:   cached run keeps pointing at the old name; invalidate only the cache and the
#:   stale file is silently reused. A crop-policy change needs both.
#: ``tight`` -> ``tight2`` when the stray-band trim was added: a ``tight``
#: asset was cut with the old rule and still carries the ghost line, so reusing
#: it by name would hide the fix exactly the way the original ``-tight`` rename
#: existed to prevent.
DELIVERED_CROP_POLICY_TAG: str = "tight2"

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
FIGURE_ASSET_DPI_TAG = f"dpi{FIGURE_ASSET_DPI}-{DELIVERED_CROP_POLICY_TAG}"


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


#: Gap left between a delivered crop's edge and a neighbouring text line's ink.
#: One raster pixel at 300 DPI is 0.24pt. A text rect is the *layout* box, and
#: glyph ink can overshoot it by a fraction of a point, so the clearance used is
#: a full point: the margin shrinks to the distance left, and a sub-pixel
#: overshoot would still show the neighbour's glyph tops in the delivered image
#: (measured on Eq. (3.1): a 0.83pt clearance with a 0.2pt guard still shipped a
#: sliver of the line below). Shrinking the margin never enters the block's own
#: region, so a generous guard cannot clip the block itself.
_CLEARANCE_EPS_PT: float = 1.0


def delivered_crop_insets(
    pdf_path: Path | str,
    page_num: int,
    bbox: BoundingBox | tuple[float, float, float, float],
    *,
    cap: float = DELIVERED_BLEED_PT,
) -> dict[str, float]:
    """Per-side margin a *delivered* crop may grow without catching a neighbour.

    A delivered crop is placed on a rebuilt page, so its margin is not free
    space — it is a window onto the source page that can swallow the adjacent
    paragraph. The extractor's bbox is the region its layout model assigned to
    the block and already bounds that block's own ink (measured on the reference
    chapter: a ``bleed_pt=0`` crop of an equation is pixel-identical to the inner
    region of a padded one, apart from the padding itself). The safe margin is
    therefore not a constant — it is the *clearance* to the nearest neighbouring
    text run on each side.

    Each side is measured only against runs that overlap the block on the
    perpendicular axis, so a line that merely shares a page row with the block
    (a wide equation's number, a two-column neighbour) does not shrink a side it
    cannot reach.

    Returns ``{"left", "right", "bottom", "top"}`` in points, each in
    ``[0, cap]``. All four are ``cap`` when nothing else is near, which is the
    common case (74 of 96 delivered crops on the reference chapter keep the
    full margin; the other 22 are the ones that were shipping a sliced
    neighbour).
    """
    from ubt.adapters.pdf.textgeom import extract_text_rects

    if isinstance(bbox, BoundingBox):
        bx0, by0, bx1, by1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    else:
        bx0, by0, bx1, by1 = bbox
    low_x, high_x = min(bx0, bx1), max(bx0, bx1)
    low_y, high_y = min(by0, by1), max(by0, by1)

    try:
        rects = extract_text_rects(Path(pdf_path), page_num)
    except Exception as exc:  # noqa: BLE001 - no geometry means the safe answer is "no margin"
        # Fail closed: a crop with no margin is faithfully the block; one grown
        # to the cap might carry a neighbour's sentence onto the page.
        logger.debug("delivered crop insets unavailable for page %s: %s", page_num, exc)
        return {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}

    insets = {"left": cap, "right": cap, "bottom": cap, "top": cap}
    for x0, y0, x1, y1 in rects:
        if x1 <= x0 or y1 <= y0:
            continue
        if x1 > low_x and x0 < high_x:
            if y1 <= low_y:
                insets["bottom"] = min(insets["bottom"], max(0.0, low_y - y1 - _CLEARANCE_EPS_PT))
            if y0 >= high_y:
                insets["top"] = min(insets["top"], max(0.0, y0 - high_y - _CLEARANCE_EPS_PT))
        if y1 > low_y and y0 < high_y:
            if x1 <= low_x:
                insets["left"] = min(insets["left"], max(0.0, low_x - x1 - _CLEARANCE_EPS_PT))
            if x0 >= high_x:
                insets["right"] = min(insets["right"], max(0.0, x0 - high_x - _CLEARANCE_EPS_PT))
    return insets


#: Blank pixel rows that separate a stray band from the block's own content.
#: One at 300 DPI is 0.24pt; the ghost under Eq. (3.5) sat 40 rows of whitespace
#: below the equation, while a formula's own glyph bands are flush against each
#: other (0 rows). Four rows is ~1pt: enough to absorb antialiasing, far below
#: the gap a real line of text leaves.
_STRAY_BAND_GAP_PX: int = 4


def _trim_detached_bands(
    image: PILImage.Image, *, gap_px: int = _STRAY_BAND_GAP_PX
) -> PILImage.Image:
    """Drop stray ink bands at an edge that a blank gap separates from the body.

    A delivered crop is a window onto the source page, and the extractor's box
    sometimes reaches into the line below by a point or two. Withholding the
    margin cannot help there — the ink is inside the box, not in its padding —
    so the delivered page showed the equation with the next line's glyph tops
    welded to its underside, reading as a ghost.

    Geometry cannot decide this: the intruding line and the formula's own glyphs
    have the same height and both overshoot the bbox, so a text-rect test either
    cuts every formula or none. Pixels can. A neighbour forms a *separate* band
    of ink with whitespace between it and the block; the block's own glyphs are
    continuous. Measured on the reference chapter: the ghost under Eq. (3.5) is
    separated by 40 blank rows, while Eq. (A.15a)'s own overshooting glyphs are
    flush (0 rows).

    Only the outermost band on each side is considered, and only while the body
    is non-empty, so a crop that is one solid band is returned untouched.
    """
    # PIL only: ``numpy`` lives in the ``ocr``/``academic`` extras, and this
    # module sits on the base render path — importing it here would make a
    # plain `pip install ubt` fail to cut a figure.
    grey = image.convert("L")
    width, _ = grey.size
    raw = grey.tobytes()
    # ``tobytes`` is one byte per pixel for mode "L" (row-major). A row counts
    # as inked when any pixel is darker than 200, the same threshold the rest of
    # the visual gate uses for "ink" against a white page.
    rows = [min(raw[y * width : (y + 1) * width]) < 200 for y in range(grey.size[1])]
    if not any(rows) or all(rows):
        return image

    def _trim(axis_rows: list[bool], from_top: bool) -> int:
        """How many rows to drop from this edge, keeping the body intact.

        The body is the *widest* run of ink, not the first one: a stray band is
        a sliver by definition (9 rows against the equation's 133 on Eq. (3.5)),
        whereas the block can legitimately open with a small band of its own
        (an accent, a superscript). Only a band at the extreme edge, separated
        from the body by a blank gap, is dropped.
        """
        sequence = list(axis_rows if from_top else axis_rows[::-1])
        # Leading blank rows are padding, not a band.
        start = 0
        while start < len(sequence) and not sequence[start]:
            start += 1
        if start >= len(sequence):
            return 0
        # Band boundaries, in trimmed coordinates.
        bands: list[tuple[int, int]] = []
        cursor = start
        while cursor < len(sequence):
            if not sequence[cursor]:
                cursor += 1
                continue
            band_start = cursor
            while cursor < len(sequence) and sequence[cursor]:
                cursor += 1
            bands.append((band_start, cursor))
        if len(bands) < 2:
            return 0
        body_height = max(end - begin for begin, end in bands)
        # The edge band must be both detached and smaller than the body.
        edge_begin, edge_end = bands[0]
        body_begin = next(begin for begin, end in bands if end - begin == body_height)
        if edge_end - edge_begin >= body_height:
            return 0
        gap = body_begin - edge_end
        if gap < gap_px:
            return 0
        return body_begin

    top = _trim(rows, True)
    bottom = _trim(rows, False)
    width, height = image.size
    if top:
        image = image.crop((0, top, width, height))
        height = image.size[1]
    if bottom and bottom < height:
        image = image.crop((0, 0, width, height - bottom))
    return image


@pdfium_serialized
def delivered_crop_pil(
    pdf_path: Path | str,
    page_num: int,
    bbox: BoundingBox | tuple[float, float, float, float],
    dpi: int = FIGURE_ASSET_DPI,
    bleed_pt: float = DELIVERED_BLEED_PT,
) -> PILImage.Image:
    """Crop a block for *delivery*: the block's own region, no neighbour's ink.

    The single entry point for an asset that will be placed on a page. Use
    :func:`crop_block_pil` only for a crop a model reads — its bleed is the
    prompt margin and is not bounded by the page's later geometry.

    ``bleed_pt`` caps each side; the effective margin per side is the smaller of
    the cap and the clearance to the nearest neighbouring text run.
    """
    insets = delivered_crop_insets(pdf_path, page_num, bbox, cap=bleed_pt)
    # Compose the per-side margins by cropping each side separately: the shared
    # ``_compute_crop_coords`` takes one bleed for all four, and a crop grown on
    # the tight side is exactly what sweeps in the neighbour's line.
    padded = crop_block_pil(pdf_path, page_num, bbox, dpi=dpi, bleed_pt=bleed_pt)
    scale = dpi / 72.0
    pad_px = int(round(bleed_pt * scale))
    pixels = {
        "left": int(round(insets["left"] * scale)),
        "right": int(round(insets["right"] * scale)),
        "bottom": int(round(insets["bottom"] * scale)),
        "top": int(round(insets["top"] * scale)),
    }
    width, height = padded.size
    # ``crop_block_pil`` clamps at the page edge, so the actual padding on a side
    # can be smaller than ``pad_px``; never crop past the real image.
    left = min(max(0, pad_px - pixels["left"]), width)
    right = min(max(0, pad_px - pixels["right"]), width - left)
    top = min(max(0, pad_px - pixels["top"]), height)
    bottom = min(max(0, pad_px - pixels["bottom"]), height - top)
    if left or right or top or bottom:
        padded = padded.crop((left, top, width - right, height - bottom))
    return _trim_detached_bands(padded)


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
