"""Sample the colour behind a region, so a fallback mask blends into the page.

The micro-mask is the fallback path: when the source text under a region cannot
be stripped (a page-shared form, an unreadable resource, a strip that aborted),
the region is painted over and the target drawn on top. That paint used to be a
hardcoded pure white. Most books are not printed on white -- cream stock, a
tinted band, a shaded table row, a scanned page's off-white -- and there a white
rectangle is a visible scar across every replaced line.

The colour is sampled from the page instead: rasterize the page once at a low
dpi and read the median pixel of each region. The median, not the mean: the
region is mostly background with the source text's ink on top, so the middle
value of each channel falls on the background side of the ink/paper split,
while a mean would be dragged toward the ink by every glyph. A median also
follows the page's polarity -- light text on a dark band comes back dark --
which a fixed percentile would get exactly backwards.

Sampling is best-effort. Anything that fails (no rasterizer, an empty crop, a
region off the page) yields ``None`` for that region and the caller falls back
to white, because a visible mask is still better than source text left legible
under the translation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

#: The fill a region gets when nothing can be sampled -- the historical colour.
DEFAULT_MASK_BACKGROUND: tuple[float, float, float] = (1.0, 1.0, 1.0)

#: Sampling resolution. A median needs a few hundred pixels per region, not a
#: sharp image: at 48 dpi an A4 page is 397x561 px (~0.7 MB), and the raster is
#: reused for every other region on the same page.
_SAMPLE_DPI = 48
#: Trim this fraction of a region's shorter side (capped) before sampling. A
#: border rule drawn on the region's own edge -- a boxed table cell, a framed
#: callout -- would otherwise pull the median toward the rule's colour.
_INSET_RATIO = 0.15
_INSET_MAX_PT = 2.0


def sample_backgrounds(
    pdf_path: str | Path,
    page_no: int,
    rects: Sequence[tuple[float, float, float, float]],
) -> list[tuple[float, float, float] | None]:
    """The colour behind each rect, in order (``None`` when it cannot be sampled).

    ``rects`` are PDF points in the page's *unrotated* user space -- the frame
    the overlay boxes are extracted in -- and the rasterizer is asked for the
    same frame, so a ``/Rotate`` page samples the region the mask will paint.
    """
    return [_sample_region(pdf_path, page_no, rect) for rect in rects]


def _sample_region(
    pdf_path: str | Path,
    page_no: int,
    rect: tuple[float, float, float, float],
) -> tuple[float, float, float] | None:
    x0, y0, x1, y1 = rect
    inset = min(min(x1 - x0, y1 - y0) * _INSET_RATIO, _INSET_MAX_PT)
    inner = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    # Imported here, not at module scope: the render tree is importable without
    # the PDF stack, and this path is a fallback that most runs never take.
    from ubt.adapters.pdf.visual_scalpel import crop_block_pil  # noqa: PLC0415

    try:
        image = crop_block_pil(pdf_path, page_no, inner, dpi=_SAMPLE_DPI, bleed_pt=0.0)
    except Exception as exc:  # pdfium missing, empty/off-page crop, bad page
        logger.debug("background sample p%s %s failed: %s", page_no, inner, exc)
        return None
    if image.width < 1 or image.height < 1:
        return None
    return _median_rgb(image.convert("RGB"))


def _median_rgb(image: PILImage.Image) -> tuple[float, float, float]:
    """The per-channel median of an RGB image, as PDF 0-1 components.

    Read off the histogram: the smallest level whose cumulative count is more
    than half the pixels. On an even split that is the upper of the two middle
    values, which is all a background estimate needs; averaging them would
    invent a colour that is on the page nowhere.
    """
    middle = (image.width * image.height) // 2
    components: list[float] = []
    for band in (0, 1, 2):
        seen = 0
        value = 0
        for level, count in enumerate(image.getchannel(band).histogram()):
            seen += count
            if seen > middle:
                value = level
                break
        components.append(round(value / 255.0, 3))
    return (components[0], components[1], components[2])
