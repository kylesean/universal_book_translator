"""A: Coordinate resolution and bounding box normalization for scanned/OCR PDFs.

Bridges diverse coordinate systems into standard PDF bottom-left points:
1. Normalized [0, 1] or [0, 1000] coordinates (LayoutLM, PaddleOCR, MinerU, DocLayNet).
2. Image pixel space (top-left origin, Y downward).
3. PDF top-left vs bottom-left orientation and page rotation (90°, 180°, 270°).
4. Synthetic line-box generation for textless scanned pages lacking embedded text streams.

Zero-AGPL: 100% permissive (MIT/BSD/Apache-2.0). Never imports fitz or pymupdf.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class PageBBoxResolver:
    """Resolves arbitrary OCR/VLM bounding boxes to canonical PDF bottom-left points."""

    page_width: float
    page_height: float
    image_width: float | None = None
    image_height: float | None = None
    rotation: int = 0  # 0, 90, 180, 270

    def resolve_bbox(
        self,
        bbox: Sequence[float] | None,
        origin: str = "auto",
        coord_system: str = "auto",
    ) -> Rect | None:
        """Resolve arbitrary raw bbox to canonical PDF bottom-left points (x0, y0, x1, y1).

        coord_system can be:
        - "auto": automatic detection from this single box. Ambiguous when the
          box fits inside the page (a normalized-1000 box in an A4 page's
          top-left is indistinguishable from native points); prefer
          :meth:`infer_coord_system` once per page and pass the result.
        - "normalized_1": [0, 1] relative coordinates.
        - "normalized_1000": [0, 1000] integer-grid coordinates.
        - "image_pixel": pixel coordinates from rendered/scanned image.
        - "pdf_points": points directly on PDF canvas.

        origin can be:
        - "auto": top-left for normalized/pixel, bottom-left for native PDF.
        - "top-left": top-left origin (y=0 at top, y increases downwards).
        - "bottom-left": standard PDF origin (y=0 at bottom, y increases upwards).
        """
        if bbox is None or len(bbox) < 4:
            return None

        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            return None

        # Normalize ordering
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        # Reject a degenerate box in the source space: every mapping below is an
        # affine scale/reflection, so a zero-extent input can only produce a
        # zero-extent rect.
        if x0 >= x1 or y0 >= y1:
            return None

        pw, ph = self.page_width, self.page_height

        # 1. Normalized [0, 1] space
        if coord_system in ("normalized_1", "norm_1") or (
            coord_system == "auto" and x1 <= 1.05 and y1 <= 1.05
        ):
            rx0 = x0 * pw
            rx1 = x1 * pw
            ry0 = (1.0 - y1) * ph
            ry1 = (1.0 - y0) * ph

        # 2. Normalized [0, 1000] space (PaddleOCR / MinerU standard)
        elif coord_system in ("normalized_1000", "norm_1000", "paddle") or (
            coord_system == "auto" and (x1 <= 1005 and y1 <= 1005) and (x1 > pw or y1 > ph)
        ):
            rx0 = (x0 / 1000.0) * pw
            rx1 = (x1 / 1000.0) * pw
            ry0 = (1.0 - y1 / 1000.0) * ph
            ry1 = (1.0 - y0 / 1000.0) * ph

        # 3. Image pixel space
        elif coord_system == "image_pixel" or (
            coord_system == "auto"
            and self.image_width
            and self.image_height
            and (x1 > pw or y1 > ph)
        ):
            iw = self.image_width or pw
            ih = self.image_height or ph
            sx = pw / iw
            sy = ph / ih
            rx0 = x0 * sx
            rx1 = x1 * sx
            ry0 = ph - (y1 * sy)
            ry1 = ph - (y0 * sy)

        # 4. Explicit origin or point coordinates
        elif origin == "top-left":
            rx0 = x0
            rx1 = x1
            ry0 = ph - y1
            ry1 = ph - y0
        else:
            # Bottom-left (native PDF)
            rx0 = x0
            rx1 = x1
            ry0 = y0
            ry1 = y1

        # Bounds clamp
        rx0 = max(0.0, min(pw, rx0))
        rx1 = max(0.0, min(pw, rx1))
        ry0 = max(0.0, min(ph, ry0))
        ry1 = max(0.0, min(ph, ry1))

        if rx1 <= rx0 or ry1 <= ry0:
            return None

        return self._apply_rotation(rx0, ry0, rx1, ry1)

    @staticmethod
    def infer_coord_system(
        boxes: Sequence[Sequence[float] | None],
        page_width: float,
        page_height: float,
        image_width: float | None = None,
        image_height: float | None = None,
    ) -> str:
        """Infer ONE coordinate system for a whole page from the boxes on it.

        Per-box ``auto`` cannot separate a normalized-1000 box that happens to
        sit in the top-left of an A4 page (``x1 <= 595, y1 <= 842``) from a
        native PDF-point box: both fit the page. The page's *extremes*
        disambiguate -- a real OCR page has boxes near the right/bottom edge, so
        the largest coordinate reaches the grid maximum. Deciding once per page
        also keeps every box on that page in the same space; mixed per-box
        detection scaled only the small boxes, misplacing them by a factor.

        Returns ``"auto"`` when no usable box is present, so the caller falls
        back to :meth:`resolve_bbox`.
        """
        max_x = 0.0
        max_y = 0.0
        for box in boxes:
            if box is None or len(box) < 4:
                continue
            try:
                max_x = max(max_x, abs(float(box[0])), abs(float(box[2])))
                max_y = max(max_y, abs(float(box[1])), abs(float(box[3])))
            except (TypeError, ValueError):
                continue
        if max_x == 0.0 and max_y == 0.0:
            return "auto"
        m = max(max_x, max_y)
        page_max = max(page_width, page_height)
        if m <= 1.05:
            return "normalized_1"
        if (
            image_width
            and image_height
            and m > page_max
            and m <= max(image_width, image_height) * 1.05
        ):
            return "image_pixel"
        if m <= 1005.0 and m > page_max:
            return "normalized_1000"
        return "pdf_points"

    def _apply_rotation(self, x0: float, y0: float, x1: float, y1: float) -> Rect:
        """Apply page rotation if needed (PDF coordinate rotation)."""
        rot = self.rotation % 360
        if rot == 0:
            return (x0, y0, x1, y1)

        pw, ph = self.page_width, self.page_height
        if rot == 90:
            # 90 deg clockwise
            return (y0, pw - x1, y1, pw - x0)
        elif rot == 180:
            # 180 deg
            return (pw - x1, ph - y1, pw - x0, ph - y0)
        elif rot == 270:
            # 270 deg clockwise
            return (ph - y1, x0, ph - y0, x1)

        return (x0, y0, x1, y1)


def synthesize_line_boxes_for_blocks(
    blocks: Sequence[IRBlock],
    page_size: tuple[float, float],
    image_size: tuple[float, float] | None = None,
) -> list[LineBox]:
    """Synthesize LineBox objects for blocks on scanned pages lacking text streams.

    If block has provenance.vlm_lines, uses them.
    Otherwise, segments block.bbox into proportional horizontal line bands.
    """
    pw, ph = page_size
    img_w, img_h = image_size if image_size else (None, None)
    resolver = PageBBoxResolver(
        page_width=pw,
        page_height=ph,
        image_width=img_w,
        image_height=img_h,
    )

    lines: list[LineBox] = []

    for block in blocks:
        # Check if vlm_lines are already present in provenance
        vlm_members = (getattr(block, "provenance", None) or {}).get("vlm_lines") or []
        if vlm_members:
            for member in vlm_members:
                try:
                    text = str(member.get("text", "")).strip()
                    raw_box = member.get("box", ())
                    box = resolver.resolve_bbox(raw_box)
                    if text and box is not None:
                        lines.append(LineBox(text, box))
                except Exception:
                    continue
            continue

        # Otherwise, synthesize from block.bbox and source_text
        if block.bbox is None:
            continue

        resolved_bbox = resolver.resolve_bbox(
            [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1],
            coord_system="pdf_points",
            origin="bottom-left",
        )
        if resolved_bbox is None:
            continue

        bx0, by0, bx1, by1 = resolved_bbox
        bh = max(by1 - by0, 1.0)

        raw_text = (block.source_text or "").strip()
        if not raw_text:
            continue

        text_lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
        if not text_lines:
            text_lines = [raw_text]

        n_lines = len(text_lines)
        if n_lines == 1:
            lines.append(LineBox(text_lines[0], (bx0, by0, bx1, by1)))
        else:
            # Segment block bbox into n_lines horizontal slices from top to bottom
            line_h = bh / n_lines
            for i, line_text in enumerate(text_lines):
                # i=0 is top line (highest y in PDF points)
                ly1 = by1 - i * line_h
                ly0 = max(by0, ly1 - line_h)
                lines.append(LineBox(line_text, (bx0, ly0, bx1, ly1)))

    return lines


__all__ = [
    "PageBBoxResolver",
    "Rect",
    "synthesize_line_boxes_for_blocks",
]
