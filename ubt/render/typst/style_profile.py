"""DocumentStyleProfile and PageStyleProfile: Visual style cloning for Typst rendering.

Clones visual styling (primary heading color, body font size baseline, and per-page metrics)
from source document IR blocks and geometry. Resolves G3 (docs/design/adaptive_delivery_prd.md §10.2).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.core.ir.models import BlockType, IRBlock

DEFAULT_PRIMARY_HEADING_HEX = "#0078ae"
_BLACK_HEX_SET = {"#000000", "#000", "#00000000", "black", "#111111", "#222222"}


@dataclass(frozen=True, slots=True)
class PageStyleProfile:
    """Visual style profile for an individual page."""

    page_no: int
    primary_heading_hex: str | None = None
    body_size_pt: float = 10.0
    body_leading_pt: float = 13.33
    is_dark_header: bool = False
    dominant_heading_font: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentStyleProfile:
    """Extracted book-wide visual style and typographic profile."""

    primary_heading_hex: str = DEFAULT_PRIMARY_HEADING_HEX
    secondary_heading_hex: str = DEFAULT_PRIMARY_HEADING_HEX
    body_size_pt: float = 10.0
    body_leading_pt: float = 13.33
    page_width_pt: float = 595.28
    page_height_pt: float = 841.89
    columns: int = 1
    has_colored_headings: bool = False
    per_page_profiles: dict[int, PageStyleProfile] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize profile for manifest.metadata."""
        return {
            "primary_heading_hex": self.primary_heading_hex,
            "secondary_heading_hex": self.secondary_heading_hex,
            "body_size_pt": self.body_size_pt,
            "body_leading_pt": self.body_leading_pt,
            "page_width_pt": self.page_width_pt,
            "page_height_pt": self.page_height_pt,
            "columns": self.columns,
            "has_colored_headings": self.has_colored_headings,
        }


def extract_document_style_profile(
    blocks: Sequence[IRBlock],
    *,
    source_path: Path | None = None,
    page_dims: Sequence[tuple[float, float]] = (),
    fallback_heading_color: str = DEFAULT_PRIMARY_HEADING_HEX,
) -> DocumentStyleProfile:
    """Extract book-wide visual style profile from document blocks.

    1. Dominant heading color: identifies the most frequent non-black color across all
       headings in the document.
    2. Body text size: calculates median measured body font size.
    3. Document geometry: determines mode page dimensions and columns.
    """
    heading_colors: list[str] = []
    body_sizes: list[float] = []
    colors_by_page: dict[int, list[str]] = {}
    sizes_by_page: dict[int, list[float]] = {}

    for b in blocks:
        page_no = b.bbox.page if b.bbox and b.bbox.page else 1
        if b.block_type == BlockType.HEADING and b.style:
            c = getattr(b.style, "color_hex", None) or getattr(b.style, "text_color_hex", None)
            if c:
                c_str = str(c).strip()
                if c_str and c_str.lower() not in _BLACK_HEX_SET:
                    heading_colors.append(c_str)
                    colors_by_page.setdefault(page_no, []).append(c_str)
        elif b.block_type == BlockType.NARRATIVE and b.style:
            fsz = getattr(b.style, "font_size", None)
            if fsz is not None and 6.0 <= float(fsz) <= 20.0:
                body_sizes.append(float(fsz))
                sizes_by_page.setdefault(page_no, []).append(float(fsz))

    has_colored = len(heading_colors) > 0
    if has_colored:
        primary_color = Counter(heading_colors).most_common(1)[0][0]
    else:
        primary_color = fallback_heading_color

    body_size = round(statistics.median(body_sizes), 1) if body_sizes else 10.0
    body_leading = round(body_size * 1.333, 2)

    pw, ph = (595.28, 841.89)
    if page_dims:
        widths = [d[0] for d in page_dims if d[0] > 0]
        heights = [d[1] for d in page_dims if d[1] > 0]
        if widths and heights:
            pw = round(statistics.median(widths), 2)
            ph = round(statistics.median(heights), 2)

    per_page: dict[int, PageStyleProfile] = {}
    all_pages = set(colors_by_page.keys()) | set(sizes_by_page.keys())
    for p in all_pages:
        p_color = Counter(colors_by_page[p]).most_common(1)[0][0] if p in colors_by_page else None
        p_size = round(statistics.median(sizes_by_page[p]), 1) if p in sizes_by_page else body_size
        per_page[p] = PageStyleProfile(
            page_no=p,
            primary_heading_hex=p_color,
            body_size_pt=p_size,
            body_leading_pt=round(p_size * 1.333, 2),
        )

    return DocumentStyleProfile(
        primary_heading_hex=primary_color,
        secondary_heading_hex=primary_color,
        body_size_pt=body_size,
        body_leading_pt=body_leading,
        page_width_pt=pw,
        page_height_pt=ph,
        has_colored_headings=has_colored,
        per_page_profiles=per_page,
    )
