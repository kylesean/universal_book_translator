"""``DocumentStyleProfile`` / ``PageStyleProfile``: visual style cloned from the source.

The heading color and body size a document really uses are measured, not guessed:
``TypstDocumentRenderer`` reads these profiles so a translated page keeps its
source's typography instead of a hard-coded default
(G3, ARCHITECTURE.md § The render path).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.core.ir.models import BlockType, IRBlock

#: Fallback heading color for a document whose headings measure *no* color:
#: every heading carried black (or carried no style at all). Neutral black is
#: the only honest reading of that measurement — painting such headings a brand
#: accent recolors a document whose own headings are black, which is exactly
#: what shipped on the ATC'25 replay (every section heading came out branded
#: blue on a paper whose headings are black). The accent stays available to
#: callers that measure a real color; it is not a default anymore.
DEFAULT_PRIMARY_HEADING_HEX = "#000000"
_BLACK_HEX_SET = {"#000000", "#000", "#00000000", "black", "#111111", "#222222"}


@dataclass(frozen=True, slots=True)
class PageStyleProfile:
    page_no: int
    primary_heading_hex: str | None = None
    body_size_pt: float = 10.0
    body_leading_pt: float = 13.33
    is_dark_header: bool = False
    dominant_heading_font: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentStyleProfile:
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
        """The subset written into ``manifest.metadata``; per-page profiles stay out."""
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
    """Book-wide style, measured over every block.

    Heading color is the most frequent *non-black* measurement and body size the
    median: one tinted heading or one oversized drop cap must not define how the
    document looks, and a paper whose headings are black must not come out branded
    (see ``DEFAULT_PRIMARY_HEADING_HEX`` for that incident).
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
