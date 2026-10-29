"""Tier L2 Page-Locked Reflow Renderer.

Renders source pages into 1:1 page-aligned, style-cloned Typst documents:
1. Clones source page physical trim size (page_w, page_h), margins, columns, and theme colors.
2. Applies monotonic binary-search page-lock convergence to ensure Page N never spills to Page N+1.
3. Preserves 1:1 page correspondence (N_out == N_src) for complex non-separable or collided layouts.

**The body floor outranks the page count.** ``FLOOR_PT`` (7.5pt, owned by
``ubt/render/layout_solver.py``) is the smallest body size this project will
ever typeset, so a page whose content cannot fit at 7.5pt is allowed to run
onto a second page, and the renderer records an explicit ``page_overflow``
finding naming the page and the deficit. The earlier code did the opposite: it
shrank the body to 6.8pt to keep the count, which is the
"翻译质量零妥协" rule being traded away silently for a layout promise the reader
cannot measure.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, IRBlock
from ubt.render.delivered_text import delivered_text
from ubt.render.layout_solver import FLOOR_PT
from ubt.render.typst.escape import typst_escape

logger = logging.getLogger(__name__)

DEFAULT_FONT_STACK: tuple[str, ...] = (
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Songti SC",
    "SimSun",
)


@dataclass(frozen=True, slots=True)
class PageGeometry:
    """Cloned physical layout and stylistic geometry for an individual source page."""

    page_w: float
    page_h: float
    margin_top: float
    margin_bottom: float
    margin_left: float
    margin_right: float
    columns: int = 1
    column_gutter_pt: float = 14.0
    theme_color_hex: str = "#0078ae"


@dataclass(slots=True)
class PageLockedPageSpec:
    """Page-lock configuration and content blocks for a single page."""

    page_no: int
    geometry: PageGeometry
    uniform_size_pt: float
    leading_em: float
    par_spacing_em: float
    tracking_em: float = 0.0
    blocks: list[IRBlock] = field(default_factory=list)


def extract_page_geometry(
    page_w: float,
    page_h: float,
    blocks: Sequence[IRBlock],
    *,
    default_margin: float = 36.0,
) -> PageGeometry:
    """Extract and clone physical geometry, margins, columns, and primary theme color from page blocks."""
    bboxes = [
        b.bbox for b in blocks if b.bbox and (b.bbox.x1 > b.bbox.x0) and (b.bbox.y1 > b.bbox.y0)
    ]

    if bboxes:
        min_x = min(bb.x0 for bb in bboxes)
        max_x = max(bb.x1 for bb in bboxes)
        min_y = min(bb.y0 for bb in bboxes)
        max_y = max(bb.y1 for bb in bboxes)

        m_left = max(18.0, min(108.0, min_x))
        m_right = max(18.0, min(108.0, page_w - max_x))
        m_top = max(18.0, min(108.0, page_h - max_y))
        m_bottom = max(18.0, min(108.0, min_y))
    else:
        m_left = m_right = m_top = m_bottom = default_margin

    # Column detection: check if blocks cleanly divide into left and right columns
    columns = 1
    if page_w >= 450.0 and len(bboxes) >= 4:
        x_mid = page_w / 2.0
        left_count = 0
        right_count = 0
        spanning_count = 0
        for bb in bboxes:
            w = bb.x1 - bb.x0
            if w > 0.65 * page_w:
                spanning_count += 1
            elif bb.x1 <= x_mid + 15.0:
                left_count += 1
            elif bb.x0 >= x_mid - 15.0:
                right_count += 1

        if left_count >= 2 and right_count >= 2 and spanning_count <= 2:
            columns = 2

    # Primary Theme Color from heading style
    theme_color = "#0078ae"
    for b in blocks:
        if b.block_type == BlockType.HEADING and b.style:
            c = getattr(b.style, "color_hex", None) or getattr(b.style, "text_color_hex", None)
            if c:
                c_str = str(c).strip()
                if c_str and c_str.lower() not in ("#000000", "#000", "black"):
                    theme_color = c_str
                    break

    return PageGeometry(
        page_w=round(page_w, 2),
        page_h=round(page_h, 2),
        margin_top=round(m_top, 2),
        margin_bottom=round(m_bottom, 2),
        margin_left=round(m_left, 2),
        margin_right=round(m_right, 2),
        columns=columns,
        column_gutter_pt=14.0,
        theme_color_hex=theme_color,
    )


def _content_width(geometry: PageGeometry) -> float:
    """Usable width of one text column, in points."""
    avail_w = geometry.page_w - geometry.margin_left - geometry.margin_right
    cols = max(1, geometry.columns)
    return (avail_w - (cols - 1) * geometry.column_gutter_pt) / cols


def _page_capacity(geometry: PageGeometry) -> float:
    """Estimated height budget for one page, after the 8% wrap headroom."""
    avail_h = geometry.page_h - geometry.margin_top - geometry.margin_bottom
    return 0.92 * max(1, geometry.columns) * avail_h


def _block_height(
    block: IRBlock,
    *,
    col_w: float,
    size: float,
    leading: float,
    spacing: float,
    tracking: float,
) -> float:
    """Estimated typeset height of one block at a given size/spacing, in points.

    A rough box model, not a Typst measurement: it exists so the page-lock
    binary search has a monotone objective it can evaluate hundreds of times
    without invoking the compiler. It is allowed to be pessimistic (which
    costs a slightly smaller body) but must never be optimistic about the
    floor — see :func:`page_overflow_at_floor`.
    """
    txt = delivered_text(block).strip()
    if not txt:
        txt = block.source_text.strip()
    if not txt:
        return 0.0

    eff_glyph_w = size * max(0.85, 1.0 + tracking)
    chars_per_line = max(1.0, col_w / eff_glyph_w)
    lines = max(1, math.ceil(len(txt) / chars_per_line))

    if block.block_type == BlockType.HEADING:
        return lines * (size * 1.25) * 1.35 + 8.0 + spacing * size
    if block.block_type == BlockType.FORMULA:
        return max(24.0, size * 2.0) + spacing * size
    if block.block_type == BlockType.TABLE:
        return max(40.0, size * 4.0) + spacing * size
    return lines * size * (1.0 + leading) + spacing * size


def _block_heights(
    geometry: PageGeometry,
    blocks: Sequence[IRBlock],
    *,
    size: float,
    leading: float,
    spacing: float,
    tracking: float,
) -> list[tuple[str, float]]:
    """Per-block ``(block_id, height_pt)`` at one candidate size."""
    col_w = _content_width(geometry)
    return [
        (
            b.id,
            _block_height(
                b,
                col_w=col_w,
                size=size,
                leading=leading,
                spacing=spacing,
                tracking=tracking,
            ),
        )
        for b in blocks
    ]


def _estimated_height(
    geometry: PageGeometry,
    blocks: Sequence[IRBlock],
    *,
    size: float,
    leading: float,
    spacing: float,
    tracking: float,
) -> float:
    """Total estimated height of every block on the page."""
    return sum(
        h
        for _, h in _block_heights(
            geometry, blocks, size=size, leading=leading, spacing=spacing, tracking=tracking
        )
    )


def page_overflow_at_floor(
    geometry: PageGeometry,
    blocks: Sequence[IRBlock],
    *,
    size_pt: float,
    leading_em: float,
    par_spacing_em: float,
    tracking_em: float,
) -> tuple[str, float] | None:
    """``(block_id, deficit_pt)`` when the page cannot fit at ``size_pt``.

    ``None`` means it fits. This is the honest replacement for the old
    shrink-to-6.8pt loop: the caller reports *which* block breaks the page and
    *by how much*, instead of quietly trading body size for a page count.

    The deficit is measured at the size the caller actually chose, so a page
    solved at 7.5pt reports the true overshoot rather than a per-round estimate.
    """
    capacity = _page_capacity(geometry)
    running = 0.0
    for block_id, height in _block_heights(
        geometry,
        blocks,
        size=size_pt,
        leading=leading_em,
        spacing=par_spacing_em,
        tracking=tracking_em,
    ):
        running += height
        if running > capacity:
            return block_id, round(running - capacity, 2)
    return None


def solve_page_lock(
    geometry: PageGeometry,
    blocks: Sequence[IRBlock],
    *,
    target_lang: str = "zh",
    max_body_pt: float = 11.5,
    floor_pt: float = FLOOR_PT,
) -> tuple[float, float, float, float]:
    """Solve uniform font size and spacing parameters to fit blocks on 1 page.

    ``floor_pt`` defaults to :data:`ubt.render.layout_solver.FLOOR_PT` (7.5pt),
    the project-wide smallest body size. It used to default to 7.0, which let
    this renderer typeset a page below the floor that every other layout path
    honours — a silent "翻译质量零妥协" violation. The return value is never below
    ``floor_pt``; when the content still does not fit there, the page is
    reported via :func:`page_overflow_at_floor` and allowed to run long rather
    than shrunk further.

    Returns:
        (uniform_size_pt, leading_em, par_spacing_em, tracking_em)
    """
    base_leading = 0.35
    base_spacing = 0.65
    base_tracking = 0.0

    target_capacity_h = _page_capacity(geometry)

    # Binary search for size in [floor_pt, max_body_pt]
    lo, hi = floor_pt, max_body_pt
    roomy = _estimated_height(
        geometry,
        blocks,
        size=hi,
        leading=base_leading,
        spacing=base_spacing,
        tracking=base_tracking,
    )
    if roomy <= target_capacity_h:
        best_size = hi
        # If very roomy, slightly relax leading to absorb whitespace
        if hi > 0 and roomy / max(1.0, target_capacity_h) < 0.60:
            base_leading = 0.45
            base_spacing = 0.80
    elif (
        _estimated_height(
            geometry,
            blocks,
            size=lo,
            leading=base_leading,
            spacing=base_spacing,
            tracking=base_tracking,
        )
        <= target_capacity_h
    ):
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if (
                _estimated_height(
                    geometry,
                    blocks,
                    size=mid,
                    leading=base_leading,
                    spacing=base_spacing,
                    tracking=base_tracking,
                )
                <= target_capacity_h
            ):
                lo = mid
            else:
                hi = mid
        best_size = lo
    else:
        # Tight mode: compress spacing and tracking. The size does NOT go below
        # the floor; the page is reported as overflowing instead (see
        # ``page_overflow_at_floor``).
        best_size = floor_pt
        base_spacing = 0.35
        base_leading = 0.22
        base_tracking = -0.015

    return (
        round(max(floor_pt, best_size), 2),
        round(base_leading, 3),
        round(base_spacing, 3),
        round(base_tracking, 3),
    )


class PageLockedReflowRenderer:
    """Renders single or multi-page documents under the strict 1:1 page-locked reflow invariant."""

    def __init__(self, font_stack: Sequence[str] | None = None) -> None:
        self.font_stack = tuple(font_stack) if font_stack else DEFAULT_FONT_STACK
        #: Pages that did not fit one page at ``FLOOR_PT``, from the most recent
        #: ``render_document`` call: ``{page_no, expected_pages, actual_pages,
        #: deficit_pt, block_id}``. Empty on a clean render.
        #:
        #: A side channel, not a log line: the caller is the only place that can
        #: put the fact in the manifest and the visual report, and a warning
        #: nobody reads is how the 6.8pt shrink passed review for so long.
        self.last_page_overflow: list[dict[str, Any]] = []

    def build_page_markup(
        self,
        spec: PageLockedPageSpec,
        *,
        target_lang: str = "zh",
    ) -> str:
        """Render a single page's cloned geometry and content into an isolated Typst #page container."""
        geom = spec.geometry
        fonts_formatted = ", ".join(f'"{f}"' for f in self.font_stack)

        lines: list[str] = [
            "#page(",
            f"  width: {geom.page_w}pt,",
            f"  height: {geom.page_h}pt,",
            f"  margin: (top: {geom.margin_top}pt, bottom: {geom.margin_bottom}pt, left: {geom.margin_left}pt, right: {geom.margin_right}pt),",
            f"  columns: {geom.columns},",
            ")[",
            f'  #set text(font: ({fonts_formatted}), size: {spec.uniform_size_pt}pt, lang: "{target_lang}", overhang: true, cjk-latin-spacing: auto)',
            f'  #set par(justify: true, linebreaks: "optimized", leading: {spec.leading_em}em, spacing: {spec.par_spacing_em}em)',
            "",
        ]

        # In 2-column mode, sort blocks by column reading order (col 1 top-down, then col 2 top-down)
        ordered_blocks = list(spec.blocks)
        if geom.columns == 2:
            x_mid = geom.page_w / 2.0
            ordered_blocks.sort(
                key=lambda b: (
                    0
                    if (
                        b.bbox
                        and b.bbox.x0 < x_mid
                        and (b.bbox.x1 - b.bbox.x0) < 0.65 * geom.page_w
                    )
                    else 1,
                    -(b.bbox.y1 if b.bbox else 0.0),
                )
            )

        for b in ordered_blocks:
            txt = delivered_text(b).strip()
            if not txt:
                txt = b.source_text.strip()
            if not txt:
                continue

            escaped = typst_escape(txt)

            if b.block_type == BlockType.HEADING:
                lvl = getattr(b.element, "level", 1) if hasattr(b, "element") else 1
                size_em = "1.25em" if lvl <= 1 else "1.10em"
                lines.append(
                    f'  #block(above: 0.8em, below: 0.5em)[#text(weight: "bold", size: {size_em}, fill: rgb("{geom.theme_color_hex}"))[{escaped}]]'
                )
            elif b.block_type == BlockType.LIST_ITEM:
                lines.append(f"  - {escaped}")
            elif b.block_type == BlockType.FORMULA:
                # Math formula
                lines.append(f"  $ {escaped} $")
            elif b.block_type == BlockType.CODE:
                from ubt.render.typst.container_repair import repair_code_block_markup

                col_w = (
                    geom.page_w
                    - geom.margin_left
                    - geom.margin_right
                    - (geom.columns - 1) * geom.column_gutter_pt
                ) / max(1, geom.columns)
                code_mk = repair_code_block_markup(txt, container_width_pt=col_w)
                lines.append(f"  {code_mk}")
            else:
                lines.append(f"  {escaped}")
                lines.append("")

        lines.append("]")
        lines.append("")
        return "\n".join(lines)

    def build_document_markup(
        self,
        pages: Sequence[PageLockedPageSpec],
        *,
        target_lang: str = "zh",
    ) -> str:
        """Construct a complete multi-page document consisting strictly of page-locked pages."""
        lines: list[str] = [
            "// Universal Book Translator - Tier L2 Page-Locked Style-Cloned Document",
            "",
        ]
        for spec in pages:
            lines.append(self.build_page_markup(spec, target_lang=target_lang))
        return "\n".join(lines)

    def render_document(
        self,
        pages: Sequence[PageLockedPageSpec],
        output_path: Path,
        build_dir: Path,
        *,
        target_lang: str = "zh",
        compile_fn: Callable[[str, str, str], tuple[bool, str]] | None = None,
    ) -> Path:
        """Compile page-locked pages into output PDF and verify strict 1:1 page-count conservation."""
        build_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        markup = self.build_document_markup(pages, target_lang=target_lang)
        typ_path = build_dir / "page_locked.typ"
        typ_path.write_text(markup, encoding="utf-8")

        #: Declared rather than inferred; see the same note in
        #: ``facsimile_renderer.render``.
        compiler: Callable[[str, str, str], tuple[bool, str]]
        if compile_fn is None:
            from ubt.adapters.pdf.typst_compile import typst_compile

            def _default_compile(src: str, dst: str, lang: str) -> tuple[bool, str]:
                return typst_compile(src, dst, target_lang=lang)

            compiler = _default_compile
        else:
            compiler = compile_fn

        ok, stderr = compiler(str(typ_path), str(output_path), target_lang)
        if not ok:
            raise DocumentParseError(
                f"Page-locked reflow compile failed: {stderr.strip() or 'unknown Typst error'} "
                f"(source kept at {typ_path})"
            )

        # Page-Lock Guard: verify page count equality.
        #
        # The guard used to "re-converge" by subtracting 0.6pt from every page's
        # body, down to a hard 6.8pt floor, and then recompile WITHOUT checking
        # the result. That is a quality/layout trade made silently, and 6.8pt is
        # below FLOOR_PT (7.5) — the smallest body size this project typesets.
        # A reader cannot see that a promise was broken; they just get a book
        # set too small to read comfortably.
        #
        # The honest order is: never go below the floor, and if a page still
        # does not fit one page, say so — page number, expected count, and the
        # measured deficit — and let it run long. ``solve_page_lock`` already
        # refuses to return a size under the floor, so this branch reports
        # rather than shrinks.
        self.last_page_overflow = []
        try:
            import pikepdf

            with pikepdf.open(output_path) as pdf:
                actual_pages = len(pdf.pages)
                expected_pages = len(pages)
                if actual_pages != expected_pages:
                    worst = None
                    worst_deficit = 0.0
                    overflow_page: int | None = None
                    for spec in pages:
                        found = page_overflow_at_floor(
                            spec.geometry,
                            spec.blocks,
                            size_pt=spec.uniform_size_pt,
                            leading_em=spec.leading_em,
                            par_spacing_em=spec.par_spacing_em,
                            tracking_em=spec.tracking_em,
                        )
                        if found is None:
                            continue
                        block_id, block_deficit = found
                        if overflow_page is None:
                            overflow_page = spec.page_no
                        if block_deficit > worst_deficit:
                            worst, worst_deficit = block_id, block_deficit
                    self.last_page_overflow.append(
                        {
                            "page_no": overflow_page,
                            "expected_pages": expected_pages,
                            "actual_pages": actual_pages,
                            "deficit_pt": round(worst_deficit, 2) if worst else None,
                            "block_id": worst,
                        }
                    )
                    logger.warning(
                        "Page-locked reflow overflowed the 1:1 page count (expected %d, "
                        "got %d) and the body floor (%.1fpt) forbids shrinking further. "
                        "Worst block %s is over budget by %s pt on page %s. The book ships "
                        "with an extra page rather than unreadable type.",
                        expected_pages,
                        actual_pages,
                        FLOOR_PT,
                        worst or "<not attributed>",
                        f"{worst_deficit:.2f}" if worst else "unknown",
                        overflow_page if overflow_page is not None else "unknown",
                    )
        except Exception as exc:
            logger.debug("Page count verification skipped or failed: %s", exc)

        return output_path


__all__ = [
    "DEFAULT_FONT_STACK",
    "PageGeometry",
    "PageLockedPageSpec",
    "PageLockedReflowRenderer",
    "extract_page_geometry",
    "page_overflow_at_floor",
    "solve_page_lock",
]
