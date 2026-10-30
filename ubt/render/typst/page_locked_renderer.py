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

import hashlib
import logging
import math
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, IRBlock
from ubt.render.delivered_text import delivered_text
from ubt.render.layout_solver import FLOOR_PT
from ubt.render.typst.escape import typst_escape

#: The bundled CJK face's own line height, in em, measured from a compiled
#: artifact: a 20pt Noto Serif CJK SC paragraph with ``leading: 0em`` has a
#: baseline pitch of 14.66pt (0.733em). Typst's ``par(leading:)`` is a *gap*
#: added on top of that, while both solvers express a *pitch ratio*
#: (``1.0 + leading_em``). Passing the ratio straight through shipped a 1.35
#: intent as a 1.083 pitch, so every path converts through this term.
FONT_LINE_HEIGHT_EM: float = 0.733


def _text_set_line(
    fonts_formatted: str, size_pt: float, tracking_em: float, target_lang: str
) -> str:
    """The ``#set text`` line for a page, carrying the group's tracking.

    Typst 0.15 accepts ``text(tracking:)``, so the document body group's letter
    spacing reaches the artifact instead of only feeding the height estimator.
    Left out when zero: an explicit ``0em`` is the same value, but the emitted
    page then differs from the pre-Stage-3A markup for no reason.
    """
    tracking = f", tracking: {tracking_em}em" if tracking_em else ""
    return (
        f"  #set text(font: ({fonts_formatted}), size: {size_pt}pt{tracking}, "
        f'lang: "{target_lang}", overhang: true, cjk-latin-spacing: auto)'
    )


def leading_gap_em(pitch_ratio: float) -> float:
    """Typst ``par(leading:)`` gap for a target line-height ratio.

    ``pitch_ratio`` is what the solvers optimise (1.35 = 1.35x the font size).
    Typst adds the gap to the font's own line height, so the gap is
    ``pitch_ratio - FONT_LINE_HEIGHT_EM``. Clamped at zero: a ratio below the
    font's base cannot ask Typst for negative leading.
    """
    return max(0.0, float(pitch_ratio) - FONT_LINE_HEIGHT_EM)


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


def _central_band_is_occupied(
    bboxes: Sequence[Any],
    page_w: float,
    *,
    band_pt: float = 6.0,
    coverage: float = 0.40,
) -> bool:
    """Whether blocks cover enough of the page's central vertical band to rule out a gutter.

    A two-column page has a vertical gutter: a band around the page midpoint
    that no body block crosses. A single-column page has full-measure prose
    across that band, even when its display formulas are centred and its short
    labels sit on the left.
    """
    if not bboxes or page_w <= 0:
        return False
    mid = page_w / 2.0
    left, right = mid - band_pt, mid + band_pt
    intervals: list[tuple[float, float]] = []
    for bb in bboxes:
        bx0 = float(bb.x0)
        bx1 = float(bb.x1)
        # A box occupies the central gutter if it substantially straddles the page midpoint.
        # A box strictly to the left (bx1 <= mid) or strictly to the right (bx0 >= mid) is
        # within its column and does not cross the gutter.
        if bx1 <= mid or bx0 >= mid:
            continue
        if bx1 <= left or bx0 >= right:
            continue
        lo = min(float(bb.y0), float(bb.y1))
        hi = max(float(bb.y0), float(bb.y1))
        if hi > lo:
            intervals.append((lo, hi))
    if not intervals:
        return False
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    covered = sum(hi - lo for lo, hi in merged)
    content_h = max(float(bb.y1) for bb in bboxes) - min(float(bb.y0) for bb in bboxes)
    return content_h > 0 and covered / content_h >= coverage


def extract_page_geometry(
    page_w: float,
    page_h: float,
    blocks: Sequence[IRBlock],
    *,
    default_margin: float = 36.0,
    default_theme_color: str = "#0078ae",
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

    # Column detection: a gutter is a band the body does not cross, not a
    # count of boxes whose x0 happens to sit near the midpoint.
    columns = 1
    if page_w >= 450.0 and len(bboxes) >= 4 and not _central_band_is_occupied(bboxes, page_w):
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
    theme_color = default_theme_color
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


def _block_measure(geometry: PageGeometry, block: IRBlock) -> float:
    """The width one block was promised, in points.

    The block's own source box, floored at 10pt and capped at the column measure.
    A page-locked canvas paints each block at its source rectangle, so that
    rectangle — not the abstract column — is the container a table has to fit:
    the L1 canvas already hands ``render_table`` ``max(10.0, x1 - x0)`` for its
    placements, and using the same measure here is what makes an exhibit lay out
    identically whichever tier its page routed to. Blocks with no bbox fall back
    to the column width.
    """
    if block.bbox is None:
        return _content_width(geometry)
    width = float(block.bbox.x1) - float(block.bbox.x0)
    return min(max(10.0, width), _content_width(geometry))


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
    if block.block_type is BlockType.IMAGE:
        # An image block's "text" is the parser's asset *path*, never prose.
        # In flow it is drawn from the extracted asset, so it reserves the
        # height the source region occupied scaled to the column: pessimistic
        # when the asset turns out unresolvable at markup time (and the block
        # is dropped), and never optimistic about the floor -- which is exactly
        # the error direction this model is allowed to have.
        if block.bbox is None:
            return 0.0
        w = max(0.0, float(block.bbox.x1) - float(block.bbox.x0))
        h = max(0.0, float(block.bbox.y1) - float(block.bbox.y0))
        if w <= 0.0 or h <= 0.0:
            return 0.0
        drawn_w = min(w, col_w)
        drawn_h = h * (drawn_w / w)
        return drawn_h + spacing * size
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
        table_lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        row_count = sum(1 for ln in table_lines if not re.match(r"^\|?[\s\-:|]+\|?$", ln))
        row_count = max(2, row_count)
        row_h = size * (1.0 + leading) + 8.0
        bbox_h = (
            max(0.0, float(block.bbox.y1) - float(block.bbox.y0)) if block.bbox is not None else 0.0
        )
        return max(row_count * row_h + 12.0, bbox_h) + spacing * size
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
    preferred_pt: float | None = None,
) -> tuple[float, float, float, float]:
    """Solve uniform font size and spacing parameters to fit blocks on 1 page.

    ``floor_pt`` defaults to :data:`ubt.render.layout_solver.FLOOR_PT` (7.5pt),
    the project-wide smallest body size. It used to default to 7.0, which let
    this renderer typeset a page below the floor that every other layout path
    honours — a silent "翻译质量零妥协" violation. The return value is never below
    ``floor_pt``; when the content still does not fit there, the page is
    reported via :func:`page_overflow_at_floor` and allowed to run long rather
    than shrunk further.

    ``preferred_pt`` is the document's body-group size (G4 Stage 3A). A page-lock
    page is a *different placement strategy*, not a different style: it may
    re-typeset the page, but it starts from the group's size and only goes below
    it when the content genuinely cannot fit. Going below is reported by the
    caller as a style-group exception, because it is one.

    Returns:
        (uniform_size_pt, leading_em, par_spacing_em, tracking_em)
    """
    base_leading = 0.35
    base_spacing = 0.65
    base_tracking = 0.0

    target_capacity_h = _page_capacity(geometry)

    # Binary search for size in [floor_pt, ceiling]
    hi = max_body_pt if preferred_pt is None else min(max_body_pt, max(floor_pt, preferred_pt))
    lo = floor_pt
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


def make_asset_copier(
    build_dir: Path, assets_root: Path | None = None
) -> Callable[[str], str | None]:
    """Return a copier that places one extracted asset inside ``build_dir``.

    The parser stores an asset's absolute cache path as the block's
    ``source_text``; Typst addresses ``image()`` relative to the source file, so
    the file has to be copied next to it. The name is content-addressed (the
    same reasoning as the publish renderer's ``_copy_asset``: flat build
    directories plus shared basenames silently overwrite each other), and any
    unresolvable path yields ``None`` so the caller drops the block rather than
    emitting an ``image()`` Typst will refuse to compile -- the original defect
    here was printing the cache path as *prose* in the figure's place.

    ``build_dir`` is created on demand. A caller that builds markup before the
    render directory exists (a preflight, a preview, a test) would otherwise get
    ``None`` for *every* asset and an artifact with no figures at all -- the
    silent-loss failure this function exists to prevent, arriving through a
    missing ``mkdir`` rather than a missing file.
    """

    def copy(asset_path: str) -> str | None:
        if not asset_path:
            return None
        candidate = Path(asset_path)
        if not (candidate.is_absolute() and candidate.exists()) and assets_root is not None:
            rooted = assets_root / candidate.name
            candidate = rooted if rooted.exists() else candidate
        if not candidate.exists():
            return None
        try:
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()[:16]
        except OSError:
            return None
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate.stem).strip("-") or "asset"
        suffix = re.sub(r"[^A-Za-z0-9.]+", "", candidate.suffix)[:10].lower()
        target = build_dir / f"{stem}-{digest}{suffix}"
        if target.exists():
            return target.name
        try:
            build_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
        except OSError:
            return None
        return target.name

    return copy


def make_formula_cropper(
    build_dir: Path, source_pdf: Path | None = None
) -> Callable[[IRBlock], str | None]:
    """Return a 300 DPI cropper that saves an uncompilable formula's source region into ``build_dir``.

    When a display ``FORMULA`` block cannot be transpiled to Typst math (or is
    rejected by ``TypstMathProbe``), degrading a multi-line equation or matrix
    to flat Unicode strips its 2D layout. When the source PDF and the block's
    bounding box are available, cropping the original equation at 300 DPI via
    :func:`delivered_crop_pil` preserves exact mathematical fidelity (§8.9).
    """

    def crop(block: IRBlock) -> str | None:
        if source_pdf is None or not source_pdf.exists():
            return None
        bbox = getattr(block, "bbox", None)
        if bbox is None or getattr(bbox, "page", 0) < 1:
            return None
        w = float(bbox.x1) - float(bbox.x0)
        h = float(bbox.y1) - float(bbox.y0)
        if w < 8.0 or h < 4.0:
            return None
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", str(block.id)).strip("-") or "formula"
        target = build_dir / f"formula-crop-{safe_id}-p{int(bbox.page)}.png"
        if target.exists():
            return target.name
        try:
            from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI, delivered_crop_pil

            cropped = delivered_crop_pil(source_pdf, int(bbox.page), bbox, dpi=FIGURE_ASSET_DPI)
            if getattr(cropped, "width", 0) < 4 or getattr(cropped, "height", 0) < 4:
                return None
            build_dir.mkdir(parents=True, exist_ok=True)
            cropped.save(target)
        except Exception as exc:
            logger.debug("Formula raster fallback failed for %s: %s", block.id, exc)
            return None
        return target.name

    return crop


class PageLockedReflowRenderer:
    """Renders single or multi-page documents under the strict 1:1 page-locked reflow invariant."""

    def __init__(
        self,
        font_stack: Sequence[str] | None = None,
        *,
        math_gate: Callable[[str], bool] | None = None,
    ) -> None:
        self.font_stack = tuple(font_stack) if font_stack else DEFAULT_FONT_STACK
        #: The Typst compile gate for a transpiled math body — see
        #: ``TypstDocumentRenderer.math_gate``. A body the transpiler accepts but
        #: Typst rejects (``frac(a, )``) fails the *whole* document at one compile,
        #: so an L2 page cannot emit a converted formula without this verdict.
        #: ``None`` skips the gate, which is what a caller without a probe wants.
        self._math_gate = math_gate
        #: Pages that did not fit one page at ``FLOOR_PT``, from the most recent
        #: ``render_document`` call: ``{page_no, expected_pages, actual_pages,
        #: deficit_pt, block_id}``. Empty on a clean render.
        #:
        #: A side channel, not a log line: the caller is the only place that can
        #: put the fact in the manifest and the visual report, and a warning
        #: nobody reads is how the 6.8pt shrink passed review for so long.
        self.last_page_overflow: list[dict[str, Any]] = []

    def _display_math_markup(
        self,
        block: IRBlock,
        text: str,
        *,
        col_w: float | None = None,
        formula_cropper: Callable[[IRBlock], str | None] | None = None,
    ) -> str:
        r"""Typst math for one ``FORMULA`` block, 300 DPI source crop, or readable Unicode.

        The branch this replaces was ``f"  $ {typst_escape(txt)} $"``: escape the
        LaTeX, then wrap the *escaped* text in math delimiters. That is broken in
        both directions. Typst math mode has no escapes, so ``N_{ch}`` became
        ``N\_\{ch\}`` and Typst read the backslash-escaped brace as a token,
        failing the whole document — ``error: unknown variable: ch`` (reproduced
        verbatim, not inferred). And an escape-only body is never math: a formula
        that did survive came out as literal ``\psi\_1``.

        So a formula block is transpiled, exactly as the reflow channel's
        ``block_type == "formula"`` branch does. A body the transpiler cannot
        convert, or that the compile gate rejects, first attempts a 300 DPI
        source-PDF crop via ``formula_cropper`` (§8.9), and otherwise degrades
        to readable Unicode rather than leaking command names.
        """
        from ubt.render.typst.document_renderer import escape_markup
        from ubt.render.typst.math_transpiler import (
            clean_math_fallback_text,
            strip_math_delimiters,
            typstify_math,
        )

        converted = typstify_math(strip_math_delimiters(text))
        if converted is not None and self._math_gate is not None and not self._math_gate(converted):
            converted = None
        if converted:
            return f"$ {converted} $"
        if formula_cropper is not None:
            crop_name = formula_cropper(block)
            if crop_name:
                bbox = getattr(block, "bbox", None)
                raw_w = float(bbox.x1 - bbox.x0) if bbox is not None else (col_w or 240.0)
                max_w = col_w if col_w is not None and col_w > 0 else raw_w
                img_w = min(max(8.0, raw_w), max_w)
                return f'#figure(image("{crop_name}", width: {img_w:.1f}pt), numbering: none)'
        fallback = clean_math_fallback_text(text)
        if fallback:
            return escape_markup(fallback)
        # Neither transpilable nor degradable: nothing readable is left, and a
        # bare ``$ $`` is a compile error. The block id keeps this attributable --
        # it is the same reason the reflow channel counts ``formulas_dropped``
        # instead of failing silently.
        logger.warning("Page-locked: formula block %s has no deliverable body", block.id)
        return ""

    def build_page_markup(
        self,
        spec: PageLockedPageSpec,
        *,
        target_lang: str = "zh",
        asset_copier: Callable[[str], str | None] | None = None,
        formula_cropper: Callable[[IRBlock], str | None] | None = None,
    ) -> str:
        """Render a single page's cloned geometry and content into an isolated Typst #page container.

        An L2 page reflows text in reading order, so it carries no full-page
        stripped background (fixed-coordinate source ink under reflowed prose
        would collide by construction). Figures are instead drawn in flow via
        ``asset_copier``, which maps the extracted-asset path stored in
        ``source_text`` to a build-relative file name; if no copier is given or
        the file is missing, the figure block is dropped rather than printing its
        cache path as prose.
        """
        geom = spec.geometry
        fonts_formatted = ", ".join(f'"{f}"' for f in self.font_stack)

        lines: list[str] = [
            "#page(",
            f"  width: {geom.page_w}pt,",
            f"  height: {geom.page_h}pt,",
            f"  margin: (top: {geom.margin_top}pt, bottom: {geom.margin_bottom}pt, left: {geom.margin_left}pt, right: {geom.margin_right}pt),",
            f"  columns: {geom.columns},",
            ")[",
            _text_set_line(
                fonts_formatted,
                spec.uniform_size_pt,
                spec.tracking_em,
                target_lang,
            ),
            f'  #set par(justify: true, linebreaks: "optimized", '
            f"leading: {leading_gap_em(1.0 + spec.leading_em):.3f}em, "
            f"spacing: {spec.par_spacing_em}em)",
            "",
        ]

        # In 2-column mode, sort blocks by column reading order (col 1 top-down, then col 2 top-down).
        # In single-column mode, sort blocks top-down by -bbox.y1 when all blocks carry valid
        # bounding boxes so extractor reading-order inversions (e.g., formula lead-ins or figure
        # labels emitted out of vertical order) never invert the page layout.
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
        elif ordered_blocks and all(
            b.bbox is not None and b.bbox.y1 > b.bbox.y0 for b in ordered_blocks
        ):
            ordered_blocks.sort(
                key=lambda b: (
                    -round(float(b.bbox.y1), 1) if b.bbox else 0.0,
                    round(float(b.bbox.x0), 1) if b.bbox else 0.0,
                )
            )

        for b in ordered_blocks:
            # An image block's text is the asset *path* the parser stored
            # (``source_text=str(asset_path)``), never reader-visible prose.
            # L2 pages reflow without a background layer, so the figure is drawn
            # in flow from the extracted asset at its source width; if no copier
            # or file is available, the block is dropped rather than typesetting
            # ``/home/<user>/.cache/...png`` as prose.
            if b.block_type is BlockType.IMAGE:
                if asset_copier is None:
                    continue
                # Suppress tiny decorative icons / badges (e.g. author ORCID badges, license marks < 55x55pt)
                if b.bbox and (b.bbox.x1 - b.bbox.x0 < 55.0) and (b.bbox.y1 - b.bbox.y0 < 55.0):
                    continue
                name = asset_copier(b.source_text)
                if not name:
                    continue
                img_w = (b.bbox.x1 - b.bbox.x0) if b.bbox else _content_width(geom)
                img_w = min(max(1.0, img_w), _content_width(geom))
                lines.append(f'  #figure(image("{name}", width: {img_w:.1f}pt), numbering: none)')
                lines.append("")
                continue
            txt = delivered_text(b).strip()
            if not txt:
                txt = b.source_text.strip()
            if not txt:
                continue

            # ``render_prose`` is the reflow channel's one owner of "escape the
            # prose, transpile the ``$…$``": the escaped route below and this one
            # must not be chosen by hand per branch, or a span that lands in the
            # wrong half ships its LaTeX source to the reader.
            #
            # A function-local import: ``document_renderer`` is the heavier module
            # (it pulls the table compiler, page trim and inline spans) and the
            # facsimile renderer reaches this class from it, so a top-level edge
            # here would put an import cycle one refactor away. Same reason
            # ``facsimile_renderer._placement_content`` imports ``render_table``
            # lazily.
            #
            # THE DELIVERY BUG THIS FIXES: a translated page of a technical book
            # carries inline math the model wrote as LaTeX, and this branch used
            # to run ``typst_escape`` over the WHOLE block -- a route with no math
            # branch at all. ``$`` and ``_`` are both in the escape set, so
            # ``$\psi_1$`` shipped as the literal ``\$\\psi\_1\$``: measured on a
            # 26-page BSIM-CMG chapter, 265 escaped ``\$`` across 44 lines and
            # 93 raw LaTeX macro leaks (``\text`` 35, ``\beta`` 33, ``\psi`` 16,
            # ``\mu`` 4, ``\frac`` 3, ``\epsilon`` 2) in a file billed as a
            # finished translation. Escaping is correct for prose punctuation and
            # catastrophic for a formula, which is exactly the distinction
            # ``render_prose`` was written to make.
            from ubt.render.typst.document_renderer import render_prose, render_table

            escaped = typst_escape(txt)

            if b.block_type == BlockType.HEADING:
                lvl = getattr(b.element, "level", 1) if hasattr(b, "element") else 1
                size_em = "1.25em" if lvl <= 1 else "1.10em"
                lines.append(
                    f'  #block(above: 0.8em, below: 0.5em)[#text(weight: "bold", size: {size_em}, fill: rgb("{geom.theme_color_hex}"))[{render_prose(txt, is_math_ok=self._math_gate)}]]'
                )
            elif b.block_type == BlockType.LIST_ITEM:
                lines.append(f"  - {render_prose(txt, is_math_ok=self._math_gate)}")
            elif b.block_type == BlockType.TABLE:
                # A Markdown grid must be typeset, not printed: escaped, its pipes
                # and dashes are literal text.
                #
                # ``page_columns=1`` deliberately, even on a two-column page, and
                # the container is the *column* the block was solved into. With
                # ``page_columns=geom.columns``, ``render_table``'s full-page
                # promotion (``page_columns > 1 and too_wide``) re-lays a wide grid
                # across the whole page measure, which on a page-locked canvas
                # paints over the neighbouring column -- the block's own box is
                # what the reader was promised. The L1 canvas already passes
                # ``page_columns=1`` for the same reason (``_placement_content``);
                # matching it means the same grid lays out the same way whichever
                # tier its page routed to.
                col_w = _block_measure(geom, b)
                table_markup = render_table(
                    txt,
                    is_math_ok=self._math_gate,
                    page_columns=1,
                    content_width_pt=col_w,
                    target_lang=target_lang,
                )
                if table_markup:
                    lines.append(f"  {table_markup}")
                else:
                    # An unparseable grid degrades to escaped text rather than
                    # vanishing: a table the compiler cannot lay out is still
                    # content the reader needs. Same rule as the L1 canvas.
                    logger.warning(
                        "Page-locked: block %s looked like a table but did not parse; delivering it as text",
                        b.id,
                    )
                    lines.append(f"  {escaped}")
                    lines.append("")
            elif b.block_type == BlockType.FORMULA:
                col_w = _block_measure(geom, b)
                lines.append(
                    f"  {self._display_math_markup(b, txt, col_w=col_w, formula_cropper=formula_cropper)}"
                )
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
                lines.append(f"  {render_prose(txt, is_math_ok=self._math_gate)}")
                lines.append("")

        lines.append("]")
        lines.append("")
        return "\n".join(lines)

    def build_document_markup(
        self,
        pages: Sequence[PageLockedPageSpec],
        *,
        target_lang: str = "zh",
        asset_copier: Callable[[str], str | None] | None = None,
        formula_cropper: Callable[[IRBlock], str | None] | None = None,
    ) -> str:
        """Construct a complete multi-page document consisting strictly of page-locked pages."""
        lines: list[str] = [
            "// Universal Book Translator - Tier L2 Page-Locked Style-Cloned Document",
            "",
        ]
        for spec in pages:
            lines.append(
                self.build_page_markup(
                    spec,
                    target_lang=target_lang,
                    asset_copier=asset_copier,
                    formula_cropper=formula_cropper,
                )
            )
        return "\n".join(lines)

    def render_document(
        self,
        pages: Sequence[PageLockedPageSpec],
        output_path: Path,
        build_dir: Path,
        *,
        target_lang: str = "zh",
        asset_copier: Callable[[str], str | None] | None = None,
        formula_cropper: Callable[[IRBlock], str | None] | None = None,
        compile_fn: Callable[[str, str, str], tuple[bool, str]] | None = None,
    ) -> Path:
        """Compile page-locked pages into output PDF and verify strict 1:1 page-count conservation."""
        build_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        markup = self.build_document_markup(
            pages,
            target_lang=target_lang,
            asset_copier=asset_copier,
            formula_cropper=formula_cropper,
        )
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
    "make_asset_copier",
    "make_formula_cropper",
    "page_overflow_at_floor",
    "solve_page_lock",
]
