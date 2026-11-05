"""1:1 Facsimile (Canvas) Typst Document Renderer.

Typst draws the translated text over a text-stripped vector page (``bg.pdf``), so the
source's diagrams, rules and colors survive and only the ink is replaced; geometry and
type size come from ``layout_solver.py``.

A placed block is drawn at the *page's* uniform body size unless it carries its
own ``font_size_pt``. That exception is what makes a heading look like a heading:
the page solves for one body size, so without it a 16.2pt chapter title was
drawn at ~10pt and the delivered page read as a flat wall of uniform text.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ubt.core.exceptions import DocumentParseError
from ubt.core.fs_perms import restrict_dir_to_owner
from ubt.render.layout_solver import FLOOR_PT
from ubt.render.typst.page_locked_renderer import (
    DEFAULT_FONT_STACK,
    PageLockedPageSpec,
    PageLockedReflowRenderer,
    leading_gap_em,
)

logger = logging.getLogger(__name__)

#: Smallest font scale the canvas will apply to make a box's text fit.
#:
#: The canvas places translated text into the source block's rectangle; when the
#: translation is longer than the space (Chinese reference lists rendered at the
#: body group's size are the measured case: 615pt of flow in a 460pt box) Typst
#: clips at the block height, so the tail of the text is drawn but cut — and the
#: clipped glyphs stay in the text layer, which the delivery contract reads as
#: text overlapping whatever the next block drew there. Shrinking the font is the
#: only in-box remedy: below the effective floor the page genuinely cannot hold
#: its text at a readable size and clipping (with the contract's block) is the
#: honest outcome. ``_fit_to_box`` never scales below this ratio *or* below
#: ``FLOOR_PT``, whichever is the stronger bound: a 10.9pt page bottoms out at
#: 7.5pt, not the old fixed 6.5pt.
_MIN_FIT_SCALE = 0.6

#: Font weights the renderer will pass through to Typst verbatim. Anything else
#: falls back to the block's own bold/regular classification rather than being
#: interpolated into the markup unchecked (a measured ``"semibold condensed"``
#: would have produced an invalid ``weight:`` value and failed the whole page).
_KNOWN_WEIGHTS: frozenset[str] = frozenset({"regular", "medium", "semibold", "bold", "black"})


@dataclass(frozen=True, slots=True)
class FacsimileBlockPlacement:
    """A single placed block on a facsimile page."""

    block_id: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    is_heading: bool = False
    heading_level: int = 1
    text_color_hex: str = "#000000"
    font_weight: str = "regular"
    align: str = "justify"
    skip_render: bool = False
    #: The block's own measured size, in points. ``None`` means "use the page's
    #: uniform body size". A heading needs this: the page solves for the body.
    font_size_pt: float | None = None
    #: The block is a Markdown table grid and must be typeset as ``#table``,
    #: not escaped and printed as literal pipe text.
    is_table: bool = False
    #: The block is a code listing and must be typeset in a monospace face with
    #: its line structure intact.
    is_code: bool = False
    #: Styled inline runs (superscripts, citations, bold, color) located by substring
    runs: tuple[Any, ...] = ()
    #: Original unrelaxed source bounding box vertical coordinates (used for layout stream continuity)
    src_y0: float | None = None
    src_y1: float | None = None
    #: Horizontal glyph stretch percentage (e.g. 90.0, 95.0, 100.0) for variable font elastic compaction
    stretch_pct: float = 100.0
    #: The block is a figure caption rigidly anchored to a figure
    is_caption: bool = False
    #: ID of the parent figure (e.g. image) this caption is anchored to
    parent_figure_id: str | None = None


@dataclass(slots=True)
class FacsimilePageSpec:
    """One page's full facsimile configuration."""

    page_no: int
    page_w: float
    page_h: float
    uniform_size_pt: float
    leading_em: float
    tracking_em: float = 0.0
    stretch_pct: float = 100.0
    placements: list[FacsimileBlockPlacement] = field(default_factory=list)
    tier: str = "L1"  # "L1" (canvas) or "L2" (page-locked reflow)
    l2_spec: PageLockedPageSpec | None = None


def _normalise_weight(font_weight: str, *, is_heading: bool, bold: bool) -> str:
    """The Typst ``weight`` value for a block.

    ``font_weight`` is what the extractor measured; ``bold`` is the block's own
    classification, used when the measurement is not a value Typst accepts.

    The previous expression was ``"bold" if font_weight in ("regular", "")``,
    which inverted the sense: every heading whose measured weight was regular
    was emitted bold, and a genuinely bold one was passed through only by luck.
    """
    candidate = (font_weight or "").strip().lower()
    if candidate in _KNOWN_WEIGHTS:
        return candidate
    if is_heading or bold:
        return "bold"
    return "regular"


class FacsimileRenderer:
    """Produces and compiles 1:1 facsimile Typst documents."""

    def __init__(
        self,
        font_stack: Sequence[str] | None = None,
        *,
        math_gate: Callable[[str], bool] | None = None,
        enable_column_flow: bool = False,
    ) -> None:
        self.font_stack = tuple(font_stack) if font_stack else DEFAULT_FONT_STACK
        self.enable_column_flow = enable_column_flow
        #: The Typst compile gate for a transpiled math body, forwarded to the L2
        #: pages it owns. A body the transpiler accepts but Typst rejects (
        #: ``frac(a, )``) fails the whole document at the single compile below, so
        #: gating here is what keeps one malformed formula from costing every
        #: page. ``None`` skips the gate (tests, and callers with no probe).
        #: Ids of placements the canvas intentionally did not draw because the
        #: translated content cannot fit its fixed box (currently only tables
        #: that fail even at the minimum scale). The source background image
        #: still shows these blocks, so the adapter records them as preserved
        #: rather than as a successful translation.
        self.preserved_placements: list[tuple[str, str]] = []
        self._math_gate = math_gate

    def _placement_content(
        self,
        pl: FacsimileBlockPlacement,
        *,
        width_pt: float,
        target_lang: str,
        base_font_size_pt: float | None = None,
    ) -> str:
        """Typst markup for one placement's text, sized and styled by its block type."""
        if pl.is_table:
            from ubt.render.table_compiler import compile_typst_booktabs
            from ubt.render.typst.document_renderer import render_table

            # First try dedicated academic booktabs compiler. ``require_fit``
            # makes a table that cannot fit its fixed box return empty instead
            # of a clipped min-scale grid; the source background still carries
            # the table, and the caller records the block as preserved.
            table_markup = compile_typst_booktabs(
                pl.text,
                content_width_pt=width_pt,
                content_height_pt=max(10.0, float(pl.y1) - float(pl.y0)),
                base_font_size_pt=pl.font_size_pt or base_font_size_pt,
                min_font_size_pt=6.5,
                stretch_pct=pl.stretch_pct,
                target_lang=target_lang,
                is_math_ok=self._math_gate,
                enable_continuation=True,
                table_uid=f"src-{pl.block_id}",
                require_fit=True,
            )
            if table_markup:
                return table_markup

            table_markup = render_table(
                pl.text,
                is_math_ok=self._math_gate,
                page_columns=1,
                content_width_pt=width_pt,
                content_height_pt=max(10.0, float(pl.y1) - float(pl.y0)),
                base_font_size_pt=base_font_size_pt,
                stretch_pct=pl.stretch_pct,
                target_lang=target_lang,
                enable_continuation=True,
                table_uid=f"src-{pl.block_id}",
                require_fit=True,
            )
            if table_markup:
                return table_markup
            # Neither compiler can lay this grid out inside its box at a
            # readable size. Draw nothing: the L1 background image still shows
            # the source table, and the adapter records the block as preserved
            # (not as a rebuilt table whose last rows were clipped away).
            logger.warning(
                "Facsimile: block %s does not fit its table box; preserving the source table",
                pl.block_id,
            )
            self.preserved_placements.append((pl.block_id, "table_preserved_source"))
            return ""

        if pl.is_code:
            from ubt.render.typst.container_repair import repair_code_block_markup

            return repair_code_block_markup(pl.text.strip(), container_width_pt=width_pt)

        # Same rule as the L2 branch it mirrors: the canvas is not exempt from
        # LaTeX leaking. ``typst_escape`` alone put ``\$\\psi(x, y)\$`` on the page
        # for a translated paragraph of a BSIM-CMG chapter (page 5 of the measured
        # run, an L1 page), because ``$`` and ``_`` are both in the escape set and
        # this branch had no math handling. ``render_prose`` escapes the prose and
        # transpiles only a span ``_is_math_span`` recognises, so the formula
        # becomes real math and a stray ``$`` stays a literal character.
        #
        # A function-local import: ``document_renderer`` pulls the table compiler,
        # page trim and inline spans, and this module's L2 pages come from
        # ``page_locked_renderer``, which now delegates prose to this same module.
        # Keeping the edge lazy is what stops the two files forming a cycle.
        from ubt.render.typst.document_renderer import render_prose

        text = pl.text.strip()
        # ``render_prose`` covers the whole escape set this branch needs: the
        # mid-line structural characters and the line-initial ``=``/``-``/``N.``
        # markers (verified: ``"= 5 and $\\psi_1$ ok"`` -> ``"\\= 5 and $psi_1$ ok"``).
        escaped = render_prose(text, runs=pl.runs, is_math_ok=self._math_gate)
        size = f"size: {pl.font_size_pt}pt, " if pl.font_size_pt else ""
        stretch = (
            f"stretch: {pl.stretch_pct:.0f}%, " if pl.stretch_pct and pl.stretch_pct < 100.0 else ""
        )

        if pl.is_caption:
            # No absolute size here: the fixed-box wrapper sets the caption's
            # measured size and scales it to the box. Pinning 9pt inside the
            # content overrode that wrapper and drew long captions past their
            # 20pt box with only the drawing clipped.
            fill = f'fill: rgb("{pl.text_color_hex}")' if pl.text_color_hex != "#000000" else ""
            fill_arg = f", {fill}" if fill else ""
            cap_align = pl.align if pl.align in {"left", "center", "right"} else "left"
            return (
                f"#set par(justify: false, leading: 0.50em)\n"
                f'    #align({cap_align})[#text({stretch}weight: "regular"{fill_arg})[{escaped}]]'
            )

        if pl.is_heading:
            weight = _normalise_weight(pl.font_weight, is_heading=True, bold=True)
            body = (
                f'#text({size}{stretch}weight: "{weight}", '
                f'fill: rgb("{pl.text_color_hex}"))[{escaped}]'
            )
            # A centered source title carries ``align="center"`` from the
            # layout annotator; re-center it instead of drawing it flush left.
            # Other alignments keep the block's own box (left/justify).
            if pl.align == "center":
                body = f"#align(center)[{body}]"
            return f"#set par(justify: false)\n    {body}"

        if size or stretch or pl.font_weight not in ("", "regular"):
            weight = _normalise_weight(pl.font_weight, is_heading=False, bold=False)
            fill = f'fill: rgb("{pl.text_color_hex}"), ' if pl.text_color_hex != "#000000" else ""
            return f'#text({size}{stretch}{fill}weight: "{weight}")[{escaped}]'
        return escaped

    def _fit_to_box(
        self,
        *,
        dx: float,
        dy: float,
        width: float,
        height: float,
        base_size_pt: float,
        outset: str,
        content: str,
        floor_pt: float = FLOOR_PT,
        ladder: bool = True,
        page_tracking_em: float = 0.0,
    ) -> str:
        """Place *content* in its source box, shrinking it until the flow fits.

        ``page_tracking_em`` is the page-level tracking the plan resolved (one
        owner: the style-group rung). It is the *base* of the in-box ladder —
        the ladder may only tighten it, never replace it with its own table —
        so the page-level knob actually reaches the artifact. Passing the page
        value into the emitted Typst keeps the fit-height estimator (which
        already uses the same rung) and the drawn glyphs in agreement.

        ``#block(height: ..., clip: true)`` does not bound the *text*: Typst
        lays the overflow out past the block and only clips the drawing, so the
        glyphs remain in the PDF text layer below the block. Two consequences
        were measured on a real faithful run: the delivery contract's overlap
        check compared that invisible text with the next block's visible line
        (8 critical findings on a bibliography page), and the cut line itself
        rendered as a half-height sliver.

        ``measure`` runs in a ``context`` so it sees the page's own font stack
        and ``#set par`` (leading, justification). The ratio is length/length,
        applied as a font-scale; reflow makes the result slightly shorter than
        the box, which is the safe direction. Content that pins its own size
        (headings, captions) keeps it, and the flow is measured around it.

        Applies an Elastic Typography Ladder when content overflows:
        1. Leading compaction (0.55em -> 0.42em)
        2. Micro-tracking compaction (-0.01em to -0.02em)
        3. Horizontal glyph stretch / condensing (95% to 88%)
        4. Proportional font-size scaling down to the effective floor

        The floor is ``max(_MIN_FIT_SCALE, floor_pt / base_size_pt)``. For body
        text ``floor_pt`` is ``FLOOR_PT``: the canvas must never draw prose
        below the same 7.5pt the solver promises. A *caption* is allowed below
        the body floor (a figure's 8pt caption is not body text), so its caller
        passes ``floor_pt=6.5`` and ``ladder=False``: the elastic tracking /
        glyph-stretch rungs are body micro-typography, not caption styling.
        """
        effective_floor = floor_pt if floor_pt > 0.0 else FLOOR_PT
        min_fit = (
            min(1.0, max(_MIN_FIT_SCALE, effective_floor / base_size_pt))
            if base_size_pt > 0.0
            else _MIN_FIT_SCALE
        )
        # The page-level rung is the tracking's single owner. The in-box ladder
        # adds a *bounded* extra squeeze on top of it (never a replacement), and
        # captions opt out of the ladder entirely: their micro-typography is the
        # page value alone. Without this the page-level -0.030em was overwritten
        # by a box-local 0em/-0.02em table, so the plan's compression never
        # reached the artifact (measured: identical last-line x ranges for a
        # page set at 0 and at -0.030em).
        if ladder:
            # The same floor the style-group ladder promises (``TRACKING_FLOOR_EM``),
            # restated locally because core→render is a forbidden edge. The extra
            # squeeze is whatever headroom the page rung has NOT already spent.
            tracking_floor_em = -0.030
            base_track = max(page_tracking_em, tracking_floor_em)
            extra = max(0.0, min(0.02, base_track - tracking_floor_em))
            track_stretch = (
                f"    let base-track = {base_track:.4f}em\n"
                "    let eff-track = if ratio < 0.90 { base-track - "
                f"{extra:.4f}em }} else if ratio < 0.96 {{ base-track - "
                f"{extra / 2.0:.4f}em }} else {{ base-track }}\n"
                "    let eff-stretch = if ratio < 0.88 { 90% } else if ratio < 0.95 { 95% } else { 100% }\n"
            )
        else:
            track_stretch = (
                f"    let eff-track = {page_tracking_em:.4f}em\n    let eff-stretch = 100%\n"
            )
        return (
            "  #context {\n"
            f"    let box-h = {height}pt\n"
            f"    let natural = measure(block(width: {width}pt)[\n{content}\n    ]).height\n"
            "    let ratio = if natural > 0pt { box-h / natural } else { 1.0 }\n"
            f"    let fit = calc.max({min_fit:.4f}, calc.min(1.0, ratio))\n"
            f"{track_stretch}"
            f"    place(top + left, dx: {dx}pt, dy: {dy}pt, "
            f"block(width: {width}pt, height: {height}pt, clip: true, outset: {outset})[\n"
            f"      #set text(size: {base_size_pt}pt * fit, tracking: eff-track, stretch: eff-stretch)\n"
            f"{content}\n"
            "    ])\n"
            "  }"
        )

    def _resolve_placement_collisions(
        self,
        items: list[list[Any]],
        page_w: float,
        page_h: float,
    ) -> list[list[Any]]:
        """Resolve bounding-box overlap between adjacent blocks on a canvas page.

        Docling/OCR extraction occasionally extracts proximate or stacked elements
        (e.g. table caption and sub-panel heading, or figure labels) with overlapping
        bounding boxes (e.g. 2-4pt delta). In a 1:1 facsimile canvas where items are
        placed via absolute `#place(dx, dy)`, rendering them at raw extracted coordinates
        causes acute text collision (often >80% overlap), failing the delivery contract.

        This pass performs a stable top-down cascade relaxation:
        Whenever item B succeeds item A in reading order, shares horizontal span
        (same column or full-width overlap), and vertically overlaps with A by >= 15%,
        item B's vertical placement `dy` is nudged down to `A.bottom + gap`, provided
        it stays within the safe page boundary.
        """
        min_gap = 2.5
        max_y = page_h - 24.0
        for i in range(len(items)):
            a_pl, a_dx, a_dy, a_w, a_h, a_c = items[i]
            a_x0, a_x1 = a_dx, a_dx + a_w
            a_y0, a_y1 = a_dy, a_dy + a_h
            for j in range(i + 1, len(items)):
                b_pl, b_dx, b_dy, b_w, b_h, b_c = items[j]
                b_x0, b_x1 = b_dx, b_dx + b_w
                b_y0, b_y1 = b_dy, b_dy + b_h

                # Check horizontal overlap
                h_overlap = max(0.0, min(a_x1, b_x1) - max(a_x0, b_x0))
                smaller_w = min(a_w, b_w)
                if smaller_w <= 0 or (h_overlap / smaller_w < 0.25):
                    continue

                # Check vertical overlap
                v_overlap = max(0.0, min(a_y1, b_y1) - max(a_y0, b_y0))
                smaller_h = min(a_h, b_h)
                if smaller_h <= 0:
                    continue

                # Severe overlap threshold (matching delivery contract _OVERLAP_RATIO 0.15)
                overlap_ratio = v_overlap / smaller_h
                overlap_area = h_overlap * v_overlap
                if overlap_ratio >= 0.15 and overlap_area >= 30.0:
                    new_b_dy = round(a_y1 + min_gap, 2)
                    if new_b_dy > b_dy and new_b_dy + b_h <= max_y:
                        items[j][2] = new_b_dy
                        # Update local reference for subsequent iterations in loop
                        b_dy = new_b_dy
                        b_y0 = b_dy
                        b_y1 = b_dy + b_h
        return items

    def build_markup(
        self,
        pages: Sequence[FacsimilePageSpec],
        bg_pdf_filename: str,
        *,
        target_lang: str = "zh",
        asset_copier: Callable[[str], str | None] | None = None,
        formula_cropper: Callable[[Any], str | None] | None = None,
    ) -> str:
        """Generate Typst markup placing translated blocks on top of the stripped background.

        ``asset_copier`` and ``formula_cropper`` are forwarded to the L2 pages
        only. L1 pages do not need them: their figures and formulas are source
        ink carried by the stripped background layer. L2 pages get no background
        (they reflow), so without a copier their artwork is dropped and -- before
        the IMAGE branch existed -- its extracted asset *path* was typeset as
        prose in its place.
        """
        fonts_formatted = ", ".join(f'"{f}"' for f in self.font_stack)
        is_cjk = target_lang.lower().startswith(("zh", "ja", "ko"))
        self.preserved_placements = []
        lines: list[str] = [
            "// Universal Book Translator - 1:1 Facsimile Canvas Document",
            # ``ligatures: false`` is a text-layer decision, not a cosmetic one.
            # The delivered PDF is read back by the delivery contract, the
            # review workbench and copy/paste; a ligature glyph has to carry a
            # character mapping. Measured on a 31-page run, Typst's ligature
            # glyph for "ffi" extracted as ``Ï`` (U+00CF) once the document was
            # large enough to subset the font, so a ledger sentence
            # "a flexible and efficient ..." no longer appeared in the page text
            # and the contract reported the block as a truncated 88-char prefix.
            # Without ligatures the extracted text is exactly what was written.
            f'#set text(font: ({fonts_formatted}), lang: "{target_lang}", overhang: true, '
            "cjk-latin-spacing: auto, ligatures: false)",
            "",
        ]

        l2_renderer = PageLockedReflowRenderer(
            font_stack=self.font_stack, math_gate=self._math_gate
        )
        for p in pages:
            if p.tier == "L2" and p.l2_spec is not None:
                # An L2 page carries NO stripped background: it re-flows text in
                # reading order, and absolute source ink under a re-flowed page
                # is an ink collision by construction. Its figures and formulas
                # therefore have to be *drawn* -- re-laid in the flow from the
                # extracted assets, exactly as the reflow channel does -- or
                # routing a page to L2 drops all of its artwork (measured: 94
                # assets on a 26-page chapter, and their cache paths typeset as
                # prose in their place).
                lines.append(
                    l2_renderer.build_page_markup(
                        p.l2_spec,
                        target_lang=target_lang,
                        asset_copier=asset_copier,
                        formula_cropper=formula_cropper,
                    )
                )
                continue

            lines.append(
                f"#page(width: {p.page_w}pt, height: {p.page_h}pt, margin: 0pt, "
                f'background: image("{bg_pdf_filename}", page: {p.page_no}, width: 100%, height: 100%))['
            )
            # CJK characters have zero side-bearings; negative tracking causes
            # ideograph stroke collisions. Clamp negative tracking to 0 for CJK.
            effective_tracking = 0.0 if (is_cjk and p.tracking_em < 0) else p.tracking_em
            tracking = f", tracking: {effective_tracking}em" if effective_tracking else ""
            stretch = (
                f", stretch: {p.stretch_pct:.0f}%"
                if p.stretch_pct and p.stretch_pct < 100.0
                else ""
            )
            lines.append(f"  #set text(size: {p.uniform_size_pt}pt{tracking}{stretch})")
            # ``p.leading_em`` is a *ratio* (solver contract); Typst wants the
            # *gap* on top of the font's own line height. Without the
            # subtraction a 1.35 line height arrives as ~1.08 and captions
            # overlap.
            leading_gap = leading_gap_em(1.0 + p.leading_em)
            if is_cjk:
                # Chinese characters lack Latin ascender/descender padding; a 0.35em leading
                # results in glyph stroke claustrophobia. Enforce minimum 0.55em leading for CJK.
                leading_gap = max(0.55, leading_gap)
            lines.append(
                f'  #set par(justify: true, linebreaks: "optimized", leading: {leading_gap:.3f}em)'
            )

            # Extract placement records with top-down coordinates
            valid_items = []
            for pl in p.placements:
                if pl.skip_render or not pl.text.strip():
                    continue
                dx = round(pl.x0, 2)
                dy = round(p.page_h - pl.y1, 2)
                width = round(max(10.0, pl.x1 - pl.x0), 2)
                height = round(max(10.0, pl.y1 - pl.y0), 2)
                content = self._placement_content(
                    pl,
                    width_pt=width,
                    target_lang=target_lang,
                    base_font_size_pt=p.uniform_size_pt,
                )
                if not content:
                    # A preserved table: the background image already carries
                    # the source block, and the adapter records the skip.
                    continue
                valid_items.append([pl, dx, dy, width, height, content])

            valid_items = self._resolve_placement_collisions(valid_items, p.page_w, p.page_h)

            # Group contiguous narrative/heading prose within the same column into a flow group
            groups: list[list[Any]] = []
            for item in valid_items:
                f_pl: FacsimileBlockPlacement = cast(FacsimileBlockPlacement, item[0])
                f_dx: float = cast(float, item[1])
                f_dy: float = cast(float, item[2])
                f_w: float = cast(float, item[3])
                if not groups:
                    groups.append([item])
                    continue

                prev_group = groups[-1]
                prev_item = prev_group[-1]
                prev_pl: FacsimileBlockPlacement = cast(FacsimileBlockPlacement, prev_item[0])
                prev_dx: float = cast(float, prev_item[1])
                prev_dy: float = cast(float, prev_item[2])
                prev_w: float = cast(float, prev_item[3])

                # Criteria for intra-column natural flow:
                # 1. Neither block is a table or code block
                # 2. Both blocks sit in the same column track (or single-column track)
                # 3. Reading order descends down the column
                # 4. In the source document, they were adjacent paragraphs in the same text flow
                col = 0 if f_dx < 0.5 * p.page_w else 1
                prev_col = 0 if prev_dx < 0.5 * p.page_w else 1

                # Single-column documents (width > 60% of page_w) form a single track down the page.
                # The width tolerance is deliberately wider than the paragraph measure: a
                # lead-in sentence's bbox starts at its first-line indent, so it is ~50pt
                # narrower than the prose around it. Treating that as a different track left
                # a measured 120pt hole on page 2 of arXiv 2609.22978.
                is_single_col_pair = (
                    f_w > 0.60 * p.page_w
                    and prev_w > 0.60 * p.page_w
                    and abs(f_dx - prev_dx) <= 24.0
                    and abs(f_w - prev_w) <= 80.0
                )
                # Multi-column documents require both blocks to sit within the same column
                is_multi_col_pair = (
                    f_w <= 0.60 * p.page_w
                    and prev_w <= 0.60 * p.page_w
                    and col == prev_col
                    and abs(f_dx - prev_dx) <= 24.0
                )
                same_track = is_single_col_pair or is_multi_col_pair

                orig_prev_y0 = prev_pl.src_y0 if prev_pl.src_y0 is not None else prev_pl.y0
                orig_curr_y1 = f_pl.src_y1 if f_pl.src_y1 is not None else f_pl.y1
                src_gap = orig_prev_y0 - orig_curr_y1

                can_group = (
                    self.enable_column_flow
                    and not f_pl.is_table
                    and not prev_pl.is_table
                    and not f_pl.is_code
                    and not prev_pl.is_code
                    and not f_pl.is_caption
                    and not prev_pl.is_caption
                    and same_track
                    and f_dy >= prev_dy
                    and -4.0 <= src_gap <= 36.0
                )

                if can_group:
                    prev_group.append(item)
                else:
                    groups.append([item])

            for grp_idx, grp in enumerate(groups):
                # Intra-page safe vertical slack calculation:
                # If there is empty vertical space before the next item in the same track or page footer,
                # allow the box height budget to absorb up to that gap (minus safety margin) rather than hard-clipping.
                grp_bottom = (grp[-1][2] + grp[-1][4]) if len(grp) > 1 else (grp[0][2] + grp[0][4])
                slack_pt = 0.0
                if grp_idx + 1 < len(groups):
                    next_item = groups[grp_idx + 1][0]
                    if next_item[2] >= grp_bottom and abs(next_item[1] - grp[0][1]) <= 60.0:
                        slack_pt = max(0.0, (next_item[2] - grp_bottom) - 8.0)
                else:
                    slack_pt = max(0.0, (p.page_h - 36.0) - grp_bottom)
                base_h = (grp[-1][2] + grp[-1][4]) - grp[0][2] if len(grp) > 1 else grp[0][4]
                safe_slack = round(min(slack_pt, max(12.0, base_h * 0.4), 45.0), 2)

                if len(grp) == 1:
                    pl, dx, dy, width, height, content = grp[0]
                    fit_h = round(height + safe_slack, 2)
                    lines.append(f"  // bbox: {dx} {dy} {width} {height}")
                    # Table mask: paint a white background to completely eliminate ghost vector rules
                    if pl.is_table:
                        lines.append(
                            f"  #place(top + left, dx: {dx}pt, dy: {dy}pt, "
                            f'block(width: {width}pt, height: {height}pt, fill: rgb("#ffffff"), clip: true, outset: (top: 2.5pt, bottom: 2.5pt))[\n'
                            f"    {content}\n"
                            f"  ])"
                        )
                    elif pl.is_caption:
                        # Captions used to be drawn with ``clip: false`` and a
                        # hardcoded 9pt fallback, so a long caption simply ran
                        # past its 20pt box (measured: two lines outside it)
                        # while the text layer kept every word. Fit it to its
                        # own box; the caption floor is lower than body's
                        # because a figure's 8pt caption is not body prose.
                        lines.append(
                            self._fit_to_box(
                                dx=dx,
                                dy=dy,
                                width=width,
                                height=fit_h,
                                base_size_pt=pl.font_size_pt or 9.0,
                                outset="(top: 2.0pt, bottom: 2.0pt)",
                                content=content,
                                floor_pt=6.5,
                                ladder=False,
                                page_tracking_em=effective_tracking,
                            )
                        )
                    else:
                        eff_stretch = min(pl.stretch_pct, p.stretch_pct)
                        if eff_stretch < 100.0:
                            scale_factor = eff_stretch / 100.0
                            inner_w = round(width / scale_factor, 2)
                            lines.append(
                                f"  #place(top + left, dx: {dx}pt, dy: {dy}pt, "
                                f"block(width: {width}pt, height: {fit_h}pt, clip: true, outset: (top: 2.5pt, bottom: 2.5pt))[\n"
                                f"    #scale(x: {eff_stretch:.0f}%, origin: left + top)[\n"
                                f"      #block(width: {inner_w}pt)[\n"
                                f"        {content}\n"
                                f"      ]\n"
                                f"    ]\n"
                                f"  ])"
                            )
                        else:
                            lines.append(
                                self._fit_to_box(
                                    dx=dx,
                                    dy=dy,
                                    width=width,
                                    height=fit_h,
                                    base_size_pt=p.uniform_size_pt,
                                    outset="(top: 2.5pt, bottom: 2.5pt)",
                                    content=f"        {content}",
                                    page_tracking_em=effective_tracking,
                                )
                            )
                else:
                    # Flow group of contiguous paragraphs in column
                    grp_dx = round(min(it[1] for it in grp), 2)
                    grp_dy = round(grp[0][2], 2)
                    grp_width = round(max(it[3] for it in grp), 2)
                    grp_height = round((grp[-1][2] + grp[-1][4]) - grp[0][2], 2)
                    fit_grp_h = round(grp_height + safe_slack, 2)
                    lines.append(
                        f"  // flow-group: {len(grp)} blocks dx={grp_dx} dy={grp_dy} w={grp_width} h={grp_height}"
                    )
                    flow_body = []
                    for idx, (it_pl, _, _, _, _, it_content) in enumerate(grp):
                        if idx > 0:
                            # Preserve the source paragraph/section gap instead of a
                            # fixed 0.65em: a section heading sits ~24pt above its
                            # body in the source, and collapsing that to 0.4em
                            # made the heading look attached to the previous paragraph.
                            prev_pl = grp[idx - 1][0]
                            prev_y0 = prev_pl.src_y0 if prev_pl.src_y0 is not None else prev_pl.y0
                            curr_y1 = it_pl.src_y1 if it_pl.src_y1 is not None else it_pl.y1
                            gap_em = min(
                                2.2, max(0.35, (prev_y0 - curr_y1) / max(1.0, p.uniform_size_pt))
                            )
                            flow_body.append(f"    #v({gap_em:.3f}em)")
                        flow_body.append(f"    #block[\n      {it_content}\n    ]")
                    flow_content_str = "\n".join(flow_body)
                    eff_stretch = min(min(it[0].stretch_pct for it in grp), p.stretch_pct)
                    if eff_stretch < 100.0:
                        scale_factor = eff_stretch / 100.0
                        inner_w = round(grp_width / scale_factor, 2)
                        lines.append(
                            f"  #place(top + left, dx: {grp_dx}pt, dy: {grp_dy}pt, "
                            f"block(width: {grp_width}pt, height: {fit_grp_h}pt, clip: true, outset: (top: 3.0pt, bottom: 3.0pt))[\n"
                            f"    #scale(x: {eff_stretch:.0f}%, origin: left + top)[\n"
                            f"      #block(width: {inner_w}pt)[\n"
                            f"{flow_content_str}\n"
                            f"      ]\n"
                            f"    ]\n"
                            f"  ])"
                        )
                    else:
                        lines.append(
                            self._fit_to_box(
                                dx=grp_dx,
                                dy=grp_dy,
                                width=grp_width,
                                height=fit_grp_h,
                                base_size_pt=p.uniform_size_pt,
                                outset="(top: 3.0pt, bottom: 3.0pt)",
                                content=flow_content_str,
                                page_tracking_em=effective_tracking,
                            )
                        )

            lines.append("]")
            lines.append("")

        return "\n".join(lines)

    def render_document(
        self,
        pages: Sequence[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        *,
        target_lang: str = "zh",
        asset_copier: Callable[[str], str | None] | None = None,
        formula_cropper: Callable[[Any], str | None] | None = None,
        compile_fn: Callable[[str, str, str], tuple[bool, str]] | None = None,
    ) -> Path:
        """Compile the facsimile pages into the final output PDF."""
        restrict_dir_to_owner(build_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        local_bg = build_dir / "bg_stripped.pdf"
        if bg_pdf_path.resolve() != local_bg.resolve():
            shutil.copyfile(bg_pdf_path, local_bg)

        markup = self.build_markup(
            pages,
            local_bg.name,
            target_lang=target_lang,
            asset_copier=asset_copier,
            formula_cropper=formula_cropper,
        )
        typ_path = build_dir / "facsimile.typ"
        typ_path.write_text(markup, encoding="utf-8")

        #: One callable type for both branches. Annotating ``compiler`` from
        #: scratch (rather than letting the ``if``/``else`` infer it) matters:
        #: without it mypy pins the variable to the nested function's def type
        #: and then rejects the caller's ``Callable``, which is the same shape.
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
                f"Facsimile compile failed: {stderr.strip() or 'unknown Typst error'} "
                f"(source kept at {typ_path})"
            )
        return output_path


__all__ = [
    "DEFAULT_FONT_STACK",
    "FacsimileBlockPlacement",
    "FacsimilePageSpec",
    "FacsimileRenderer",
]
