"""Region assembly for the rigid typesetter.

The rigid paradigm replaces per-line box pairing with **block zones**:
a paragraph owns a rectangular region (its docling bbox grown over the rows
that textually belong to it, bounded by figures, other blocks and the
chrome bands) and the translated text is typeset into that region with an
adaptive font size. No per-row coverage gate, no claim order, no merge
windows.

Zone evidence (fail-closed, deliberately simpler than pairing):

- a row belongs to a block when its token precision against the block's
  source clears the prune bar (``PREC_MIN`` / ``HITS_MIN``) — the same
  text rung pairing uses to reject equation shards;
- the docling bbox is the seed; owned rows connect through blank gaps and
  stop at the first foreign row (figure furniture, neighbouring
  paragraphs), so a first-zone bbox still grows over its own paragraph but
  never over a plot;
- unowned runs on the block's own page may rejoin it when they match
  (post-equation halves, side-by-side column continuations); runs on later
  pages need the strong exact-substring rung
  (:func:`rigid.rows._row_is_continuation`), mirroring the cross-page
  continuation evidence discipline;
- image bounds and block bboxes are hard guards: zones never include them.

Pure functions over extracted geometry — no PDF, no Typst, unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

from ubt.adapters.pdf.rigid.rows import (
    _reclaimable,
    _row_is_continuation,
    _row_precision_hits,
    content_token_count,
)
from ubt.adapters.pdf.textgeom import LineBox, _aggregate_line_styles, column_order
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.policy.layout_policy import HEADER_BAND_PT, ROW_MERGE_GAP_PT
from ubt.model.ast import RegionKind

Rect = tuple[float, float, float, float]

PREC_MIN = 0.6
HITS_MIN = 2
BASE_HEIGHT_RATIO = 0.95
BASE_SIZE_MIN = 8.0
BASE_SIZE_MAX = 24.0
ZONE_PAD_PT = 1.5
ZONE_GAP_PT = 3.5
# Connected owned rows may be at most this many line heights apart.
CONNECT_H_FACTOR = 2.5
# A same-page unowned run rejoins the block only within this distance below
# its bottom zone (post-equation halves sit right after the guard).
TAIL_SEARCH_PT = 60.0


@dataclass(frozen=True)
class PageFacts:
    """Extracted page geometry the zone builder consumes (no PDF access)."""

    page: int
    width: float
    height: float
    lines: tuple[LineBox, ...] = ()
    images: tuple[Rect, ...] = ()
    # Sampled page background for textless (scanned) pages: the raster keeps
    # its own glyphs, so painted zones must cover them with an opaque rect
    # before the translation is drawn. None on born-digital pages.
    bg: str | None = None


@dataclass(frozen=True)
class Zone:
    """One paintable region of one block on one page (bottom-up coords)."""

    block_id: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    base_size: float
    rows: tuple[str, ...] = ()
    kind: str = "own"
    bold: bool = False
    italic: bool = False
    align: str = "left"

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def rect(self) -> Rect:
        return (self.x0, self.y0, self.x1, self.y1)

    def with_y1(self, y1: float) -> Zone:
        return Zone(
            self.block_id,
            self.page,
            self.x0,
            self.y0,
            self.x1,
            y1,
            self.base_size,
            self.rows,
            self.kind,
            self.bold,
            self.italic,
            self.align,
        )

    def with_y0(self, y0: float) -> Zone:
        return Zone(
            self.block_id,
            self.page,
            self.x0,
            y0,
            self.x1,
            self.y1,
            self.base_size,
            self.rows,
            self.kind,
            self.bold,
            self.italic,
            self.align,
        )


# Bottom clamp: keep zones above the page-number band. FOOTER_BAND_PT (55)
# would swallow legitimate caption tails that sit low on the page; the page
# number itself is never a matched row.
CONTENT_BOTTOM_PT = 30.0


def content_window(facts: PageFacts) -> tuple[float, float]:
    """Page content band in bottom-up coords (bottom clamp, header)."""
    return (CONTENT_BOTTOM_PT, facts.height - HEADER_BAND_PT - 2.0)


# Chrome painting (translate_chrome) opts out of the header/footer bands:
# a running head lives inside the band the body window deliberately excludes.
CHROME_MARGIN_PT = 2.0


def _clamp_window(
    block: IRBlock,
    facts: PageFacts,
    region: tuple[float, float],
    allow_chrome_bands: bool,
) -> tuple[float, float]:
    """Clamp a zone's y-span to the block's allowed page band."""
    if allow_chrome_bands and block.region in (RegionKind.HEADER, RegionKind.FOOTER):
        window = (CHROME_MARGIN_PT, facts.height - CHROME_MARGIN_PT)
    else:
        window = content_window(facts)
    y0, y1 = region
    return (max(y0, window[0]), min(y1, window[1]))


def _median_height(lines: tuple[LineBox, ...], default: float = 10.0) -> float:
    heights = [ln.rect[3] - ln.rect[1] for ln in lines if ln.rect[3] > ln.rect[1]]
    return median(heights) if heights else default


def _row_matches(text: str, source: str) -> bool:
    from ubt.adapters.pdf.docling_blocks import parse_toc_entry_line

    parsed = parse_toc_entry_line(text or "")
    cmp_text = parsed[0] if parsed is not None else (text or "")
    prec, hits = _row_precision_hits(cmp_text, source or "")
    if prec >= PREC_MIN and hits >= HITS_MIN:
        return True
    # Single-word blocks (e.g. standalone 'Proof.', 'Abstract', 'References')
    # only have 1 token in ``source``, so ``hits`` can never reach ``HITS_MIN=2``.
    src_tokens = re.findall(r"[a-z0-9]+", (source or "").lower())
    return len(src_tokens) == 1 and hits == 1 and prec >= 0.99


def _in_other_bbox(row: LineBox, block: IRBlock, others: tuple[IRBlock, ...], page: int) -> bool:
    """True when another block's bbox on ``page`` already claims this row."""
    cx = (row.rect[0] + row.rect[2]) / 2
    cy = (row.rect[1] + row.rect[3]) / 2
    for other in others:
        if other.id == block.id or other.bbox is None or other.bbox.page != page:
            continue
        bb = other.bbox
        if bb.x0 - 1 <= cx <= bb.x1 + 1 and bb.y0 - 1 <= cy <= bb.y1 + 1:
            return True
    return False


def _edge_row(row: LineBox, source: str) -> bool:
    """A garbled prose edge row: weaker bar, still source vocabulary.

    OCR drops subscripts and hyphens ("Nch" for ``N_ch``), which pushes a
    genuine paragraph edge row below :func:`_row_matches`; one such row per
    zone edge may be absorbed when most of its words still come from the
    block. Figure furniture ("Metal Gate", "0.9") shares no vocabulary.
    """
    text = row.text or ""
    if sum(1 for c in text if c.isalpha()) < 6:
        return False
    prec, _hits = _row_precision_hits(text, source or "")
    alpha_tokens = [t for t in re.findall(r"[A-Za-z]+", text.lower()) if len(t) > 1]
    vocab = set(re.findall(r"[a-z0-9]+", (source or "").lower()))
    alpha_hits = sum(1 for t in alpha_tokens if t in vocab)
    needed = 1 if len(alpha_tokens) <= 3 else 2
    return prec >= 0.5 and alpha_hits >= needed


def _rect_of(lines: tuple[LineBox, ...]) -> Rect:
    return (
        min(ln.rect[0] for ln in lines) - ZONE_PAD_PT,
        min(ln.rect[1] for ln in lines) - ZONE_PAD_PT,
        max(ln.rect[2] for ln in lines) + ZONE_PAD_PT,
        max(ln.rect[3] for ln in lines) + ZONE_PAD_PT,
    )


def _base_size(lines: tuple[LineBox, ...]) -> float:
    fsz, _bold, _italic = _aggregate_line_styles(lines)
    if fsz >= 4.5:
        return min(BASE_SIZE_MAX, max(BASE_SIZE_MIN, fsz))
    height = _median_height(lines)
    return min(BASE_SIZE_MAX, max(BASE_SIZE_MIN, height * BASE_HEIGHT_RATIO))


def _zone_style(block: IRBlock, lines: tuple[LineBox, ...]) -> tuple[bool, bool]:
    _fsz, is_bold, is_italic = _aggregate_line_styles(lines)
    if block.block_type == BlockType.HEADING:
        is_bold = True
    return is_bold, is_italic


def _resolve_horizontal_span_and_align(
    block: IRBlock,
    facts: PageFacts,
    best: list[LineBox],
    all_rows: list[LineBox],
    others: tuple[IRBlock, ...],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float, str]:
    """Detect symmetric center alignment, single-line right-margin expansion, and paragraph justify."""
    if block.provenance.get("toc_entry"):
        return x0, x1, "left"
    page_w = facts.width
    if page_w <= 0.0:
        return x0, x1, "left"
    page_mid = page_w / 2.0
    block_mid = (x0 + x1) / 2.0
    content_left = min((r.rect[0] for r in all_rows), default=70.0)
    content_right = max((r.rect[2] for r in all_rows), default=page_w - 70.0)

    # Check if any foreign block or image occupies the same vertical band [y0, y1]
    owned_ids = {id(ln) for ln in best}
    has_left_obstacle = False
    has_right_obstacle = False
    right_limit = max(x1, min(page_w - 48.0, max(content_right, page_w - 72.0)))

    for other in others:
        if other.id == block.id or other.bbox is None or other.bbox.page != facts.page:
            continue
        obb = other.bbox
        if min(y1, obb.y1) - max(y0, obb.y0) > 1.5:
            if obb.x1 <= x0 + 4.0:
                has_left_obstacle = True
            if obb.x0 >= x1 - 4.0:
                has_right_obstacle = True
                right_limit = min(right_limit, obb.x0 - ZONE_GAP_PT)

    for img_x0, img_y0, img_x1, img_y1 in facts.images:
        if min(y1, img_y1) - max(y0, img_y0) > 1.5:
            if img_x1 <= x0 + 4.0:
                has_left_obstacle = True
            if img_x0 >= x1 - 4.0:
                has_right_obstacle = True
                right_limit = min(right_limit, img_x0 - ZONE_GAP_PT)

    for r in all_rows:
        if id(r) in owned_ids:
            continue
        rx0, ry0, rx1, ry1 = r.rect
        if min(y1, ry1) - max(y0, ry0) > 2.0:
            if rx1 <= x0 + 4.0:
                has_left_obstacle = True
            if rx0 >= x1 - 4.0:
                has_right_obstacle = True
                right_limit = min(right_limit, rx0 - ZONE_GAP_PT)

    # 1. Center alignment: block and all its lines are symmetric around page midpoint.
    # Indented list items (BlockType.LIST_ITEM) are never centered blocks even when
    # their lines happen to reach near the right margin.
    lines_centered = all(abs((ln.rect[0] + ln.rect[2]) / 2.0 - page_mid) <= 18.0 for ln in best)
    is_indented_from_margins = x0 > content_left + 12.0
    is_centered_heading = block.block_type == BlockType.HEADING and len(best) <= 2
    if (
        block.block_type != BlockType.LIST_ITEM
        and not has_left_obstacle
        and not has_right_obstacle
        and abs(block_mid - page_mid) <= 14.0
        and lines_centered
        and (is_indented_from_margins or is_centered_heading)
    ):
        sym_margin = max(36.0, min(x0, page_w - x1, content_left))
        return sym_margin, page_w - sym_margin, "center"

    # 2. Single-line block / Heading / Footnote with unoccupied right margin:
    # release x1 into blank space so translated text (or wider CJK/Latin mix) never wraps/clips
    if (
        not has_right_obstacle
        and right_limit > x1 + 4.0
        and (
            len(best) == 1
            or block.block_type == BlockType.HEADING
            or block.region == RegionKind.FOOTNOTE
        )
    ):
        return x0, right_limit, "left"

    # 3. Multi-line narrative paragraph or list item spanning column width -> justify
    if (
        block.block_type in (BlockType.NARRATIVE, BlockType.LIST_ITEM)
        and len(best) >= 2
        and (x1 - x0) >= 0.38 * page_w
    ):
        return x0, x1, "justify"

    return x0, x1, "left"


def visual_rows(lines: tuple[LineBox, ...]) -> list[LineBox]:
    """Fold same-line fragments into one visual row (pdfium word soup).

    ``extract_lines`` mostly merges fragments already, but sub/superscript
    shards and double emissions survive on some pages; pairing them as
    separate rows would both inflate base size and break run connectivity.
    Folding is capped at ``ROW_MERGE_GAP_PT`` on the x axis so side-by-side
    column lines sharing a baseline never merge into one cross-gutter row.
    """
    bands: list[list[LineBox]] = []
    for ln in sorted(lines, key=lambda v: (-v.rect[3], v.rect[0])):
        h = ln.rect[3] - ln.rect[1]
        hit = None
        for band in bands:
            y0 = min(v.rect[1] for v in band)
            y1 = max(v.rect[3] for v in band)
            overlap = min(y1, ln.rect[3]) - max(y0, ln.rect[1])
            if not (h > 0 and overlap >= 0.5 * h):
                continue
            bx0 = min(v.rect[0] for v in band)
            bx1 = max(v.rect[2] for v in band)
            gap = max(ln.rect[0] - bx1, bx0 - ln.rect[2])
            if gap > ROW_MERGE_GAP_PT:
                continue
            hit = band
            break
        if hit is None:
            bands.append([ln])
        else:
            hit.append(ln)
    out: list[LineBox] = []
    for band in bands:
        band.sort(key=lambda v: v.rect[0])
        text = " ".join((v.text or "").strip() for v in band if (v.text or "").strip())
        x0 = min(v.rect[0] for v in band)
        y0 = min(v.rect[1] for v in band)
        x1 = max(v.rect[2] for v in band)
        y1 = max(v.rect[3] for v in band)
        fsz, is_bold, is_italic = _aggregate_line_styles(band)
        out.append(
            LineBox(
                text,
                (x0, y0, x1, y1),
                font_size=fsz,
                bold=is_bold,
                italic=is_italic,
            )
        )
    out.sort(key=lambda v: -v.rect[3])
    return out


def _connected_runs(rows: list[LineBox]) -> list[list[LineBox]]:
    """Group top-down rows into runs with blank-gap tolerance."""
    h = _median_height(tuple(rows)) if rows else 10.0
    runs: list[list[LineBox]] = []
    for ln in sorted(rows, key=lambda v: -v.rect[3]):
        if runs and runs[-1][-1].rect[1] - ln.rect[3] <= CONNECT_H_FACTOR * h:
            runs[-1].append(ln)
        else:
            runs.append([ln])
    return runs


def own_zone(
    block: IRBlock,
    facts: PageFacts,
    others: tuple[IRBlock, ...] = (),
    *,
    allow_chrome_bands: bool = False,
) -> Zone | None:
    """The block's region on its home page (bbox seed grown over owned rows)."""
    bbox = block.bbox
    if bbox is None or bbox.page != facts.page:
        return None
    source = block.source_text or ""
    rows = visual_rows(tuple(ln for ln in facts.lines if not ln.table_band))
    if not rows:
        return _bbox_zone(block, facts, allow_chrome_bands=allow_chrome_bands)
    h = _median_height(tuple(rows))
    # Seed: rows overlapping the docling band. Growth is DOWNWARD only —
    # docling stores a block's first zone, and upward precision matches are
    # how the previous paragraph's tail gets absorbed (same vocabulary).
    seed = [
        ln
        for ln in rows
        if ln.rect[3] > bbox.y0
        and ln.rect[1] < bbox.y1
        and ln.rect[0] < bbox.x1
        and ln.rect[2] > bbox.x0
        and (_row_matches(ln.text or "", source) or _edge_row(ln, source))
    ]
    if not seed:
        # Caption layout: docling's first zone is the "FIG. 3.3" label box
        # (28pt wide, no content hits); the caption itself is the nearest
        # matching run below it.
        best_run: list[LineBox] | None = None
        gap = float("inf")
        matched = [
            ln
            for ln in rows
            if _row_matches(ln.text or "", source)
            and not _in_other_bbox(ln, block, others, facts.page)
        ]
        for run_candidate in _connected_runs(matched):
            top = max(ln.rect[3] for ln in run_candidate)
            if top > bbox.y0 + 1.0:
                continue
            if not any(ln.rect[0] < bbox.x1 and ln.rect[2] > bbox.x0 for ln in run_candidate):
                continue
            candidate_gap = bbox.y0 - top
            if candidate_gap < gap:
                best_run, gap = run_candidate, candidate_gap
        if best_run is None or gap > CONNECT_H_FACTOR * h:
            # A prose block sitting on table bands — a table row/cell extracted as
            # prose — gets no seed because ``rows`` drops table rows. Its bbox is
            # still authoritative, so paint it strictly inside that box: no growth
            # and no right-margin expansion into the next column, which is exactly
            # the overlap a table cell must not cause.
            table_rows = [
                ln
                for ln in facts.lines
                if ln.table_band
                and ln.rect[3] > bbox.y0
                and ln.rect[1] < bbox.y1
                and ln.rect[0] < bbox.x1
                and ln.rect[2] > bbox.x0
                and (_row_matches(ln.text or "", source) or _edge_row(ln, source))
            ]
            if not table_rows:
                return None
            ty0, ty1 = _clamp_window(block, facts, (bbox.y0, bbox.y1), allow_chrome_bands)
            if ty1 - ty0 < 1.0 or bbox.x1 - bbox.x0 < 1.0:
                return None
            is_bold, is_italic = _zone_style(block, tuple(table_rows))
            return Zone(
                block_id=block.id,
                page=facts.page,
                x0=bbox.x0,
                y0=ty0,
                x1=bbox.x1,
                y1=ty1,
                base_size=_base_size(tuple(table_rows)),
                rows=(),
                kind="own",
                bold=is_bold,
                italic=is_italic,
            )
        seed = best_run
    run = list(seed)
    bottom = min(ln.rect[1] for ln in run)
    for ln in sorted(rows, key=lambda v: -v.rect[3]):
        if ln.rect[3] > bottom + 1.0:
            continue
        if id(ln) in {id(r) for r in run}:
            continue
        if bottom - ln.rect[3] > CONNECT_H_FACTOR * h:
            break
        # A row another block's own bbox claims always ends the growth —
        # topical vocabulary must not swallow the next heading/paragraph.
        if _in_other_bbox(ln, block, others, facts.page):
            break
        # A line indented >50pt past a left-margin paragraph's x0 is a centered
        # display equation sharing math variables with the preceding theorem/prose,
        # never a continuation line of the paragraph.
        if ln.rect[0] > bbox.x0 + 50.0:
            break
        if _row_matches(ln.text or "", source):
            run.append(ln)
            bottom = ln.rect[1]
            continue
        break
    # One garbled trailing edge row (caption tails, hyphen-dropped rows).
    trailing = [
        ln
        for ln in sorted(rows, key=lambda v: -v.rect[3])
        if ln.rect[3] <= bottom + 1.0
        and id(ln) not in {id(r) for r in run}
        and bottom - ln.rect[3] <= 1.2 * h
        and ln.rect[0] <= bbox.x0 + 50.0
        and _edge_row(ln, source)
        and not _in_other_bbox(ln, block, others, facts.page)
    ]
    if trailing:
        run.append(trailing[0])
        bottom = trailing[0].rect[1]
    best = run
    top = max(ln.rect[3] for ln in best)
    if max(0.0, bbox.y0 - top, bottom - bbox.y1) > CONNECT_H_FACTOR * h:
        return None
    region = _rect_of(tuple(best))
    x0 = min(bbox.x0, region[0])
    y0 = min(bbox.y0, region[1])
    x1 = max(bbox.x1, region[2])
    y1 = max(bbox.y1, region[3])
    base_sz = _base_size(tuple(best))
    is_bold, is_italic = _zone_style(block, tuple(best))
    # Single-line blocks whose letters lack descenders (e.g. '1. Introduction' or
    # '1.2.3. The Coarse-Grained Workaround') have ink height smaller than their true
    # point size. Ensure vertical box height accommodates at least 1 line at base_sz.
    min_single_h = base_sz * 0.85 + ZONE_PAD_PT * 1.5
    if len(best) == 1 and (y1 - y0) < min_single_h:
        y0 = y1 - min_single_h
    y0, y1 = _clamp_window(block, facts, (y0, y1), allow_chrome_bands)
    if y1 - y0 < 1.0:
        return None
    x0, x1, align = _resolve_horizontal_span_and_align(
        block, facts, best, rows, others, x0, y0, x1, y1
    )
    return Zone(
        block_id=block.id,
        page=facts.page,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        base_size=base_sz,
        rows=tuple(ln.text or "" for ln in best),
        kind="own",
        bold=is_bold,
        italic=is_italic,
        align=align,
    )


def _bbox_zone(
    block: IRBlock, facts: PageFacts, *, allow_chrome_bands: bool = False
) -> Zone | None:
    """Seed-only zone for a page with no usable text rows (belt and braces).

    Textless pages normally arrive with authored line rows (OCR member
    lines or bbox slices), so this only triggers when even those are
    missing: the docling/OCR bbox is still a valid region, and the
    typesetter fails closed on its own if the box cannot hold the text.
    """
    bbox = block.bbox
    if bbox is None:
        return None
    y0, y1 = _clamp_window(block, facts, (bbox.y0, bbox.y1), allow_chrome_bands)
    if y1 - y0 < 1.0 or bbox.x1 - bbox.x0 < 1.0:
        return None
    return Zone(
        block_id=block.id,
        page=facts.page,
        x0=bbox.x0,
        y0=y0,
        x1=bbox.x1,
        y1=y1,
        base_size=_base_size(()),
        rows=(),
        kind="own",
        bold=(block.block_type == BlockType.HEADING),
    )


def _column_disjoint(run: tuple[LineBox, ...], zones: list[Zone]) -> bool:
    """True when the run shares no x-band with any of the block's zones."""
    rx0 = min(ln.rect[0] for ln in run)
    rx1 = max(ln.rect[2] for ln in run)
    return all(rx1 <= z.x0 + 1.0 or rx0 >= z.x1 - 1.0 for z in zones)


def _covered_by(rect: Rect, covers: tuple[Rect, ...], slack: float = 1.0) -> bool:
    cx = (rect[0] + rect[2]) / 2
    cy = (rect[1] + rect[3]) / 2
    for c in covers:
        if c[0] - slack <= cx <= c[2] + slack and c[1] - slack <= cy <= c[3] + slack:
            return True
    return False


def orphan_runs(facts: PageFacts, covers: tuple[Rect, ...]) -> list[tuple[LineBox, ...]]:
    """Contiguous rows owned by no zone and no guard, top-down."""
    rows = [
        ln
        for ln in visual_rows(tuple(facts.lines))
        if not ln.table_band and not _covered_by(ln.rect, covers) and _reclaimable(ln)
    ]
    return [tuple(run) for run in _connected_runs(rows)]


def _run_zone(
    block: IRBlock,
    run: tuple[LineBox, ...],
    page: int,
    *,
    strong: bool,
    others: tuple[IRBlock, ...] = (),
) -> Zone | None:
    """Turn an orphan run into a continuation zone (strong/weak evidence)."""
    source = block.source_text or ""
    if strong:
        matched = [ln for ln in run if _row_is_continuation(ln.text or "", source)]
    else:
        matched = [ln for ln in run if _row_matches(ln.text or "", source)]
    # Wrapped fragments ("expressed by") carry too few content tokens to be
    # judged; they neither count in the denominator nor veto a run whose
    # substantive rows match. A run with nothing judgeable falls back to the
    # strict whole-run ratio.
    judged = [ln for ln in run if content_token_count(ln.text or "") >= HITS_MIN] or list(run)
    if not matched or len(matched) < len(judged) * 0.6:
        return None
    # Evidence is the matched rows; geometry takes the whole run when its
    # unmatched edge rows are still source vocabulary (OCR subscript
    # damage: "dependence, Nch is the channel doping ..."). Strong runs are
    # gated by _row_is_continuation, which _row_matches need not agree with
    # (a short tail like "agent." clears the tail rung but not the substring
    # matcher) -- the matched rows must survive into the geometry.
    matched_ids = {id(ln) for ln in matched}
    kept = tuple(
        ln
        for ln in run
        if id(ln) in matched_ids
        or _row_matches(ln.text or "", source)
        or (_edge_row(ln, source) and not _in_other_bbox(ln, block, others, page))
    )
    if not kept:
        return None
    region = _rect_of(kept)
    is_bold, is_italic = _zone_style(block, kept)
    cont_align = "justify" if block.block_type == BlockType.NARRATIVE and len(kept) >= 2 else "left"
    return Zone(
        block_id=block.id,
        page=page,
        x0=region[0],
        y0=region[1],
        x1=region[2],
        y1=region[3],
        base_size=_base_size(kept),
        rows=tuple(ln.text or "" for ln in kept),
        kind="continuation",
        bold=is_bold,
        italic=is_italic,
        align=cont_align,
    )


def build_zones(
    pages: dict[int, PageFacts],
    blocks: list[IRBlock],
    *,
    allow_chrome_bands: bool = False,
) -> dict[str, tuple[Zone, ...]]:
    """Own + continuation zones per prose block, reading order within each block.

    ``blocks`` is the full block list: non-prose blocks (formulas, images,
    tables) and chrome seed no zones but are hard guards, exactly like
    images — a paragraph region never swallows an equation.
    """
    from ubt.core.policy.layout_policy import PROSE_BLOCK_TYPES

    prose = [b for b in blocks if b.block_type in PROSE_BLOCK_TYPES]
    by_id: dict[str, list[Zone]] = {}
    accepted: dict[int, list[Zone]] = {}

    def _accept(zone: Zone) -> bool:
        min_h = max(zone.base_size * 0.8, 8.0)
        for other in accepted.get(zone.page, []):
            if zone.x0 < other.x1 and other.x0 < zone.x1:
                if other.y0 >= zone.y0:
                    if zone.y1 > other.y0 - ZONE_GAP_PT:
                        clipped = zone.with_y1(other.y0 - ZONE_GAP_PT)
                        if clipped.height < min_h:
                            return False
                        zone = clipped
                elif zone.y0 >= other.y0 and zone.y0 < other.y1 + ZONE_GAP_PT:
                    clipped = zone.with_y0(other.y1 + ZONE_GAP_PT)
                    if clipped.height < min_h:
                        return False
                    zone = clipped
        accepted.setdefault(zone.page, []).append(zone)
        by_id.setdefault(zone.block_id, []).append(zone)
        return True

    for block in prose:
        if block.bbox is None:
            continue
        facts = pages.get(block.bbox.page)
        if facts is None:
            continue
        zone = own_zone(block, facts, tuple(blocks), allow_chrome_bands=allow_chrome_bands)
        if zone is not None:
            _accept(zone)

    guards: dict[int, tuple[Rect, ...]] = {}
    for page, facts in pages.items():
        rects: list[Rect] = list(facts.images)
        rects.extend(
            (b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1)
            for b in blocks
            if b.bbox is not None and b.bbox.page == page
        )
        guards[page] = tuple(rects)

    runs_by_page: dict[int, list[tuple[LineBox, ...]]] = {}
    for page, facts in pages.items():
        covers = tuple(z.rect for z in accepted.get(page, [])) + guards[page]
        runs_by_page[page] = orphan_runs(facts, covers)

    def _consume(page: int, run: tuple[LineBox, ...]) -> None:
        runs = runs_by_page.get(page, [])
        runs_by_page[page] = [r for r in runs if r is not run]

    for block in prose:
        if block.bbox is None or not (block.source_text or "").strip():
            continue
        page = block.bbox.page
        if page in runs_by_page:
            # Top-down so each accepted tail lowers the reference bottom: a
            # paragraph whose fragments are separated by several display
            # formulas claims every one of them, not just the first.
            for run in sorted(runs_by_page[page], key=lambda r: -max(ln.rect[3] for ln in r)):
                if all(run is not pending for pending in runs_by_page.get(page, ())):
                    continue  # already consumed by an earlier block
                zones = by_id.get(block.id, [])
                bottom = min((z.y0 for z in zones), default=None)
                if bottom is None:
                    break
                top = max(ln.rect[3] for ln in run)
                if top > bottom:
                    # Side-by-side column continuation: the run starts above
                    # the zone bottom because it is the next column of the
                    # same paragraph. Same-band runs stay blocked by the
                    # downward-only growth rule.
                    if not _column_disjoint(run, zones):
                        continue
                else:
                    if min(ln.rect[0] for ln in run) > min(z.x0 for z in zones) + 50.0:
                        continue
                    gap = bottom - top
                    # Across a display-formula guard the gap may be large; an
                    # open stretch of that size is a different block's furniture.
                    crossed_guard = any(
                        g[1] >= top - 1.0 and g[3] <= bottom + 1.0 for g in guards.get(page, ())
                    )
                    if gap > (TAIL_SEARCH_PT if crossed_guard else 2.5 * _median_height(run)):
                        continue
                tail = _run_zone(block, run, page, strong=False, others=tuple(blocks))
                if tail is None or not _accept(tail):
                    continue
                _consume(page, run)
        for page_no in sorted(runs_by_page):
            if page_no <= page:
                continue
            if page_no > page + 2:
                break
            for run in list(runs_by_page[page_no]):
                tail = _run_zone(block, run, page_no, strong=True, others=tuple(blocks))
                if tail is None or not _accept(tail):
                    continue
                _consume(page_no, run)

    # Axiom B safety net: a prose block whose region is a sub-line fragment gets
    # no accepted zone above, and the rigid engine would otherwise ship its
    # source (render:no_zone). Give it its own region anyway -- its matched rows,
    # or its bbox -- so the translation is painted (the engine then shrinks or
    # overflows as needed). Overlap with a neighbour is accepted: the agreed
    # priority puts translation completeness above pixel layout.
    for block in prose:
        if block.bbox is None or block.id in by_id:
            continue
        facts = pages.get(block.bbox.page)
        if facts is None:
            continue
        zone = own_zone(block, facts, tuple(blocks), allow_chrome_bands=allow_chrome_bands)
        if zone is None:
            zone = _bbox_zone(block, facts, allow_chrome_bands=allow_chrome_bands)
        if zone is not None and zone.height >= 1.0:
            by_id.setdefault(block.id, []).append(zone)

    return {bid: tuple(_order_block_zones(zones, pages)) for bid, zones in by_id.items() if zones}


def _order_block_zones(zones: list[Zone], pages: dict[int, PageFacts]) -> list[Zone]:
    """Reading order for one block's zones: pages, then columns, then top-down.

    A single-page single-column block keeps its top-down order. Multi-zone
    pages reuse :func:`~ubt.adapters.pdf.textgeom.column_order` on the zone
    rects, so a paragraph that flows into a side-by-side column is paginated
    left column first — the source reading order — instead of by height.
    """
    ordered: list[Zone] = []
    for page_no in sorted({z.page for z in zones}):
        page_zones = [z for z in zones if z.page == page_no]
        facts = pages.get(page_no)
        if len(page_zones) == 1 or facts is None:
            ordered.extend(sorted(page_zones, key=lambda z: -z.y1))
            continue
        proxies = [LineBox("", z.rect) for z in page_zones]
        by_identity = {id(proxies[i]): page_zones[i] for i in range(len(proxies))}
        ordered.extend(by_identity[id(p)] for p in column_order(proxies, facts.width))
    return ordered
