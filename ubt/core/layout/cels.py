"""CELS Stage 1–2 geometry: column tracks, pinned obstacles, elastic segments.

Design: ``docs/design/adaptive_delivery_prd.md`` §7.3. Staged plan:
``docs/guides/ADAPTIVE_DELIVERY_TECH_DESIGN.md`` §7.1.

This module is deliberately pure: measured line boxes in, tracks / pins /
segments out. It opens no PDF, calls no measurement service and solves no
constraint. That split is what lets Stage 1 be unit-tested against synthetic
pages while Stage 5 does the content-stream surgery in the adapter.

Why a projection profile, and why it needs three guards
------------------------------------------------------
The only evidence CELS gets about the source page is its geometry, so Stage 1
reads the page as what it physically is: columns of ink separated by empty
bands. Three measured traps make a bare "widest empty band" split wrong, and
each produces a refusal rather than a repair:

* **blocking density** — a bin blocks a gutter only when the boxes crossing it
  carry :data:`PROFILE_BLOCK_SHARE` of the text block's vertical extent. A
  full-width title or a display rule covers the gutter's x range for one line,
  and with a naive profile that single line erases a real 24pt gutter. At 2 %
  of the block's height it is not a column wall, so the gutter survives.
* **ink share** — p12 has a wide empty band in its right margin. By width alone
  it is the widest gutter on the page, but the ink beside it is a margin
  fragment. Accepting it turns one column into a 96 % "left" and a 4 % "right"
  column, and every translated line is then forced into that 4 % strip.
  ``GUTTER_MIN_INK_SHARE`` refuses it.
* **vertical overlap** — a gutter only splits a page when the two sides are
  *stacked beside each other*. A lone fragment to the right of one line has no
  vertical overlap with the text block; it is an equation number or a margin
  note, not a column.
* **straddler budget** — a figure that genuinely spans both columns has no
  empty band around it, so it cannot be cut through. When the ink crossing a
  candidate gutter exceeds :data:`STRADDLER_MAX_SHARE` the split is refused
  instead of guessed, and the figure is pinned by :func:`assign_lines`.

Every refusal is returned with its reason. G3's acceptance is *"0 cross-column
penetration"*, and the evidence that a page was **not** split is the accepted
track count plus a reason for each near miss.

Scope limit: the cut is recursive up to :data:`MAX_TRACKS`, which covers 1–4
column pages. Reading order is left-to-right per track; interleaving columns
into one reading stream is Stage 1 work for G4 and is not assumed here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

#: Bin width of the horizontal projection profile, in points. One point is the
#: smallest unit a PDF producer reliably distinguishes between a word gap and a
#: gutter, and it keeps a US-Letter page's profile at 612 bins.
PROFILE_BIN_PT: float = 1.0

#: A bin is a wall when the boxes crossing it stack up to this share of the
#: text block's vertical extent. Body columns score 0.75–1.0; a full-width
#: title or rule scores 0.02–0.09, so 0.15 separates "column" from "one line
#: crossing the page".
PROFILE_BLOCK_SHARE: float = 0.15

#: Minimum empty band that can be a gutter. Body word gaps on the corpus are
#: 2.5–5pt and column gutters 12–24pt, so 8pt sits between them with margin on
#: both sides (the same reasoning as ``layout_policy``'s gutter threshold).
GUTTER_MIN_WIDTH_PT: float = 8.0

#: Each side of an accepted gutter must carry this share of the group's ink
#: area. A margin fragment beside 20 body lines carries ≈0.3 %, and a
#: two-line sidebar beside a full column ≈9 %, so 0.15 refuses both while a
#: genuine 3-column page (smallest share ≈0.28) passes.
GUTTER_MIN_INK_SHARE: float = 0.15

#: The two sides must be vertically stacked over this share of the shorter
#: span. A margin fragment beside one line of a 500pt-tall block scores ≈0.02.
GUTTER_MIN_VERTICAL_OVERLAP: float = 0.5

#: Ink crossing a candidate gutter may not exceed this share of the group's
#: ink; above it, a spanning figure or table owns that band and the page is not
#: split. Set below :data:`GUTTER_MIN_INK_SHARE` on purpose: one straddling
#: element must not outweigh a side that is not there.
STRADDLER_MAX_SHARE: float = 0.15

#: Recursion cap. 1 cut = 2 columns, 2 cuts = up to 4.
MAX_TRACKS: int = 4

#: Horizontal tolerance, in points, for "this box is on that side of the
#: gutter" and for overrun reporting. Half a point is the measured glyph-bearing
#: slop on the corpus (see ``PEN_PAD_EM`` in ``strip_pdf_text``).
EDGE_TOL_PT: float = 0.5

#: ``straddler`` marks a flowable block wider than one column of a multi-column
#: page (a spanning title or table): it is pinned so gutter detection and the
#: column flow route around it instead of collapsing the columns. Only the
#: coordinator may classify a block as a straddler, and only after the narrower
#: flow has actually revealed a multi-column layout -- on a single-column page
#: the body itself is wide and must keep flowing.
ObstacleKind = Literal[
    "figure", "formula", "rule", "header", "footer", "full_width", "straddler", "other"
]


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in PDF user coordinates (points, origin bottom-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True, slots=True)
class ColumnTrack:
    """One column CELS may reflow into.

    ``x0``/``x1`` are the track's hard horizontal limits: a translated line
    must not paint past either. ``y0``/``y1`` are the measured vertical extent
    of the source text in this track, which is where Stage 3B starts its
    vertical solve.
    """

    index: int
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def x_right(self) -> float:
        """The right limit under the name used in the PRD's acceptance text."""
        return self.x1

    def contains_horizontally(self, box: Box, tol: float = EDGE_TOL_PT) -> bool:
        return box.x0 >= self.x0 - tol and box.x1 <= self.x1 + tol


@dataclass(frozen=True, slots=True)
class InlineRigidBox:
    """Inline rigid obstacle (formula, superscript, unshrinkable unit).

    Design: ``docs/design/adaptive_delivery_prd.md`` §8.2 & §1194. Resolves GAP-14.
    An unshrinkable rigid box embedded inside an elastic text line. It does not
    scale with font size or tracking (Tz / Tc), and in Stage 5 is repositioned
    via Form XObject translation: ``q 1 0 0 1 x y cm /Fm Do Q``.
    """

    token_id: str
    width_pt: float
    height_pt: float
    baseline_offset_pt: float = 0.0
    xobject_ref: str = ""


@dataclass(frozen=True, slots=True)
class PinnedObstacle:
    """Something CELS may move around but never overwrite.

    ``flow_beside`` separates the two behaviours Stage 1 must not confuse: a
    figure or rule can be flowed *past* on both sides, while a header/footer or
    a full-width block ends the flow at that band.
    """

    box: Box
    kind: ObstacleKind
    flow_beside: bool = False


@dataclass(frozen=True, slots=True)
class ElasticFlowSegment:
    """A paragraph stream assigned to one track: the unit Stage 3A measures.

    Kept as plain data on purpose. ``char_budget`` is Stage 4's to compute, from
    a real :class:`~ubt.delivery.measurement.MeasuredBlock`, never from a
    characters-per-line guess.
    """

    track_index: int
    y_top: float
    y_bottom: float
    source: tuple[Box, ...] = ()

    @property
    def height(self) -> float:
        return max(0.0, self.y_top - self.y_bottom)


@dataclass(frozen=True, slots=True)
class GutterCandidate:
    """A vertical non-blocking band and the verdict Stage 1 reached about it.

    Reported even when rejected: the G3 acceptance is *"0 cross-column
    penetration"*, and the evidence that a page was **not** split is an
    accepted-track count plus the rejection reason for every near miss.
    """

    x0: float
    x1: float
    left_ink_share: float
    right_ink_share: float
    vertical_overlap: float
    straddler_share: float = 0.0
    reason: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def accepted(self) -> bool:
        return not self.reason


@dataclass(frozen=True, slots=True)
class ColumnDetection:
    """Stage 1 output for one page."""

    tracks: tuple[ColumnTrack, ...]
    accepted_gutters: tuple[GutterCandidate, ...] = ()
    rejected_gutters: tuple[GutterCandidate, ...] = ()
    #: Lines that span more than one track and therefore split the vertical
    #: flow into bands instead of participating in a column.
    full_width: tuple[Box, ...] = ()
    #: Horizontal ink extent the detector worked on (left/right of the text
    #: block, not the page box), so a report can show margins rather than
    #: treating them as columns.
    ink_extent: tuple[float, float] = (0.0, 0.0)

    @property
    def is_multicolumn(self) -> bool:
        return len(self.tracks) > 1


@dataclass(frozen=True, slots=True)
class TrackOverrun:
    """A line that paints past its track's right limit.

    G3's acceptance is "左栏右借位不越过 ``x_right``"; this is the machine
    check behind that sentence, and it is what G4's QA replays on every page.
    """

    track_index: int
    box: Box
    overrun_pt: float


def detect_columns(
    lines: Sequence[Box],
    *,
    page_width: float,
    page_height: float,
    min_gutter_pt: float = GUTTER_MIN_WIDTH_PT,
    min_ink_share: float = GUTTER_MIN_INK_SHARE,
    min_vertical_overlap: float = GUTTER_MIN_VERTICAL_OVERLAP,
    block_share: float = PROFILE_BLOCK_SHARE,
    straddler_share: float = STRADDLER_MAX_SHARE,
    max_tracks: int = MAX_TRACKS,
) -> ColumnDetection:
    """Cut a page's measured line boxes into column tracks.

    Recursive on accepted gutters: each accepted cut splits its group and the
    halves are re-examined, so a 3-column page yields 3 tracks. Every near-miss
    candidate is returned with its rejection reason instead of being dropped.
    """
    boxes = tuple(box for box in lines if box.width > 0.0 and box.height > 0.0)
    if not boxes:
        return ColumnDetection(
            tracks=(ColumnTrack(0, 0.0, page_width, 0.0, page_height),),
            ink_extent=(0.0, page_width),
        )

    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    accepted: list[GutterCandidate] = []
    rejected: list[GutterCandidate] = []
    groups: list[tuple[Box, ...]] = [boxes]

    for _ in range(max(0, max_tracks - 1)):
        split_happened = False
        next_groups: list[tuple[Box, ...]] = []
        for group in groups:
            halves = _split_group(
                group,
                accepted=accepted,
                rejected=rejected,
                min_gutter_pt=min_gutter_pt,
                min_ink_share=min_ink_share,
                min_vertical_overlap=min_vertical_overlap,
                block_share=block_share,
                straddler_share=straddler_share,
            )
            if halves is None:
                next_groups.append(group)
            else:
                next_groups.extend(halves)
                split_happened = True
        groups = next_groups
        if not split_happened:
            break

    tracks = tuple(
        ColumnTrack(
            index=index,
            x0=min(box.x0 for box in group),
            x1=max(box.x1 for box in group),
            y0=min(box.y0 for box in group),
            y1=max(box.y1 for box in group),
        )
        for index, group in enumerate(sorted(groups, key=lambda g: min(b.x0 for b in g)))
    )
    return ColumnDetection(
        tracks=tracks,
        accepted_gutters=tuple(accepted),
        rejected_gutters=tuple(rejected),
        ink_extent=(left, right),
    )


def assign_lines(
    boxes: Sequence[Box],
    tracks: Sequence[ColumnTrack],
) -> tuple[dict[int, tuple[Box, ...]], tuple[Box, ...]]:
    """Sort boxes into tracks, or out of the flow entirely.

    The leftover bucket is deliberate rather than a silent default: a box that
    overlaps two tracks is a full-width barrier (it cannot be reflowed into
    either column), and a box inside no track is a margin object. Both are
    returned so the caller can pin them instead of guessing a column.
    """
    assigned: dict[int, list[Box]] = {track.index: [] for track in tracks}
    leftover: list[Box] = []
    for box in boxes:
        hits = [
            track for track in tracks if _horizontal_overlap_fraction(box, track) >= _STRADDLE_SHARE
        ]
        if len(hits) == 1:
            assigned[hits[0].index].append(box)
        else:
            leftover.append(box)
    return {index: tuple(items) for index, items in assigned.items()}, tuple(leftover)


def track_overruns(
    tracks: Sequence[ColumnTrack],
    boxes: Sequence[Box],
    *,
    tol: float = EDGE_TOL_PT,
) -> tuple[TrackOverrun, ...]:
    """Every line that paints past the right limit of the track it belongs to."""
    assignments, _ = assign_lines(boxes, tracks)
    by_index = {track.index: track for track in tracks}
    found: list[TrackOverrun] = []
    for index, items in assignments.items():
        track = by_index[index]
        for box in items:
            overrun = box.x1 - track.x1
            if overrun > tol:
                found.append(TrackOverrun(index, box, overrun))
    return tuple(found)


def pin_from(
    box: Box,
    kind: ObstacleKind = "full_width",
    *,
    flow_beside: bool = False,
) -> PinnedObstacle:
    """Build a pin from a measured box (kept here so callers don't re-derive one)."""
    return PinnedObstacle(box=box, kind=kind, flow_beside=flow_beside)


#: A box counts as straddling when this share of its width sits in a track. A
#: full-width title scores 0.5 in each of two columns; a line that merely
#: overshoots its own track by a glyph bearing scores ≈0.99 in one and ≈0.01 in
#: the other, so 0.2 separates "belongs here" from "spans both".
_STRADDLE_SHARE: float = 0.2


def _horizontal_overlap_fraction(box: Box, track: ColumnTrack) -> float:
    if box.width <= 0.0:
        return 0.0
    overlap = min(box.x1, track.x1) - max(box.x0, track.x0)
    return max(0.0, overlap) / box.width


def _split_group(
    group: Sequence[Box],
    *,
    accepted: list[GutterCandidate],
    rejected: list[GutterCandidate],
    min_gutter_pt: float,
    min_ink_share: float,
    min_vertical_overlap: float,
    block_share: float,
    straddler_share: float,
) -> tuple[tuple[Box, ...], tuple[Box, ...]] | None:
    """Cut one group at its widest accepted gutter, or report why it could not."""
    if len(group) < 2:
        return None
    left_edge = min(box.x0 for box in group)
    right_edge = max(box.x1 for box in group)
    extent = right_edge - left_edge
    if extent < 2.0 * min_gutter_pt:
        return None
    y_extent = max(box.y1 for box in group) - min(box.y0 for box in group)
    if y_extent <= 0.0:
        return None

    bins = max(1, math.ceil(extent / PROFILE_BIN_PT))
    density = [0.0] * bins
    for box in group:
        first = max(0, math.floor((box.x0 - left_edge) / PROFILE_BIN_PT))
        last = min(bins, math.ceil((box.x1 - left_edge) / PROFILE_BIN_PT))
        for index in range(first, last):
            density[index] += box.height
    wall = block_share * y_extent
    total_ink = sum(box.area for box in group)

    best: tuple[float, tuple[Box, ...], tuple[Box, ...]] | None = None
    run_start: int | None = None
    for index in range(bins + 1):
        blocked = index < bins and density[index] >= wall
        if index < bins and not blocked:
            if run_start is None:
                run_start = index
            continue
        if run_start is None:
            continue
        cand_x0 = left_edge + run_start * PROFILE_BIN_PT
        cand_x1 = left_edge + index * PROFILE_BIN_PT
        run_start = None
        if cand_x1 - cand_x0 < min_gutter_pt:
            continue

        left_side = tuple(box for box in group if box.x1 <= cand_x0 + EDGE_TOL_PT)
        right_side = tuple(box for box in group if box.x0 >= cand_x1 - EDGE_TOL_PT)
        left_ids = {id(box) for box in left_side}
        right_ids = {id(box) for box in right_side}
        straddlers = tuple(
            box for box in group if id(box) not in left_ids and id(box) not in right_ids
        )
        shares = (
            sum(box.area for box in left_side) / total_ink if total_ink else 0.0,
            sum(box.area for box in right_side) / total_ink if total_ink else 0.0,
        )
        overlap = _vertical_overlap(left_side, right_side)
        straddler_total = sum(box.area for box in straddlers) / total_ink if total_ink else 0.0

        candidate = GutterCandidate(
            x0=cand_x0,
            x1=cand_x1,
            left_ink_share=shares[0],
            right_ink_share=shares[1],
            vertical_overlap=overlap,
            straddler_share=straddler_total,
            reason=_refusal_reason(
                shares,
                overlap,
                straddler_total,
                min_ink_share,
                min_vertical_overlap,
                straddler_share,
            ),
        )
        if candidate.reason:
            rejected.append(candidate)
            continue
        accepted.append(candidate)
        if best is None or candidate.width > best[0]:
            best = (candidate.width, left_side, right_side)
    if best is None:
        return None
    return (best[1], best[2])


def _refusal_reason(
    shares: tuple[float, float],
    overlap: float,
    straddler_share: float,
    min_ink_share: float,
    min_vertical_overlap: float,
    straddler_room: float,
) -> str:
    if shares[0] <= 0.0 or shares[1] <= 0.0:
        side = "left" if shares[0] <= 0.0 else "right"
        return f"no ink on the {side} side of the band; it is a margin, not a gutter"
    if min(shares) < min_ink_share:
        return (
            f"ink share {shares[0]:.3f}/{shares[1]:.3f} below {min_ink_share:.3f}; "
            "the band is a margin note beside a column, not a gutter"
        )
    if overlap < min_vertical_overlap:
        return (
            f"vertical overlap {overlap:.3f} below {min_vertical_overlap:.3f}; "
            "the sides are not stacked beside each other"
        )
    if straddler_share > straddler_room:
        return (
            f"straddling ink {straddler_share:.3f} above {straddler_room:.3f}; "
            "a full-width element owns this band and cannot be cut through"
        )
    return ""


def _vertical_overlap(left: Sequence[Box], right: Sequence[Box]) -> float:
    if not left or not right:
        return 0.0
    left_span = (min(box.y0 for box in left), max(box.y1 for box in left))
    right_span = (min(box.y0 for box in right), max(box.y1 for box in right))
    overlap = min(left_span[1], right_span[1]) - max(left_span[0], right_span[0])
    shorter = min(left_span[1] - left_span[0], right_span[1] - right_span[0])
    if shorter <= 0.0:
        return 0.0
    return max(0.0, overlap) / shorter


__all__ = [
    "Box",
    "ColumnDetection",
    "ColumnTrack",
    "EDGE_TOL_PT",
    "ElasticFlowSegment",
    "GUTTER_MIN_INK_SHARE",
    "GUTTER_MIN_VERTICAL_OVERLAP",
    "GUTTER_MIN_WIDTH_PT",
    "GutterCandidate",
    "InlineRigidBox",
    "MAX_TRACKS",
    "PROFILE_BIN_PT",
    "PROFILE_BLOCK_SHARE",
    "PinnedObstacle",
    "STRADDLER_MAX_SHARE",
    "TrackOverrun",
    "assign_lines",
    "detect_columns",
    "pin_from",
    "track_overruns",
]
