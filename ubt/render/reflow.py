"""Page-local vertical reflow (Band Reflow).

The compositor draws each translated paragraph into a box the size of its
*source* paragraph and top-aligns the fragment. A CJK target is usually shorter
than its English source, so the leftover height of every box shows as blank
space *between* paragraphs -- the page reads as a stack of gaps.

This module redistributes that leftover. It groups a page's reflowable prose
into **bands**: maximal runs of consecutive paragraphs with no fixed obstacle
(figure, table, formula, heading, multi-box paragraph) between them. Within a
band every fragment's natural height is measured at the size it will be drawn,
then the fragments are stacked top-down with a paragraph gap; the band's slack
is spent widening those gaps (up to a cap), and whatever remains stays at the
band's bottom. Nothing ever moves *down* past the band's own envelope or *up*
past the first paragraph, so a reflowed page cannot collide with the figures and
tables that bound the band.

The transform is pure: it takes the compositor's :class:`Overlay` list and the
obstacle boxes, calls the injected measure, and returns overlays carrying the
new geometry plus the original source boxes to mask (see ``Overlay.mask_boxes``).
A band whose target does not fit its source envelope is left untouched, so a
long translation keeps its (correct, if gappy) source layout rather than
overlapping its neighbours.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace

from ubt.model.span import PhysicalBox
from ubt.render.outputs import Overlay, StyledRun

#: ``measure_many(items) -> heights``; items are
#: ``(text, width_pt, size_pt, indent_pt, kind, is_bold, runs)``.
MeasureMany = Callable[
    [Sequence[tuple[str, float, float, float | None, str, bool, tuple[StyledRun, ...]]]],
    Sequence[float],
]
#: ``cap_size(kind, source_font_size) -> draw size`` (the compositor's own cap).
CapSize = Callable[[str, float | None], float]

#: A vertical gap larger than this many line heights splits a band (a section
#: break, a column change), so unrelated prose is never pulled together. A
#: source that separates paragraphs with a blank line leaves a gap near two line
#: heights (a run-in "Action Fusion." paragraph, say); 1.6 was too tight and left
#: such a page with no band at all, so it never reflowed and its text fell back
#: to the smaller document-wide fitted size. Structural breaks that carry no
#: figure/table/heading are the only thing this guards, and those gaps sit well
#: above 2.5 line heights.
_BAND_GAP_FACTOR = 2.5
#: Assumed line height (em) when turning a font size into a band-gap threshold.
_LINE_HEIGHT_EM = 1.2
#: Horizontal overlap (pt) two boxes need to belong to the same band.
_X_OVERLAP_TOL = 1.0
#: Extra box height (pt) below each fragment so a rounding difference between the
#: measure and the draw cannot clip the last line.
_BOX_EPS = 0.5
#: Font size assumed when a block carries none, for the band-gap threshold.
_DEFAULT_FONT_PT = 10.0
#: A band must have at least this many paragraphs to be worth reflowing.
_MIN_BAND = 2


def _is_reflowable(overlay: Overlay) -> bool:
    """A single-box prose paragraph the reflow pass may move.

    Headings, formulas, TOC rows, bilingual overlays, and multi-box continuation
    paragraphs are structural anchors: moving them would break the reading
    structure or the flow solver, so they bound the bands instead.
    """
    return (
        not overlay.boxes
        and overlay.kind == "text"
        and not overlay.source
        and not overlay.fixed_box
        and bool(overlay.text.strip())
    )


def _x_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > _X_OVERLAP_TOL


def _line_height(overlay: Overlay) -> float:
    size = overlay.font_size if overlay.font_size and overlay.font_size > 0 else _DEFAULT_FONT_PT
    return size * _LINE_HEIGHT_EM


def _obstacle_between(prev: Overlay, cur: Overlay, obstacles: Sequence[PhysicalBox]) -> bool:
    """True when a fixed box sits vertically between two stacked paragraphs."""
    for box in obstacles:
        if box.page != prev.page:
            continue
        ob = box.bbox
        if not _x_overlap(ob, prev.bbox):
            continue
        # prev is above cur (PDF y grows upward): an obstacle in the gap has its
        # top at or below prev's bottom and its bottom at or above cur's top.
        if ob[3] <= prev.bbox[1] + 1.0 and ob[1] >= cur.bbox[3] - 1.0:
            return True
    return False


def _form_bands(
    ordered: Sequence[Overlay], obstacles: Sequence[PhysicalBox]
) -> list[list[Overlay]]:
    bands: list[list[Overlay]] = []
    current: list[Overlay] = []
    for overlay in ordered:
        if not current:
            current = [overlay]
            continue
        prev = current[-1]
        gap = prev.bbox[1] - overlay.bbox[3]
        max_gap = _BAND_GAP_FACTOR * max(_line_height(prev), _line_height(overlay))
        if (
            _x_overlap(prev.bbox, overlay.bbox)
            and gap <= max_gap
            and not _obstacle_between(prev, overlay, obstacles)
        ):
            current.append(overlay)
        else:
            bands.append(current)
            current = [overlay]
    if current:
        bands.append(current)
    return bands


def _layout_band(
    band: Sequence[Overlay],
    class_sizes: dict[tuple[str, int], float],
    measure_many: MeasureMany,
    *,
    gap_max_em: float,
) -> list[tuple[Overlay, tuple[float, float, float, float]]] | None:
    """New geometry for one band, or ``None`` to leave it on the source boxes."""
    if len(band) < _MIN_BAND:
        return None
    sizes = [class_sizes[(ov.kind, round(ov.font_size) if ov.font_size else 0)] for ov in band]
    widths = [ov.bbox[2] - ov.bbox[0] for ov in band]
    if any(width <= 0 for width in widths):
        return None
    items = [
        (ov.text, width, size, ov.indent_pt, ov.kind, ov.is_bold, ov.runs)
        for ov, width, size in zip(band, widths, sizes, strict=True)
    ]
    naturals = list(measure_many(items))
    if len(naturals) != len(band) or any(not math.isfinite(h) or h <= 0 for h in naturals):
        return None
    band_top = band[0].bbox[3]
    band_bottom = band[-1].bbox[1]
    budget = band_top - band_bottom
    ref = max(sizes)
    # Every box carries ``_BOX_EPS`` extra height below its text.
    needed = sum(naturals) + len(band) * _BOX_EPS
    if needed > budget + 0.5:
        # The target is longer than the source envelope: keep the source layout
        # (the fit search will shrink it) rather than overlap the next obstacle.
        return None
    g_max = gap_max_em * ref
    # ``_MIN_BAND`` guarantees at least two paragraphs, so there is a gap to widen.
    g = min(g_max, max(0.0, (budget - needed) / (len(band) - 1)))
    placed: list[tuple[Overlay, tuple[float, float, float, float]]] = []
    top = band_top
    for overlay, natural in zip(band, naturals, strict=True):
        y1 = top
        y0 = top - (natural + _BOX_EPS)
        placed.append((overlay, (overlay.bbox[0], y0, overlay.bbox[2], y1)))
        top = y0 - g
    return placed


def _partition_columns(overlays: Sequence[Overlay]) -> list[list[Overlay]]:
    """Partition a page's reflowable overlays into column clusters.

    In multi-column pages (academic papers, multi-column books), overlays from
    different columns must not be interleaved into a single top-down sequence;
    otherwise horizontal overlap checks fail between adjacent elements from
    different columns, collapsing all bands to singletons and dropping them.

    This function detects column gutters and splits overlays into distinct
    column buckets. Single-column layouts remain intact as a single cluster.
    """
    if len(overlays) <= 1:
        return [list(overlays)]

    min_x = min(ov.bbox[0] for ov in overlays)
    max_x = max(ov.bbox[2] for ov in overlays)
    span = max_x - min_x
    if span <= 20.0:
        return [list(overlays)]

    candidates = [ov for ov in overlays if (ov.bbox[2] - ov.bbox[0]) <= 0.75 * span]
    if len(candidates) < 2:
        return [list(overlays)]

    def center(ov: Overlay) -> float:
        return (ov.bbox[0] + ov.bbox[2]) / 2.0

    def side_span(cands: list[Overlay]) -> float:
        if not cands:
            return 0.0
        return max(ov.bbox[2] for ov in cands) - min(ov.bbox[0] for ov in cands)

    def crossings(split: float) -> int:
        return sum(1 for ov in candidates if ov.bbox[0] < split < ov.bbox[2])

    split_points = sorted({ov.bbox[0] for ov in candidates} | {ov.bbox[2] for ov in candidates})
    best_split: float | None = None
    best_crossings = len(candidates) + 1
    for split in split_points:
        left = [ov for ov in candidates if center(ov) < split]
        right = [ov for ov in candidates if center(ov) >= split]
        if not left or not right:
            continue
        if side_span(left) <= 0.15 * span or side_span(right) <= 0.15 * span:
            continue
        crossed = crossings(split)
        if crossed < best_crossings:
            best_crossings = crossed
            best_split = split

    if best_split is None or best_crossings > 0.1 * len(candidates):
        return [list(overlays)]

    full_width = [ov for ov in overlays if ov.bbox[0] < best_split < ov.bbox[2]]
    left = [ov for ov in overlays if ov not in full_width and center(ov) < best_split]
    right = [ov for ov in overlays if ov not in full_width and center(ov) >= best_split]

    res: list[list[Overlay]] = []
    if full_width:
        res.append(full_width)
    if left:
        res.extend(_partition_columns(left))
    if right:
        res.extend(_partition_columns(right))
    return res


def reflow_overlays(
    overlays: Sequence[Overlay],
    obstacles: Sequence[PhysicalBox],
    *,
    measure_many: MeasureMany,
    cap_size: CapSize,
    gap_max_em: float = 2.0,
) -> tuple[Overlay, ...]:
    """Repack single-box prose paragraphs into their page's bands.

    ``obstacles`` are the boxes that must not be crossed (figures, tables,
    formulas, headings, multi-box paragraphs); the non-reflowable overlays are
    added to them internally. Overlays the pass cannot improve are returned
    unchanged, so the caller can always use the result.
    """
    reflowable = [ov for ov in overlays if _is_reflowable(ov)]
    if not reflowable:
        return tuple(overlays)
    reflowable_ids = {overlay.element_id for overlay in reflowable}
    obstacle_boxes: list[PhysicalBox] = list(obstacles)
    for overlay in overlays:
        if overlay.element_id in reflowable_ids:
            continue
        obstacle_boxes.extend(overlay.flow_boxes)
    # One draw size per style class, so same-size body text stays uniform after
    # the reflow. It is the source-size cap the compositor's fit path also uses,
    # so a reflowed paragraph and a lone fitted one land on the same size.
    class_sizes: dict[tuple[str, int], float] = {}
    for overlay in reflowable:
        cls = (overlay.kind, round(overlay.font_size) if overlay.font_size else 0)
        size = cap_size(overlay.kind, overlay.font_size)
        class_sizes[cls] = min(class_sizes.get(cls, size), size)
    obstacles_by_page: dict[int, list[PhysicalBox]] = {}
    for box in obstacle_boxes:
        obstacles_by_page.setdefault(box.page, []).append(box)
    by_page: dict[int, list[Overlay]] = {}
    for overlay in reflowable:
        by_page.setdefault(overlay.page, []).append(overlay)
    new_by_id: dict[str, Overlay] = {}

    def _class_size(overlay: Overlay) -> float:
        return class_sizes[(overlay.kind, round(overlay.font_size) if overlay.font_size else 0)]

    for page, page_overlays in by_page.items():
        columns = _partition_columns(page_overlays)
        for col_overlays in columns:
            ordered = sorted(col_overlays, key=lambda ov: (-ov.bbox[3], ov.bbox[0]))
            for band in _form_bands(ordered, obstacles_by_page.get(page, [])):
                placed = _layout_band(
                    band,
                    class_sizes,
                    measure_many,
                    gap_max_em=gap_max_em,
                )
                if placed is None:
                    continue
                for overlay, new_bbox in placed:
                    new_by_id[overlay.element_id] = replace(
                        overlay,
                        bbox=new_bbox,
                        boxes=(),
                        mask_boxes=(PhysicalBox.of(overlay.page, overlay.bbox),),
                        fixed_box=True,
                        # The box height is this size's natural height, so the
                        # compositor must draw at exactly it: re-deriving the size
                        # from the block's own cap can wrap one more line past the
                        # box, which ``clip`` hides as an ink-less line.
                        draw_size_pt=_class_size(overlay),
                    )
    if not new_by_id:
        return tuple(overlays)
    return tuple(new_by_id.get(overlay.element_id, overlay) for overlay in overlays)


__all__ = ["CapSize", "MeasureMany", "reflow_overlays"]
