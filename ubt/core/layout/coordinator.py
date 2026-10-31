"""CELS layout coordinator: chains Stage 1 column tracks to the Stage 3B solve.

The CELS modules are each pure and each refuses to know about the others:
:func:`~ubt.core.layout.cels.detect_columns` turns boxes into tracks,
:func:`~ubt.core.layout.solver.solve_vertical` turns a track into block
positions, :func:`~ubt.core.layout.distortion.compute_page_distortion` scores
the result. This module is where they meet, and it owns the wiring only -- which
blocks may be flowed, which are pinned instead, which track each segment belongs
to, and when a solved deficit above :data:`DEFICIT_TRIGGER_PT` becomes a
:class:`Stage4CondensationItem` carrying a ``char_budget`` for condensation.

Keeping the chaining here is what lets the geometry be tested against synthetic
pages and the solver against an injected measure function; the coordinator is the
only piece that needs both at once, so it is the only one that takes a renderer's
numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ubt.core.layout.cels import (
    Box,
    ColumnDetection,
    ColumnTrack,
    ElasticFlowSegment,
    PinnedObstacle,
    assign_lines,
    detect_columns,
    pin_from,
)
from ubt.core.layout.distortion import PageDistortionScore, compute_page_distortion
from ubt.core.layout.solver import (
    DEFICIT_TRIGGER_PT,
    SolvedBlockSpec,
    VerticalBlockInput,
    VerticalTrack,
    solve_vertical,
)

__all__ = [
    "CELSPageLayoutSolution",
    "PageLayoutBlock",
    "Stage4CondensationItem",
    "TrackLayoutSolution",
    "solve_page_cels",
]


@dataclass(frozen=True, slots=True)
class PageLayoutBlock:
    """One layout block presented to the CELS coordinator."""

    block_id: str
    box: Box
    text: str
    source_font_size_pt: float
    required_height_pt: float
    is_pinned: bool = False
    source_gap_after_pt: float = 0.0


@dataclass(frozen=True, slots=True)
class Stage4CondensationItem:
    """A block flagged for Stage 4 LLM constrained condensation due to deficit."""

    block_id: str
    track_index: int
    deficit_pt: float
    allocated_height_pt: float
    container_width_pt: float
    char_budget: int
    current_text: str
    locked_numbers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrackLayoutSolution:
    """The solved vertical layout for one column track."""

    track: ColumnTrack
    solved_blocks: tuple[SolvedBlockSpec, ...]
    elastic_segments: tuple[ElasticFlowSegment, ...]
    block_ids: tuple[str, ...] = ()

    @property
    def specs_by_id(self) -> dict[str, SolvedBlockSpec]:
        return dict(zip(self.block_ids, self.solved_blocks, strict=False))


@dataclass(frozen=True, slots=True)
class CELSPageLayoutSolution:
    """Complete CELS layout solution for one page."""

    page_no: int
    detection: ColumnDetection
    track_solutions: tuple[TrackLayoutSolution, ...]
    pinned_obstacles: tuple[PinnedObstacle, ...]
    distortion: PageDistortionScore
    total_deficit_pt: float
    requires_llm_condensation: bool
    stage4_requests: tuple[Stage4CondensationItem, ...] = ()

    @property
    def solved_blocks_by_id(self) -> dict[str, SolvedBlockSpec]:
        res: dict[str, SolvedBlockSpec] = {}
        for tr in self.track_solutions:
            res.update(tr.specs_by_id)
        return res

    def to_container_fit_specs(self) -> list[Any]:
        """Convert Stage 4 items to ContainerFitSpec for repair_bottleneck_containers."""
        from ubt.render.typst.container_repair import ContainerFitSpec

        return [
            ContainerFitSpec(
                seg_id=item.block_id,
                current_translation=item.current_text,
                max_target_chars=item.char_budget,
                locked_numbers=list(item.locked_numbers),
            )
            for item in self.stage4_requests
        ]


def solve_page_cels(
    *,
    page_no: int,
    blocks: Sequence[PageLayoutBlock],
    page_width: float,
    page_height: float,
    printable_box: Box,
    source_body_size_pt: float,
    realized_body_size_pt: float,
    min_gap_pt: float = 2.0,
    target_lang: str = "zh",
) -> CELSPageLayoutSolution:
    """Run the complete CELS layout pipeline on one page."""
    if not blocks:
        detection = detect_columns((), page_width=page_width, page_height=page_height)
        distortion = compute_page_distortion(
            page_no=page_no,
            boxes=(),
            tracks=(),
            printable_box=printable_box,
            source_body_size_pt=source_body_size_pt,
            realized_body_size_pt=realized_body_size_pt,
        )
        return CELSPageLayoutSolution(
            page_no=page_no,
            detection=detection,
            track_solutions=(),
            pinned_obstacles=(),
            distortion=distortion,
            total_deficit_pt=0.0,
            requires_llm_condensation=False,
            stage4_requests=(),
        )

    # Only flowable text may decide the tracks: a pinned block is not evidence of
    # a column.
    # A block wider than 60% of the page cannot fit into one column of a
    # multi-column page and would poison gutter detection, so it rides as a
    # pinned straddler obstacle -- but only when the narrower flow actually
    # reveals columns. On a single-column page the body itself is wider than
    # 60% of the page; pinning it would leave nothing flowable and collapse
    # the solve, so there the wide blocks keep flowing.
    flowable_blocks = [b for b in blocks if not b.is_pinned]
    wide_limit = 0.60 * page_width
    narrow_blocks = [b for b in flowable_blocks if b.box.width <= wide_limit]
    wide_blocks = [b for b in flowable_blocks if b.box.width > wide_limit]
    spanning_pinned: list[PinnedObstacle] = []
    if wide_blocks and narrow_blocks:
        narrow_detection = detect_columns(
            [b.box for b in narrow_blocks], page_width=page_width, page_height=page_height
        )
        if narrow_detection.is_multicolumn:
            spanning_pinned = [pin_from(b.box, kind="straddler") for b in wide_blocks]
            flowable_blocks = narrow_blocks
    flowable_boxes = [b.box for b in flowable_blocks]
    explicit_pinned = [
        pin_from(b.box, kind="other") for b in blocks if b.is_pinned
    ] + spanning_pinned
    detection = detect_columns(flowable_boxes, page_width=page_width, page_height=page_height)

    # A box touching two tracks -- or none -- has no column to belong to, so it is
    # pinned and the flow routes around it instead of a guess being made.
    lines_by_track, straddler_boxes = assign_lines(flowable_boxes, detection.tracks)
    straddler_pinned = [pin_from(box, kind="other") for box in straddler_boxes]
    all_pinned = tuple(explicit_pinned) + tuple(straddler_pinned)

    # The geometry modules trade in ``Box``es, not blocks, and they hand back the
    # same objects they were given -- never copies -- so identity is a safe key for
    # recovering the block a box came from.
    block_by_box_id = {id(b.box): b for b in blocks}

    # Solve one track at a time: a column's vertical budget is its own, and
    # borrowing space across a gutter would be the overlap this layout exists to
    # avoid.
    track_solutions: list[TrackLayoutSolution] = []
    total_deficit = 0.0
    needs_condensation = False
    stage4_requests: list[Stage4CondensationItem] = []
    solved_boxes_for_distortion: list[Box] = []

    for track in detection.tracks:
        track_boxes = lines_by_track.get(track.index, ())
        track_blocks = [
            block_by_box_id[id(box)] for box in track_boxes if id(box) in block_by_box_id
        ]
        # Sort top-to-bottom in PDF coordinates (descending y1, ascending x0)
        track_blocks.sort(key=lambda b: (-b.box.y1, b.box.x0))

        if not track_blocks:
            track_solutions.append(
                TrackLayoutSolution(
                    track=track,
                    solved_blocks=(),
                    elastic_segments=(),
                )
            )
            continue

        # Build vertical inputs with top-down coordinates
        v_inputs = [
            VerticalBlockInput(
                index=i,
                required_height_pt=b.required_height_pt,
                source_height_pt=b.required_height_pt if not b.is_pinned else b.box.height,
                source_top_pt=b.box.y1,
            )
            for i, b in enumerate(track_blocks)
        ]

        # Stage 3B: Cassowary Kiwi vertical solver
        # The vertical track must be bounded by the blocks and any pinned obstacles in this column,
        # rather than unconditionally expanding to the entire page printable area.
        blocks_y_min = min(b.box.y0 for b in track_blocks)
        blocks_y_max = max(b.box.y1 for b in track_blocks)
        track_y_min = max(printable_box.y0, blocks_y_min - 15.0)
        track_y_max = min(printable_box.y1, blocks_y_max + 15.0)

        for p in all_pinned:
            if min(p.box.x1, track.x1) > max(p.box.x0, track.x0) + 1.0:
                if p.box.y0 >= blocks_y_max - 5.0:
                    track_y_max = min(track_y_max, p.box.y0 - min_gap_pt)
                elif p.box.y1 <= blocks_y_min + 5.0:
                    track_y_min = max(track_y_min, p.box.y1 + min_gap_pt)

        track_y_min = min(track_y_min, blocks_y_min)
        track_y_max = max(track_y_max, blocks_y_max)
        v_track = VerticalTrack(y_min_pt=track_y_min, y_max_pt=track_y_max)
        solved_specs = solve_vertical(v_inputs, track=v_track, min_gap_pt=min_gap_pt)

        for i, spec in enumerate(solved_specs):
            b = track_blocks[i]
            total_deficit += spec.deficit_pt
            if spec.needs_stage4 or spec.deficit_pt > DEFICIT_TRIGGER_PT:
                needs_condensation = True
                # Two independent estimates of how much text the granted height can
                # hold -- the block's own share of its source text, and what the
                # track's geometry can carry -- and the smaller one wins, because
                # Stage 4 must be given a budget that neither estimate would break.
                ratio = max(0.1, min(1.0, spec.height_pt / max(1.0, b.required_height_pt)))
                proportional_budget = max(1, math.floor(len(b.text) * ratio * 0.92))
                from ubt.render.typst.container_repair import (
                    compute_container_char_budget,
                    extract_locked_numbers,
                )

                est_lines = max(
                    1, math.floor(spec.height_pt / max(1.0, realized_body_size_pt * 1.15))
                )
                geom_budget = compute_container_char_budget(
                    container_width_pt=track.width,
                    max_lines=est_lines,
                    floor_pt=realized_body_size_pt,
                    target_lang=target_lang,
                )
                char_budget = min(proportional_budget, geom_budget)
                nums = tuple(extract_locked_numbers(b.text))
                stage4_requests.append(
                    Stage4CondensationItem(
                        block_id=b.block_id,
                        track_index=track.index,
                        deficit_pt=spec.deficit_pt,
                        allocated_height_pt=spec.height_pt,
                        container_width_pt=track.width,
                        char_budget=char_budget,
                        current_text=b.text,
                        locked_numbers=nums,
                    )
                )

            # Store box with valid axis orientation: y0 (bottom) <= y1 (top)
            solved_boxes_for_distortion.append(
                Box(
                    x0=track.x0,
                    y0=spec.y_bottom_pt,
                    x1=track.x1,
                    y1=spec.y_top_pt,
                )
            )

        segment = ElasticFlowSegment(
            track_index=track.index,
            y_top=printable_box.y1,
            y_bottom=printable_box.y0,
            source=tuple(track_boxes),
        )
        track_solutions.append(
            TrackLayoutSolution(
                track=track,
                solved_blocks=tuple(solved_specs),
                elastic_segments=(segment,),
                block_ids=tuple(b.block_id for b in track_blocks),
            )
        )

    # A pinned obstacle is still ink on the delivered page, so the score has to
    # see it: a page that "solved" itself by pinning half its content would
    # otherwise score as clean.
    for p in all_pinned:
        solved_boxes_for_distortion.append(p.box)

    distortion = compute_page_distortion(
        page_no=page_no,
        boxes=solved_boxes_for_distortion,
        tracks=detection.tracks,
        printable_box=printable_box,
        source_body_size_pt=source_body_size_pt,
        realized_body_size_pt=realized_body_size_pt,
        source_boxes=tuple(b.box for b in blocks),
    )

    return CELSPageLayoutSolution(
        page_no=page_no,
        detection=detection,
        track_solutions=tuple(track_solutions),
        pinned_obstacles=tuple(all_pinned),
        distortion=distortion,
        total_deficit_pt=round(total_deficit, 4),
        requires_llm_condensation=needs_condensation,
        stage4_requests=tuple(stage4_requests),
    )
