"""Fast per-page fit models for the document style group (G4 Stage 3A).

The style group has to be decided *before* the page loop can start, because the
decision is a single body size for the whole document. The loop itself cannot
answer that: it builds placements, copies assets and crops formulas as it goes,
and the per-page size it used to compute is exactly the number that must not
vary.

So this module builds a cheap, side-effect-free model of each page — the same
column records the 2D solver consumes, minus the multi-region flow splitting
that needs the heading size — and lets
:func:`ubt.core.layout.style_group.plan_body_group` choose the size and the
micro-typography rung from it.

The model is an approximation on purpose, and the renderer is the authority:

* here, a block spans one record built from its own bbox;
* there, a block that crosses columns has its text split across several boxes.

When the two disagree the renderer's verdict wins (it is the side that placed
the glyphs) and the disagreement is recorded as an exception instead of being
smoothed away. What this module must not do is claim space it cannot measure:
records carry ``right_budget`` only when the page itself shows a same-column
right edge to borrow, and the ladder is capped at a few points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.layout.style_group import (
    PageFit,
    StyleGroupException,
    StyleGroupPlan,
    plan_body_group,
)
from ubt.model.span import PhysicalBox, boxes_from_provenance
from ubt.render.delivered_text import delivered_text
from ubt.render.flow import approximate_measure, solve_flow
from ubt.render.layout_solver import FitReport, check_source_overlap, fit_at_size

#: A block wider than this share of the page is a full-width element, not a
#: column member. Same threshold the renderer buckets by.
FULL_WIDTH_SHARE = 0.65

#: Records whose left edge differs by less than this are treated as belonging to
#: the same column when computing a right-edge borrow budget. Without it a
#: full-width block parked in the same bucket as left-column prose would lend
#: the prose the whole page's right margin.
_SAME_COLUMN_X_TOL_PT = 12.0


@dataclass(frozen=True, slots=True)
class PageFitModel:
    """One page's solver records plus the metrics the group needs from it."""

    page: int
    columns: tuple[tuple[dict[str, Any], ...], ...]
    source_size_pt: float
    source_pitch: float
    text_weight: int
    source_overlap: bool

    def as_records(self) -> list[list[dict[str, Any]]]:
        """Fresh mutable copies, because the borrow step rewrites widths."""
        return [[dict(record) for record in column] for column in self.columns]


def _glyph_aspect(text: str) -> float:
    """Share of the text's advance that a CJK glyph occupies (the renderer's rule)."""
    ascii_count = sum(1 for char in text if ord(char) < 128)
    return (len(text) - ascii_count + ascii_count * 0.50) / max(1, len(text))


def _right_edge_budgets(records: Sequence[dict[str, Any]]) -> None:
    """Give each record the right edge its own column already reaches.

    The budget is the rightmost ``x1`` among sibling records that share this
    record's left edge, so Δw can only spend whitespace the column itself uses
    elsewhere on the page. A full-width block parked in the same bucket does not
    lend its right margin to left-column prose, because their left edges differ
    by more than the tolerance.
    """
    for record in records:
        x0 = float(record["x0"])
        same_column = [
            float(other["x1"])
            for other in records
            if abs(float(other["x0"]) - x0) <= _SAME_COLUMN_X_TOL_PT
        ]
        record["right_budget"] = max(same_column, default=float(record["x1"]))


def _physical_box_region(pbox: PhysicalBox, page_width: float) -> tuple[int, int]:
    """The ``(page, column)`` region one physical box belongs to."""
    bw = pbox.bbox[2] - pbox.bbox[0]
    bmid = (pbox.bbox[0] + pbox.bbox[2]) / 2.0
    bcol = 0 if bw > FULL_WIDTH_SHARE * page_width or bmid < page_width * 0.50 else 1
    return (int(pbox.page), bcol)


def group_multi_region_boxes(block: IRBlock, page_width: float) -> list[PhysicalBox] | None:
    """Return grouped disjoint region boxes when ``block`` spans multiple columns or pages."""
    if (
        not block.provenance
        or not block.provenance.physical_boxes
        or len(block.provenance.physical_boxes) <= 1
    ):
        return None
    raw_boxes = boxes_from_provenance(block.provenance.physical_boxes)
    distinct_regions = {_physical_box_region(bx, page_width) for bx in raw_boxes}
    if len(distinct_regions) <= 1:
        return None
    grouped_boxes: list[PhysicalBox] = []
    curr_reg: tuple[int, int] | None = None
    curr_page = 0
    ux0, uy0, ux1, uy1 = 0.0, 0.0, 0.0, 0.0
    for bx in raw_boxes:
        reg = _physical_box_region(bx, page_width)
        if reg != curr_reg:
            if curr_reg is not None:
                grouped_boxes.append(PhysicalBox.of(curr_page, (ux0, uy0, ux1, uy1)))
            curr_reg = reg
            curr_page = int(bx.page)
            ux0, uy0, ux1, uy1 = (
                float(bx.bbox[0]),
                float(bx.bbox[1]),
                float(bx.bbox[2]),
                float(bx.bbox[3]),
            )
        else:
            ux0 = min(ux0, float(bx.bbox[0]))
            uy0 = min(uy0, float(bx.bbox[1]))
            ux1 = max(ux1, float(bx.bbox[2]))
            uy1 = max(uy1, float(bx.bbox[3]))
    if curr_reg is not None:
        grouped_boxes.append(PhysicalBox.of(curr_page, (ux0, uy0, ux1, uy1)))
    return grouped_boxes


def build_page_fit_models(
    blocks_by_page: Mapping[int, Sequence[IRBlock]],
    page_dims: Sequence[tuple[float, float]],
    metrics_by_page: Mapping[int, tuple[float, float]],
    *,
    total_pages: int,
) -> dict[int, PageFitModel]:
    """Build one model per page, without touching the filesystem or the renderer.

    ``metrics_by_page`` maps page -> ``(source_size_pt, source_pitch)``; a page
    with no measurement is skipped, because a page the group cannot describe is
    a page the group cannot promise anything about.
    """
    models: dict[int, PageFitModel] = {}
    for page_no in range(1, total_pages + 1):
        metrics = metrics_by_page.get(page_no)
        if metrics is None:
            continue
        source_size_pt, source_pitch = metrics
        pw, ph = page_dims[page_no - 1] if page_no - 1 < len(page_dims) else (595.0, 842.0)

        buckets: dict[int, list[dict[str, Any]]] = {}
        obstacles: dict[int, list[float]] = {}
        weight = 0
        for block in blocks_by_page.get(page_no, []):
            if block.bbox is None:
                continue
            width = max(0.0, float(block.bbox.x1) - float(block.bbox.x0))
            height = max(0.0, float(block.bbox.y1) - float(block.bbox.y0))
            if width <= 0.0 or height <= 0.0:
                continue

            preserved = block.skip_translate or block.block_type in (
                BlockType.FORMULA,
                BlockType.IMAGE,
            )
            if preserved:
                if block.bbox.page != page_no:
                    continue
                # A preserved figure/formula is a vertical obstacle for the prose
                # above it: glue may not expand into it. Same rule as the loop.
                obs_top = float(block.bbox.y1)
                if width > FULL_WIDTH_SHARE * pw:
                    obstacles.setdefault(0, []).append(obs_top)
                    obstacles.setdefault(1, []).append(obs_top)
                else:
                    mid_x = (float(block.bbox.x0) + float(block.bbox.x1)) / 2.0
                    obstacles.setdefault(0 if mid_x < pw * 0.50 else 1, []).append(obs_top)
                continue

            text = delivered_text(block) or block.source_text
            if not text.strip():
                continue

            grouped_boxes = group_multi_region_boxes(block, pw)
            if grouped_boxes is not None:
                if not any(gbox.page == page_no for gbox in grouped_boxes):
                    continue
                measure = approximate_measure(
                    font_size_pt=source_size_pt,
                    leading_pt=max(1.0, source_pitch * source_size_pt),
                )
                flowed = solve_flow(text, grouped_boxes, measure)
                for fl_idx, fl_pl in enumerate(flowed):
                    if fl_pl.box.page != page_no or not fl_pl.text.strip():
                        continue
                    fl_x0, fl_y0, fl_x1, fl_y1 = fl_pl.box.bbox
                    fl_w = max(0.0, fl_x1 - fl_x0)
                    fl_h = max(0.0, fl_y1 - fl_y0)
                    if fl_w <= 0.0 or fl_h <= 0.0:
                        continue
                    fl_mid_x = (fl_x0 + fl_x1) / 2.0
                    fl_col = 0 if fl_w > FULL_WIDTH_SHARE * pw or fl_mid_x < pw * 0.50 else 1
                    buckets.setdefault(fl_col, []).append(
                        {
                            "id": f"{block.id}_flow_{fl_idx}",
                            "top": fl_y1,
                            "bot": fl_y0,
                            "w": fl_w,
                            "x0": fl_x0,
                            "x1": fl_x1,
                            "n": max(1, round(fl_h / max(1.0, source_pitch * source_size_pt))),
                            "glyphs": len(fl_pl.text.strip()),
                            "glyph_aspect": _glyph_aspect(fl_pl.text),
                        }
                    )
                    weight += len(fl_pl.text.strip())
                continue

            if block.bbox.page != page_no:
                continue
            mid_x = (float(block.bbox.x0) + float(block.bbox.x1)) / 2.0
            column = 0 if width > FULL_WIDTH_SHARE * pw or mid_x < pw * 0.50 else 1
            buckets.setdefault(column, []).append(
                {
                    "id": block.id,
                    "top": float(block.bbox.y1),
                    "bot": float(block.bbox.y0),
                    "w": width,
                    "x0": float(block.bbox.x0),
                    "x1": float(block.bbox.x1),
                    "n": max(1, round(height / max(1.0, source_pitch * source_size_pt))),
                    "glyphs": len(text.strip()),
                    "glyph_aspect": _glyph_aspect(text),
                }
            )
            weight += len(text.strip())

        columns: list[list[dict[str, Any]]] = []
        for column_idx in sorted(buckets):
            records = sorted(
                buckets[column_idx], key=lambda record: float(record["top"]), reverse=True
            )
            _right_edge_budgets(records)
            for rec in records:
                below = [
                    obs_top
                    for obs_top in obstacles.get(column_idx, [])
                    if obs_top <= float(rec["bot"]) + 2.0
                ]
                if below:
                    rec["obstacle_top"] = max(below)
            columns.append(records)
        if not columns:
            columns = [[]]

        models[page_no] = PageFitModel(
            page=page_no,
            columns=tuple(tuple(record for record in column) for column in columns),
            source_size_pt=source_size_pt,
            source_pitch=source_pitch,
            text_weight=weight,
            source_overlap=check_source_overlap(columns),
        )
    return models


def fit_page_model(
    model: PageFitModel,
    *,
    size_pt: float,
    pitch: float,
    tracking_em: float,
    width_borrow_pt: float,
) -> FitReport:
    """Feasibility of one page at the group's size and rung."""
    return fit_at_size(
        model.as_records(),
        size_pt=size_pt,
        pitch=pitch,
        tracking_em=tracking_em,
        width_borrow_pt=width_borrow_pt,
    )


def plan_facsimile_body_group(
    models: Mapping[int, PageFitModel],
    *,
    group_id: str = "body:document",
    block_count: int = 0,
) -> StyleGroupPlan:
    """Choose the document body group from the per-page models.

    A page whose source boxes overlap cannot be judged by a fixed-box probe: it
    will be re-typeset by the page-locked reflow, so its fit is decided *later*,
    at the same group size. Such pages are handed to the planner as deferred —
    they join the group but do not get a vote on its size, because their model
    here is an artefact of boxes that were never compatible to begin with.
    """
    metrics = {
        page: (model.source_size_pt, model.source_pitch, model.text_weight)
        for page, model in models.items()
    }
    deferred = tuple(page for page, model in models.items() if model.source_overlap)

    def probe(page: int, size_pt: float, rung: Any, pitch: float) -> PageFit:
        model = models.get(page)
        if model is None:  # pragma: no cover - the planner only asks known pages
            return PageFit(page=page, fits=True)
        report = fit_page_model(
            model,
            size_pt=size_pt,
            pitch=pitch,
            tracking_em=rung.tracking_em,
            width_borrow_pt=rung.width_borrow_pt,
        )
        return PageFit(
            page=page,
            fits=report.fits,
            deficit_pt=report.deficit_pt,
            block_ids=report.block_ids,
        )

    return plan_body_group(
        metrics,
        probe,
        group_id=group_id,
        block_count=block_count,
        deferred_pages=deferred,
    )


def exceptions_from_fit(
    page_fits: Mapping[int, FitReport],
    *,
    rung: str,
    skipped_pages: Sequence[int] = (),
) -> tuple[StyleGroupException, ...]:
    """Turn the renderer's own per-page verdicts into the recorded exceptions.

    Called after the page loop, so the exceptions describe the artifact that was
    actually built rather than the pre-pass guess. ``skipped_pages`` are pages
    the renderer could not model at all (no translatable text); they carry no
    deficit and are still reported, because "no text to fit" must not read as
    "fits".
    """
    out: list[StyleGroupException] = []
    for page in sorted(set(page_fits) | set(skipped_pages)):
        report = page_fits.get(page)
        if report is None:
            out.append(
                StyleGroupException(
                    page=page, block_ids=(), deficit_pt=0.0, rung=rung, reason="unmodeled"
                )
            )
            continue
        if report.fits:
            continue
        out.append(
            StyleGroupException(
                page=page,
                block_ids=report.block_ids,
                deficit_pt=report.deficit_pt,
                rung=rung,
                # The renderer knows *why*: an L1 height shortfall and an L2
                # shrink are both exceptions, but they need different follow-ups
                # (content compression vs a placement-strategy decision).
                reason=report.reason,
            )
        )
    return tuple(out)


__all__ = [
    "FULL_WIDTH_SHARE",
    "PageFitModel",
    "build_page_fit_models",
    "exceptions_from_fit",
    "fit_page_model",
    "plan_facsimile_body_group",
]
