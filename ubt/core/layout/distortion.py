"""LayoutDistortionScore calculation for adaptive delivery.

Design: ``docs/design/adaptive_delivery_prd.md`` §8.10 &
``docs/guides/ADAPTIVE_DELIVERY_TECH_DESIGN.md`` §7.4. Resolves GAP-17.

Formula:
    LayoutDistortionScore(page) =
      0.30 * overlap_ratio
    + 0.25 * out_of_bounds_ratio
    + 0.20 * style_drift
    + 0.15 * gutter_violation
    + 0.10 * whitespace_void

Thresholds:
    < 0.08:      Clean faithful deliverable
    0.08 - 0.15: Acceptable with major review finding
    > 0.15:      Excessive distortion; trigger page_reflow or document readable degradation
    Critical:    overlap_ratio > 0.001 or out_of_bounds_ratio > 0.001 triggers
                 critical fail-closed (the epsilon absorbs float noise).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ubt.core.layout.cels import Box, ColumnTrack


@dataclass(frozen=True, slots=True)
class DistortionSubscores:
    overlap_ratio: float
    out_of_bounds_ratio: float
    style_drift: float
    gutter_violation: float
    whitespace_void: float


@dataclass(frozen=True, slots=True)
class PageDistortionScore:
    page_no: int
    score: float
    subscores: DistortionSubscores
    is_critical: bool
    verdict: str  # "faithful_deliverable" | "review_major" | "degrade_reflow" | "critical_fail"


def compute_page_distortion(
    *,
    page_no: int,
    boxes: Sequence[Box],
    tracks: Sequence[ColumnTrack],
    printable_box: Box,
    source_body_size_pt: float,
    realized_body_size_pt: float,
    tail_void_area: float = 0.0,
    source_boxes: Sequence[Box] | None = None,
) -> PageDistortionScore:
    """Compute the 5-component LayoutDistortionScore for one page."""
    n_boxes = len(boxes)
    if n_boxes == 0:
        sub = DistortionSubscores(0.0, 0.0, 0.0, 0.0, 0.0)
        return PageDistortionScore(
            page_no=page_no,
            score=0.0,
            subscores=sub,
            is_critical=False,
            verdict="faithful_deliverable",
        )

    def _calc_overlap(box_list: Sequence[Box]) -> float:
        ov = 0.0
        n = len(box_list)
        for i in range(n):
            b1 = box_list[i]
            for j in range(i + 1, n):
                b2 = box_list[j]
                x_ov = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                y_ov = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if x_ov > 0.5 and y_ov > 0.5:
                    ov += x_ov * y_ov
        return ov

    # 1. Overlap ratio: overlapping area / total box area (evaluates both solved and source boxes)
    total_area = sum(b.area for b in boxes) or 1.0
    overlap_area = _calc_overlap(boxes)
    if source_boxes:
        overlap_area = max(overlap_area, _calc_overlap(source_boxes))
    overlap_ratio = min(1.0, overlap_area / total_area)

    # 2. Out-of-bounds ratio: boxes outside printable box / total boxes
    oob_count = 0
    for b in boxes:
        if (
            b.x0 < printable_box.x0 - 1.0
            or b.x1 > printable_box.x1 + 1.0
            or b.y0 < printable_box.y0 - 1.0
            or b.y1 > printable_box.y1 + 1.0
        ):
            oob_count += 1
    out_of_bounds_ratio = float(oob_count) / float(n_boxes)

    # 3. Style drift: |realized - source| / source
    if source_body_size_pt > 0:
        drift = abs(realized_body_size_pt - source_body_size_pt) / source_body_size_pt
        style_drift = min(1.0, drift)
    else:
        style_drift = 0.0

    # 4. Gutter violation: boxes crossing track boundaries / total boxes
    gutter_viol_count = 0
    if len(tracks) >= 2:
        for b in boxes:
            # A box violates track gutters if it straddles a track boundary without being a pinned obstacle
            in_track = any((b.x0 >= t.x0 - 1.0 and b.x1 <= t.x1 + 1.0) for t in tracks)
            if not in_track:
                gutter_viol_count += 1
    gutter_violation = float(gutter_viol_count) / float(n_boxes)

    # 5. Whitespace void: unused void area / printable area
    printable_area = printable_box.area or 1.0
    whitespace_void = min(1.0, max(0.0, tail_void_area / printable_area))

    subscores = DistortionSubscores(
        overlap_ratio=round(overlap_ratio, 4),
        out_of_bounds_ratio=round(out_of_bounds_ratio, 4),
        style_drift=round(style_drift, 4),
        gutter_violation=round(gutter_violation, 4),
        whitespace_void=round(whitespace_void, 4),
    )

    score = (
        0.30 * subscores.overlap_ratio
        + 0.25 * subscores.out_of_bounds_ratio
        + 0.20 * subscores.style_drift
        + 0.15 * subscores.gutter_violation
        + 0.10 * subscores.whitespace_void
    )
    score = round(score, 4)

    is_critical = subscores.overlap_ratio > 0.001 or subscores.out_of_bounds_ratio > 0.001
    if is_critical:
        verdict = "critical_fail"
    elif score < 0.08:
        verdict = "faithful_deliverable"
    elif score <= 0.15:
        verdict = "review_major"
    else:
        verdict = "degrade_reflow"

    return PageDistortionScore(
        page_no=page_no,
        score=score,
        subscores=subscores,
        is_critical=is_critical,
        verdict=verdict,
    )
