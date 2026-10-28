"""1:1 Facsimile 2D monotonic relaxation layout solver.

Solves:
1. Single uniform body size per page (s*) via monotonic binary search.
2. Dynamic vertical leading allocation (leading_em) absorbing EN->ZH contraction without bottom gaps.
3. Local overlap resolution and source overlap detection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

FLOOR_PT: float = 7.5
CEILING_RATIO: float = 1.10
MAX_BODY_PT: float = 12.0
TOL_PT: float = 0.5
DEFAULT_EN_PER_ZH: float = 3.53


@dataclass(frozen=True, slots=True)
class PageLayoutSolution:
    """The layout solution for one 1:1 facsimile page."""

    uniform_size_pt: float
    leading_em: float
    tracking_em: float
    fits: bool
    source_overlap: bool
    delivery_tier: str  # "L1" | "L2" | "L3"
    bottleneck_seg_ids: list[str]


def estimate_needed_height(
    record: dict[str, Any],
    size: float,
    em_ratio: float,
    tracking_em: float = 0.0,
) -> float:
    """Estimated vertical height (pt) this record's text occupies at size and em_ratio."""
    w = float(record.get("w", 0.0))
    if w <= 0.0:
        return float(record.get("n", 1)) * em_ratio * size
    eff_glyph_w = size * max(0.85, (1.0 + tracking_em)) * float(record.get("glyph_aspect", 1.0))
    per_line = max(1.0, w / eff_glyph_w)
    glyphs = float(record.get("glyphs", 0.0))
    lines = max(1, math.ceil(glyphs / per_line))
    return float(lines) * em_ratio * size


def fits_all(
    columns_records: Sequence[Sequence[dict[str, Any]]],
    size: float,
    em_ratio: float,
    tracking_em: float = 0.0,
    tol_pt: float = TOL_PT,
) -> bool:
    """Check if all paragraphs fit within their vertical bounds across all columns."""
    for records in columns_records:
        for idx, rec in enumerate(records):
            limit = float(rec["bot"])
            # In vertical glue mode, the paragraph can expand into the gap before the next
            for later in records[idx + 1 :]:
                if float(later["top"]) < float(rec["top"]) - tol_pt:
                    limit = float(later["top"])
                    break
            needed = estimate_needed_height(rec, size, em_ratio, tracking_em)
            if float(rec["top"]) - needed < limit - tol_pt:
                return False
    return True


def check_source_overlap(
    columns_records: Sequence[Sequence[dict[str, Any]]],
    tol_pt: float = TOL_PT,
) -> bool:
    """Return True if any column has source paragraphs whose bounding boxes overlap."""
    for records in columns_records:
        for i in range(len(records) - 1):
            if float(records[i]["bot"]) < float(records[i + 1]["top"]) - tol_pt:
                return True
    return False


def resolve_local_overlaps(
    records: Sequence[dict[str, Any]],
    max_merge_overlap_pt: float = 2.0,
) -> list[dict[str, Any]]:
    """消解同栏相邻文本块的轻度框交叠.

    When adjacent records in the same column have a tiny boundary overlap (<= max_merge_overlap_pt),
    adjust the boundary so false-positive source_overlap is eliminated.
    """
    if not records:
        return []
    resolved: list[dict[str, Any]] = [dict(records[0])]
    for nxt in records[1:]:
        prev = resolved[-1]
        prev_bot = float(prev["bot"])
        nxt_top = float(nxt["top"])
        if prev_bot < nxt_top - 0.1:
            overlap = nxt_top - prev_bot
            if overlap <= max_merge_overlap_pt:
                # Slight micro-overlap between adjacent paragraphs: align boundary to mid-point
                mid = (prev_bot + nxt_top) / 2.0
                prev["bot"] = mid
                nxt_copy = dict(nxt)
                nxt_copy["top"] = mid
                resolved.append(nxt_copy)
                continue
        resolved.append(dict(nxt))
    return resolved


def _find_bottlenecks(
    columns_records: Sequence[Sequence[dict[str, Any]]],
    size: float,
    em_ratio: float,
    tracking_em: float = 0.0,
    tol_pt: float = TOL_PT,
) -> list[str]:
    """Identify which paragraph/segment ids failed to fit at (size, em_ratio)."""
    bottlenecks: list[str] = []
    for records in columns_records:
        for idx, rec in enumerate(records):
            limit = float(rec["bot"])
            for later in records[idx + 1 :]:
                if float(later["top"]) < float(rec["top"]) - tol_pt:
                    limit = float(later["top"])
                    break
            needed = estimate_needed_height(rec, size, em_ratio, tracking_em)
            if float(rec["top"]) - needed < limit - tol_pt:
                seg_id = str(rec.get("id") or f"col_{idx}")
                bottlenecks.append(seg_id)
    return bottlenecks


def solve_page_2d(
    columns_records: Sequence[Sequence[dict[str, Any]]],
    *,
    source_size_pt: float,
    source_leading_pt: float,
    floor_pt: float = FLOOR_PT,
    ceiling_ratio: float = CEILING_RATIO,
    max_body_pt: float = MAX_BODY_PT,
    tol_pt: float = TOL_PT,
    resolve_overlaps: bool = True,
) -> PageLayoutSolution:
    """Solve uniform font size and leading for a facsimile page.

    Step 1: Check source overlap. If resolve_overlaps is True, resolves micro-overlaps.
    Step 2: Binary search for uniform font size s* in [floor_pt, ceiling].
    Step 3: In contraction mode (EN->ZH), dynamically stretch leading to absorb whitespace.
    """
    effective_cols = (
        [resolve_local_overlaps(col) for col in columns_records]
        if resolve_overlaps
        else columns_records
    )

    overlapped = check_source_overlap(effective_cols, tol_pt=tol_pt)
    if overlapped:
        return PageLayoutSolution(
            uniform_size_pt=round(source_size_pt, 2),
            leading_em=0.33,
            tracking_em=0.0,
            fits=False,
            source_overlap=True,
            delivery_tier="L2",
            bottleneck_seg_ids=[],
        )

    base_em = source_leading_pt / source_size_pt if source_size_pt > 0 else 1.33
    base_em = max(1.15, min(1.65, base_em))
    ceiling = min(max_body_pt, source_size_pt * ceiling_ratio)
    tracking = 0.0
    em_used = base_em

    # Step 1: Binary search for uniform size s*
    if fits_all(effective_cols, ceiling, base_em, tol_pt=tol_pt):
        best_size = ceiling
    elif fits_all(effective_cols, floor_pt, base_em, tol_pt=tol_pt):
        lo, hi = floor_pt, ceiling
        for _ in range(26):
            mid = (lo + hi) / 2.0
            if fits_all(effective_cols, mid, base_em, tol_pt=tol_pt):
                lo = mid
            else:
                hi = mid
        best_size = lo
    else:
        # Tight mode: shrink leading and letter-spacing slightly
        em_used = 1.15
        tracking = -0.015
        if not fits_all(effective_cols, floor_pt, em_used, tracking_em=tracking, tol_pt=tol_pt):
            return PageLayoutSolution(
                uniform_size_pt=round(floor_pt, 2),
                leading_em=round(em_used - 1.0, 3),
                tracking_em=tracking,
                fits=False,
                source_overlap=False,
                delivery_tier="L2",
                bottleneck_seg_ids=_find_bottlenecks(
                    effective_cols, floor_pt, em_used, tracking_em=tracking, tol_pt=tol_pt
                ),
            )
        lo, hi = floor_pt, ceiling
        for _ in range(26):
            mid = (lo + hi) / 2.0
            if fits_all(effective_cols, mid, em_used, tracking_em=tracking, tol_pt=tol_pt):
                lo = mid
            else:
                hi = mid
        best_size = lo

    # Step 2: Dynamic leading stretch when size is capped and room is plentiful (EN->ZH)
    if best_size >= ceiling - 0.05 and fits_all(effective_cols, best_size, base_em, tol_pt=tol_pt):
        lo_em, hi_em = base_em, max(base_em, 1.50)
        for _ in range(20):
            mid_em = (lo_em + hi_em) / 2.0
            if fits_all(effective_cols, best_size, mid_em, tol_pt=tol_pt):
                lo_em = mid_em
            else:
                hi_em = mid_em
        em_used = lo_em

    return PageLayoutSolution(
        uniform_size_pt=round(best_size, 2),
        leading_em=round(max(0.15, em_used - 1.0), 3),
        tracking_em=tracking,
        fits=True,
        source_overlap=False,
        delivery_tier="L1",
        bottleneck_seg_ids=[],
    )


__all__ = [
    "CEILING_RATIO",
    "DEFAULT_EN_PER_ZH",
    "FLOOR_PT",
    "MAX_BODY_PT",
    "PageLayoutSolution",
    "TOL_PT",
    "check_source_overlap",
    "estimate_needed_height",
    "fits_all",
    "resolve_local_overlaps",
    "solve_page_2d",
]
