"""Adaptive Policy: render-route resolution and the (single) execution granularity.

Decision point for the "Single Pipeline" architecture. Two orthogonal axes:

1. Granularity — always :data:`Granularity.MICRO` (frozen-math block keyset
   pagination). The retired whole-section ``macro`` mode is rejected by
   ``UBTConfig``; the enum keeps only the value the pipeline actually runs.
2. Render engine (publication, rigid, auto) — resolved from physical layout
   facts by :func:`resolve_render_engine_from_signals`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ubt.core.config import UBTConfig, canonical_render_engine
from ubt.core.ir.models import BlockType, BookManifest, IRBlock
from ubt.core.router_mode import RouteDecision

logger = logging.getLogger(__name__)

# Structural block share above which the rigid engine wins the auto
# dispatch: reflowing a formula/table-heavy page risks collisions that
# region-rigid typesetting cannot produce (figures stay untouched).
STRUCT_SHARE_AUTO = 0.20

# Page-level multi-column share above which the rigid engine wins the auto
# dispatch. Multi-column extraction order is exactly what a reflow re-typeset
# mangles (arXiv 2609.20519: a two-column paper whose 7/15 columnar pages lost
# 4 of 6 figures and shattered Table 1). Unlike ``STRUCT_SHARE_AUTO`` this is a
# page ratio, so it does not drift with parser chunk granularity, and the
# signal is bimodal (single-column prose ~0.02 vs a two-column body ~0.4+), so
# the cutoff has wide margin.
MULTICOLUMN_SHARE_AUTO = 0.25


class Granularity(StrEnum):
    """Execution chunking granularity for the unified translation pipeline.

    Only ``MICRO`` remains: the retired whole-section ``macro`` mode produced
    unbounded whole-chapter synthesis and was replaced by the frozen-math block
    architecture. ``UBTConfig`` rejects ``granularity='macro'`` outright.
    """

    MICRO = "micro"  # Atomic block / paragraph level (geometry backfill, scans, bounded context)


@dataclass(frozen=True)
class AdaptivePolicy:
    """Consolidated policy driving stages 1 through 6 in the unified pipeline."""

    granularity: Granularity
    render_engine: str  # "publication" | "rigid" | "auto"
    fast_lane_bible: bool
    visual_blocking: bool
    deterministic_glossary: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity.value,
            "render_engine": self.render_engine,
            "fast_lane_bible": self.fast_lane_bible,
            "visual_blocking": self.visual_blocking,
            "deterministic_glossary": self.deterministic_glossary,
            "reason": self.reason,
        }


def resolve_adaptive_policy(
    manifest: BookManifest,
    route_decision: RouteDecision,
    config: UBTConfig,
    forced_granularity: str | None = None,
) -> AdaptivePolicy:
    """Resolve adaptive execution policy from document facts and config.

    Granularity is always :data:`Granularity.MICRO`: the retired whole-section
    ``macro`` mode is rejected by ``UBTConfig``. Short born-digital documents
    additionally keep the fast-lane seed bible and the deterministic glossary,
    but run the same frozen-math block pipeline, so math masking and bounded
    context are never traded away for speed.
    """
    norm_engine = canonical_render_engine(config.render_engine or "publication")

    # Defensive only: UBTConfig rejects granularity='macro' at parse time, so
    # this catches a direct caller of this function. It warns rather than
    # silently downgrading, because a silent downgrade would let a user believe
    # whole-section synthesis is running when it is not.
    if forced_granularity == "macro":
        logger.warning(
            "Granularity 'macro' (legacy whole-section synthesis) has been retired in favor "
            "of the canonical frozen-math micro-block architecture; executing under micro "
            "granularity. Remove the setting — UBTConfig rejects it."
        )

    is_short = route_decision.mode == "short"
    return AdaptivePolicy(
        granularity=Granularity.MICRO,
        render_engine=norm_engine
        if norm_engine in ("publication", "rigid", "auto")
        else "publication",
        fast_lane_bible=is_short,
        visual_blocking=True if is_short else config.visual_blocking_gate_enabled,
        deterministic_glossary=is_short,
        reason=(
            f"Canonical frozen-math publication pipeline ({route_decision.pages}pp"
            + (", fast-lane seed bible" if is_short else "")
            + ")"
        ),
    )


def resolve_render_engine_from_signals(
    requested: str,
    *,
    has_math: bool,
    struct_share: float,
    has_geometry: bool = True,
    multicolumn_share: float = 0.0,
) -> str:
    """Canonical render-route decision from document facts, without IR blocks.

    The single source of truth for the ``auto`` dispatch, shared by the runtime
    renderer (which derives the facts from IR blocks via
    :func:`resolve_pdf_engine`) and by the assessor/advisor (which estimate them
    from cheap document probes), so the route a user is quoted is the route that
    actually runs.

    ``publication``/``reflow`` and ``rigid`` pass through; ``auto`` routes to
    rigid when the document carries math, is structure-dense, or is laid out in
    columns **and** has usable geometry. Rigid typesets from source boxes, so a
    plain-text fallback that stamps zero-area bboxes must reflow instead.
    Unknown names fall back to publication.
    """
    norm = canonical_render_engine(requested) if requested else "auto"
    if norm in ("publication", "rigid"):
        return norm
    if norm != "auto":
        logger.warning("Unknown render_engine=%r, falling back to 'publication'", requested)
        return "publication"
    if not has_geometry:
        logger.warning(
            "render_engine='auto': no usable geometry (plain-text fallback "
            "extraction); routing to 'publication' reflow"
        )
        return "publication"
    # Structure-dense documents take the rigid engine: reflow rebuilds the
    # page from extracted structure, and extraction is exactly where these
    # documents fail -- multi-row table headers shatter into single-character
    # cells and TikZ/matplotlib figures go missing wholesale (arXiv
    # 2609.20519: 4 of 6 figures lost, Table 1 unusable). Rigid keeps the
    # source page as the canvas, so untouched geometry cannot be corrupted.
    # ``multicolumn_share`` is the page-level form of the same tell: a
    # multi-column body is what a reflow re-typeset mangles most, and the
    # 2609.20519 case is caught by it (7/15 columnar pages) even though its
    # IR block-count share (8.6%) and FORMULA-block count (0) both miss.
    # Plain prose has no such risk and gets the reflow route's better typography.
    return (
        "rigid"
        if has_math
        or struct_share >= STRUCT_SHARE_AUTO
        or multicolumn_share >= MULTICOLUMN_SHARE_AUTO
        else "publication"
    )


def resolve_pdf_engine(
    requested: str,
    blocks: Sequence[IRBlock],
    manifest: BookManifest | None = None,
) -> str:
    """Resolve the PDF render engine for one render call from its IR blocks and manifest.

    Derives the routing signals the canonical dispatcher needs
    (``has_math`` / ``struct_share`` / ``multicolumn_share`` / ``has_geometry``)
    from the block mix and manifest facts, then delegates to
    :func:`resolve_render_engine_from_signals`.
    """
    materialized = list(blocks)
    has_geometry = any(
        b.bbox is not None and b.bbox.x1 > b.bbox.x0 and b.bbox.y1 > b.bbox.y0 for b in materialized
    )
    has_math = False
    struct_share = 0.0
    multicolumn_share = 0.0
    if materialized:
        # ``has_math`` is driven by explicit FORMULA blocks only. An inline
        # ``$x$`` anywhere in prose used to flip this True via
        # ``count_math_spans``, which routed a whole 300-page prose book to the
        # rigid engine -- and rigid is monolingual, so the requested bilingual
        # delivery silently became mono. Reflow renders inline math fine;
        # display/formula blocks are the extraction risk rigid exists for.
        has_math = any(b.block_type == BlockType.FORMULA for b in materialized)
        struct_share = sum(
            1
            for b in materialized
            if b.block_type in (BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE, BlockType.CODE)
        ) / len(materialized)

    if manifest is not None:
        rd = getattr(manifest.run, "route_decision", None)
        if isinstance(rd, dict):
            if rd.get("formula_heavy"):
                has_math = True
            # Page-level census carried by the router. Preferred over the
            # block-count share because it does not drift with parser chunking.
            try:
                multicolumn_share = float(rd.get("multicolumn_page_share") or 0.0)
            except (TypeError, ValueError):
                multicolumn_share = 0.0

    return resolve_render_engine_from_signals(
        requested,
        has_math=has_math,
        struct_share=struct_share,
        has_geometry=has_geometry,
        multicolumn_share=multicolumn_share,
    )
