"""Adaptive Policy: the (single) execution granularity.

Decision point for the "Single Pipeline" architecture:

Granularity — always :data:`Granularity.MICRO` (frozen-math block keyset
pagination). The retired whole-section ``macro`` mode is rejected by
``UBTConfig``; the enum keeps only the value the pipeline actually runs.

There is no render-route axis. Every PDF composes through the single
source-canvas ``overlay`` engine (``ubt.core.config.RENDER_ENGINE``), so the
former density dispatch, its threshold constants and the ``auto``/``rigid``/
``reflow``/``publication``/``composite`` vocabulary were deleted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ubt.core.config import UBTConfig
from ubt.core.ir.models import BookManifest
from ubt.core.router_mode import RouteDecision

logger = logging.getLogger(__name__)


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
    fast_lane_bible: bool
    visual_blocking: bool
    deterministic_glossary: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity.value,
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
    # ``deterministic_glossary`` is the export-time literal term enforcement
    # layer, and it is short-route only: on a whole book an Aho-Corasick
    # substring swap can splice across word boundaries, so the book route leans
    # on the bible + QE defenses instead. There is no config switch to force it
    # on for books.
    return AdaptivePolicy(
        granularity=Granularity.MICRO,
        fast_lane_bible=is_short,
        visual_blocking=is_short or config.visual_blocking_gate_enabled,
        deterministic_glossary=is_short,
        reason=(
            f"Canonical frozen-math publication pipeline ({route_decision.pages}pp"
            + (", fast-lane seed bible" if is_short else "")
            + ")"
        ),
    )
