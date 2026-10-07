"""Short/long execution policy derived from the route decision.

Three booleans and a reason string, all functions of ``route_decision.mode``
plus one config flag. The former ``Granularity`` axis is gone: the pipeline is
always the frozen-math micro-block architecture (the whole-section ``macro``
mode is retired and ``UBTConfig`` no longer carries the knob).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ubt.core.config import UBTConfig
from ubt.core.ir.models import BookManifest
from ubt.core.router_mode import RouteDecision


@dataclass(frozen=True)
class AdaptivePolicy:
    """The short/long flags stages read, plus the human-readable reason."""

    fast_lane_bible: bool
    visual_blocking: bool
    deterministic_glossary: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        # ``granularity`` stays in the run-metadata payload as a constant: the
        # report schema still names the axis, and every run is micro.
        return {
            "granularity": "micro",
            "fast_lane_bible": self.fast_lane_bible,
            "visual_blocking": self.visual_blocking,
            "deterministic_glossary": self.deterministic_glossary,
            "reason": self.reason,
        }


def resolve_adaptive_policy(
    manifest: BookManifest,
    route_decision: RouteDecision,
    config: UBTConfig,
) -> AdaptivePolicy:
    """Resolve the short/long execution policy from the route decision.

    Short born-digital documents keep the fast-lane seed bible and the
    deterministic glossary, but run the same frozen-math block pipeline, so
    math masking and bounded context are never traded away for speed.
    """
    del manifest  # kept in the signature: policy is a function of the document
    is_short = route_decision.mode == "short"
    # ``deterministic_glossary`` is the export-time literal term enforcement
    # layer, and it is short-route only: on a whole book an Aho-Corasick
    # substring swap can splice across word boundaries, so the book route leans
    # on the bible + QE defenses instead. There is no config switch to force it
    # on for books.
    return AdaptivePolicy(
        fast_lane_bible=is_short,
        visual_blocking=is_short or config.visual_blocking_gate_enabled,
        deterministic_glossary=is_short,
        reason=(
            f"Canonical frozen-math publication pipeline ({route_decision.pages}pp"
            + (", fast-lane seed bible" if is_short else "")
            + ")"
        ),
    )
