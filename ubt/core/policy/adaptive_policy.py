"""Short/long execution policy derived from the route decision.

Three booleans and a reason string, all functions of ``route_decision.mode``
plus one config flag. There is no granularity axis: the pipeline is always the
frozen-math micro-block architecture, and ``UBTConfig`` carries no knob for it.
"""

from __future__ import annotations

from collections.abc import Mapping
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

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        fallback: AdaptivePolicy,
    ) -> AdaptivePolicy:
        """Read a policy persisted as ``manifest.run.adaptive_policy``.

        A re-export must reproduce the policy the original run resolved, not
        re-derive it from today's config: the route decision that produced it
        is gone, and a short document's ``visual_blocking`` /
        ``deterministic_glossary`` are both true there while the config
        defaults are false. Missing or malformed fields fall back to the
        caller's config-derived value, so old manifests keep working.
        """
        if not isinstance(payload, Mapping):
            return fallback

        def flag(key: str, default: bool) -> bool:
            # A JSON ``true``/``false`` is a bool; anything else (a quoted
            # ``"false"``, a 0/1 from an older writer) is not a value this
            # policy can trust, so keep the config-derived default instead of
            # applying Python truthiness to it.
            value = payload.get(key)
            return value if isinstance(value, bool) else default

        reason = payload.get("reason")
        return cls(
            fast_lane_bible=flag("fast_lane_bible", fallback.fast_lane_bible),
            visual_blocking=flag("visual_blocking", fallback.visual_blocking),
            deterministic_glossary=flag("deterministic_glossary", fallback.deterministic_glossary),
            reason=reason.strip()
            if isinstance(reason, str) and reason.strip()
            else fallback.reason,
        )

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
    # The manifest is not an input today; keep the parameter so the call surface
    # stays stable while the policy grows.
    del manifest
    is_short = route_decision.mode == "short"
    # ``deterministic_glossary`` is the export-time literal term enforcement
    # layer. Short-route by default; ``deterministic_glossary_long_route``
    # opts a whole book in (the enforcer's word-boundary, protected-span,
    # approved-compound and jieba fail-closed guards make the old
    # "Aho-Corasick splices across word boundaries" concern a measurable
    # claim rather than a premise — enable after an on/off comparison on
    # your own corpus).
    return AdaptivePolicy(
        fast_lane_bible=is_short,
        visual_blocking=is_short or config.visual_blocking_gate_enabled,
        deterministic_glossary=is_short or config.deterministic_glossary_long_route,
        reason=(
            f"Canonical frozen-math publication pipeline ({route_decision.pages}pp"
            + (", fast-lane seed bible" if is_short else "")
            + (
                ", deterministic glossary (config opt-in)"
                if config.deterministic_glossary_long_route
                else ""
            )
            + ")"
        ),
    )
