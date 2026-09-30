"""Values one stage produces and a later one consumes (ADR-0001 explicit params).

The shared :class:`~ubt.core.engine.stage_context.StageContext` used to carry
these as mutable fields, which made "what flows between stages" indistinguishable
from "what the run was configured with". The plan now owns the value and hands it
to the stage that reads it, so the flow is a named, typed parameter.

A producer cannot *return* its value: every stage is an async generator yielding
progress events (``async for event in ...``), and an async generator has no return
value. The producer therefore fills the plan-owned value and the plan passes it
on -- the hand-off is explicit in the call, not hidden on a context object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ubt.core.config import DualMode
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.policy.bilingual_advisor import Advisory
from ubt.core.qe.base import BaseQERunner


@dataclass
class Terminology:
    """The run's enforced glossary and abbreviation channel.

    Produced by the bible stage and consumed by every stage that drafts, repairs,
    scores or exports text. ``glossary_dicts`` are the translated terms the
    glossary enforcers and validators use; ``abbreviation_entries`` are the
    keep-unchanged channel the draft prompt carries.
    """

    glossary_dicts: list[dict[str, Any]] = field(default_factory=list)
    abbreviation_entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LayoutAdvisory:
    """The bilingual layout decision.

    Phase 1 (mode advisory, before drafting) fills it; phase 2 (difficulty
    advisory, after triage) reads it to decide whether to downgrade a step.
    """

    advisory: Advisory | None = None
    tier_basis: DualMode = "inline"
    enforcement: Literal["advise", "auto"] = "advise"


@dataclass
class Scoring:
    """The terminology-bound scoring collaborators the quality gate produces.

    The gate binds the run's glossary onto the QE runner so a terminology
    violation scores at its own band, and builds the repair loop around that
    runner; repair, triage and consistency then re-score with the same one.
    ``None`` means "use the run's own" (``ctx.qe_runner`` / ``ctx.repair_loop``),
    so a caller that injects a runner/loop keeps it.
    """

    qe_runner: BaseQERunner | None = None
    repair_loop: RepairLoop | None = None


@dataclass
class RunFacts:
    """The values a run threads between its stages.

    The orchestrator owns one and hands it to the plan, so the plan threads a
    named value to each stage and the terminal hook (which needs the same facts
    the stages filled) can read them.
    """

    terminology: Terminology = field(default_factory=Terminology)
    layout: LayoutAdvisory = field(default_factory=LayoutAdvisory)
    scoring: Scoring = field(default_factory=Scoring)


__all__ = ["LayoutAdvisory", "RunFacts", "Scoring", "Terminology"]
