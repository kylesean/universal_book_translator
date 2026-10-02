"""Compatibility re-export of :mod:`ubt.core.engine.facts`.

The inter-stage value contract moved into the core engine so the stages no
longer import a compiler package; this shim keeps the old import path working
for external callers.
"""

from __future__ import annotations

from ubt.core.engine.facts import (
    LayoutAdvisory,
    RenderOutcome,
    RenderPlan,
    RunFacts,
    Scoring,
    Terminology,
)

__all__ = [
    "LayoutAdvisory",
    "RenderOutcome",
    "RenderPlan",
    "RunFacts",
    "Scoring",
    "Terminology",
]
