"""Turn document-level terminology drift into bounded constrained re-translations.

The deterministic scan (:mod:`ubt.core.qe.term_metrics`) already reports which
blocks render a glossary term with a different surface than the canonical one.
Rewriting those surfaces in place is not safe — the variant surface is unknown
and string surgery can damage prose — so the accepted mechanism is a
**constrained re-translation** of just those blocks: the block is repaired with
an explicit "render X as Y" instruction and kept only when it passes the same
structural and QE gates as any other repair.

This module is the deterministic planner: drift in, a bounded task list out.
"""

from __future__ import annotations

from dataclasses import dataclass

from ubt.core.qe.term_metrics import TermMetrics

#: Prefix that makes the constraint greppable in ledgers/reports. The repair
#: prompt surfaces ``error_flags`` verbatim under "Issues to Fix", so the flag
#: doubles as the instruction.
CONSTRAINT_PREFIX = "terminology_consistency:"


@dataclass(frozen=True, slots=True)
class ConsistencyTask:
    """One block that must be re-translated to restore a canonical rendering."""

    block_id: str
    source: str
    expected: str

    @property
    def flag(self) -> str:
        """The repair instruction, carried as an error flag."""
        return f"{CONSTRAINT_PREFIX} render '{self.source}' as '{self.expected}'"


def plan_consistency_tasks(terms: TermMetrics, max_tasks: int = 50) -> tuple[ConsistencyTask, ...]:
    """One task per (block, drifted term), stable order, capped at ``max_tasks``.

    ``max_tasks <= 0`` disables planning. Ordering is ``(block_id, source)`` so
    the same drift always yields the same plan (grep-able, testable, and cheap
    to diff between runs).
    """
    if max_tasks <= 0:
        return ()
    seen: set[tuple[str, str]] = set()
    tasks: list[ConsistencyTask] = []
    for hit in terms.per_hit:
        if hit.exact:
            continue
        key = (hit.block_id, hit.source)
        if key in seen:
            continue
        seen.add(key)
        tasks.append(
            ConsistencyTask(block_id=hit.block_id, source=hit.source, expected=hit.expected)
        )
    tasks.sort(key=lambda task: (task.block_id, task.source))
    return tuple(tasks[:max_tasks])
