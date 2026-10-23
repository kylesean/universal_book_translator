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

from ubt.core.qe.defect_taxonomy import CONSTRAINT_PREFIX, format_defect_flag
from ubt.core.qe.term_metrics import TermMetrics

#: Prefix that makes the constraint greppable in ledgers/reports. The repair
#: prompt surfaces ``error_flags`` verbatim under "Issues to Fix", so the flag
#: doubles as the instruction. Defined in :mod:`ubt.core.qe.defect_taxonomy`
#: because that module's marker tables classify this same string; re-exported
#: here for the planner's callers.
__all__ = ["CONSTRAINT_PREFIX", "ConsistencyTask", "plan_consistency_tasks"]

#: Terminology drift is a Major defect by the triage contract (it damages
#: consistency, not facts), which is why it is absent from the critical table.
CONSTRAINT_SEVERITY = "major"


@dataclass(frozen=True, slots=True)
class ConsistencyTask:
    """One block that must be re-translated to restore a canonical rendering."""

    block_id: str
    source: str
    expected: str

    @property
    def flag(self) -> str:
        """The repair instruction, carried as a structured error flag.

        ``<prefix>:<severity>:<instruction>`` — the same shape every new flag
        writer uses (see ``format_defect_flag``). It still starts with
        ``CONSTRAINT_PREFIX``, so the structural table, the deterministic
        glossary pass and ``_flags_are_recheckable`` keep matching it by prefix,
        while triage reads the declared severity instead of guessing from prose.
        """
        return format_defect_flag(
            CONSTRAINT_PREFIX.rstrip(":"),
            CONSTRAINT_SEVERITY,
            f"render '{self.source}' as '{self.expected}'",
        )


def plan_consistency_tasks(terms: TermMetrics, max_blocks: int = 50) -> tuple[ConsistencyTask, ...]:
    """One task per (block, drifted term), stable order, capped by *blocks*.

    ``max_blocks <= 0`` disables planning. The cap counts distinct blocks, not
    tasks: ``consistency_max_repairs`` budgets re-translations and the paid unit
    is a block. A selected block keeps *every* drifted-term task, or its repair
    is under-constrained and could "fix" one term while leaving another drifted.
    Ordering is ``(block_id, source)`` so the same drift always yields the same
    plan (grep-able, testable, and cheap to diff between runs).
    """
    if max_blocks <= 0:
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
    # Tasks are sorted by block, so a selected block's tasks are contiguous and
    # all of them survive the cap.
    selected: set[str] = set()
    bounded: list[ConsistencyTask] = []
    for task in tasks:
        if task.block_id not in selected:
            if len(selected) >= max_blocks:
                continue
            selected.add(task.block_id)
        bounded.append(task)
    return tuple(bounded)
