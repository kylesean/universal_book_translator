"""Turn document-level terminology drift into bounded constrained re-translations.

The planner is deterministic: drift in, a stable task list out. Pinned contracts:

* ``ConsistencyTask.flag`` is the greppable repair instruction carried verbatim
  as an error flag;
* one task per ``(block, drifted term)`` — exact renderings are skipped and
  duplicate hits collapse;
* ordering is ``(block_id, source)`` so the same drift always yields the same
  plan;
* the cap counts distinct **blocks**, not tasks, and a selected block keeps
  every one of its drifted-term tasks (or its repair is under-constrained);
* ``max_blocks <= 0`` disables planning.
"""

from __future__ import annotations

import pytest

from ubt.core.qe.consistency_enforce import (
    CONSTRAINT_PREFIX,
    ConsistencyTask,
    plan_consistency_tasks,
)
from ubt.core.qe.term_metrics import TermHit, TermMetrics

pytestmark = pytest.mark.fast


def _metrics(*hits: TermHit) -> TermMetrics:
    return TermMetrics(
        terms_expected=0,
        terms_rendered=0,
        term_precision=0.0,
        fuzzy_term_precision=0.0,
        term_recall=0.0,
        per_hit=hits,
    )


def test_flag_is_the_repair_instruction() -> None:
    assert ConsistencyTask("b", "X", "Y").flag == ("terminology_consistency: render 'X' as 'Y'")
    assert CONSTRAINT_PREFIX in ConsistencyTask("b", "X", "Y").flag


def test_planning_is_disabled_for_non_positive_caps() -> None:
    metrics = _metrics(TermHit("A", "EA", "b1", exact=False, fuzzy=False))
    assert plan_consistency_tasks(metrics, max_blocks=0) == ()
    assert plan_consistency_tasks(metrics, max_blocks=-1) == ()


def test_exact_hits_are_not_planned() -> None:
    metrics = _metrics(TermHit("A", "EA", "b1", exact=True, fuzzy=True))
    assert plan_consistency_tasks(metrics) == ()


def test_duplicate_hits_collapse() -> None:
    metrics = _metrics(
        TermHit("A", "EA", "b1", exact=False, fuzzy=False),
        TermHit("A", "EA", "b1", exact=False, fuzzy=False),
    )
    tasks = plan_consistency_tasks(metrics)
    assert len(tasks) == 1
    assert (tasks[0].block_id, tasks[0].source) == ("b1", "A")


def test_tasks_are_ordered_by_block_then_source() -> None:
    metrics = _metrics(
        TermHit("A", "EA", "b2", exact=False, fuzzy=False),
        TermHit("B", "EB", "b1", exact=False, fuzzy=False),
        TermHit("A", "EA", "b1", exact=False, fuzzy=False),
    )
    tasks = plan_consistency_tasks(metrics)
    assert [(t.block_id, t.source) for t in tasks] == [("b1", "A"), ("b1", "B"), ("b2", "A")]


def test_the_cap_counts_blocks_and_keeps_every_task_of_a_selected_block() -> None:
    metrics = _metrics(
        TermHit("A", "EA", "b1", exact=False, fuzzy=False),
        TermHit("B", "EB", "b1", exact=False, fuzzy=False),
        TermHit("C", "EC", "b2", exact=False, fuzzy=False),
    )
    tasks = plan_consistency_tasks(metrics, max_blocks=1)
    assert [(t.block_id, t.source) for t in tasks] == [("b1", "A"), ("b1", "B")]
