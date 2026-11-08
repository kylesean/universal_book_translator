"""One decision tree for the status a ledger row may report.

``job_meta.status`` has no ``running`` value: the row is written
``initialized`` at ingest and stays that way for the whole run, so the column
alone cannot tell a live job from one whose process died. Three surfaces used
to answer that question with three private decision trees — the REST
``/status`` disk fallback, the MCP ``ubt_job_status`` fallback, and the
``/jobs`` listing — and the trees had drifted: REST failed a ledger whose
writer had just died while MCP gave it a quiet grace window, and MCP checked
a persisted ``failed`` *after* the all-blocks-terminal counter while the other
two honored terminal rows first. A new reader surface would have had to pick
which tree to copy.

The tree here is the one they must all call. The only caller-visible knob is
the quiet grace window: MCP deliberately reports RUNNING for a ledger whose
writer lease is gone but that was written recently (a live run in another
process spends minutes inside one model call; telling it to re-run collides
with the writer lock or pays the provider twice), while the REST and listing
surfaces have no staleness signal and fail immediately. Everything else —
terminal rows pass through, all-terminal blocks mean completed even when the
finalize never landed, agent states are honored only after that completion
check because they are intentionally lock-free, and an unwritten lease means
failed — is shared, so the surfaces cannot disagree about one ledger again.
"""

from __future__ import annotations

from dataclasses import dataclass

from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES

#: The agent states an Agent-mode job parks in between tool calls. Neither a
#: writer lease nor a quiet ledger is evidence of a crash while one of these is
#: the session's current state.
PARKED_AGENT_STATES = ("awaiting_agent", "needs_attention")


@dataclass(frozen=True)
class ProjectedStatus:
    """The reportable status plus whether it was reached by declaring a crash.

    ``interrupted`` is true exactly when the tree concluded the writer is gone
    (``failed`` via the no-writer branch, never via a persisted terminal row),
    so a caller can attach its own "the run was interrupted" note without
    re-deriving the decision.
    """

    status: str
    interrupted: bool = False


def project_ledger_status(
    *,
    stored: str,
    total: int,
    done: int,
    agent_state: str | None = None,
    has_live_writer: bool,
    assume_running_when_unwritten: bool = False,
) -> ProjectedStatus:
    """Project a ``job_meta`` row onto the status a client may see.

    ``stored`` is the persisted row value; ``total``/``done`` are the block
    counters (``done`` counts every terminal block: completed, failed,
    needs_human, blocked_human); ``agent_state`` is the persisted session
    state or ``None``; ``has_live_writer`` answers
    :meth:`ubt.core.engine.writer_lock.LedgerWriterLock.lease_holder`;
    ``assume_running_when_unwritten`` is the MCP quiet-grace window — pass
    ``True`` only when the caller has its own staleness signal and the ledger
    is NOT stale.
    """
    if stored in TERMINAL_JOB_STATUSES:
        return ProjectedStatus(stored)
    if total > 0 and done >= total:
        return ProjectedStatus("completed")
    if agent_state in PARKED_AGENT_STATES:
        return ProjectedStatus(agent_state)
    if has_live_writer:
        return ProjectedStatus("running")
    if assume_running_when_unwritten:
        return ProjectedStatus("running")
    return ProjectedStatus("failed", interrupted=True)
