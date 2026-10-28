"""Lifecycle state machine and transition guards for IRBlock.

Defines the formal transition matrix and predecessor relations for BlockStatus.
Every status transition in-memory or persisted to SQLite must satisfy these rules,
preventing backward skips, terminal resurrection, or unverified leaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ubt.core.exceptions import InvalidBlockStatusTransitionError

if TYPE_CHECKING:
    from ubt.core.ir.models import BlockStatus


def _init_transitions() -> tuple[
    dict[BlockStatus, frozenset[BlockStatus]],
    dict[BlockStatus, frozenset[BlockStatus]],
]:
    from ubt.core.ir.models import BlockStatus

    transitions: dict[BlockStatus, frozenset[BlockStatus]] = {
        # PENDING: initial state of an extracted block
        BlockStatus.PENDING: frozenset(
            {
                BlockStatus.PENDING,  # idempotent
                BlockStatus.DRAFTED,  # drafted by translation engine
                BlockStatus.MTQE_PASSED,  # TM hit or skip_translate during ingest/draft
                BlockStatus.REPAIR_PENDING,  # dirty tag restoration requiring repair
                BlockStatus.FAILED,  # drafting failure / budget exhausted / timeout
            }
        ),
        # DRAFTED: initial draft generated, awaiting QE audit
        BlockStatus.DRAFTED: frozenset(
            {
                BlockStatus.DRAFTED,  # idempotent
                BlockStatus.MTQE_PASSED,  # passed QE threshold with no MQM blockers
                BlockStatus.REPAIR_PENDING,  # QE flagged below threshold or format glitch
                BlockStatus.NEEDS_HUMAN,  # MQM triage major defect requiring post-edit
                BlockStatus.BLOCKED_HUMAN,  # MQM triage critical defect (safety/hallucination)
                BlockStatus.FAILED,  # unrecoverable QE evaluation error / budget cap
            }
        ),
        # REPAIR_PENDING: queued for targeted LLM repair
        BlockStatus.REPAIR_PENDING: frozenset(
            {
                BlockStatus.REPAIR_PENDING,  # multi-round intermediate retry
                BlockStatus.REPAIRED,  # targeted repair succeeded
                BlockStatus.MTQE_PASSED,  # alternative repair / verified translation
                BlockStatus.NEEDS_HUMAN,  # repair budget/rounds exhausted, non-critical defect
                BlockStatus.BLOCKED_HUMAN,  # repair budget/rounds exhausted, critical blocker
                BlockStatus.FAILED,  # fatal repair error / unrecoverable failure
            }
        ),
        # MTQE_PASSED: quality passed, released directly (terminal unless Bible inconsistency or human PE revision)
        BlockStatus.MTQE_PASSED: frozenset(
            {
                BlockStatus.MTQE_PASSED,  # idempotent
                BlockStatus.REPAIRED,  # Human PE post-edit or targeted terminology fix
                BlockStatus.REPAIR_PENDING,  # Consistency stage: Translation Bible terminology violation
                BlockStatus.NEEDS_HUMAN,  # Consistency stage: inter-chapter terminology divergence
                BlockStatus.BLOCKED_HUMAN,  # Critical safety/compliance audit escalation
                BlockStatus.FAILED,  # Fatal export failure
            }
        ),
        # REPAIRED: successfully repaired by flagship model (terminal unless Bible inconsistency or human PE revision)
        BlockStatus.REPAIRED: frozenset(
            {
                BlockStatus.REPAIRED,  # idempotent
                BlockStatus.MTQE_PASSED,  # Human review approved after inspection
                BlockStatus.REPAIR_PENDING,  # Consistency stage: Bible violation re-opened for repair
                BlockStatus.NEEDS_HUMAN,  # Consistency stage: human review needed
                BlockStatus.BLOCKED_HUMAN,  # Critical compliance audit escalation
                BlockStatus.FAILED,  # Fatal export failure
            }
        ),
        # NEEDS_HUMAN: MQM major defect, shippable but queued for human PE (terminal unless human PE acts)
        BlockStatus.NEEDS_HUMAN: frozenset(
            {
                BlockStatus.NEEDS_HUMAN,  # idempotent
                BlockStatus.MTQE_PASSED,  # Human review approved as-is
                BlockStatus.REPAIRED,  # Human post-edited text saved
                BlockStatus.REPAIR_PENDING,  # Human requested re-repair by LLM
                BlockStatus.BLOCKED_HUMAN,  # Escalated to critical blocker
                BlockStatus.FAILED,  # Rejected by human
            }
        ),
        # BLOCKED_HUMAN: MQM critical defect, MUST NOT ship machine output (terminal unless human PE acts)
        BlockStatus.BLOCKED_HUMAN: frozenset(
            {
                BlockStatus.BLOCKED_HUMAN,  # idempotent
                BlockStatus.MTQE_PASSED,  # Human explicit override / approved
                BlockStatus.REPAIRED,  # Human post-edited text saved
                BlockStatus.NEEDS_HUMAN,  # Unblocked to review
                BlockStatus.FAILED,  # Rejected by human
            }
        ),
        # FAILED: unrecoverable failure (terminal)
        BlockStatus.FAILED: frozenset(
            {
                BlockStatus.FAILED,  # idempotent
                # Note: With allow_override=True, FAILED can transition to REPAIR_PENDING (triage) or PENDING (reset)
            }
        ),
    }

    predecessors: dict[BlockStatus, frozenset[BlockStatus]] = {
        target: frozenset(src for src, dsts in transitions.items() if target in dsts)
        for target in BlockStatus
    }

    return transitions, predecessors


_TRANSITIONS_CACHE: dict[BlockStatus, frozenset[BlockStatus]] | None = None
_PREDECESSORS_CACHE: dict[BlockStatus, frozenset[BlockStatus]] | None = None


def get_valid_transitions() -> dict[BlockStatus, frozenset[BlockStatus]]:
    """Return the mapping of valid transitions for each BlockStatus."""
    global _TRANSITIONS_CACHE, _PREDECESSORS_CACHE
    if _TRANSITIONS_CACHE is None:
        _TRANSITIONS_CACHE, _PREDECESSORS_CACHE = _init_transitions()
    return _TRANSITIONS_CACHE


def get_allowed_predecessors() -> dict[BlockStatus, frozenset[BlockStatus]]:
    """Return the mapping of valid predecessor statuses for each BlockStatus."""
    global _TRANSITIONS_CACHE, _PREDECESSORS_CACHE
    if _PREDECESSORS_CACHE is None:
        _TRANSITIONS_CACHE, _PREDECESSORS_CACHE = _init_transitions()
    return _PREDECESSORS_CACHE


def can_transition(
    from_status: BlockStatus | str,
    to_status: BlockStatus | str,
    *,
    allow_override: bool = False,
) -> bool:
    """Return True if transitioning from `from_status` to `to_status` is permitted."""
    from ubt.core.ir.models import BlockStatus

    try:
        src = from_status if isinstance(from_status, BlockStatus) else BlockStatus(str(from_status))
        dst = to_status if isinstance(to_status, BlockStatus) else BlockStatus(str(to_status))
    except ValueError:
        return False

    if src == dst:
        return True

    if allow_override:
        return True

    transitions = get_valid_transitions()
    allowed = transitions.get(src)
    if allowed is None:
        return False
    return dst in allowed


def validate_status_transition(
    from_status: BlockStatus | str,
    to_status: BlockStatus | str,
    *,
    block_id: str | None = None,
    allow_override: bool = False,
) -> None:
    """Validate a status transition, raising InvalidBlockStatusTransitionError if forbidden."""
    from ubt.core.ir.models import BlockStatus

    try:
        src = from_status if isinstance(from_status, BlockStatus) else BlockStatus(str(from_status))
    except ValueError as exc:
        raise InvalidBlockStatusTransitionError(
            str(from_status),
            str(to_status),
            block_id=block_id,
            reason=f"Unknown source status: {from_status!r}",
        ) from exc

    try:
        dst = to_status if isinstance(to_status, BlockStatus) else BlockStatus(str(to_status))
    except ValueError as exc:
        raise InvalidBlockStatusTransitionError(
            str(from_status),
            str(to_status),
            block_id=block_id,
            reason=f"Unknown destination status: {to_status!r}",
        ) from exc

    if not can_transition(src, dst, allow_override=allow_override):
        raise InvalidBlockStatusTransitionError(
            src.value,
            dst.value,
            block_id=block_id,
            reason=f"Transition from '{src.value}' to '{dst.value}' is forbidden in the lifecycle state machine",
        )
