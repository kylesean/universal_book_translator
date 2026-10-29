"""Tests for IRBlock lifecycle state machine and transition guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import InvalidBlockStatusTransitionError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BoundingBox,
    ChapterIR,
    IRBlock,
    make_element,
)
from ubt.core.ir.state_machine import (
    can_transition,
    validate_status_transition,
)

pytestmark = pytest.mark.fast


def _make_block(block_id: str = "ch1#1", status: BlockStatus = BlockStatus.PENDING) -> IRBlock:
    el = make_element(
        id=block_id,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Test source text.",
        bbox=BoundingBox(page=1, x0=10, y0=20, x1=100, y1=200),
    )
    return IRBlock(element=el, status=status)


def test_idempotent_self_transitions_allowed_for_all_statuses() -> None:
    for status in BlockStatus:
        assert can_transition(status, status) is True
        validate_status_transition(status, status)


def test_valid_lifecycle_transitions() -> None:
    # Standard drafting flow
    assert can_transition(BlockStatus.PENDING, BlockStatus.DRAFTED) is True
    # Fast path: skip_translate or exact TM hit directly passes MTQE
    assert can_transition(BlockStatus.PENDING, BlockStatus.MTQE_PASSED) is True
    # Tag restoration glitch flagged during draft
    assert can_transition(BlockStatus.PENDING, BlockStatus.REPAIR_PENDING) is True
    # Drafting fatal failure
    assert can_transition(BlockStatus.PENDING, BlockStatus.FAILED) is True

    # QE audit flow
    assert can_transition(BlockStatus.DRAFTED, BlockStatus.MTQE_PASSED) is True
    assert can_transition(BlockStatus.DRAFTED, BlockStatus.REPAIR_PENDING) is True
    assert can_transition(BlockStatus.DRAFTED, BlockStatus.NEEDS_HUMAN) is True
    assert can_transition(BlockStatus.DRAFTED, BlockStatus.BLOCKED_HUMAN) is True
    assert can_transition(BlockStatus.DRAFTED, BlockStatus.FAILED) is True

    # Repair flow
    assert can_transition(BlockStatus.REPAIR_PENDING, BlockStatus.REPAIRED) is True
    assert can_transition(BlockStatus.REPAIR_PENDING, BlockStatus.MTQE_PASSED) is True
    assert can_transition(BlockStatus.REPAIR_PENDING, BlockStatus.NEEDS_HUMAN) is True
    assert can_transition(BlockStatus.REPAIR_PENDING, BlockStatus.BLOCKED_HUMAN) is True
    assert can_transition(BlockStatus.REPAIR_PENDING, BlockStatus.FAILED) is True

    # Consistency flow: Bible violation re-opens passed / repaired block
    assert can_transition(BlockStatus.MTQE_PASSED, BlockStatus.REPAIR_PENDING) is True
    assert can_transition(BlockStatus.REPAIRED, BlockStatus.REPAIR_PENDING) is True

    # Human PE review flow
    assert can_transition(BlockStatus.NEEDS_HUMAN, BlockStatus.MTQE_PASSED) is True
    assert can_transition(BlockStatus.NEEDS_HUMAN, BlockStatus.REPAIRED) is True
    assert can_transition(BlockStatus.BLOCKED_HUMAN, BlockStatus.REPAIRED) is True
    assert can_transition(BlockStatus.BLOCKED_HUMAN, BlockStatus.MTQE_PASSED) is True


def test_forbidden_transitions_without_override() -> None:
    # Cannot jump from PENDING straight to REPAIRED without drafting or repair
    assert can_transition(BlockStatus.PENDING, BlockStatus.REPAIRED) is False
    with pytest.raises(InvalidBlockStatusTransitionError, match="forbidden"):
        validate_status_transition(BlockStatus.PENDING, BlockStatus.REPAIRED)

    # Cannot jump from PENDING to human triage
    assert can_transition(BlockStatus.PENDING, BlockStatus.NEEDS_HUMAN) is False
    assert can_transition(BlockStatus.PENDING, BlockStatus.BLOCKED_HUMAN) is False

    # Cannot downgrade finalized states to DRAFTED
    assert can_transition(BlockStatus.MTQE_PASSED, BlockStatus.DRAFTED) is False
    assert can_transition(BlockStatus.REPAIRED, BlockStatus.DRAFTED) is False
    assert can_transition(BlockStatus.FAILED, BlockStatus.DRAFTED) is False

    # FAILED cannot transition without explicit override
    assert can_transition(BlockStatus.FAILED, BlockStatus.REPAIR_PENDING) is False
    with pytest.raises(InvalidBlockStatusTransitionError):
        validate_status_transition(BlockStatus.FAILED, BlockStatus.REPAIR_PENDING)


def test_allow_override_permits_recovery_transitions() -> None:
    # Triage paid escalated repair over a failed row
    assert (
        can_transition(BlockStatus.FAILED, BlockStatus.REPAIR_PENDING, allow_override=True) is True
    )
    validate_status_transition(BlockStatus.FAILED, BlockStatus.REPAIR_PENDING, allow_override=True)

    # Administrative reset back to pending
    assert can_transition(BlockStatus.FAILED, BlockStatus.PENDING, allow_override=True) is True
    assert can_transition(BlockStatus.MTQE_PASSED, BlockStatus.PENDING, allow_override=True) is True


def test_unknown_status_raises_error() -> None:
    with pytest.raises(InvalidBlockStatusTransitionError, match="Unknown destination"):
        validate_status_transition(BlockStatus.PENDING, "non_existent_status")
    with pytest.raises(InvalidBlockStatusTransitionError, match="Unknown source"):
        validate_status_transition("invalid_source", BlockStatus.DRAFTED)


def test_ir_block_transition_to_method() -> None:
    block = _make_block(status=BlockStatus.PENDING)

    # Valid transition
    block.transition_to(BlockStatus.DRAFTED)
    assert block.status == BlockStatus.DRAFTED

    # String input coerced to BlockStatus. Read the value back through the
    # enum constructor: mypy narrows ``block.status`` to the literal
    # ``DRAFTED`` after the assertion above, so a direct comparison reads as
    # non-overlapping. The round-trip keeps the check meaningful.
    block.transition_to("mtqe_passed")
    assert BlockStatus(block.status) is BlockStatus.MTQE_PASSED

    # Forbidden transition raises and leaves status intact.
    #
    # The comparison is written against ``BlockStatus(block.status)`` rather
    # than the attribute directly: after the assertion above mypy narrows
    # ``block.status`` to the literal ``MTQE_PASSED``, so re-comparing it with
    # ``MTQE_PASSED`` reads as a tautology and mypy reports a non-overlapping
    # equality. Reading the value back through the enum keeps the assertion
    # meaningful -- it fails if the status actually moved.
    with pytest.raises(InvalidBlockStatusTransitionError):
        block.transition_to(BlockStatus.DRAFTED)
    assert BlockStatus(block.status) is BlockStatus.MTQE_PASSED

    # Override allows administrative reset (same enum round-trip as above)
    block.transition_to(BlockStatus.PENDING, allow_override=True)
    assert BlockStatus(block.status) is BlockStatus.PENDING


def test_ir_block_model_copy_status_guard() -> None:
    block = _make_block(status=BlockStatus.PENDING)

    # Valid model_copy
    copied = block.model_copy(update={"status": BlockStatus.DRAFTED})
    assert copied.status == BlockStatus.DRAFTED

    # Forbidden model_copy raises
    with pytest.raises(InvalidBlockStatusTransitionError):
        block.model_copy(update={"status": BlockStatus.REPAIRED})

    # model_copy with allow_override succeeds
    copied_override = block.model_copy(
        update={"status": BlockStatus.REPAIRED, "allow_override": True}
    )
    assert copied_override.status == BlockStatus.REPAIRED


def test_ledger_checkpoint_state_machine_guard(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "test-job"
    ledger.init_job_placeholder(job_id, "/tmp/in.pdf", "zh")

    # Seed block in PENDING
    ch = ChapterIR(
        doc_id="doc",
        chapter_id="ch1",
        title="Ch 1",
        spine_index=1,
        blocks=[_make_block("ch1#1", status=BlockStatus.PENDING)],
    )
    ledger.append_chapter(job_id, ch)

    # 1. PENDING cannot jump directly to REPAIRED (predecessor check rejects it)
    updated = ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.REPAIRED}],
        job_id=job_id,
    )
    assert updated == 0
    reloaded = ledger.get_block("ch1#1", job_id=job_id)
    # A ``None`` here means the block vanished, which is a different
    # failure from the wrong status; asserting presence keeps the two apart.
    assert reloaded is not None
    assert reloaded.status == BlockStatus.PENDING

    # 2. In strict mode, invalid transition raises InvalidBlockStatusTransitionError
    with pytest.raises(InvalidBlockStatusTransitionError, match="Predecessor condition not met"):
        ledger.save_checkpoints_batch(
            [{"block_id": "ch1#1", "status": BlockStatus.REPAIRED}],
            job_id=job_id,
            strict=True,
        )

    # 3. Valid transition to DRAFTED succeeds
    updated = ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED, "draft_text": "Drafted"}],
        job_id=job_id,
    )
    assert updated == 1
    reloaded = ledger.get_block("ch1#1", job_id=job_id)
    # A ``None`` here means the block vanished, which is a different
    # failure from the wrong status; asserting presence keeps the two apart.
    assert reloaded is not None
    assert reloaded.status == BlockStatus.DRAFTED

    # 4. Valid transition from DRAFTED to MTQE_PASSED succeeds
    updated = ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.MTQE_PASSED, "target_text": "Final"}],
        job_id=job_id,
    )
    assert updated == 1
    reloaded = ledger.get_block("ch1#1", job_id=job_id)
    # A ``None`` here means the block vanished, which is a different
    # failure from the wrong status; asserting presence keeps the two apart.
    assert reloaded is not None
    assert reloaded.status == BlockStatus.MTQE_PASSED

    # 5. MTQE_PASSED cannot be demoted back to DRAFTED without override
    updated = ledger.save_checkpoints_batch(
        [{"block_id": "ch1#1", "status": BlockStatus.DRAFTED}],
        job_id=job_id,
    )
    assert updated == 0
    reloaded = ledger.get_block("ch1#1", job_id=job_id)
    # A ``None`` here means the block vanished, which is a different
    # failure from the wrong status; asserting presence keeps the two apart.
    assert reloaded is not None
    assert reloaded.status == BlockStatus.MTQE_PASSED

    # 6. Invalid status string raises InvalidBlockStatusTransitionError
    with pytest.raises(InvalidBlockStatusTransitionError, match="Unknown target block status"):
        ledger.save_checkpoints_batch(
            [{"block_id": "ch1#1", "status": "corrupted_garbage_status"}],
            job_id=job_id,
        )
