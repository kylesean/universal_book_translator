"""The export-time NEEDS_HUMAN gate: unverified machine output must not ship
as a finished book with nothing marking it (fail-closed default; an explicit
allow flag turns the block list into the report's audit trail)."""

from __future__ import annotations

import pytest

from ubt.core.engine.stages.export import _check_needs_human_gate
from ubt.core.exceptions import IntegrityViolationError
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(block_id: str, status: BlockStatus) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Sentence.",
            bbox=BoundingBox(page=1, x0=0, y0=0, x1=100, y1=10),
        ),
        target_text="句子。",
        status=status,
    )


def test_export_refuses_needs_human_blocks_by_default() -> None:
    blocks = [
        _block("b001", BlockStatus.MTQE_PASSED),
        _block("b002", BlockStatus.NEEDS_HUMAN),
    ]
    with pytest.raises(IntegrityViolationError) as exc:
        _check_needs_human_gate("job", blocks, allow=False)
    message = str(exc.value)
    assert "b002" in message
    assert "1 block(s)" in message
    assert "--allow-needs-human" in message


def test_export_allow_flag_ships_needs_human_blocks_knowingly() -> None:
    blocks = [
        _block("b001", BlockStatus.MTQE_PASSED),
        _block("b002", BlockStatus.NEEDS_HUMAN),
    ]
    # Explicit operator authorization: no raise, no mutation.
    _check_needs_human_gate("job", blocks, allow=True)
    assert [b.status for b in blocks] == [BlockStatus.MTQE_PASSED, BlockStatus.NEEDS_HUMAN]


def test_export_gate_is_silent_without_a_needs_human_queue() -> None:
    blocks = [
        _block("b001", BlockStatus.MTQE_PASSED),
        _block("b002", BlockStatus.REPAIRED),
        _block("b003", BlockStatus.BLOCKED_HUMAN),
    ]
    # BLOCKED_HUMAN is a different queue with its own handling (source renders
    # behind a <mark> wrap); this gate covers the *unmarked* NEEDS_HUMAN ship.
    _check_needs_human_gate("job", blocks, allow=False)
