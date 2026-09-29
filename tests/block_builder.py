"""Canonical IRBlock test factory for unit and integration test suites."""

from __future__ import annotations

from typing import Any

from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    StyleMeta,
)


def make_test_block(
    id: str = "b1",
    *,
    flow_id: FlowID = FlowID.MAIN_STORY,
    spine_index: int = 1,
    block_type: BlockType = BlockType.NARRATIVE,
    source_text: str = "Test source sentence.",
    draft_text: str | None = None,
    target_text: str | None = None,
    status: BlockStatus = BlockStatus.PENDING,
    mtqe_score: float | None = None,
    bbox: BoundingBox | None = None,
    style: StyleMeta | None = None,
    **kwargs: Any,
) -> IRBlock:
    """Construct an IRBlock with sensible inert defaults for testing."""
    return IRBlock(
        id=id,
        flow_id=flow_id,
        spine_index=spine_index,
        block_type=block_type,
        source_text=source_text,
        draft_text=draft_text,
        target_text=target_text,
        status=status,
        mtqe_score=mtqe_score,
        bbox=bbox,
        style=style,
        **kwargs,
    )
