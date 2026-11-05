from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ubt.core.engine.stages.ingest import _fuse_chapter_boundaries
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(block_id: str, text: str, page: int) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text=text,
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=page),
        )
    )


class _FakeLedger:
    def __init__(self, blocks: list[IRBlock]) -> None:
        self.blocks = blocks
        self.merged: list[dict[str, Any]] = []

    def get_all_blocks(self, _job_id: str) -> list[IRBlock]:
        return list(self.blocks)

    def apply_continuation_merge(self, survivor_id: str, absorbed_id: str, **kwargs: Any) -> int:
        self.merged.append({"survivor": survivor_id, "absorbed": absorbed_id, **kwargs})
        return 1


def test_a_paragraph_split_at_the_page_chunk_boundary_is_fused() -> None:
    ledger = _FakeLedger(
        [
            _block("c1#b1", "This sentence continues", page=1),
            _block("c2#b1", "onto the next page.", page=2),
        ]
    )

    count = asyncio.run(_fuse_chapter_boundaries(ledger, "job", ["c1", "c2"]))  # type: ignore[arg-type]

    assert count == 1
    assert ledger.merged[0]["survivor"] == "c1#b1"
    assert ledger.merged[0]["absorbed"] == "c2#b1"
    assert "This sentence continues" in ledger.merged[0]["source_text"]
    assert "onto the next page." in ledger.merged[0]["source_text"]
