from __future__ import annotations

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.stages.draft import _DraftInputs, chunk_by_section
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(bid: str, block_type: BlockType) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=bid, spine_index=1, block_type=block_type, source_text="Some source text here."
        )
    )


def _prepared(*types: BlockType) -> list[tuple[IRBlock, _DraftInputs]]:
    return [
        (
            _block(f"b{i}", t),
            _DraftInputs(
                masked_source="x",
                code_map=None,
                cite_map=None,
                math_map=None,
                glossary_table="",
                neighbor_ctx="",
                macro_ctx="",
                few_shot_reference="",
            ),
        )
        for i, t in enumerate(types)
    ]


def test_macro_chunk_size_defaults_to_the_chapter_level_path() -> None:
    assert UBTConfig().macro_chunk_size > 1


def test_narrative_blocks_pack_into_one_chunk() -> None:
    prepared = _prepared(*(BlockType.NARRATIVE for _ in range(6)))
    chunks = chunk_by_section(prepared, chunk_size=8)
    assert len(chunks) == 1
    assert len(chunks[0]) == 6
